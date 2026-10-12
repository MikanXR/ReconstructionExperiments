"""Held-out frame evaluation.

The candidate mesh is ray cast from each held-out ARKit pose at LiDAR
resolution and compared with that frame's confidence-2 LiDAR depth. Every
number is a comparison against sensor pixels the mesh never saw during fusion.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .capture import Capture, DEPTH_WIDTH, DEPTH_HEIGHT, unproject, transform_points


# ---- ray casting ---------------------------------------------------------------

def make_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    if len(mesh.triangles):
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def camera_rays(K: np.ndarray, T_WC: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """World-space ray origins and unit directions, (H*W,3) each, plus the
    camera-space z component of each direction for depth conversion."""
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dz = dirs[:, 2].copy()
    dirs_w = dirs @ T_WC[:3, :3].T
    origins = np.broadcast_to(T_WC[:3, 3], dirs_w.shape).astype(np.float32)
    return np.concatenate([origins, dirs_w.astype(np.float32)], axis=1), dz


def raycast(scene: o3d.t.geometry.RaycastingScene, K: np.ndarray, T_WC: np.ndarray,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_depth HxW with inf on miss, world normals HxWx3, hit mask HxW)."""
    rays, dz = camera_rays(K, T_WC, width, height)
    out = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = out["t_hit"].numpy()
    hit = np.isfinite(t_hit)
    z = np.where(hit, t_hit * dz, np.inf).reshape(height, width)
    normals = out["primitive_normals"].numpy().reshape(height, width, 3)
    return z, normals, hit.reshape(height, width)


# ---- silhouette edges -----------------------------------------------------------

EDGE_CLIP_PX = 20.0      # an edge with no counterpart within this many pixels counts as this far
EDGE_NEAR_PX = 2.0       # "edge matched" tolerance
EDGE_BORDER_PX = 2       # image border excluded from edge detection


def depth_edges(depth: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    """Pixels where depth jumps relatively by more than `threshold` to a
    4-neighbour that is also valid, or where a valid pixel borders a valid
    pixel with infinite depth (the mesh ends: a silhouette against nothing).
    Only pixels inside `valid`, away from the image border, can be edges."""
    valid = valid.copy()
    valid[:EDGE_BORDER_PX] = valid[-EDGE_BORDER_PX:] = False
    valid[:, :EDGE_BORDER_PX] = valid[:, -EDGE_BORDER_PX:] = False
    d = np.where(np.isfinite(depth), depth, 1e6)
    edges = np.zeros_like(valid)
    for dy, dx in ((0, 1), (1, 0)):
        a = d[dy:, dx:]
        b = d[: d.shape[0] - dy, : d.shape[1] - dx]
        both = valid[dy:, dx:] & valid[: d.shape[0] - dy, : d.shape[1] - dx]
        rel = np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-6)
        jump = both & (rel > threshold)
        edges[dy:, dx:] |= jump
        edges[: d.shape[0] - dy, : d.shape[1] - dx] |= jump
    return edges


def edge_distance(edges_lidar: np.ndarray, edges_mesh: np.ndarray) -> dict:
    """Nearest-pixel distances between the two edge sets, both ways. Means are
    clipped at EDGE_CLIP_PX so a single stray edge pixel cannot dominate a
    frame, and the median and the matched fraction are reported alongside."""
    import cv2

    def dists(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if not src.any():
            return np.zeros(0)
        if not dst.any():
            return np.full(int(src.sum()), EDGE_CLIP_PX)
        dist = cv2.distanceTransform((~dst).astype(np.uint8), cv2.DIST_L2, 5)
        return np.minimum(dist[src], EDGE_CLIP_PX)

    d_lm = dists(edges_lidar, edges_mesh)
    d_ml = dists(edges_mesh, edges_lidar)
    both = np.concatenate([d_lm, d_ml])
    return {
        "lidar_to_mesh_px": float(d_lm.mean()) if len(d_lm) else float("nan"),
        "mesh_to_lidar_px": float(d_ml.mean()) if len(d_ml) else float("nan"),
        "symmetric_px": float(both.mean()) if len(both) else float("nan"),
        "median_px": float(np.median(both)) if len(both) else float("nan"),
        "matched_fraction": float((both <= EDGE_NEAR_PX).mean()) if len(both) else float("nan"),
        "lidar_edge_pixels": int(edges_lidar.sum()),
        "mesh_edge_pixels": int(edges_mesh.sum()),
    }


# ---- per-frame comparison -------------------------------------------------------

def evaluate_frame(scene, capture: Capture, frame_id: int, cfg: dict) -> dict:
    K = capture.intrinsics_depth(frame_id)
    T_WC = capture.pose(frame_id)
    z, _, hit = raycast(scene, K, T_WC, DEPTH_WIDTH, DEPTH_HEIGHT)
    lidar = capture.depth(frame_id)
    conf = capture.confidence(frame_id)
    valid = (conf == 2) & (lidar > 0) & (lidar <= cfg["depth_max"])
    n_valid = int(valid.sum())
    covered = valid & hit & (z <= cfg["depth_max"])
    err = z[covered] - lidar[covered]
    abs_err = np.abs(err)

    edges_lidar = depth_edges(lidar, valid, cfg["edge_gradient"])
    edges_mesh = depth_edges(z, valid, cfg["edge_gradient"])

    out = {
        "frame": int(frame_id),
        "lidar_pixels": n_valid,
        "completeness": float(covered.sum() / max(n_valid, 1)),
        "mae_m": float(abs_err.mean()) if len(abs_err) else float("nan"),
        "rmse_m": float(np.sqrt((err ** 2).mean())) if len(err) else float("nan"),
        "median_abs_m": float(np.median(abs_err)) if len(abs_err) else float("nan"),
        "floater_fraction": float((err < -cfg["floater_margin"]).sum() / max(len(err), 1)),
        "edges": edge_distance(edges_lidar, edges_mesh),
    }
    for t in cfg["near_thresholds"]:
        out[f"within_{int(round(t * 100))}cm"] = float((abs_err <= t).sum() / max(len(abs_err), 1))
    return out


def summarize_frames(frames: list[dict], cfg: dict) -> dict:
    def mean(key):
        vals = [f[key] for f in frames if np.isfinite(f[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {k: mean(k) for k in ["completeness", "mae_m", "rmse_m", "median_abs_m", "floater_fraction"]}
    for t in cfg["near_thresholds"]:
        k = f"within_{int(round(t * 100))}cm"
        summary[k] = mean(k)
    for key, out_key in (("symmetric_px", "edge_symmetric_px"), ("median_px", "edge_median_px"),
                         ("matched_fraction", "edge_matched_fraction")):
        vals = [f["edges"][key] for f in frames if np.isfinite(f["edges"][key])]
        summary[out_key] = float(np.mean(vals)) if vals else float("nan")
    return summary


# ---- whole-mesh metrics ---------------------------------------------------------

def sample_surface(mesh: o3d.geometry.TriangleMesh, n: int) -> np.ndarray:
    if not len(mesh.triangles):
        return np.zeros((0, 3))
    return np.asarray(mesh.sample_points_uniformly(number_of_points=n).points)


def training_lidar_cloud(capture: Capture, train_ids, depth_max: float, voxel: float = 0.01) -> np.ndarray:
    """Every LiDAR point fusion could have used (confidence 1 and 2)."""
    pts = [capture.lidar_points(f, min_confidence=1, depth_max=depth_max) for f in train_ids]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    return np.asarray(pcd.voxel_down_sample(voxel).points)


def unsupported_surface_fraction(mesh, lidar_cloud: np.ndarray, cfg: dict) -> float:
    """Fraction of the mesh surface farther than support_distance from any
    training LiDAR point: geometry the sensor never observed."""
    pts = sample_surface(mesh, cfg["support_samples"])
    if not len(pts) or not len(lidar_cloud):
        return float("nan")
    d, _ = cKDTree(lidar_cloud).query(pts, k=1, workers=-1)
    return float((d > cfg["support_distance"]).mean())


def chamfer_against(mesh, reference, cfg: dict) -> dict:
    """Accuracy (mesh -> reference), completeness (reference -> mesh), chamfer,
    and F-score at fscore_threshold, all point-to-surface distances."""
    n = cfg["chamfer_samples"]
    tau = cfg["fscore_threshold"]
    pts_mesh = sample_surface(mesh, n)
    pts_ref = sample_surface(reference, n)
    if not len(pts_mesh) or not len(pts_ref):
        return {"accuracy_m": float("nan"), "completeness_m": float("nan"), "chamfer_m": float("nan"), "fscore": float("nan")}
    d_acc = make_scene(reference).compute_distance(o3d.core.Tensor(pts_mesh.astype(np.float32))).numpy()
    d_comp = make_scene(mesh).compute_distance(o3d.core.Tensor(pts_ref.astype(np.float32))).numpy()
    precision = float((d_acc <= tau).mean())
    recall = float((d_comp <= tau).mean())
    f = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"accuracy_m": float(d_acc.mean()), "completeness_m": float(d_comp.mean()),
            "chamfer_m": float((d_acc.mean() + d_comp.mean()) / 2), "precision": precision, "recall": recall, "fscore": f}


# ---- entry point ----------------------------------------------------------------

def evaluate(mesh, capture: Capture, train_ids, holdout_ids, cfg: dict,
             baseline_mesh=None) -> dict:
    scene = make_scene(mesh)
    frames = [evaluate_frame(scene, capture, int(f), cfg) for f in holdout_ids]
    summary = summarize_frames(frames, cfg)
    # Edge distances are measured at LiDAR resolution; report them at colour
    # resolution too so they can be read against the video frame.
    summary["edge_symmetric_px_color"] = summary["edge_symmetric_px"] * capture.color_size[0] / DEPTH_WIDTH
    summary["unsupported_surface_fraction"] = unsupported_surface_fraction(
        mesh, training_lidar_cloud(capture, train_ids, cfg["depth_max"]), cfg)
    result = {"summary": summary, "frames": frames}
    if baseline_mesh is not None and len(baseline_mesh.triangles):
        result["vs_baseline"] = chamfer_against(mesh, baseline_mesh, cfg)
    return result
