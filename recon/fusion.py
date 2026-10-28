"""Dense TSDF fusion on the GPU, identical for every method.

A dense float grid over the training LiDAR's bounding box is small enough at
room scale (tens of millions of voxels, a few hundred MB) that voxel hashing is
not worth its complexity, and a dense grid gives per-pixel weights and a
gradient cull for free, which Open3D's integrators do not expose. Every voxel
update is one gather from the depth image, so a frame integrates in
milliseconds on the GPU.

Per frame: cull flying pixels at depth discontinuities, project voxel centres,
read depth and weight, compute the truncated signed distance, and fold it into
the running weighted average. Marching cubes (scikit-image) extracts the zero
crossing; Open3D cleans the mesh.

With fusion.frame_to_model enabled the loop becomes KinectFusion's: before a
frame is integrated the volume so far is ray cast from the frame's predicted
pose and the frame is registered against that model map (recon/register.py).
The prediction is the ARKit pose carried forward by the previous frame's
correction, so the refinement tracks accumulated drift rather than solving it
afresh each frame. The refined pose of every frame in the take, held-out ones
taking their nearest training frame's correction, is returned beside the mesh
so the evaluation scores the mesh in the frame it was built in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch
from skimage.measure import marching_cubes

from .capture import Capture
from .depthcache import DepthCache
from .register import register_frame, rotation_angle_deg


@dataclass
class FusionResult:
    mesh_raw: o3d.geometry.TriangleMesh
    mesh: o3d.geometry.TriangleMesh
    stats: dict
    # (N,4,4) refined camera-to-world for every frame of the capture when
    # frame-to-model registration ran, else None.
    poses: np.ndarray | None = None


def training_bounds(capture: Capture, train_ids, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of the confidence-2 LiDAR cloud over the training
    frames, padded by bounds_margin. Every method fuses into the same box."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for fid in train_ids[::4]:
        pts = capture.lidar_points(fid, min_confidence=2, depth_max=cfg["depth_max"])
        if len(pts):
            lo = np.minimum(lo, np.percentile(pts, 0.5, axis=0))
            hi = np.maximum(hi, np.percentile(pts, 99.5, axis=0))
    return lo - cfg["bounds_margin"], hi + cfg["bounds_margin"]


def cull_flying_pixels(depth: torch.Tensor, threshold: float, keep_near: bool = False, radius: int = 1) -> torch.Tensor:
    """Mask out pixels whose depth jumps by more than `threshold` (relative)
    to any 4-neighbour. These sit on silhouettes where the sensor or the model
    interpolates between foreground and background. Both pixels of a bad pair
    go, or with keep_near only the farther one, so a near object keeps its
    outermost silhouette pixels. With keep_near and a radius past 1, every pixel
    farther than the nearest valid depth within the radius by the threshold goes
    too: background pixels that close to a near edge would otherwise carve the
    near object's side as free space. Mirrors MikanXR's TsdfVolume::cullFlyingPixels."""
    valid = torch.isfinite(depth) & (depth > 0)
    d = torch.where(valid, depth, torch.zeros_like(depth))
    keep = valid.clone()
    for dy, dx in ((0, 1), (1, 0)):
        a = d[dy:, dx:]
        b = d[: d.shape[0] - dy, : d.shape[1] - dx]
        va = valid[dy:, dx:] & valid[: d.shape[0] - dy, : d.shape[1] - dx]
        rel = (a - b).abs() / torch.maximum(torch.minimum(a, b), torch.full_like(a, 1e-6))
        bad = va & (rel > threshold)
        if keep_near:
            keep[dy:, dx:] &= ~(bad & (a > b))
            keep[: d.shape[0] - dy, : d.shape[1] - dx] &= ~(bad & (b > a))
        else:
            keep[dy:, dx:] &= ~bad
            keep[: d.shape[0] - dy, : d.shape[1] - dx] &= ~bad
    if keep_near and radius > 1:
        far = torch.where(valid, d, torch.full_like(d, float("inf")))
        nearest = -torch.nn.functional.max_pool2d(-far[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
        rel = (d - nearest).abs() / torch.maximum(torch.minimum(d, nearest), torch.full_like(d, 1e-6))
        keep &= ~(valid & (rel > threshold))
    return keep


class DenseTSDF:
    def __init__(self, lo: np.ndarray, hi: np.ndarray, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        voxel = float(cfg["voxel_size"])
        dims = np.ceil((hi - lo) / voxel).astype(int) + 1
        while int(np.prod(dims)) > int(cfg["max_voxels"]):
            voxel *= 1.1
            dims = np.ceil((hi - lo) / voxel).astype(int) + 1
        self.voxel = voxel
        self.origin = lo.astype(np.float32)
        self.dims = dims
        self.trunc = float(cfg["sdf_trunc"])
        n = int(np.prod(dims))
        self.tsdf = torch.ones(n, device=device)
        self.weight = torch.zeros(n, device=device)
        # Voxel centre world coordinates, built once.
        zi, yi, xi = torch.meshgrid(
            torch.arange(dims[2], device=device),
            torch.arange(dims[1], device=device),
            torch.arange(dims[0], device=device),
            indexing="ij")
        idx = torch.stack([xi, yi, zi], dim=-1).reshape(-1, 3).float()
        self.points = idx * voxel + torch.as_tensor(self.origin, device=device)
        self.frames_integrated = 0

    # ---- sampling and ray casting -----------------------------------------

    def _sample(self, points_w: torch.Tensor) -> torch.Tensor:
        """Trilinear TSDF at world points (M,3), NaN wherever any of the eight
        surrounding voxels is unobserved or the point is outside the grid.
        A sample at the rim of an observed patch would otherwise blend the
        observed band with the unobserved default of +1 and read as a zero
        crossing with an in-plane normal: a fake wall around every patch,
        which registration then slides toward."""
        dims = torch.as_tensor(self.dims, device=self.device, dtype=torch.float32)
        idx = (points_w - torch.as_tensor(self.origin, device=self.device)) / self.voxel
        norm = idx / (dims - 1) * 2 - 1
        grid = norm.reshape(1, 1, 1, -1, 3)
        shape = (1, 1, int(self.dims[2]), int(self.dims[1]), int(self.dims[0]))
        masked = torch.where(self.weight > 0, self.tsdf, torch.full_like(self.tsdf, float("nan"))).reshape(shape)
        t = torch.nn.functional.grid_sample(masked, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        t = t.reshape(-1)
        outside = (norm < -1).any(dim=1) | (norm > 1).any(dim=1)
        return torch.where(outside, torch.full_like(t, float("nan")), t)

    def raycast(self, K: np.ndarray, T_WC: np.ndarray, width: int, height: int,
                step_fraction: float = 0.5, normal_stencil: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render the model from a camera: world-space hit points (H,W,3),
        normals (H,W,3) pointing out of the surface, and a hit mask. Rays
        march in steps of step_fraction * truncation from depth_min to
        depth_max and stop at the first observed positive-to-negative zero
        crossing, refined by linear interpolation of the two samples.

        Normals are the TSDF gradient by central differences normal_stencil
        voxels either side of the hit. One voxel reads the interpolation's
        own kinks as 2 to 3 percent of normal noise, which registration then
        mistakes for a constraint along a plane; two voxels bring that under
        one percent without reaching past the truncation band."""
        cfg = self.cfg
        dev = self.device
        u, v = torch.meshgrid(torch.arange(width, device=dev, dtype=torch.float32),
                              torch.arange(height, device=dev, dtype=torch.float32), indexing="xy")
        dirs_c = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], dim=-1).reshape(-1, 3)
        T = torch.as_tensor(T_WC, device=dev, dtype=torch.float32)
        dirs_w = dirs_c @ T[:3, :3].T
        origin_w = T[:3, 3]
        step = self.trunc * step_fraction
        ts = torch.arange(float(cfg["depth_min"]), float(cfg["depth_max"]), step, device=dev)
        n_rays, n_steps = dirs_w.shape[0], ts.shape[0]
        samples_w = origin_w + dirs_w[:, None, :] * ts[None, :, None]
        tsdf = self._sample(samples_w.reshape(-1, 3)).reshape(n_rays, n_steps)
        observed = torch.isfinite(tsdf)
        tsdf = torch.nan_to_num(tsdf, nan=1.0)
        prev_pos = observed[:, :-1] & (tsdf[:, :-1] > 0)
        cur_neg = observed[:, 1:] & (tsdf[:, 1:] <= 0)
        crossing = prev_pos & cur_neg
        hit = crossing.any(dim=1)
        first = torch.argmax(crossing.int(), dim=1)
        rows = torch.arange(n_rays, device=dev)
        s_prev = tsdf[rows, first]
        s_cur = tsdf[rows, first + 1]
        frac = s_prev / torch.clamp(s_prev - s_cur, min=1e-6)
        t_hit = ts[first] + frac * step
        points_w = origin_w + dirs_w * t_hit[:, None]
        # Normal from the TSDF gradient at the hit, central differences one voxel apart.
        eps = self.voxel * normal_stencil
        grad = []
        for axis in range(3):
            offset = torch.zeros(3, device=dev)
            offset[axis] = eps
            grad.append(self._sample(points_w + offset) - self._sample(points_w - offset))
        normals = torch.stack(grad, dim=-1)
        hit &= torch.isfinite(normals).all(dim=-1)
        normals = torch.nan_to_num(normals, nan=0.0)
        length = normals.norm(dim=-1, keepdim=True)
        hit &= length[:, 0] > 1e-6
        normals = normals / torch.clamp(length, min=1e-6)
        shape = (height, width)
        return (points_w.reshape(*shape, 3).cpu().numpy(),
                normals.reshape(*shape, 3).cpu().numpy(),
                hit.reshape(shape).cpu().numpy())

    def integrate(self, depth: np.ndarray, weight: np.ndarray, K: np.ndarray, T_WC: np.ndarray) -> None:
        cfg = self.cfg
        dev = self.device
        d = torch.as_tensor(depth, device=dev, dtype=torch.float32)
        w = torch.as_tensor(weight, device=dev, dtype=torch.float32)
        keep = cull_flying_pixels(d, cfg["gradient_cull"], cfg.get("gradient_cull_keep_near", False),
                                  cfg.get("gradient_cull_radius", 1)) & (w >= cfg["min_weight"])
        keep &= (d >= cfg["depth_min"]) & (d <= cfg["depth_max"])
        if not bool(keep.any()):
            return
        d = torch.where(keep, d, torch.zeros_like(d))
        w = torch.where(keep, w, torch.zeros_like(w))
        h, wd = d.shape

        T_CW = torch.as_tensor(np.linalg.inv(T_WC), device=dev, dtype=torch.float32)
        Kt = torch.as_tensor(K, device=dev, dtype=torch.float32)
        pc = self.points @ T_CW[:3, :3].T + T_CW[:3, 3]
        z = pc[:, 2]
        front = z > cfg["depth_min"]
        u = (pc[:, 0] / z) * Kt[0, 0] + Kt[0, 2]
        v = (pc[:, 1] / z) * Kt[1, 1] + Kt[1, 2]
        ui = torch.round(u).long()
        vi = torch.round(v).long()
        inside = front & (ui >= 0) & (ui < wd) & (vi >= 0) & (vi < h)
        ui = ui.clamp(0, wd - 1)
        vi = vi.clamp(0, h - 1)
        dz = d[vi, ui]
        wz = w[vi, ui]
        valid = inside & (dz > 0)
        sdf = dz - z
        valid &= sdf > -self.trunc
        if not bool(valid.any()):
            return
        tsdf_new = torch.clamp(sdf / self.trunc, max=1.0)
        W = self.weight
        idx = valid.nonzero(as_tuple=True)[0]
        w_old = W[idx]
        w_new = wz[idx]
        self.tsdf[idx] = (self.tsdf[idx] * w_old + tsdf_new[idx] * w_new) / (w_old + w_new)
        W[idx] = w_old + w_new
        self.frames_integrated += 1

    def extract(self) -> o3d.geometry.TriangleMesh:
        tsdf = self.tsdf.reshape(self.dims[2], self.dims[1], self.dims[0]).cpu().numpy()
        weight = self.weight.reshape(self.dims[2], self.dims[1], self.dims[0]).cpu().numpy()
        observed = weight > 0
        if observed.sum() < 8:
            return o3d.geometry.TriangleMesh()
        # Unobserved voxels hold tsdf=1. Left alone they form a zero crossing
        # against the observed negative band behind every surface, which
        # extracts as a back shell two voxels behind the real one (a thin
        # closed slab per surface). Filling each unobserved voxel with the
        # sign of its nearest observed voxel keeps the band negative behind
        # surfaces and positive in free space, so only real crossings remain.
        # Marching cubes still runs under the observed mask, dilated by one
        # voxel so cubes on the boundary of the observed region are kept.
        from scipy import ndimage

        nearest = ndimage.distance_transform_edt(~observed, return_distances=False, return_indices=True)
        filled = np.where(observed, tsdf, np.sign(tsdf[tuple(nearest)]) + 0.0)
        filled[~observed & (filled == 0)] = 1.0
        mask = ndimage.binary_dilation(observed, iterations=1)
        verts, faces, normals, _ = marching_cubes(filled, level=0.0, mask=mask, allow_degenerate=False)
        verts_world = verts[:, ::-1] * self.voxel + self.origin   # (z,y,x) index -> (x,y,z) world
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts_world.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)))
        mesh.compute_vertex_normals()
        return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_component_triangles: int) -> tuple[o3d.geometry.TriangleMesh, int]:
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    cluster_idx, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_idx = np.asarray(cluster_idx)
    cluster_sizes = np.asarray(cluster_sizes)
    small = cluster_sizes[cluster_idx] < min_component_triangles
    removed = int(small.sum())
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, removed


def refine_poses_frame_to_model(volume: DenseTSDF, cache: DepthCache, capture: Capture, train_ids,
                                cfg: dict) -> tuple[np.ndarray, dict]:
    """Integrate the training frames with frame-to-model registration and
    return the refined pose of every frame in the capture, plus a summary.

    Each training frame's correction is the transform from its ARKit pose to
    its refined pose. The next frame starts from the ARKit pose carried
    forward by that correction, and every frame of the capture, held out or
    not, takes the correction of the nearest integrated training frame by
    index, so a held-out frame is scored where the mesh around it was built.
    A frame whose registration is rejected is skipped rather than integrated
    at a pose the data disagree with: the frames after an ARKit tracking
    jump read that way until a step large enough to absorb it is accepted."""
    f2m = cfg["frame_to_model"]
    reg_cfg = {**f2m, "depth_min": cfg["depth_min"], "depth_max": cfg["depth_max"]}
    train_ids = [int(f) for f in train_ids]
    corrections = {}
    delta = np.eye(4)
    accepted = 0
    rejected = 0
    skipped = 0
    translations = []
    rotations = []
    rms_before = []
    rms_after = []
    per_frame = []
    width, height = capture.depth_size
    for i, fid in enumerate(train_ids):
        if not cache.has(fid):
            raise KeyError(f"{cache.method}: depth cache lacks training frame {fid}")
        depth, weight, K = cache.read(fid)
        T_arkit = capture.pose(fid)
        T_prior = delta @ T_arkit
        T_use = T_prior
        integrate = True
        if i >= int(f2m["warmup_frames"]):
            model_points, model_normals, model_hit = volume.raycast(
                K, T_prior, width, height, normal_stencil=float(f2m["normal_stencil_voxels"]))
            result = register_frame(depth, weight, K, T_prior, model_points, model_normals, model_hit,
                                    T_prior, reg_cfg)
            if result.accepted:
                accepted += 1
                T_use = result.T_WC
                translations.append(result.translation_m)
                rotations.append(result.rotation_deg)
                rms_before.append(result.rms_before_m)
                rms_after.append(result.rms_after_m)
            else:
                rejected += 1
                integrate = bool(f2m.get("integrate_rejected", False))
            per_frame.append({
                "frame": fid, "accepted": bool(result.accepted), "iterations": result.iterations,
                "correspondences": result.correspondences,
                "rms_before_mm": float(result.rms_before_m * 1000), "rms_after_mm": float(result.rms_after_m * 1000),
                "translation_mm": float(result.translation_m * 1000), "rotation_deg": float(result.rotation_deg),
            })
        if not integrate:
            skipped += 1
            continue
        volume.integrate(depth, weight, K, T_use)
        delta = T_use @ np.linalg.inv(T_arkit)
        corrections[fid] = delta

    anchors = np.array(sorted(corrections))
    poses = np.zeros_like(capture.T_WC)
    for j in range(len(capture)):
        fid = int(capture.frame_ids[j])
        nearest = anchors[np.argmin(np.abs(anchors - fid))]
        poses[j] = corrections[int(nearest)] @ capture.T_WC[j]
    final = corrections[int(anchors[-1])]
    summary = {
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "mean_step_translation_mm": float(np.mean(translations) * 1000) if translations else 0.0,
        "mean_step_rotation_deg": float(np.mean(rotations)) if rotations else 0.0,
        "mean_rms_before_mm": float(np.mean(rms_before) * 1000) if rms_before else float("nan"),
        "mean_rms_after_mm": float(np.mean(rms_after) * 1000) if rms_after else float("nan"),
        "final_correction_translation_mm": float(np.linalg.norm(final[:3, 3]) * 1000),
        "final_correction_rotation_deg": rotation_angle_deg(final[:3, :3]),
        "per_frame": per_frame,
    }
    return poses, summary


def fuse(cache: DepthCache, capture: Capture, train_ids, cfg: dict, device: str | None = None) -> FusionResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    lo, hi = training_bounds(capture, train_ids, cfg)
    volume = DenseTSDF(lo, hi, cfg, device)
    poses = None
    f2m_summary = None
    if (cfg.get("frame_to_model") or {}).get("enabled", False):
        poses, f2m_summary = refine_poses_frame_to_model(volume, cache, capture, train_ids, cfg)
    else:
        for fid in train_ids:
            if not cache.has(fid):
                raise KeyError(f"{cache.method}: depth cache lacks training frame {fid}")
            depth, weight, K = cache.read(fid)
            volume.integrate(depth, weight, K, capture.pose(fid))
    t_integrate = time.perf_counter() - t0
    mesh_raw = volume.extract()
    mesh, removed = clean_mesh(mesh_raw, cfg["min_component_triangles"])
    stats = {
        "voxel_size": volume.voxel,
        "grid_dims": [int(x) for x in volume.dims],
        "bounds_lo": [float(x) for x in lo],
        "bounds_hi": [float(x) for x in hi],
        "frames_integrated": volume.frames_integrated,
        "integrate_seconds": t_integrate,
        "fuse_seconds": time.perf_counter() - t0,
        "triangles_raw": len(mesh_raw.triangles),
        "triangles_clean": len(mesh.triangles),
        "triangles_removed_as_islands": removed,
    }
    if f2m_summary is not None:
        stats["frame_to_model"] = f2m_summary
    return FusionResult(mesh_raw, mesh, stats, poses)
