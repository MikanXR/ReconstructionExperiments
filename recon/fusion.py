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


@dataclass
class FusionResult:
    mesh_raw: o3d.geometry.TriangleMesh
    mesh: o3d.geometry.TriangleMesh
    stats: dict


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


def cull_flying_pixels(depth: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mask out pixels whose depth jumps by more than `threshold` (relative)
    to any 4-neighbour. These sit on silhouettes where the sensor or the model
    interpolates between foreground and background."""
    valid = torch.isfinite(depth) & (depth > 0)
    d = torch.where(valid, depth, torch.zeros_like(depth))
    keep = valid.clone()
    for dy, dx in ((0, 1), (1, 0)):
        a = d[dy:, dx:]
        b = d[: d.shape[0] - dy, : d.shape[1] - dx]
        va = valid[dy:, dx:] & valid[: d.shape[0] - dy, : d.shape[1] - dx]
        rel = (a - b).abs() / torch.maximum(torch.minimum(a, b), torch.full_like(a, 1e-6))
        bad = va & (rel > threshold)
        keep[dy:, dx:] &= ~bad
        keep[: d.shape[0] - dy, : d.shape[1] - dx] &= ~bad
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

    def integrate(self, depth: np.ndarray, weight: np.ndarray, K: np.ndarray, T_WC: np.ndarray) -> None:
        cfg = self.cfg
        dev = self.device
        d = torch.as_tensor(depth, device=dev, dtype=torch.float32)
        w = torch.as_tensor(weight, device=dev, dtype=torch.float32)
        keep = cull_flying_pixels(d, cfg["gradient_cull"]) & (w >= cfg["min_weight"])
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


def fuse(cache: DepthCache, capture: Capture, train_ids, cfg: dict, device: str | None = None) -> FusionResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    lo, hi = training_bounds(capture, train_ids, cfg)
    volume = DenseTSDF(lo, hi, cfg, device)
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
    return FusionResult(mesh_raw, mesh, stats)
