"""Mesh renders for the visual sheet.

Rendering goes through the ray caster rather than Open3D's offscreen
renderer, so it needs no window, GPU context, or Filament setup. Shading is a
headlight Lambert term on the hit normal, which is enough to judge surface
shape and silhouettes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .capture import Capture, scale_intrinsics
from .evaluate import make_scene, raycast

VIEW_WIDTH = 640
VIEW_HEIGHT = 480


def shade(normals: np.ndarray, hit: np.ndarray, view_dirs: np.ndarray) -> np.ndarray:
    lambert = np.abs(np.einsum("hwc,hwc->hw", normals, -view_dirs))
    gray = (60 + 180 * np.clip(lambert, 0, 1)).astype(np.uint8)
    img = np.stack([gray] * 3, axis=-1)
    img[~hit] = (255, 255, 255)
    return img


def render_view(scene, K: np.ndarray, T_WC: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (shaded RGB image, z depth) from a pinhole camera."""
    z, normals, hit = raycast(scene, K, T_WC, width, height)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_w = dirs @ T_WC[:3, :3].T
    return shade(normals, hit, dirs_w), z


def depth_overlay(frame_rgb: np.ndarray, z: np.ndarray, depth_max: float) -> np.ndarray:
    """Video frame with the rendered depth blended in (red near, blue far),
    the same reading as MikanXR's capture overlay. The colour range spans the
    frame's own 1st to 99th depth percentile so close-range scenes do not
    collapse to one colour."""
    h, w = z.shape
    frame = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
    hit = np.isfinite(z) & (z <= depth_max)
    if hit.any():
        lo, hi = np.percentile(z[hit], [1, 99])
    else:
        lo, hi = 0.0, depth_max
    t = np.clip((np.where(hit, z, hi) - lo) / max(hi - lo, 1e-3), 0, 1)
    colour = cv2.applyColorMap((255 - t * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    out = frame.copy()
    out[hit] = (0.45 * frame[hit] + 0.55 * colour[hit]).astype(np.uint8)
    return out


def render_topdown(scene, lo: np.ndarray, hi: np.ndarray, pixels: int) -> np.ndarray:
    """Orthographic view straight down (-Y), shaded by normal."""
    span = hi - lo
    aspect = span[2] / max(span[0], 1e-6)
    width = pixels
    height = max(int(pixels * aspect), 1)
    xs = np.linspace(lo[0], hi[0], width, dtype=np.float32)
    zs = np.linspace(hi[2], lo[2], height, dtype=np.float32)
    gx, gz = np.meshgrid(xs, zs)
    origins = np.stack([gx, np.full_like(gx, hi[1] + 1.0), gz], axis=-1).reshape(-1, 3)
    dirs = np.tile(np.array([0, -1, 0], dtype=np.float32), (origins.shape[0], 1))
    out = scene.cast_rays(o3d.core.Tensor(np.concatenate([origins, dirs], axis=1)))
    t_hit = out["t_hit"].numpy()
    hit = np.isfinite(t_hit).reshape(height, width)
    normals = out["primitive_normals"].numpy().reshape(height, width, 3)
    view = np.broadcast_to(np.array([0, -1, 0], dtype=np.float32), normals.shape)
    return shade(normals, hit, view)


def render_sheet(mesh, capture: Capture, view_ids, out_dir: Path, bounds: tuple[np.ndarray, np.ndarray],
                 depth_max: float, topdown_pixels: int) -> dict:
    """Write per-view renders and the top-down view. Returns relative paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = make_scene(mesh)
    frames = capture.read_frames(view_ids)
    written = {"views": {}}
    for fid in view_ids:
        K_full = capture.intrinsics_color(fid)
        K = scale_intrinsics(K_full, capture.color_size, (VIEW_WIDTH, VIEW_HEIGHT))
        shaded, z = render_view(scene, K, capture.pose(fid), VIEW_WIDTH, VIEW_HEIGHT)
        frame = frames[int(fid)]
        overlay = depth_overlay(frame, z, depth_max)
        small = cv2.resize(frame, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
        strip = np.concatenate([small, shaded, overlay], axis=1)
        name = f"view_{int(fid):06d}.png"
        cv2.imwrite(str(out_dir / name), strip[..., ::-1])
        written["views"][int(fid)] = name
    top = render_topdown(scene, bounds[0], bounds[1], topdown_pixels)
    cv2.imwrite(str(out_dir / "topdown.png"), top[..., ::-1])
    written["topdown"] = "topdown.png"
    return written
