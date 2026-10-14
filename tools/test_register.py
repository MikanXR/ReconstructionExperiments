"""Synthetic check of the frame-to-model loop: ray casting a fused volume
and registering a displaced frame back onto it.

    python tools/test_register.py

A room corner (floor and two walls) is rendered as depth from a few poses,
fused, then ray cast from one of those poses: the cast depth must match the
rendered depth to well under a voxel. A frame rendered from a pose displaced
by a few centimetres and a degree, handed to the registration with the
undisplaced pose as the prior, must come back within a few millimetres of
the truth. A frame that sees only the floor must correct its height and
leave its unobservable in-plane offset at the prior, which is what the prior
term is for. Exits non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.fusion import DenseTSDF  # noqa: E402
from recon.register import register_frame, rotation_angle_deg  # noqa: E402

WIDTH, HEIGHT = 256, 192
K = np.array([[210.0, 0, 128.0], [0, 210.0, 96.0], [0, 0, 1.0]])


PLANES = ((0, 2.0), (2, 2.0), (1, -1.0))   # walls x = 2 and z = 2, floor y = -1
FLOOR = ((1, -1.0),)


def render_corner(T_WC: np.ndarray, planes=PLANES) -> np.ndarray:
    """Depth of axis-aligned planes seen from a pose, OpenCV camera axes."""
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    dirs_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    dirs_w = dirs_c @ T_WC[:3, :3].T
    origin = T_WC[:3, 3]
    best = np.full(u.shape, np.inf)
    for axis, value in planes:
        d = dirs_w[..., axis]
        t = np.where(np.abs(d) > 1e-9, (value - origin[axis]) / np.where(np.abs(d) > 1e-9, d, 1), np.inf)
        t = np.where(t > 0.05, t, np.inf)
        best = np.minimum(best, t)
    depth = np.where(np.isfinite(best), best, 0.0)
    depth[depth > 4.5] = 0.0
    return depth


def look_from(position, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    T = np.eye(4)
    T[:3, :3] = Ry @ Rx
    T[:3, 3] = position
    return T


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["fusion"]
    cfg["voxel_size"] = 0.02
    cfg["sdf_trunc"] = 0.08
    volume = DenseTSDF(np.array([-1.5, -1.5, -1.5]), np.array([2.5, 1.5, 2.5]), cfg, "cpu")

    poses = [look_from([0.0, 0.0, 0.0], yaw, pitch) for yaw in (20, 35, 50) for pitch in (10, 25)]
    for T in poses:
        depth = render_corner(T)
        volume.integrate(depth, (depth > 0).astype(np.float32), K, T)

    # Ray cast against the truth from a fused pose.
    T0 = poses[2]
    truth = render_corner(T0)
    points, normals, hit = volume.raycast(K, T0, WIDTH, HEIGHT, normal_stencil=cfg["frame_to_model"]["normal_stencil_voxels"])
    cast_depth = (points - T0[:3, 3]) @ T0[:3, 2]
    ok = hit & (truth > 0)
    err = np.abs(cast_depth[ok] - truth[ok])
    print(f"raycast: {hit.mean() * 100:.0f}% hit, depth error median {np.median(err) * 1000:.1f} mm, "
          f"p95 {np.percentile(err, 95) * 1000:.1f} mm")
    failed = np.median(err) > 0.5 * cfg["voxel_size"]

    # Register a displaced frame with the undisplaced pose as the prior.
    rng = np.random.default_rng(1)
    displaced = look_from(T0[:3, 3] + np.array([0.03, -0.02, 0.04]), 36.5, 11.0)
    frame = render_corner(displaced)
    frame += rng.normal(0, 0.003, frame.shape) * (frame > 0)
    reg_cfg = {**cfg["frame_to_model"], "depth_min": cfg["depth_min"], "depth_max": cfg["depth_max"]}
    result = register_frame(frame, (frame > 0).astype(np.float32), K, T0, points, normals, hit, T0, reg_cfg)
    residual = np.linalg.inv(displaced) @ result.T_WC
    t_err = np.linalg.norm(residual[:3, 3]) * 1000
    r_err = rotation_angle_deg(residual[:3, :3])
    start = np.linalg.inv(displaced) @ T0
    print(f"register: accepted {result.accepted}, {result.iterations} iterations, {result.correspondences} correspondences, "
          f"rms {result.rms_before_m * 1000:.1f} -> {result.rms_after_m * 1000:.1f} mm; "
          f"pose error {np.linalg.norm(start[:3, 3]) * 1000:.0f} mm {rotation_angle_deg(start[:3, :3]):.2f} deg "
          f"-> {t_err:.1f} mm {r_err:.3f} deg")
    failed |= (not result.accepted) or t_err > 8.0 or r_err > 0.3

    # Floor only: height is observable, the in-plane offset is not.
    floor_volume = DenseTSDF(np.array([-1.5, -1.5, -1.5]), np.array([2.5, 1.5, 2.5]), cfg, "cpu")
    T_floor = look_from([0.0, 0.0, 0.0], 35, 45)
    for T in (T_floor, look_from([0.1, 0.0, 0.05], 30, 50)):
        depth = render_corner(T, FLOOR)
        floor_volume.integrate(depth, (depth > 0).astype(np.float32), K, T)
    f_points, f_normals, f_hit = floor_volume.raycast(K, T_floor, WIDTH, HEIGHT, normal_stencil=reg_cfg["normal_stencil_voxels"])
    truth_floor = look_from(T_floor[:3, 3] + np.array([0.03, 0.02, 0.0]), 35, 45)   # 3 cm along the floor, 2 cm up
    frame = render_corner(truth_floor, FLOOR)
    frame += rng.normal(0, 0.003, frame.shape) * (frame > 0)
    result = register_frame(frame, (frame > 0).astype(np.float32), K, T_floor, f_points, f_normals, f_hit, T_floor, reg_cfg)
    offset = result.T_WC[:3, 3] - T_floor[:3, 3]
    print(f"floor only: accepted {result.accepted}, {result.correspondences} correspondences, "
          f"solved offset x {offset[0] * 1000:.1f} mm (unobservable, truth 30), y {offset[1] * 1000:.1f} mm (truth 20), "
          f"z {offset[2] * 1000:.1f} mm (unobservable, truth 0), rotation {result.rotation_deg:.3f} deg")
    failed |= (not result.accepted) or abs(offset[1] - 0.02) > 0.004 or abs(offset[0]) > 0.004 or abs(offset[2]) > 0.004
    failed |= result.rotation_deg > 0.15
    print("FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
