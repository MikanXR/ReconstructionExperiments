"""Where a fused take's held-out error comes from.

    python tools/take_diagnostics.py <take>.mp4 [--run lidar_tsdf] [--config configs/default.yaml]

Three views of one take and one fused mesh under out/<take>/<run>/mesh.ply:

- error by LiDAR range and by tail, so a scene problem (a surface the fusion
  lost, which shows as a heavy tail over a low median) reads differently from
  drift (the whole distribution shifting up);
- per-frame error against camera speed, which separates motion blur or pose
  lag from everything else;
- loop closure, by registering the last seconds of LiDAR against the first
  with point-to-plane ICP on the overlapping part, which is the rigid
  start-to-end drift a pose refinement would have to remove at the seam.

The held-out frames are the ones the fusion run held out, from the same
config, so the numbers sit beside the run's own metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.capture import Capture, load_mikan_take  # noqa: E402
from recon.evaluate import make_scene, raycast  # noqa: E402


def heldout_errors(scene, capture: Capture, holdout_ids) -> list[dict]:
    """Per held-out frame: the absolute depth error of every high-confidence
    pixel the mesh covers, with the LiDAR depth it was measured at."""
    width, height = capture.depth_size
    rows = []
    for frame_id in holdout_ids:
        z, _, hit = raycast(scene, capture.intrinsics_depth(frame_id), capture.pose(frame_id), width, height)
        depth = capture.depth(frame_id)
        ok = (capture.confidence(frame_id) == 2) & (depth > 0) & (depth < 5) & hit
        rows.append({"frame": int(frame_id), "error": np.abs(z[ok] - depth[ok]), "depth": depth[ok]})
    return rows


def report_range_and_tail(rows: list[dict]) -> None:
    error = np.concatenate([r["error"] for r in rows])
    depth = np.concatenate([r["depth"] for r in rows])
    print(f"held-out pixels {len(error)}: MAE {error.mean() * 100:.2f} cm, median {np.median(error) * 100:.2f} cm")
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 5)]:
        band = (depth >= lo) & (depth < hi)
        if band.any():
            print(f"  {lo}-{hi} m: {band.mean() * 100:4.1f}% of pixels, MAE {error[band].mean() * 100:5.2f} cm, "
                  f"median {np.median(error[band]) * 100:5.2f} cm")
    tail = error > 0.05
    print(f"  pixels over 5 cm: {tail.mean() * 100:.1f}%, carrying {error[tail].sum() / error.sum() * 100:.0f}% "
          f"of the total error")


def report_speed(rows: list[dict], capture: Capture) -> None:
    positions = capture.T_WC[:, :3, 3]
    dt = np.maximum(np.diff(capture.timestamps), 1e-3)
    speed = np.r_[0, np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt]
    frame_speed = np.array([speed[max(r["frame"] - 2, 0):r["frame"] + 3].mean() for r in rows])
    frame_mae = np.array([r["error"].mean() * 100 for r in rows])
    corr = float(np.corrcoef(frame_speed, frame_mae)[0, 1]) if len(rows) > 2 else float("nan")
    worst = sorted(zip(frame_mae, frame_speed, [r["frame"] for r in rows]), reverse=True)[:5]
    print(f"per-frame MAE against camera speed: correlation {corr:.2f} "
          f"(speed {frame_speed.min():.2f} to {frame_speed.max():.2f} m/s)")
    print("  worst frames: " + ", ".join(f"{f} ({m:.1f} cm at {s:.2f} m/s)" for m, s, f in worst))


def report_loop_closure(capture: Capture, seconds: float = 4.0) -> None:
    n = len(capture)
    fps = n / max(capture.timestamps[-1] - capture.timestamps[0], 1e-3)
    span = int(seconds * fps)
    if n < 3 * span:
        print("loop closure: take too short to compare its ends")
        return

    def cloud(ids) -> o3d.geometry.PointCloud:
        points = np.concatenate([capture.lidar_points(i, depth_max=4.0) for i in ids])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(0.02)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        return pc

    # The first frames of a take are sensor warm-up, so the early window starts a little in.
    early = cloud(range(10, 10 + span, 3))
    late = cloud(range(n - span, n, 3))
    distance = np.asarray(late.compute_point_cloud_distance(early))
    overlap = late.select_by_index(np.where(distance < 0.2)[0])
    if len(overlap.points) < 500:
        print(f"loop closure: the ends do not overlap ({len(overlap.points)} shared points); "
              "the take did not return to its start view")
        return

    before = np.asarray(overlap.compute_point_cloud_distance(early))
    result = o3d.pipelines.registration.registration_icp(
        overlap, early, 0.10, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    T = result.transformation
    angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    overlap.transform(T)
    after = np.asarray(overlap.compute_point_cloud_distance(early))
    print(f"loop closure over the overlapping {len(overlap.points) / len(late.points) * 100:.0f}% of the last "
          f"{seconds:.0f} s: before {np.median(before) * 1000:.0f} mm median / {np.percentile(before, 90) * 1000:.0f} mm "
          f"p90; rigid ICP offset {np.linalg.norm(T[:3, 3]) * 1000:.0f} mm, {angle:.2f} deg; "
          f"after {np.median(after) * 1000:.0f} mm / {np.percentile(after, 90) * 1000:.0f} mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("movie", help="path to <take>.mp4")
    ap.add_argument("--run", default="lidar_tsdf", help="run directory under out/<take>/ holding mesh.ply")
    ap.add_argument("--config", default="configs/default.yaml", help="the config the run used, for the frame split")
    ap.add_argument("--arkit-poses", action="store_true",
                    help="score with the recorded ARKit poses even when the run refined them")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / args.config))
    capture = load_mikan_take(args.movie)
    mesh_path = ROOT / cfg["out_root"] / capture.name / args.run / "mesh.ply"
    if not mesh_path.exists():
        print(f"no mesh at {mesh_path}; run the fusion first")
        return 1
    scene = make_scene(o3d.io.read_triangle_mesh(str(mesh_path)))
    _, holdout_ids = capture.split(cfg["keyframe_stride"], cfg["holdout_every"])

    # A run that refined its poses is scored in that frame, as run.py does.
    poses_path = mesh_path.parent / "poses_refined.npy"
    refined = poses_path.exists() and not args.arkit_poses
    if refined:
        capture.T_WC = np.load(poses_path)

    print(f"{capture.name} / {args.run}: {len(capture)} frames, {len(holdout_ids)} held out"
          + (", refined poses" if refined else ""))
    rows = heldout_errors(scene, capture, holdout_ids)
    report_range_and_tail(rows)
    report_speed(rows, capture)
    report_loop_closure(capture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
