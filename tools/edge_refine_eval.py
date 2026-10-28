"""Measure the image-space edge refinement MikanXR's DepthEdgeRefineNode runs,
on a fused mesh and the take it came from.

    python tools/edge_refine_eval.py <take>.mp4 --mesh out/<take>/<run>/mesh.ply
           [--poses out/<take>/<run>/poses_refined.npy] [--znear 0.1 --zfar 20]
           [--frames 8] [--radius 16] [--sigma 0.2] [--jump 0.05]
           [--sweep] [--radii 4,8,16,32] [--sigmas 0.05,0.1,0.2] [--jumps 0.05]
           [--out out/edge_refine]

For each held-out frame the mesh is rendered from the frame's pose at the
colour resolution into the editor's normalized linear depth (1 where nothing
was hit), refined by the same three passes the node's shaders run, and both
the prior and the refined depth are scored against the held-out LiDAR's
silhouette edges at the LiDAR resolution with the harness's edge metric.
The LiDAR carries its own edge blur, so a refined edge that follows the
video exactly can score worse than the prior where the LiDAR is blobby; the
contact sheets written beside the numbers are the check on that.

refine_depth() is the reference the node's GLSL and C++ reference follow:
window min/max over +/- radius (separable), a band where
(max - min) > jump * max, and inside the band a two-prototype colour
classification: the window's taps split into a near and a far side at the
midpoint depth, each side's mean colour is its prototype, and the pixel joins
the side whose prototype its colour is nearer to, at that side's mean depth,
unless the two prototypes are closer than color_sigma apart.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.capture import load_mikan_take  # noqa: E402
from recon.edge_refine import refine_depth  # noqa: E402
from recon.evaluate import depth_edges, edge_distance, make_scene, raycast  # noqa: E402


# ---- frames and scoring ---------------------------------------------------------

def normalized_depth(z: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    d = np.where(np.isfinite(z), (z - znear) / (zfar - znear), 1.0)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def metres_from_normalized(d: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    z = d * (zfar - znear) + znear
    return np.where(d >= 1.0, np.inf, z)


def to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    return (rgb_u8.astype(np.float32) / 255.0) ** 2.2


def score_against_lidar(capture, frame_id: int, depth_norm: np.ndarray, znear: float, zfar: float,
                        edge_gradient: float) -> dict:
    """Edge distance between a candidate depth (colour resolution) and the held-out LiDAR, at LiDAR resolution."""
    w, h = capture.depth_size
    small = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_NEAREST)
    z = metres_from_normalized(small, znear, zfar)
    lidar = capture.depth(frame_id)
    valid = (capture.confidence(frame_id) == 2) & (lidar > 0)
    edges_lidar = depth_edges(lidar, valid, edge_gradient)
    edges_candidate = depth_edges(z, valid, edge_gradient)
    return edge_distance(edges_lidar, edges_candidate)


def edge_mask_colour_res(depth_norm: np.ndarray, znear: float, zfar: float, edge_gradient: float) -> np.ndarray:
    z = metres_from_normalized(depth_norm, znear, zfar)
    return depth_edges(z, np.ones(z.shape, dtype=bool), edge_gradient)


def contact_sheet(rgb: np.ndarray, prior: np.ndarray, refined: np.ndarray, capture, frame_id: int, znear: float,
                  zfar: float, edge_gradient: float, path: Path) -> None:
    sheet = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = prior.shape
    lidar = capture.depth(frame_id)
    valid = (capture.confidence(frame_id) == 2) & (lidar > 0)
    lidar_edges = depth_edges(lidar, valid, edge_gradient).astype(np.uint8)
    lidar_edges = cv2.resize(lidar_edges, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    sheet[lidar_edges] = (255, 128, 0)
    sheet[edge_mask_colour_res(prior, znear, zfar, edge_gradient)] = (0, 0, 255)
    sheet[edge_mask_colour_res(refined, znear, zfar, edge_gradient)] = (0, 255, 0)
    cv2.imwrite(str(path), sheet)


def parse_list(text: str, cast):
    return [cast(v) for v in text.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("movie")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--poses", help="poses_refined.npy of the run; the recorded poses otherwise")
    ap.add_argument("--znear", type=float, default=0.1)
    ap.add_argument("--zfar", type=float, default=20.0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--radius", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--jump", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--radii", default="4,8,16,32")
    ap.add_argument("--sigmas", default="0.05,0.1,0.2")
    ap.add_argument("--jumps", default="0.05")
    ap.add_argument("--out", default="out/edge_refine")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    edge_gradient = cfg["eval"]["edge_gradient"]
    capture = load_mikan_take(args.movie)
    if args.poses:
        capture.T_WC = np.load(args.poses)
    scene = make_scene(o3d.io.read_triangle_mesh(args.mesh))
    _, holdout = capture.split(cfg["keyframe_stride"], cfg["holdout_every"])
    frame_ids = [int(f) for f in np.array(holdout)[np.linspace(0, len(holdout) - 1, args.frames).round().astype(int)]]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height = capture.color_size
    rgb_frames = capture.read_frames(frame_ids)
    priors = {}
    for f in frame_ids:
        z, _, _ = raycast(scene, capture.intrinsics_color(f), capture.pose(f), width, height)
        priors[f] = normalized_depth(z, args.znear, args.zfar)
    print(f"{capture.name}: {len(frame_ids)} held-out frames at {width}x{height}, zNear {args.znear} zFar {args.zfar}")
    print("note: the LiDAR edges carry the sensor's own blur, so follow the numbers with the contact sheets")

    prior_scores = [score_against_lidar(capture, f, priors[f], args.znear, args.zfar, edge_gradient) for f in frame_ids]
    prior_px = float(np.mean([s["symmetric_px"] for s in prior_scores]))
    prior_match = float(np.mean([s["matched_fraction"] for s in prior_scores]))
    print(f"prior   : edge {prior_px:.2f} px, matched {prior_match * 100:.1f}%")

    combos = (list(itertools.product(parse_list(args.radii, int), parse_list(args.sigmas, float), parse_list(args.jumps, float)))
              if args.sweep else [(args.radius, args.sigma, args.jump)])
    rows = []
    for radius, sigma, jump in combos:
        scores = []
        moved = []
        for f in frame_ids:
            rgb_lin = to_linear(rgb_frames[f])
            refined = refine_depth(priors[f], rgb_lin, radius, sigma, jump)
            scores.append(score_against_lidar(capture, f, refined, args.znear, args.zfar, edge_gradient))
            moved.append(float(np.mean(np.abs(refined - priors[f]) > 1e-4)) * 100)
            if not args.sweep:
                contact_sheet(rgb_frames[f], priors[f], refined, capture, f, args.znear, args.zfar, edge_gradient,
                              out_dir / f"frame_{f:06d}_r{radius}_s{sigma}_j{jump}.png")
        px = float(np.mean([s["symmetric_px"] for s in scores]))
        match = float(np.mean([s["matched_fraction"] for s in scores]))
        rows.append({"radius": radius, "sigma": sigma, "jump": jump, "edge_px": px, "matched": match,
                     "pixels_changed_pct": float(np.mean(moved))})
        print(f"refined : r {radius:2d} sigma {sigma:.2f} jump {jump:.2f}: edge {px:.2f} px, matched {match * 100:.1f}%, "
              f"{np.mean(moved):.2f}% of pixels changed")

    if args.sweep:
        with open(out_dir / "sweep.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        best = min(rows, key=lambda r: r["edge_px"])
        print(f"best    : r {best['radius']} sigma {best['sigma']} jump {best['jump']}: edge {best['edge_px']:.2f} px "
              f"(prior {prior_px:.2f} px); wrote {out_dir / 'sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
