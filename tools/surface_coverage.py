"""Per-vertex observation coverage of a region of a fused occlusion mesh, from the take it was fused from.

    python tools/surface_coverage.py <take.trackcamera> <mesh.glb> <refined.pose.json>
           --center x,y,z --radius 0.07 --floor <y> [--stride 3] [--band 0.015] [--out out/coverage/<name>]

For every mesh vertex within --radius (horizontally) of --center and above --floor (ARKit world,
metres, +Y up), each fused keyframe (every --stride-th frame in time order, as the builder takes
them) is checked against its own LiDAR depth at the refined pose: the vertex projects to a depth
pixel, and the measured depth there either agrees with the vertex within --band (support), lies
farther away (the frame says the vertex is empty space: a carve vote), or lies nearer (the vertex
is hidden). Support is split by the pixel's confidence and by whether the fusion's silhouette cull
(keep near, 2 px far band) drops the pixel.

Per vertex: support, high-confidence support, carve votes, directions seen from (12 azimuth bins
around the region's vertical axis), the best incidence (cosine between normal and view), and the
nearest distance. Per 30 degree sector around the axis the same are summarised beside the
surface's roughness there: the radial deviation from the median radius at each height, which is
what reads as lumps on a surface of revolution such as a cup.

Writes vertices.npz, sectors.txt, and coverage_<metric>.png renders of the region coloured by each
metric from four sides.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.fusion import cull_flying_pixels  # noqa: E402

AZ_BINS = 12


def open_take(path: Path):
    archive = tarfile.open(path)
    manifest = json.load(archive.extractfile("capture.json"))
    members = manifest["members"]
    track = json.load(archive.extractfile(members["movie_pose"]))
    w, h = track["depth_width"], track["depth_height"]
    n = len(track["frames"])

    def member_map(name: str, dtype, record: int) -> np.ndarray:
        info = archive.getmember(name)
        return np.memmap(path, dtype=dtype, mode="r", offset=info.offset_data, shape=(n, record))

    depth = member_map(members["depth"], np.uint16, w * h)
    conf = member_map(members["confidence"], np.uint8, w * h)
    return track, depth, conf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("poses", type=Path)
    parser.add_argument("--center", required=True)
    parser.add_argument("--radius", type=float, default=0.07)
    parser.add_argument("--floor", type=float, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--band", type=float, default=0.015)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or ROOT / "out" / "coverage" / args.take.stem
    out.mkdir(parents=True, exist_ok=True)
    centre = np.array([float(x) for x in args.center.split(",")])

    track, depth_records, conf_records = open_take(args.take)
    w, h = track["depth_width"], track["depth_height"]
    sx, sy = w / track["image_width"], h / track["image_height"]
    unit = track["depth_unit_mm"] / 1000.0
    refined = json.load(open(args.poses))["frames"]

    mesh = trimesh.load(args.mesh, force="mesh")
    v_all = mesh.vertices
    horiz = np.linalg.norm(v_all[:, [0, 2]] - centre[[0, 2]], axis=1)
    region = (horiz < args.radius) & (v_all[:, 1] > args.floor)
    idx = np.nonzero(region)[0]
    v = v_all[idx]
    normals = mesh.vertex_normals[idx]
    n = len(v)
    print(f"{n} vertices in the region")

    support = np.zeros(n, int)
    support_hi = np.zeros(n, int)
    culled = np.zeros(n, int)
    carve = np.zeros(n, int)
    az_seen = np.zeros((n, AZ_BINS), bool)
    best_cos = np.full(n, -1.0)
    nearest = np.full(n, np.inf)

    order = np.argsort([f["time_us"] for f in track["frames"]], kind="stable")
    keyframes = order[:: args.stride]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera axes to OpenCV
    for fi in keyframes:
        frame = track["frames"][fi]
        cam_to_world = np.array(refined[fi]["transform"], float).reshape(4, 4) @ flip
        world_to_cam = np.linalg.inv(cam_to_world)
        pc = (world_to_cam[:3, :3] @ v.T).T + world_to_cam[:3, 3]
        z = pc[:, 2]
        fx, fy = frame["fx"] * sx, frame["fy"] * sy
        cx, cy = frame["cx"] * sx, frame["cy"] * sy
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(fx * pc[:, 0] / z + cx).astype(int)
            r = np.round(fy * pc[:, 1] / z + cy).astype(int)
        inside = (z > 0.1) & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        if not inside.any():
            continue
        depth = depth_records[fi].reshape(h, w).astype(np.float32) * unit
        conf = conf_records[fi].reshape(h, w)
        keep = cull_flying_pixels(torch.as_tensor(depth), 0.08, True, 2).numpy()

        ii = np.nonzero(inside)[0]
        d = depth[r[ii], u[ii]]
        c = conf[r[ii], u[ii]]
        k = keep[r[ii], u[ii]]
        valid = (d > 0) & (c >= 1)
        agree = valid & (np.abs(d - z[ii]) < args.band)
        empty = valid & k & (d - z[ii] > args.band)

        cam = cam_to_world[:3, 3]
        to_cam = cam - v[ii]
        dist = np.linalg.norm(to_cam, axis=1)
        cosang = np.sum(normals[ii] * to_cam, axis=1) / dist
        az = np.arctan2(cam[2] - centre[2], cam[0] - centre[0])
        az_bin = int((np.degrees(az) % 360) // (360 / AZ_BINS))

        s = ii[agree & k]
        support[s] += 1
        support_hi[ii[agree & k & (c == 2)]] += 1
        culled[ii[agree & ~k]] += 1
        carve[ii[empty]] += 1
        az_seen[s, az_bin] = True
        best_cos[s] = np.maximum(best_cos[s], cosang[agree & k])
        nearest[s] = np.minimum(nearest[s], dist[agree & k])

    # Roughness: radial deviation from the median radius at each height
    rad = np.linalg.norm(v[:, [0, 2]] - centre[[0, 2]], axis=1)
    hbin = ((v[:, 1] - args.floor) // 0.01).astype(int)
    med = np.zeros(n)
    for b in np.unique(hbin):
        m = hbin == b
        med[m] = np.median(rad[m])
    dev = rad - med
    vaz = (np.degrees(np.arctan2(v[:, 2] - centre[2], v[:, 0] - centre[0])) % 360) // (360 / AZ_BINS)

    np.savez(out / "vertices.npz", index=idx, support=support, support_hi=support_hi, culled=culled, carve=carve,
             az_seen=az_seen, best_cos=best_cos, nearest=nearest, deviation=dev, sector=vaz)

    lines = [f"{len(keyframes)} keyframes, {n} vertices, band {args.band * 1000:.0f} mm",
             f"{'sector':>6s} {'verts':>5s} {'|dev| mm':>8s} {'support':>7s} {'hi conf':>7s} {'culled':>6s} "
             f"{'carve':>6s} {'dirs':>4s} {'best cos':>8s} {'near m':>6s} {'unseen':>6s}"]
    for s in range(AZ_BINS):
        m = vaz == s
        if not m.any():
            continue
        lines.append(f"{s * 30:4d}-{s * 30 + 30:<3d}{m.sum():5d} {np.median(np.abs(dev[m])) * 1000:8.1f} "
                     f"{np.median(support[m]):7.0f} {np.median(support_hi[m]):7.0f} {np.median(culled[m]):6.0f} "
                     f"{np.median(carve[m]):6.0f} {np.median(az_seen[m].sum(1)):4.0f} "
                     f"{np.median(best_cos[m]):8.2f} {np.median(nearest[m][np.isfinite(nearest[m])]) if np.isfinite(nearest[m]).any() else 0:6.2f} "
                     f"{(support[m] == 0).mean():6.2f}")
    m = np.ones(n, bool)
    corr = lambda a: np.corrcoef(a[m], np.abs(dev[m]))[0, 1]
    lines.append(f"correlation with |deviation|: support {corr(support):.2f}, hi conf {corr(support_hi):.2f}, "
                 f"carve {corr(carve):.2f}, carve share {corr(carve / np.maximum(carve + support, 1)):.2f}, "
                 f"dirs {corr(az_seen.sum(1)):.2f}, best cos {corr(best_cos):.2f}")
    text = "\n".join(lines)
    (out / "sectors.txt").write_text(text + "\n")
    print(text)

    render_views(mesh, idx, {"support": support, "carve_share": carve / np.maximum(carve + support, 1),
                             "deviation_mm": np.abs(dev) * 1000, "dirs": az_seen.sum(1).astype(float),
                             "hi_conf": support_hi.astype(float)}, centre, args.floor, out)


def render_views(mesh, idx, metrics: dict, centre: np.ndarray, floor: float, out: Path) -> None:
    """Orthographic splats of the region's vertices from four sides, coloured by each metric."""
    v = mesh.vertices[idx]
    size = 360
    for name, values in metrics.items():
        lo, hi = np.percentile(values, 2), np.percentile(values, 98)
        norm = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
        colours = cv2.applyColorMap((norm * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        tiles = []
        for k in range(4):
            a = np.radians(90 * k)
            right = np.array([np.cos(a), 0, np.sin(a)])
            fwd = np.array([-np.sin(a), 0, np.cos(a)])
            rel = v - np.array([centre[0], floor, centre[2]])
            x, y, depth = rel @ right, rel[:, 1], rel @ fwd
            scale = size / 0.2
            px = (x * scale + size / 2).astype(int)
            py = (size - 20 - y * scale).astype(int)
            img = np.full((size, size, 3), 30, np.uint8)
            zbuf = np.full((size, size), np.inf)
            for i in np.argsort(-depth):
                if 0 <= px[i] < size and 0 <= py[i] < size:
                    cv2.circle(img, (px[i], py[i]), 4, tuple(int(c) for c in colours[i]), -1)
            cv2.putText(img, f"{name} view {90 * k}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(img)
        cv2.putText(tiles[0], f"{lo:.2f} .. {hi:.2f}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"coverage_{name}.png"), np.hstack(tiles))


if __name__ == "__main__":
    main()
