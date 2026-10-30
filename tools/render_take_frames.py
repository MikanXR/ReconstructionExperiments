"""Render an occlusion mesh into chosen frames of a MikanARStreamer take, offline.

    python tools/render_take_frames.py <take.trackcamera> <mesh.glb> (--times 2,6,10 | --frames 60,180 | --stride N)
           [--poses <track.pose.json>] [--out out/<take stem>]

Movie frame i (the i-th decoded frame of the take's movie) belongs to pose track frame i in the
track's file order. --frames names those indices, --times names playback seconds from the earliest
time_us in the track (the nearest frame is taken), --stride takes every Nth frame. --poses replaces
the camera poses with those of another track of the same format (a refined or eroded one), matched
by time_us, or by index when the counts agree and the times do not. Intrinsics always come from the
take's own track. The mesh is in the take's ARKit world, the same world as the poses.

Per frame the movie is decoded sequentially (never past the last chosen frame), the mesh is ray cast
from the frame's pose at the video size, and the depth is normalized between 0.1 and 20 m with 1.0
where no mesh is hit.

Writes one folder per frame, t<seconds> when --times was used and f<index> otherwise, holding
video.png, mesh_depth.npy, refined_depth.npy (a copy of mesh_depth, since the scoring tools require
it) and meta.json. These are the folders mark_silhouette.py marks and occlusion_edge_eval.py and
score_mesh_variants.py score. One line per frame reports its folder, index, time and the percentage
of pixels the mesh covers.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import av
import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.capture import ARKIT_TO_OPENCV_CAMERA  # noqa: E402
from recon.evaluate import make_scene, raycast  # noqa: E402

ZNEAR, ZFAR = 0.1, 20.0


def normalized_depth(z: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    d = np.where(np.isfinite(z), (z - znear) / (zfar - znear), 1.0)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def parse_list(text: str, kind) -> list:
    return [kind(x) for x in text.split(",") if x.strip()]


def choose_frames(args, track: dict) -> list[tuple[int, str]]:
    """The chosen (frame index, folder name) pairs, in ascending frame order."""
    frames = track["frames"]
    if args.frames is not None:
        picks = [(i, f"f{i:05d}") for i in parse_list(args.frames, int)]
    elif args.times is not None:
        t0 = min(f["time_us"] for f in frames)
        rel = np.array([(f["time_us"] - t0) / 1e6 for f in frames])
        picks = []
        for t in parse_list(args.times, float):
            picks.append((int(np.argmin(np.abs(rel - t))), f"t{int(round(t)):02d}"))
    else:
        picks = [(i, f"f{i:05d}") for i in range(0, len(frames), args.stride)]
    for i, _ in picks:
        if not 0 <= i < len(frames):
            raise SystemExit(f"frame {i} is outside the track's {len(frames)} frames")
    return sorted(picks)


def override_transforms(base: dict, other_path: Path) -> list:
    """Transforms of another pose track for each frame of the base track, matched by time_us."""
    other = json.loads(other_path.read_text())["frames"]
    base_frames = base["frames"]
    by_time = {f["time_us"]: f["transform"] for f in other}
    if all(f["time_us"] in by_time for f in base_frames):
        return [by_time[f["time_us"]] for f in base_frames]
    if len(other) == len(base_frames):
        print(f"warning: {other_path.name} times do not match the take's, matching by index")
        return [f["transform"] for f in other]
    raise SystemExit(f"{other_path.name} has {len(other)} frames and matches the take's {len(base_frames)} "
                     f"neither by time nor by count")


def decode_frames(archive: tarfile.TarFile, member: str, wanted: set[int], out: Path):
    """Yield (index, RGB uint8 frame) for the wanted indices, stopping after the last."""
    last = max(wanted)
    try:
        container = av.open(archive.extractfile(member))
    except Exception:
        target = out / "movie.mp4"
        print(f"movie member is not readable as a stream, extracting it to {target}")
        with open(target, "wb") as f:
            f.write(archive.extractfile(member).read())
        container = av.open(str(target))
    with container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                yield i, frame.to_ndarray(format="rgb24")
            if i >= last:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--times", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if sum(x is not None for x in (args.times, args.frames, args.stride)) != 1:
        parser.error("give exactly one of --times, --frames, --stride")
    if args.stride is not None and args.stride < 1:
        parser.error("--stride must be at least 1")
    out = args.out or ROOT / "out" / args.take.stem
    out.mkdir(parents=True, exist_ok=True)

    archive = tarfile.open(args.take)
    members = json.load(archive.extractfile("capture.json"))["members"]
    track = json.load(archive.extractfile(members["movie_pose"]))
    frames = track["frames"]
    transforms = override_transforms(track, args.poses) if args.poses else [f["transform"] for f in frames]
    track_size = (int(track["image_width"]), int(track["image_height"]))
    t0 = min(f["time_us"] for f in frames)

    picks = choose_frames(args, track)
    names = dict(picks)
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    scene = make_scene(mesh)

    seen = set()
    for index, rgb in decode_frames(archive, members["movie"], set(names), out):
        seen.add(index)
        h, w = rgb.shape[:2]
        if (w, h) != track_size:
            print(f"warning: frame {index} decodes to {w}x{h} but the track records {track_size[0]}x{track_size[1]}")
        f = frames[index]
        sx, sy = w / track_size[0], h / track_size[1]
        K = np.array([[f["fx"] * sx, 0, f["cx"] * sx], [0, f["fy"] * sy, f["cy"] * sy], [0, 0, 1.0]])
        camera_to_world = np.array(transforms[index], dtype=np.float64).reshape(4, 4)
        T_WC = camera_to_world @ ARKIT_TO_OPENCV_CAMERA
        z, _, hit = raycast(scene, K, T_WC, w, h)
        depth = normalized_depth(z, ZNEAR, ZFAR)

        seconds = (f["time_us"] - t0) / 1e6
        folder = out / names[index]
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "video.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(folder / "mesh_depth.npy", depth)
        np.save(folder / "refined_depth.npy", depth)
        meta = {"frame": index, "time_seconds": seconds, "znear": ZNEAR, "zfar": ZFAR, "size": [w, h],
                "take": str(args.take), "mesh": str(args.mesh),
                "poses": str(args.poses) if args.poses else None,
                "camera_to_world": [float(x) for x in camera_to_world.reshape(16)]}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{names[index]:8s} frame {index:5d}  t {seconds:7.3f} s  mesh covers {100.0 * hit.mean():5.1f} %")

    missing = sorted(set(names) - seen)
    if missing:
        print(f"warning: the movie ended before frames {missing}")


if __name__ == "__main__":
    main()
