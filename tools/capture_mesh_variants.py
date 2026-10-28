"""Capture marked take frames once per occlusion mesh variant, rendered by the running editor.

    python tools/capture_mesh_variants.py <variant>=<model path>[,<pose track path>] ...
           [--frames take_t02,take_t06,...] [--stencil 2210] [--source 2201] [--compositor 2209]
           [--camera 2207] [--root out/live_occlusion] [--port 21120]

Each variant names a model in the project (models/<name>.glb) and optionally the refined pose
track its build wrote, so the mesh renders from the poses it was fused with. For every variant
the stencil's model and the take source's pose track are swapped in, the take camera is
composited, and each listed frame is captured at its take time (the frame folder's name ends in
t<seconds>) into <root>/variants/<variant>/<frame>, with the frame's hand marks copied beside
it. The stencil, the pose track, the compositor's camera and the take's playback are restored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from capture_live_occlusion import Automation, capture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+", help="<name>=<model path>[,<pose track path>]")
    parser.add_argument("--frames", default="take_t02,take_t06,take_t10,take_t18")
    parser.add_argument("--stencil", type=int, default=2210)
    parser.add_argument("--source", type=int, default=2201)
    parser.add_argument("--compositor", type=int, default=2209)
    parser.add_argument("--camera", type=int, default=2207)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--port", type=int, default=21120)
    args = parser.parse_args()

    auto = Automation(args.port)
    get = lambda system, cid, prop: auto.command(f"property get {system} {cid} {prop}")[0]
    put = lambda system, cid, prop, value: auto.command(f"property set {system} {cid} {prop} {value}")

    original = {
        "model": get("ModelStencilSystem", args.stencil, "model_path"),
        "track": get("FileVideoSourceSystem", args.source, "pose_track_path"),
        "camera": get("CompositorObjectSystem", args.compositor, "camera_id"),
        "state": get("FileVideoSourceSystem", args.source, "playback_state"),
    }
    frames = args.frames.split(",")
    try:
        put("CompositorObjectSystem", args.compositor, "camera_id", args.camera)
        auto.command(f"function invoke FileVideoSourceSystem {args.source} pause")
        for spec in args.variants:
            name, paths = spec.split("=", 1)
            model, _, track = paths.partition(",")
            put("ModelStencilSystem", args.stencil, "model_path", model)
            put("FileVideoSourceSystem", args.source, "pose_track_path", track or original["track"])
            time.sleep(3.0)  # the model loads and the track reloads
            # A new pose track reopens the source, which starts it playing again
            auto.command(f"function invoke FileVideoSourceSystem {args.source} pause")
            for frame in frames:
                seconds = float(frame.rsplit("_t", 1)[1])
                put("FileVideoSourceSystem", args.source, "playback_time", f"{seconds:.3f}")
                time.sleep(2.0)
                out = capture(auto, frame, args.compositor, args.camera, args.root / "variants" / name)
                for mark in ("gt_mask.png", "gt_polygons.json"):
                    shutil.copy2(args.root / frame / mark, out / mark)
    finally:
        put("ModelStencilSystem", args.stencil, "model_path", original["model"])
        put("FileVideoSourceSystem", args.source, "pose_track_path", original["track"])
        put("CompositorObjectSystem", args.compositor, "camera_id", original["camera"])
        if original["state"] == "1":
            auto.command(f"function invoke FileVideoSourceSystem {args.source} play")


if __name__ == "__main__":
    main()
