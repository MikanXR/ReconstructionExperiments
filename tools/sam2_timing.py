"""Time SAM2 per frame on this GPU: the image encoder, the prompt decoder, and the video tracker's
per-frame step, per model size.

    .venv_sam2/Scripts/python tools/sam2_timing.py [--models tiny,small,base_plus,large] [--sizes 1024]
        [--iters 50] [--warmup 10] [--image out/live_occlusion/angle_01/video.png]
        [--frames-dir out/sam2_track/frames] [--track-frames 60] [--checkpoints models/sam2] [--out out/sam2_timing]

Image path: set_image (the encoder) and predict with a box and points (the decoder) on one
1920x1440 frame, torch.cuda.synchronize around every call, bfloat16 autocast, median and p90 in ms
over the iterations. --sizes overrides model.image_size through hydra; a size the model refuses is
reported as such and skipped. In sam2 1.1.0 a bare 512 override fails inside the Hiera trunk (a view
over a non-contiguous tensor), so the 512 real-time configuration needs the config changes that go
with it and is not timed by default.

Video path, when --frames-dir holds the JPEG frames sam2_track_eval.py wrote: the first
--track-frames of them are copied aside, a box is prompted on the first, and each step of
propagate_in_video is timed, which is the encoder plus memory attention a live tracker pays per
frame. Runs at 1024 only.

Writes out/summary.txt and prints it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sam2_edge_eval as S  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


def timed(fn, iters: int, warmup: int) -> tuple[float, float]:
    """Median and p90 wall time in ms of fn, each call bracketed by cuda.synchronize."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def demo_prompts(w: int, h: int):
    box = np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7], np.float32)
    points = np.array([[w * 0.5, h * 0.5], [w * 0.45, h * 0.6], [w * 0.1, h * 0.1]], np.float32)
    labels = np.array([1, 1, 0])
    return box, points, labels


def time_image_path(model: str, size: int, image: np.ndarray, args) -> list[str]:
    cfg, ckpt = S.MODELS[model]
    overrides = [] if size == 1024 else [f"++model.image_size={size}"]
    try:
        sam = build_sam2(cfg, str(args.checkpoints / ckpt), device="cuda", hydra_overrides_extra=overrides)
        predictor = SAM2ImagePredictor(sam)
        h, w = image.shape[:2]
        box, points, labels = demo_prompts(w, h)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc = timed(lambda: predictor.set_image(image), args.iters, args.warmup)
            dec = timed(lambda: predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                  multimask_output=True), args.iters, args.warmup)
        peak = torch.cuda.max_memory_allocated() / 2**20
        line = (f"{model:10s} {size:5d}  encoder {enc[0]:7.1f} / {enc[1]:7.1f} ms   decoder {dec[0]:6.1f} / {dec[1]:6.1f} ms"
                f"   frame {enc[0] + dec[0]:7.1f} ms   peak {peak:6.0f} MB")
    except Exception as exc:  # a size the model refuses, or a missing checkpoint
        line = f"{model:10s} {size:5d}  failed: {str(exc).splitlines()[0][:120]}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return [line]


def time_video_path(model: str, frames_dir: Path, args) -> list[str]:
    cfg, ckpt = S.MODELS[model]
    subset = args.out / "track_subset"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True)
    names = sorted(frames_dir.glob("*.jpg"))[:args.track_frames]
    if len(names) < 2:
        return [f"{model:10s} video: fewer than two frames in {frames_dir}"]
    for i, p in enumerate(names):
        shutil.copy(p, subset / f"{i:05d}.jpg")
    first = cv2.imread(str(names[0]))
    h, w = first.shape[:2]
    box, points, labels = demo_prompts(w, h)
    try:
        predictor = build_sam2_video_predictor(cfg, str(args.checkpoints / ckpt), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(subset), offload_video_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=points, labels=labels, box=box)
            samples = []
            torch.cuda.synchronize()
            t_prev = time.perf_counter()
            for _ in predictor.propagate_in_video(state):
                torch.cuda.synchronize()
                now = time.perf_counter()
                samples.append((now - t_prev) * 1000.0)
                t_prev = now
        steps = np.array(samples[1:])   # the first yield is the prompted frame itself
        peak = torch.cuda.max_memory_allocated() / 2**20
        line = (f"{model:10s} video 1024  step {np.median(steps):7.1f} / {np.percentile(steps, 90):7.1f} ms"
                f" over {len(steps)} frames   peak {peak:6.0f} MB")
    except Exception as exc:
        line = f"{model:10s} video: failed: {str(exc).splitlines()[0][:120]}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return [line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(S.MODELS))
    parser.add_argument("--sizes", default="1024")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--image", type=Path, default=ROOT / "out" / "live_occlusion" / "angle_01" / "video.png")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "out" / "sam2_track" / "frames")
    parser.add_argument("--track-frames", type=int, default=60)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sam2_timing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    gpu = torch.cuda.get_device_name(0)
    lines = [f"{gpu}, torch {torch.__version__}, image {image.shape[1]}x{image.shape[0]}, "
             f"{args.iters} iterations after {args.warmup} warm-up, median / p90",
             f"{'model':10s} {'size':>5s}"]
    models = args.models.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    for model in models:
        for size in sizes:
            lines += time_image_path(model, size, image, args)
            print(lines[-1])
    if args.frames_dir.exists():
        for model in models:
            lines += time_video_path(model, args.frames_dir, args)
            print(lines[-1])
    else:
        lines.append(f"video path skipped: {args.frames_dir} does not exist (run sam2_track_eval.py first)")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:1]))


if __name__ == "__main__":
    main()
