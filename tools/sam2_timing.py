"""Time a promptable segmenter per frame on this GPU: the image encoder, the prompt decoder, and the
video tracker's per-frame step, per model.

    .venv_sam2/Scripts/python tools/sam2_timing.py [--family sam2] [--models tiny,small,base_plus,large]
        [--sizes 0] [--iters 50] [--warmup 10] [--image out/live_occlusion/angle_01/video.png]
        [--frames-dir out/sam2_track/frames] [--track-frames 60] [--checkpoints <dir>] [--out out/sam2_timing]

The families and their venvs are sam2_edge_eval.py's; run each family with its own venv's python.

Image path: set_image (the encoder) and predict with a box and points (the decoder) on one
1920x1440 frame, torch.cuda.synchronize around every call, bfloat16 autocast for the SAM2-API
families and fp32 for EfficientViT-SAM, median and p90 in ms over the iterations, with the encoder's
parameter count. --sizes overrides model.image_size through hydra, 0 being the model's own; the
predictor's hard-coded feature sizes are set from the model's size, which is what made a bare 512
override fail in sam2 1.1.0. A size the model refuses is reported as such and skipped.

Video path, when --frames-dir holds the JPEG frames sam2_track_eval.py wrote: the first
--track-frames of them are copied aside, a box is prompted on the first, and each step of
propagate_in_video is timed, which is the encoder plus memory attention a live tracker pays per
frame. Runs at the model's own size. EfficientViT-SAM has no video predictor.

Appends to out/summary.txt and prints it.
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


def time_image_path(family: str, model: str, size: int, image: np.ndarray, args) -> list[str]:
    try:
        predictor, autocast, input_size = S.build_predictor(family, model, args.checkpoints, size)
        params = sum(p.numel() for p in predictor.model.image_encoder.parameters()) / 1e6
        h, w = image.shape[:2]
        box, points, labels = demo_prompts(w, h)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            enc = timed(lambda: predictor.set_image(image), args.iters, args.warmup)
            dec = timed(lambda: predictor.predict(point_coords=points, point_labels=labels, box=box,
                                                  multimask_output=True), args.iters, args.warmup)
        peak = torch.cuda.max_memory_allocated() / 2**20
        line = (f"{family} {model:10s} {input_size:5d}  encoder {enc[0]:7.1f} / {enc[1]:7.1f} ms"
                f"   decoder {dec[0]:6.1f} / {dec[1]:6.1f} ms   frame {enc[0] + dec[0]:7.1f} ms"
                f"   peak {peak:6.0f} MB   encoder params {params:6.1f} M")
    except Exception as exc:  # a size the model refuses, or a missing checkpoint
        line = f"{family} {model:10s} {size:5d}  failed: {str(exc).splitlines()[0][:120]}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return [line]


def build_video_predictor(family: str, model: str, checkpoints: Path | None):
    spec, ckpt_path = S.model_paths(family, model, checkpoints)
    if family == "efficienttam":
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        return build_efficienttam_video_predictor(spec, str(ckpt_path), device="cuda",
                                                  hydra_overrides_extra=["++model.compile_image_encoder=False"])
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(spec, str(ckpt_path), device="cuda")


def time_video_path(family: str, model: str, frames_dir: Path, args) -> list[str]:
    if family == "efficientvit_sam":
        return [f"{family} {model:10s} video: no video predictor"]
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
        predictor = build_video_predictor(family, model, args.checkpoints)
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
        line = (f"{family} {model:10s} video  step {np.median(steps):7.1f} / {np.percentile(steps, 90):7.1f} ms"
                f" over {len(steps)} frames   peak {peak:6.0f} MB")
    except Exception as exc:
        line = f"{family} {model:10s} video: failed: {str(exc).splitlines()[0][:120]}"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return [line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="sam2", choices=list(S.FAMILIES))
    parser.add_argument("--models", default=None, help="comma separated, default every model of the family")
    parser.add_argument("--sizes", default="0", help="comma separated encoder inputs, 0 for the model's own")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--image", type=Path, default=ROOT / "out" / "live_occlusion" / "angle_01" / "video.png")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "out" / "sam2_track" / "frames")
    parser.add_argument("--track-frames", type=int, default=60)
    parser.add_argument("--checkpoints", type=Path, default=None, help="default per family")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sam2_timing")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    gpu = torch.cuda.get_device_name(0)
    lines = [f"{gpu}, torch {torch.__version__}, image {image.shape[1]}x{image.shape[0]}, "
             f"{args.iters} iterations after {args.warmup} warm-up, median / p90",
             f"{'family model':17s} {'input':>5s}"]
    models = args.models.split(",") if args.models else list(S.FAMILIES[args.family][2])
    sizes = [int(s) for s in args.sizes.split(",")]
    for model in models:
        for size in sizes:
            lines += time_image_path(args.family, model, size, image, args)
            print(lines[-1])
    if args.frames_dir.exists():
        for model in models:
            lines += time_video_path(args.family, model, args.frames_dir, args)
            print(lines[-1])
    else:
        lines.append(f"video path skipped: {args.frames_dir} does not exist (run sam2_track_eval.py first)")
    with open(args.out / "summary.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:1]))


if __name__ == "__main__":
    main()
