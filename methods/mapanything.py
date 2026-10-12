"""MapAnything (facebook/map-anything-apache) with every input it accepts:
images, intrinsics, camera poses, and the LiDAR depth.

Install (own venv, keep .venv untouched). MapAnything pins
opencv-python-headless, which fights the opencv-python in requirements.txt
for the cv2 package directory, so install it with --no-deps and add its
imports by hand (requirements-mapanything.txt holds that list):

    python -m venv .venv_mapanything
    .venv_mapanything\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .venv_mapanything\\Scripts\\python -m pip install -r requirements.txt
    git clone --depth 1 https://github.com/facebookresearch/map-anything third_party/map-anything
    .venv_mapanything\\Scripts\\python -m pip install --no-deps -e third_party/map-anything
    .venv_mapanything\\Scripts\\python -m pip install -r requirements-mapanything.txt

Run:

    .venv_mapanything\\Scripts\\python run.py mapanything

How the inputs are built. Each view gets the full-resolution RGB frame, that
frame's colour intrinsics, the camera-to-world pose (OpenCV axes, which is
what MapAnything wants), and the 256x192 LiDAR nearest-upsampled to the colour
resolution with confidence < 2 zeroed, so holes stay holes. MapAnything's
preprocess_inputs then scales everything to 518 on the long side (518x392 for
4:3), resampling depth with nearest and tracking the intrinsics through the
scale and the centre crop. The depth cache stores the prediction at that
working resolution with the intrinsics preprocess_inputs returns.

Frames are processed in sliding windows of WINDOW views overlapping by
OVERLAP. For a frame covered by two windows the prediction from the window
where it sits farther from the edge wins.

Weight = (1 - 1 / conf) * mask * non_ambiguous_mask. conf is 1 + exp(x) in
[1, inf), so the map is monotone with 0 at the minimum confidence.

The metric scale is regressed by the model's scale head (one scalar per
window) from the encoded input scales, not copied from them, so even with
metric LiDAR and poses the output carries a few percent of scale bias.
align_scale_to_lidar divides each window by the median predicted/given depth
ratio over the LiDAR pixels it was fed. It is off by default so the plain
variant reports the model's own output.

Config block (configs/default.yaml or a tagged variant):

    mapanything:
      model: facebook/map-anything-apache
      ignore_depth_inputs: false    # true for the no-LiDAR ablation
      align_scale_to_lidar: false   # per-window rescale onto the given LiDAR
      window: 24
      overlap: 6
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from recon.capture import Capture, DEPTH_WIDTH, DEPTH_HEIGHT, scale_intrinsics
from recon.depthcache import DepthCache

NAME = "mapanything"
LICENSE = "Apache 2.0 (facebook/map-anything-apache)"

DEFAULT_MODEL = "facebook/map-anything-apache"
WINDOW = 24
OVERLAP = 6


def options(cfg: dict) -> dict:
    o = {"model": DEFAULT_MODEL, "ignore_depth_inputs": False, "align_scale_to_lidar": False,
         "window": WINDOW, "overlap": OVERLAP}
    o.update(cfg.get("mapanything", {}) or {})
    o["window"] = int(o["window"])
    o["overlap"] = int(o["overlap"])
    if o["overlap"] >= o["window"]:
        raise ValueError(f"mapanything: overlap {o['overlap']} must be smaller than window {o['window']}")
    return o


def load_model(model_id: str):
    import torch
    from mapanything.models import MapAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MapAnything.from_pretrained(model_id).to(device).eval()


def lidar_at_color_resolution(capture: Capture, fid: int) -> np.ndarray:
    """LiDAR depth (metres, 0 invalid) nearest-upsampled to the colour frame, confidence < 2 dropped."""
    depth = capture.depth(fid)
    depth[capture.confidence(fid) < 2] = 0.0
    w, h = capture.color_size
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def build_views(capture: Capture, frames: dict[int, np.ndarray], fids, with_depth: bool) -> list[dict]:
    import torch

    views = []
    for fid in fids:
        view = {
            "img": frames[fid],
            "intrinsics": capture.intrinsics_color(fid).astype(np.float32),
            "camera_poses": capture.pose(fid).astype(np.float32),
            "is_metric_scale": torch.tensor([True]),
        }
        if with_depth:
            view["depth_z"] = lidar_at_color_resolution(capture, fid)
        views.append(view)
    return views


def run_window(model, capture: Capture, frames: dict[int, np.ndarray], fids, opts: dict) -> dict[int, tuple]:
    """Predict one window. Returns {fid: (depth_m, weight, K)} at the model's working resolution."""
    import torch
    from mapanything.utils.image import preprocess_inputs

    fids = [int(f) for f in fids]
    views = preprocess_inputs(build_views(capture, frames, fids, with_depth=not opts["ignore_depth_inputs"]))
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            ignore_depth_inputs=bool(opts["ignore_depth_inputs"]),
        )
    scale = 1.0
    if opts["align_scale_to_lidar"] and not opts["ignore_depth_inputs"]:
        ratios = []
        for view, pred in zip(views, preds):
            given = view["depth_z"][0].cpu().numpy()
            got = pred["depth_z"][0, ..., 0].float().cpu().numpy()
            ok = (given > 0) & (got > 0)
            ratios.append(got[ok] / given[ok])
        ratios = np.concatenate(ratios)
        if len(ratios):
            scale = 1.0 / float(np.median(ratios))
    out = {}
    for fid, view, pred in zip(fids, views, preds):
        depth = pred["depth_z"][0, ..., 0].float().cpu().numpy() * scale
        conf = pred["conf"][0].float().cpu().numpy()
        weight = 1.0 - 1.0 / np.maximum(conf, 1.0)
        if "mask" in pred:
            weight *= pred["mask"][0, ..., 0].float().cpu().numpy()
        if "non_ambiguous_mask" in pred:
            weight *= pred["non_ambiguous_mask"][0].float().cpu().numpy()
        weight[~(depth > 0)] = 0.0
        K = view["intrinsics"][0].cpu().numpy().astype(np.float64)
        out[fid] = (depth.astype(np.float32), weight.astype(np.float32), K)
    return out


def window_starts(n: int, window: int, overlap: int) -> list[int]:
    if n <= window:
        return [0]
    step = window - overlap
    starts = list(range(0, n - window, step))
    starts.append(n - window)
    return starts


def predict(capture: Capture, frame_ids, cache: DepthCache, cfg: dict) -> None:
    import torch

    opts = options(cfg)
    ids = [int(f) for f in frame_ids]
    starts = window_starts(len(ids), opts["window"], opts["overlap"])
    model = load_model(opts["model"])

    begun = False
    held: dict[int, tuple[int, tuple]] = {}   # fid -> (distance from window edge, result)
    frames: dict[int, np.ndarray] = {}
    frame_iter = capture.frames(ids)
    t_windows = []

    def flush(up_to_index: int) -> None:
        for fid in ids[:up_to_index]:
            if fid in held:
                depth, weight, K = held.pop(fid)[1]
                cache.write(fid, depth, weight, K)

    for k, start in enumerate(starts):
        wids = ids[start:start + opts["window"]]
        while any(f not in frames for f in wids):
            fid, rgb = next(frame_iter)
            frames[fid] = rgb
        t0 = time.perf_counter()
        result = run_window(model, capture, frames, wids, opts)
        t_windows.append(time.perf_counter() - t0)
        if not begun:
            h, w = next(iter(result.values()))[0].shape
            cache.begin(NAME, w, h, extra={
                "model": opts["model"],
                "ignore_depth_inputs": bool(opts["ignore_depth_inputs"]),
                "align_scale_to_lidar": bool(opts["align_scale_to_lidar"]),
                "window": opts["window"],
                "overlap": opts["overlap"],
                "input_size": list(capture.color_size),
                "weight": "(1 - 1/conf) * mask * non_ambiguous_mask",
            })
            begun = True
        for pos, fid in enumerate(wids):
            dist = min(pos, len(wids) - 1 - pos)
            if fid not in held or dist > held[fid][0]:
                held[fid] = (dist, result[fid])
        next_start = starts[k + 1] if k + 1 < len(starts) else len(ids)
        flush(next_start)
        for fid in [f for f in frames if f not in ids[next_start:]]:
            del frames[fid]
        print(f"mapanything: window {k + 1}/{len(starts)} ({len(wids)} views) {t_windows[-1]:.1f}s")

    flush(len(ids))
    cache.meta["seconds_per_window"] = float(np.mean(t_windows))
    cache.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lidar_grid_depth(depth: np.ndarray, weight: np.ndarray, K: np.ndarray, capture: Capture, fid: int) -> np.ndarray:
    """Resample a prediction onto the 256x192 LiDAR grid (nearest, holes preserved), for comparisons."""
    Kd = scale_intrinsics(capture.intrinsics_color(fid), capture.color_size, (DEPTH_WIDTH, DEPTH_HEIGHT))
    u, v = np.meshgrid(np.arange(DEPTH_WIDTH, dtype=np.float32), np.arange(DEPTH_HEIGHT, dtype=np.float32))
    mu = ((u - Kd[0, 2]) / Kd[0, 0] * K[0, 0] + K[0, 2]).astype(np.float32)
    mv = ((v - Kd[1, 2]) / Kd[1, 1] * K[1, 1] + K[1, 2]).astype(np.float32)
    d = cv2.remap(np.where(weight > 0, depth, 0.0).astype(np.float32), mu, mv, cv2.INTER_NEAREST,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return d
