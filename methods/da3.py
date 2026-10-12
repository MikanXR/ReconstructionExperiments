"""Depth Anything 3 in pose-conditioned mode, fed the ARKit poses and intrinsics.

The sequence is run in overlapping windows of consecutive training frames. DA3
normalises the input camera path, predicts depth plus its own camera poses in
that normalised frame, then (align_to_input_ext_scale=True) Umeyama-aligns the
predicted poses to the supplied ones and divides the depth by the fitted scale,
so every window comes out in the metric scale of the ARKit poses. Frames that
fall in two windows keep the prediction from the window where they sit farther
from the edge.

Depth is stored at the DA3 process resolution (longest side PROCESS_RES rounded
to a multiple of 14: 504x378 for the 4:3 capture) with the intrinsics scaled to
match. Images are resized to that size here so DA3's own resize is the identity
and the stored intrinsics equal what the model saw.

Weight: DA3's confidence is exp(x)+1, so it lives in [1, inf). The repo's GLB
exporter keeps points with conf >= 1.05 (raised adaptively to the 40th
percentile). The fusion weight is the linear ramp
clip((conf - CONF_LOW) / (CONF_HIGH - CONF_LOW), 0, 1) with CONF_LOW = 1.05
mirroring that floor and CONF_HIGH = 3.0 chosen from the smoke-test
distribution (DA3-BASE, indoor 4:3 frames: conf p5/p25/p50/p90 of about
1.8/3.1/3.9/4.7) so the median pixel saturates at 1 and the 5th percentile
lands near 0.4.

Install (separate venv, Windows, CUDA 12.8 wheels; third_party/ is gitignored):

    python -m venv .venv_da3
    .venv_da3\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .venv_da3\\Scripts\\python -m pip install -r requirements.txt
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3 third_party/Depth-Anything-3
    .venv_da3\\Scripts\\python -m pip install -r requirements-da3.txt
    .venv_da3\\Scripts\\python -m pip install -e third_party/Depth-Anything-3 --no-deps

xformers: DA3 lists it as a hard dependency but only the Giant tier's SwiGLU
FFN imports it (model/dinov2/layers/swiglu_ffn.py, inside try/except
ImportError with a plain PyTorch SwiGLUFFN fallback). Small/Base/Large use the
standard MLP and never touch it, so this module runs without xformers and no
sys.modules stub is needed. --no-deps keeps pip from replacing the CUDA torch
with the CPU wheel that an unpinned xformers from PyPI drags in. Do not stub
xformers.ops.SwiGLU with an identity if you run Giant: that silently disables
the FFN.

Config (configs/default.yaml, all optional):

    da3:
      model: depth-anything/DA3-BASE     # or DA3-SMALL, DA3-LARGE-1.1, DA3-GIANT
      window: 32
      overlap: 8
      process_res: 504
"""

from __future__ import annotations

import time

import numpy as np

from recon.capture import Capture, scale_intrinsics
from recon.depthcache import DepthCache

NAME = "da3"
LICENSE = ("Apache 2.0 for DA3-SMALL and DA3-BASE. DA3-LARGE-1.1 is listed CC BY-NC 4.0 in the GitHub README "
           "but apache-2.0 on its Hugging Face model card (disputed). DA3-GIANT is CC BY-NC 4.0.")

DEFAULT_MODEL = "depth-anything/DA3-BASE"
WINDOW = 32          # frames per inference call
OVERLAP = 8          # frames shared by consecutive windows
PROCESS_RES = 504    # DA3 default: longest image side
PATCH = 14           # DA3 rounds each side to a multiple of the ViT patch
CONF_LOW = 1.05      # weight 0 at and below: the repo's GLB export floor
CONF_HIGH = 3.0      # weight 1 at and above: see module docstring


def process_size(color_size: tuple[int, int], process_res: int = PROCESS_RES) -> tuple[int, int]:
    """(width, height) DA3 would resize a color_size image to with upper_bound_resize."""
    w, h = color_size
    scale = process_res / float(max(w, h))
    w, h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    def nearest_multiple(x: int) -> int:
        down = (x // PATCH) * PATCH
        up = down + PATCH
        return up if abs(up - x) <= abs(x - down) else down

    return max(1, nearest_multiple(w)), max(1, nearest_multiple(h))


def plan_windows(n: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges covering 0..n-1; the last window is pulled back to end at n."""
    if n <= window:
        return [(0, n)]
    step = window - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than window")
    starts = list(range(0, n - window, step)) + [n - window]
    return [(s, s + window) for s in starts]


def assign_frames(n: int, windows: list[tuple[int, int]]) -> np.ndarray:
    """For every frame index, the window index whose edge it is farthest from."""
    owner = np.full(n, -1, dtype=int)
    best = np.full(n, -1, dtype=int)
    for w, (s, e) in enumerate(windows):
        idx = np.arange(s, e)
        margin = np.minimum(idx - s, e - 1 - idx)
        take = margin > best[idx]
        owner[idx[take]] = w
        best[idx[take]] = margin[take]
    return owner


def conf_to_weight(conf: np.ndarray, depth: np.ndarray) -> np.ndarray:
    weight = np.clip((conf - CONF_LOW) / (CONF_HIGH - CONF_LOW), 0.0, 1.0).astype(np.float32)
    weight[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return weight


def load_model(model_id: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def infer_window(model, images: list[np.ndarray], T_WC: np.ndarray, K: np.ndarray, process_res: int = PROCESS_RES):
    """Run one window. images are HxWx3 uint8 already at process size, K at that size, T_WC camera-to-world.

    Returns (depth (N,H,W) float32 metres, conf (N,H,W) float32)."""
    extrinsics = np.linalg.inv(T_WC).astype(np.float32)   # DA3 wants world-to-camera, OpenCV axes
    pred = model.inference(
        images,
        extrinsics=extrinsics,
        intrinsics=K.astype(np.float32),
        align_to_input_ext_scale=True,
        ref_view_strategy="middle",
        process_res=process_res,
        process_res_method="upper_bound_resize",
    )
    depth = np.ascontiguousarray(pred.depth, dtype=np.float32)
    conf = np.ascontiguousarray(pred.conf, dtype=np.float32) if pred.conf is not None else np.full_like(depth, CONF_HIGH)
    return depth, conf


def predict(capture: Capture, frame_ids, cache: DepthCache, cfg: dict) -> None:
    import cv2
    import torch

    opts = cfg.get("da3", {}) or {}
    model_id = opts.get("model", DEFAULT_MODEL)
    window = int(opts.get("window", WINDOW))
    overlap = int(opts.get("overlap", OVERLAP))
    process_res = int(opts.get("process_res", PROCESS_RES))

    window_mode = opts.get("window_mode", "consecutive")

    frame_ids = [int(f) for f in frame_ids]
    n = len(frame_ids)
    w, h = process_size(capture.color_size, process_res)

    cache.begin(NAME, w, h, extra=dict(model=model_id, window=window, overlap=overlap, process_res=process_res,
                                       window_mode=window_mode,
                                       conf_low=CONF_LOW, conf_high=CONF_HIGH, ref_view_strategy="middle"))
    if window_mode == "strided":
        _predict_strided(capture, frame_ids, cache, model_id, window, process_res, (w, h))
        cache.finish()
        return
    if window_mode != "consecutive":
        raise ValueError(f"da3.window_mode must be consecutive or strided, got {window_mode!r}")

    windows = plan_windows(n, window, overlap)
    owner = assign_frames(n, windows)
    model = load_model(model_id)
    try:
        images: dict[int, np.ndarray] = {}     # position -> resized RGB, kept only while a pending window needs it
        pos_of = {fid: i for i, fid in enumerate(frame_ids)}
        next_window = 0
        for fid, rgb in capture.frames(frame_ids):
            i = pos_of[fid]
            images[i] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            while next_window < len(windows) and windows[next_window][1] - 1 <= i:
                s, e = windows[next_window]
                idx = list(range(s, e))
                t0 = time.perf_counter()
                depth, conf = infer_window(
                    model,
                    [images[j] for j in idx],
                    np.stack([capture.pose(frame_ids[j]) for j in idx]),
                    np.stack([scale_intrinsics(capture.intrinsics_color(frame_ids[j]), capture.color_size, (w, h)) for j in idx]),
                    process_res,
                )
                for k, j in enumerate(idx):
                    if owner[j] == next_window:
                        cache.write(frame_ids[j], depth[k], conf_to_weight(conf[k], depth[k]),
                                    scale_intrinsics(capture.intrinsics_color(frame_ids[j]), capture.color_size, (w, h)))
                print(f"da3: window {next_window + 1}/{len(windows)} frames {frame_ids[s]}..{frame_ids[e - 1]} "
                      f"in {time.perf_counter() - t0:.1f}s")
                next_window += 1
                keep_from = windows[next_window][0] if next_window < len(windows) else n
                for j in [j for j in images if j < keep_from]:
                    del images[j]
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache.finish()


def _predict_strided(capture: Capture, frame_ids: list[int], cache: DepthCache, model_id: str,
                     window: int, process_res: int, size: tuple[int, int]) -> None:
    """Strided windows: every window takes every S-th frame across the whole
    sequence (S = ceil(n / window)), so each window spans the full camera path
    instead of about three seconds of it. Each frame belongs to exactly one
    window. All frames are decoded up front at process size (about 0.6 MB
    each)."""
    import cv2
    import torch

    w, h = size
    n = len(frame_ids)
    stride = max(1, -(-n // window))
    windows = [list(range(r, n, stride)) for r in range(stride)]
    images = {}
    for fid, rgb in capture.frames(frame_ids):
        images[fid] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    model = load_model(model_id)
    try:
        for wi, idx in enumerate(windows):
            t0 = time.perf_counter()
            ids = [frame_ids[j] for j in idx]
            Ks = np.stack([scale_intrinsics(capture.intrinsics_color(f), capture.color_size, (w, h)) for f in ids])
            depth, conf = infer_window(model, [images[f] for f in ids], np.stack([capture.pose(f) for f in ids]), Ks, process_res)
            for k, f in enumerate(ids):
                cache.write(f, depth[k], conf_to_weight(conf[k], depth[k]), Ks[k])
            print(f"da3: strided window {wi + 1}/{len(windows)} ({len(ids)} frames, stride {stride}) in {time.perf_counter() - t0:.1f}s")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
