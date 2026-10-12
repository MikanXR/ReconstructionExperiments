"""Prompt Depth Anything (ViT-L): RGB plus the LiDAR frame as a metric prompt.

Code https://github.com/DepthAnything/PromptDA (Apache 2.0), weights
depth-anything/prompt-depth-anything-vitl on HuggingFace (Apache 2.0).

Install (own venv, the upstream requirements.txt pins torch 2.0.1, xformers,
and python<=3.11, so it is not used):

    python -m venv .venv_promptda
    .venv_promptda\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    .venv_promptda\\Scripts\\python -m pip install -r requirements.txt
    git clone --depth 1 https://github.com/DepthAnything/PromptDA third_party/PromptDA

No pip install of the PromptDA package: its setup.py looks for packages under
src/ (there is none) and the promptda/ tree has no __init__.py files, so an
editable install would be empty. This module puts third_party/PromptDA on
sys.path instead. The model code needs only torch, numpy, and huggingface_hub
(all in requirements.txt). The vendored DINOv2 falls back to plain attention
when xformers is absent, so xformers is not installed.

PromptDA.from_pretrained downloads model.ckpt (1.5 GB) into the HuggingFace
cache and loads it unmodified (the Lightning checkpoint loads under torch's
weights_only default). No monkeypatch is needed.

Method:
- image: long side resized to 1008, both sides floored to a multiple of 14
  (the convention of promptda.utils.io_wrapper.load_image), area
  interpolation. 1920x1440 becomes 1008x756. K scales linearly with the pure
  resize.
- prompt: the 256x192 LiDAR depth in metres. The model min/max normalizes the
  prompt and maps its sigmoid output back to that range, so holes must not be
  zero: confidence-0 and zero pixels are filled by nearest valid pixel. A
  frame with no valid LiDAR is skipped with a warning.
- weight: the model has no confidence output. Weight is 1 everywhere,
  lowered to 0.5 where the nearest-upsampled LiDAR confidence is 1 and to
  `promptda.conf0_weight` (default 0.25) where it is 0. Confidence 0 is also
  what ARKit reports beyond the LiDAR's range, where the prompt is a
  nearest-fill guess and the model extrapolates freely; a weight of 0 there
  keeps the fused mesh inside the sensed volume.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from recon.capture import Capture, scale_intrinsics
from recon.depthcache import DepthCache

NAME = "promptda"
LICENSE = "Apache 2.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTDA_DIR = REPO_ROOT / "third_party" / "PromptDA"
HF_REPO = "depth-anything/prompt-depth-anything-vitl"
PATCH = 14
DEFAULT_MAX_SIZE = 1008
DEFAULT_CONF0_WEIGHT = 0.25


def _load_model(device):
    if str(PROMPTDA_DIR) not in sys.path:
        sys.path.insert(0, str(PROMPTDA_DIR))
    from promptda.promptda import PromptDA
    from promptda.utils.logger import Log

    Log.log_on = False
    return PromptDA.from_pretrained(HF_REPO).to(device).eval()


def target_size(color_size: tuple[int, int], max_size: int = DEFAULT_MAX_SIZE) -> tuple[int, int]:
    """(width, height) after the load_image convention: long side to max_size, floored to multiples of 14."""
    w, h = color_size
    max_size = max_size // PATCH * PATCH
    scale = max_size / max(w, h) if max(w, h) > max_size else 1.0
    return int(w * scale // PATCH * PATCH), int(h * scale // PATCH * PATCH)


def fill_prompt(depth: np.ndarray, confidence: np.ndarray) -> np.ndarray | None:
    """Nearest-valid fill of zero and confidence-0 pixels. None when nothing is valid."""
    valid = (depth > 0) & (confidence > 0)
    if not valid.any():
        return None
    if valid.all():
        return depth.astype(np.float32)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return depth[tuple(idx)].astype(np.float32)


def predict(capture: Capture, frame_ids, cache: DepthCache, cfg: dict) -> None:
    import torch
    from tqdm import tqdm

    opts = cfg.get("promptda", {}) if cfg else {}
    max_size = int(opts.get("max_size", DEFAULT_MAX_SIZE))
    confidence_weight = np.array([float(opts.get("conf0_weight", DEFAULT_CONF0_WEIGHT)), 0.5, 1.0], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_w, out_h = target_size(capture.color_size, max_size)
    cache.begin(NAME, out_w, out_h, extra={"checkpoint": HF_REPO, "max_size": max_size,
                                           "prompt_fill": "nearest_valid", "weight_by_confidence": confidence_weight.tolist()})
    model = _load_model(device)
    try:
        frame_ids = [int(f) for f in frame_ids]
        for fid, rgb in tqdm(capture.frames(frame_ids), total=len(frame_ids), desc=NAME):
            prompt = fill_prompt(capture.depth(fid), capture.confidence(fid))
            if prompt is None:
                warnings.warn(f"frame {fid}: no valid LiDAR, skipped")
                continue
            image = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            image_t = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
            prompt_t = torch.from_numpy(prompt)[None, None].to(device)
            with torch.no_grad():
                depth = model.predict(image_t, prompt_t)[0, 0].float().cpu().numpy()
            depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
            conf_up = cv2.resize(capture.confidence(fid), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            weight = confidence_weight[np.clip(conf_up, 0, 2)]
            K = scale_intrinsics(capture.intrinsics_color(fid), capture.color_size, (out_w, out_h))
            cache.write(fid, depth, weight, K)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cache.finish()
