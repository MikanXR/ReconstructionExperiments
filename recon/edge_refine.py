"""The image-space depth edge refinement MikanXR's DepthEdgeRefineNode runs, as a numpy reference.

refine_depth() is the reference the node's GLSL and C++ reference follow:
window min/max over +/- radius (separable), a band where
(max - min) > jump * max, and inside the band a two-prototype colour
classification: the window's taps split into a near and a far side at the
midpoint depth, each side's mean colour is its prototype, and the pixel joins
the side whose prototype its colour is nearer to, at that side's mean depth,
unless the two prototypes are closer than color_sigma apart.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def window_min_max(depth: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Passes 1 and 2: per-pixel min and max over the (2r+1)^2 window with clamped borders."""
    size = 2 * radius + 1
    mn = minimum_filter1d(minimum_filter1d(depth, size, axis=1, mode="nearest"), size, axis=0, mode="nearest")
    mx = maximum_filter1d(maximum_filter1d(depth, size, axis=1, mode="nearest"), size, axis=0, mode="nearest")
    return mn, mx


def refine_depth(depth: np.ndarray, rgb_linear: np.ndarray, radius: int, color_sigma: float, depth_jump: float,
                 chunk: int = 2048) -> np.ndarray:
    """Pass 3 over the band pixels. depth is HxW float32 in [0, 1], rgb_linear HxWx3 float32 in [0, 1].

    Inside a band the window's taps split at the midpoint depth into a near and
    a far side, each side's mean colour is its prototype, and the pixel joins
    the side whose prototype its own colour is closer to, taking that side's
    mean depth. Two prototypes closer than color_sigma are no evidence and the
    pixel keeps its depth. Box weights, so the reach is the whole radius."""
    depth = depth.astype(np.float32)
    mn, mx = window_min_max(depth, radius)
    band = (mx - mn) > depth_jump * mx
    out = depth.copy()
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        return out

    r = radius
    k = 2 * r + 1
    depth_pad = np.pad(depth, r, mode="edge")
    rgb_pad = np.pad(rgb_linear.astype(np.float32), ((r, r), (r, r), (0, 0)), mode="edge")
    depth_win = sliding_window_view(depth_pad, (k, k))              # (H, W, k, k)
    rgb_win = sliding_window_view(rgb_pad, (k, k), axis=(0, 1))      # (H, W, 3, k, k)
    mid = 0.5 * (mn + mx)
    sigma_sq = color_sigma * color_sigma

    for start in range(0, len(ys), chunk):
        sy = ys[start:start + chunk]
        sx = xs[start:start + chunk]
        dq = depth_win[sy, sx]                                   # (n, k, k)
        iq = rgb_win[sy, sx]                                     # (n, 3, k, k)
        ip = rgb_linear[sy, sx].astype(np.float32)               # (n, 3)
        near = dq < mid[sy, sx][:, None, None]
        n_near = np.sum(near, axis=(1, 2)).astype(np.float32)
        n_far = near[0].size - n_near
        both = (n_near > 0) & (n_far > 0)
        mu_near = np.sum(iq * near[:, None], axis=(2, 3)) / np.maximum(n_near, 1)[:, None]
        mu_far = np.sum(iq * ~near[:, None], axis=(2, 3)) / np.maximum(n_far, 1)[:, None]
        d_near = np.sum(dq * near, axis=(1, 2)) / np.maximum(n_near, 1)
        d_far = np.sum(dq * ~near, axis=(1, 2)) / np.maximum(n_far, 1)
        separated = np.sum((mu_near - mu_far) ** 2, axis=1) >= sigma_sq
        pick_near = np.sum((ip - mu_near) ** 2, axis=1) <= np.sum((ip - mu_far) ** 2, axis=1)
        decided = both & separated
        result = np.where(pick_near, d_near, d_far)
        out[sy, sx] = np.where(decided, result, depth[sy, sx])
    return out
