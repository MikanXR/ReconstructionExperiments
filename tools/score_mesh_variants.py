"""Score occlusion mesh variants on hand-marked take frames: the occluder's width error and shift,
and the distance from its occluding edges to the mesh silhouette.

    python tools/score_mesh_variants.py [--root out/live_occlusion/variants] [--width-mm 66]
           [variant ...]

Reads <root>/<variant>/<frame> folders written by capture_mesh_variants.py. Per frame, through the
middle of the marked occluder (25 to 85 percent of its height, past the cap and the base), each
row's mesh silhouette edges are compared with the marked ones on the occluding sides: the mean of
the two offsets is the shift, their difference the width error. --width-mm converts the width
error to millimetres per side from the occluder's known width.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402


def edge_offsets(ctx: E.Context) -> tuple[float, float, float]:
    """Median left and right silhouette offsets (px, + = right of the marked edge) and the marked width."""
    ys, _ = np.nonzero(ctx.gt)
    occl = cv2.dilate(ctx.occluding.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    lefts, rights, widths = [], [], []
    top, bottom = ys.min(), ys.max()
    for y in range(int(top + 0.25 * (bottom - top)), int(top + 0.85 * (bottom - top))):
        gx = np.nonzero(ctx.gt[y])[0]
        if len(gx) == 0:
            continue
        gl, gr = gx.min(), gx.max()
        row = ctx.prior[y]
        c = (gl + gr) // 2
        if not row[c]:
            idx = np.nonzero(row)[0]
            if len(idx) == 0:
                continue
            c = idx[np.argmin(np.abs(idx - c))]
        l = c
        while l > 0 and row[l - 1]:
            l -= 1
        r = c
        while r < len(row) - 1 and row[r + 1]:
            r += 1
        if occl[y, gl] and abs(l - gl) < 120:
            lefts.append(l - gl)
        if occl[y, gr] and abs(r - gr) < 120:
            rights.append(r - gr)
        widths.append(gr - gl)
    left = float(np.median(lefts)) if lefts else np.nan
    right = float(np.median(rights)) if rights else np.nan
    return left, right, float(np.median(widths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion" / "variants")
    parser.add_argument("--width-mm", type=float, default=66.0)
    args = parser.parse_args()
    variants = args.variants or sorted(p.name for p in args.root.iterdir() if p.is_dir())

    print(f"{'variant':8s} {'frame':9s} {'shift px':>8s} {'width px':>8s} {'mm/side':>7s} {'occl med':>8s} {'<=5px':>6s}")
    totals = {}
    for variant in variants:
        mm_side, medians = [], []
        for frame in sorted(p for p in (args.root / variant).iterdir() if (p / "gt_mask.png").exists()):
            ctx = E.load_context(frame)
            left, right, width = edge_offsets(ctx)
            shift = np.nanmean([left, right])
            width_err = right - left
            mm = width_err / width * args.width_mm / 2
            s = E.score(ctx, ctx.prior)
            mm_side.append(mm)
            medians.append(s["gt_to_pred_median"])
            print(f"{variant:8s} {frame.name:9s} {shift:8.1f} {width_err:8.1f} {mm:7.1f} "
                  f"{s['gt_to_pred_median']:8.1f} {s['within_5px']:6.2f}")
        totals[variant] = (np.nanmean(mm_side), np.mean(medians))
    print()
    for variant, (mm, med) in totals.items():
        print(f"{variant:8s} width error {mm:5.1f} mm/side, occluding median {med:5.1f} px")


if __name__ == "__main__":
    main()
