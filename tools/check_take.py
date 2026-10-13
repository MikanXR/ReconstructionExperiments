"""Check a MikanARStreamer take before fusing it.

    python tools/check_take.py <take>.mp4 [--stride 30] [--pairs 8]

Loads the take through recon.capture.load_mikan_take, prints what it holds,
and reprojects the LiDAR between frame pairs `stride` apart. With correct
poses and axes the median absolute depth difference is a few millimetres;
a wrong camera-axis convention gives tens of centimetres or metres, and a
pose track out of step with the depth records gives a scattered result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.capture import load_mikan_take  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("movie", help="path to <take>.mp4; the sidecars are found beside it")
    ap.add_argument("--stride", type=int, default=30, help="frames between the two frames of a pair")
    ap.add_argument("--pairs", type=int, default=8, help="how many pairs to check, spread over the take")
    args = ap.parse_args()

    capture = load_mikan_take(args.movie)
    n = len(capture)
    width, height = capture.depth_size
    print(f"{capture.name}: {n} frames, colour {capture.color_size[0]}x{capture.color_size[1]}, "
          f"depth {width}x{height} at {capture.depth_unit_m * 1000:g} mm/unit, "
          f"{capture.timestamps[-1] - capture.timestamps[0]:.1f} s")

    conf = np.stack([capture.confidence(i) for i in range(0, n, max(1, n // 20))])
    high = float((conf == 2).mean())
    valid = float(np.mean([(capture.depth(i) > 0).mean() for i in range(0, n, max(1, n // 20))]))
    print(f"sampled frames: {valid:.0%} of depth pixels valid, {high:.0%} at high confidence")

    if n <= args.stride:
        print(f"take is shorter than the stride ({n} frames); nothing to compare")
        return 1

    starts = np.linspace(0, n - 1 - args.stride, num=min(args.pairs, n - args.stride), dtype=int)
    medians = []
    for a in starts:
        b = a + args.stride
        result = capture.check_pose_consistency(int(a), int(b))
        medians.append(result["median_abs_diff_m"])
        print(f"frames {a:5d} -> {b:5d}: compared {result['compared']:6d}, "
              f"median {result['median_abs_diff_m'] * 1000:7.1f} mm, p90 {result['p90_abs_diff_m'] * 1000:7.1f} mm")

    overall = float(np.nanmedian(medians)) * 1000
    print(f"median over pairs: {overall:.1f} mm")
    if overall > 50:
        print("that is far beyond ARKit drift: suspect the camera-axis convention or record alignment")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
