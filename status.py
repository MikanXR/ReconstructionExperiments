"""Show the state of every run for every capture.

    python status.py [--methods ...] [--all]

Per run the state is one of:

    done                every stage is newer than the code it depends on
    stale: fuse,eval    the named stages ran before the current fusion,
                        evaluation, or render code (or before an upstream
                        stage), so their results are out of date
    123/305 frames      depth prediction in progress or stopped partway
    fused, no eval      a mesh exists but evaluation has not run
    -                   nothing yet

Stage times come from the `stages` block run.py writes into metrics.json;
older runs without it fall back to file modification times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

DEFAULT_RUNS = ["lidar_tsdf", "promptda", "promptda_conf0", "mapanything_aligned", "da3_large"]

# Which source files each stage's result depends on.
STAGE_SOURCES = {
    "fuse": ["recon/fusion.py"],
    "eval": ["recon/evaluate.py"],
    "render": ["recon/render.py"],
}
STAGE_ORDER = ["fuse", "eval", "render"]


def expected_train_frames(capture_dir: Path, cfg: dict) -> int:
    """Count training frames the way Capture.split does, without decoding video:
    depth files, capped by the decodable frame count once a run has recorded it."""
    n = len(list((capture_dir / "depth").glob("*.png")))
    video_frames = ROOT / "out" / capture_dir.name / "video_frames.txt"
    if video_frames.exists():
        n = min(n, int(video_frames.read_text().strip()))
    keyframes = (n + cfg["keyframe_stride"] - 1) // cfg["keyframe_stride"]
    holdout_every = cfg["holdout_every"]
    holdout = len(range(holdout_every // 2, keyframes, holdout_every))
    return keyframes - holdout


def stage_time(metrics: dict, out: Path, stage: str) -> float | None:
    t = metrics.get("stages", {}).get(stage)
    if t is not None:
        return float(t)
    # Runs recorded before stage stamps existed: file times.
    if stage == "fuse" and (out / "mesh.ply").exists():
        return (out / "mesh.ply").stat().st_mtime
    if stage == "eval" and "eval" in metrics:
        return (out / "metrics.json").stat().st_mtime
    if stage == "render" and (out / "renders" / "topdown.png").exists():
        return (out / "renders" / "topdown.png").stat().st_mtime
    return None


def run_state(out: Path, expected_frames: int, code_times: dict[str, float]) -> tuple[str, bool]:
    """Return (label, up_to_date)."""
    metrics_path = out / "metrics.json"
    cache_files = len(list((out / "depth").glob("*_depth.npy"))) if (out / "depth").exists() else 0
    if not metrics_path.exists():
        return (f"{cache_files}/{expected_frames} frames" if cache_files else "-"), False
    metrics = json.loads(metrics_path.read_text())
    times = {s: stage_time(metrics, out, s) for s in STAGE_ORDER}
    if times["fuse"] is None:
        if cache_files and not metrics.get("fusion"):
            return f"{cache_files}/{expected_frames} frames", False
        return "-", False
    if "eval" not in metrics:
        return "fused, no eval", False
    stale = []
    upstream = 0.0
    stamped = metrics.get("stages", {})
    for stage in STAGE_ORDER:
        t = times[stage]
        needed = max([code_times[stage], upstream])
        if t is None or t < needed:
            stale.append(stage)
        # Only stamped times constrain later stages: a file-time fallback such
        # as metrics.json's mtime moves with every later save.
        if stage in stamped:
            upstream = max(upstream, float(stamped[stage]))
    if stale:
        return "stale: " + ",".join(stale), False
    return "done", True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--all", action="store_true", help="show every run directory present, not just --methods")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    code_times = {stage: max((ROOT / f).stat().st_mtime for f in files) for stage, files in STAGE_SOURCES.items()}

    captures = sorted(p for p in (ROOT / "captures").iterdir() if (p / "odometry.csv").exists())
    done = total = 0
    for cap in captures:
        expected = expected_train_frames(cap, cfg)
        runs = args.methods
        if args.all:
            runs = sorted(p.name for p in (ROOT / "out" / cap.name).iterdir() if p.is_dir() and p.name != "depth") if (ROOT / "out" / cap.name).exists() else []
        width = max([len(r) for r in runs] + [10])
        cells = []
        for run in runs:
            total += 1
            label, ok = run_state(ROOT / "out" / cap.name / run, expected, code_times)
            done += ok
            cells.append(f"{run:<{width}} {label:>18}")
        print(f"{cap.name}  " + "   ".join(cells))
    print(f"\n{done} / {total} runs up to date")
    summary = ROOT / "out" / "summary.md"
    if summary.exists():
        print(f"summary: {summary}")


if __name__ == "__main__":
    main()
