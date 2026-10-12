"""Re-run the fuse, eval, and render stages for every existing run, from the
settings each run recorded in its metrics.json, then rebuild every report
and the summary. For when fusion or evaluation code changes and the depth
caches are still valid.

    python refuse_all.py [--stages fuse eval render] [--captures ID ...]

Each run is replayed through run.py with its stored config written to a
temporary yaml, so tags and --opt overrides reproduce exactly. The baseline
of each capture is replayed first because the other runs' evaluation
compares against its mesh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def replay(capture: str, run_dir: Path, stages: list[str]) -> bool:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    method = metrics["method"]["name"]
    run_name = metrics["method"]["run"]
    tag = run_name[len(method) + 1:] if run_name != method else ""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=ROOT / "out") as f:
        yaml.safe_dump(metrics["config"], f)
        cfg_path = f.name
    try:
        for stage in stages:
            cmd = [sys.executable, str(ROOT / "run.py"), method, "--capture", capture, "--config", cfg_path, "--stage", stage]
            if tag:
                cmd += ["--tag", tag]
            print(f"== {capture} {run_name} {stage}", flush=True)
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                print(f"== {capture} {run_name} {stage}: FAILED", flush=True)
                return False
    finally:
        Path(cfg_path).unlink(missing_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["fuse", "eval", "render"])
    ap.add_argument("--captures", nargs="*")
    args = ap.parse_args()

    out_root = ROOT / "out"
    captures = args.captures or sorted(p.name for p in out_root.iterdir() if p.is_dir() and (ROOT / "captures" / p.name).exists())
    for capture in captures:
        runs = sorted(p.parent for p in (out_root / capture).glob("*/metrics.json"))
        runs.sort(key=lambda p: (not p.name.startswith("lidar_tsdf"), p.name))   # baselines first
        for run_dir in runs:
            if "eval" not in json.loads((run_dir / "metrics.json").read_text()):
                continue
            replay(capture, run_dir, args.stages)
        subprocess.run([sys.executable, str(ROOT / "run.py"), "lidar_tsdf", "--capture", capture, "--stage", "report"], cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "summarize.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
