"""Run a set of methods over every capture, serially, then build the summary.

    python run_all.py [--captures ID ...] [--methods ...] [--skip-existing]

Each method runs in its own venv (see VENVS), so this script only shells out
to run.py; the GPU is used by one run at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# method -> (venv python, run.py arguments after the method name)
RUNS = {
    "lidar_tsdf": (".venv", []),
    "promptda": (".venv_promptda", []),
    "promptda_conf0": (".venv_promptda", ["promptda", "--tag", "conf0", "--opt", "promptda.conf0_weight=0"]),
    "mapanything_aligned": (".venv_mapanything", ["mapanything", "--tag", "aligned", "--opt", "mapanything.align_scale_to_lidar=true"]),
    "da3_large": (".venv_da3", ["da3", "--tag", "large", "--opt", "da3.model=depth-anything/DA3-LARGE-1.1"]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*")
    ap.add_argument("--methods", nargs="*", default=["lidar_tsdf", "promptda", "mapanything_aligned"])
    ap.add_argument("--skip-existing", action="store_true", help="skip a run whose metrics.json already has an eval section")
    args = ap.parse_args()

    captures = args.captures or sorted(p.name for p in (ROOT / "captures").iterdir() if (p / "odometry.csv").exists())
    for cap in captures:
        for run in args.methods:
            venv, extra = RUNS[run]
            method_args = extra or [run]
            out_metrics = ROOT / "out" / cap / run / "metrics.json"
            if args.skip_existing and out_metrics.exists() and "eval" in json.loads(out_metrics.read_text()):
                print(f"== {cap} {run}: done, skipping")
                continue
            cmd = [str(ROOT / venv / "Scripts" / "python.exe"), str(ROOT / "run.py"), *method_args, "--capture", cap]
            print(f"== {cap} {run}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"== {cap} {run}: FAILED ({result.returncode})", flush=True)
    subprocess.run([sys.executable, str(ROOT / "summarize.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
