"""Run sam2_edge_eval.py over the four hand-marked sets for one segmenter family, in both framings,
and tabulate every family's summaries into the one table NOTES.md carries.

    <family venv>/Scripts/python tools/segmenter_sweep.py run --family edgetam [--models ...] [--size 0]
        [--variants runs_pos,runs_box_pos] [--band 24]
    python tools/segmenter_sweep.py table [--variant runs_pos] [--out out/segmenter_sweep.md]

`run` calls sam2_edge_eval.py with the running interpreter once per set and framing, writing to
out/sweep_<family>_edges<_set>/<model>[_full][_<size>], the earlier SAM2 runs' layout under a sweep
prefix so those folders stay the record (the can set has no suffix). `table` needs only the standard
library: it reads every out/sweep_*_edges*/<model>/summary.txt, keys each by
family, model, size and framing, and writes one markdown row per candidate with median / p90 /
within 5 px per set for the chosen variant, the crop framing first and the full frame second.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {"can": "live_occlusion", "cup": "mark_cup", "room": "mark_room", "duster": "mark_duster"}


def run(args) -> None:
    for set_name, folder in SETS.items():
        for full in (False, True):
            out = ROOT / "out" / (f"sweep_{args.family}_edges" + ("" if set_name == "can" else f"_{set_name}"))
            cmd = [sys.executable, str(ROOT / "tools" / "sam2_edge_eval.py"), "--family", args.family,
                   "--root", str(ROOT / "out" / folder), "--variants", args.variants, "--band", str(args.band),
                   "--out", str(out)]
            if args.models:
                cmd += ["--models", args.models]
            if args.size:
                cmd += ["--size", str(args.size)]
            if full:
                cmd.append("--full-frame")
            print("::", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=False)


def parse_summary(path: Path, variant: str) -> tuple[float, float, float] | None:
    for line in path.read_text().splitlines():
        if line.startswith(f"sam2_{variant} "):
            cols = line.split()
            return float(cols[2]), float(cols[3]), float(cols[5])
    return None


def table(args) -> None:
    rows: dict[tuple, dict] = {}
    for out_dir in sorted((ROOT / "out").glob("sweep_*_edges*")):
        m = re.fullmatch(r"sweep_(.+?)_edges(?:_(cup|room|duster))?", out_dir.name)
        if not m:
            continue
        family, set_name = m.group(1), m.group(2) or "can"
        for model_dir in sorted(p for p in out_dir.iterdir() if (p / "summary.txt").exists()):
            name = model_dir.name
            full = "_full" in name
            size = re.search(r"_(\d+)$", name.replace("_full", ""))
            model = re.sub(r"_full|_\d+$", "", name.replace("_full", ""))
            key = (family, model, int(size.group(1)) if size else 0, full)
            got = parse_summary(model_dir / "summary.txt", args.variant)
            if got:
                rows.setdefault(key, {})[set_name] = got
    lines = [f"| family | model | input | framing | " + " | ".join(SETS) + " |",
             "|---|---|---|---|" + "---|" * len(SETS)]
    for (family, model, size, full), per_set in sorted(rows.items()):
        cells = []
        for set_name in SETS:
            g = per_set.get(set_name)
            cells.append(f"{g[0]:.1f} / {g[1]:.1f} / {g[2]:.2f}" if g else "")
        lines.append(f"| {family} | {model} | {size or 'own'} | {'full frame' if full else 'crop'} | "
                     + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", required=True)
    r.add_argument("--models", default=None)
    r.add_argument("--size", type=int, default=0)
    r.add_argument("--variants", default="runs_pos,runs_box_pos")
    r.add_argument("--band", type=int, default=24)
    r.set_defaults(fn=run)
    t = sub.add_parser("table")
    t.add_argument("--variant", default="runs_pos")
    t.add_argument("--out", type=Path, default=ROOT / "out" / "segmenter_sweep.md")
    t.set_defaults(fn=table)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
