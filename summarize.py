"""Aggregate every capture's metrics.json files into out/summary.md.

    python summarize.py [--out-root out]

One table per metric with captures as rows and runs as columns, plus a mean
row, so a method's behaviour across scenes is readable at a glance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("edge mean px", lambda m: m["eval"]["summary"]["edge_symmetric_px"], "{:.1f}", "lower"),
    ("within 2 cm", lambda m: m["eval"]["summary"]["within_2cm"] * 100, "{:.1f}", "higher"),
    ("MAE cm", lambda m: m["eval"]["summary"]["mae_m"] * 100, "{:.2f}", "lower"),
    ("completeness", lambda m: m["eval"]["summary"]["completeness"] * 100, "{:.1f}", "higher"),
    ("floater hits", lambda m: m["eval"]["summary"]["floater_fraction"] * 100, "{:.2f}", "lower"),
    ("unsupported surface", lambda m: m["eval"]["summary"]["unsupported_surface_fraction"] * 100, "{:.1f}", "lower"),
    ("F-score vs LiDAR", lambda m: m["eval"].get("vs_baseline", {}).get("fscore", float("nan")), "{:.3f}", "higher"),
    ("predict s", lambda m: m["depth"]["seconds"], "{:.0f}", "lower"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--all", action="store_true",
                    help="include runs that exist for fewer than half the captures (single-capture variants)")
    args = ap.parse_args()
    root = Path(args.out_root)

    data: dict[str, dict[str, dict]] = {}
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        if "eval" not in m:
            continue
        data.setdefault(metrics_path.parent.parent.name, {})[metrics_path.parent.name] = m
    captures = sorted(data)
    all_runs = sorted({r for c in data.values() for r in c})
    if args.all:
        runs = all_runs
    else:
        runs = [r for r in all_runs if sum(r in data[c] for c in captures) * 2 >= len(captures)]
    omitted = [r for r in all_runs if r not in runs]

    lines = ["# Cross-capture summary", "",
             f"{len(captures)} captures, {len(runs)} runs. Each table: captures as rows, runs as columns, mean over the captures where the run exists. Per-capture detail and renders are in `out/<capture>/report.md`."
             + (f" Omitted single-capture variants (pass --all to include): {', '.join(omitted)}." if omitted else ""), ""]
    for name, getter, fmt, better in METRICS:
        lines.append(f"## {name} ({better} is better)")
        lines.append("")
        lines.append("| capture | " + " | ".join(runs) + " |")
        lines.append("|---|" + "---|" * len(runs))
        cols: dict[str, list[float]] = {r: [] for r in runs}
        for cap in captures:
            cells = []
            for r in runs:
                m = data[cap].get(r)
                try:
                    v = float(getter(m)) if m else float("nan")
                except (KeyError, TypeError):
                    v = float("nan")
                if np.isfinite(v):
                    cols[r].append(v)
                cells.append(fmt.format(v) if np.isfinite(v) else "")
            lines.append(f"| {cap} | " + " | ".join(cells) + " |")
        lines.append("| **mean** | " + " | ".join(fmt.format(np.mean(cols[r])) if cols[r] else "" for r in runs) + " |")
        lines.append("")
    path = root / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
