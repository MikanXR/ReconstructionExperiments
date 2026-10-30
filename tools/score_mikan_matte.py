"""Score the labels MikanCmd -occlusionMatte dumped against the hand-marked frames, next to the
Python SAM2 result on the same frames, as the parity check of the C++ port.

    .venv_sam2/Scripts/python tools/score_mikan_matte.py <dump dir> [--root out/live_occlusion] [--model small]
        [--times t02:60,t06:180,t10:300,t18:540] [--out out/mikan_matte]

The dump holds <fileIndex>_labels.png (0 removed, 128 keep, 255 added at the matte size),
<fileIndex>_depth.npy and <fileIndex>_prompts.json per segmented frame. --times maps each marked
take_tNN folder under --root to its file index. The C++ labels are applied to the marked frame's
prior (the mesh silhouette occlusion_edge_eval.load_context extracts): a pixel the labels add joins
the prior, one they remove leaves it, and the result is scored on the occluding edges like every
other method, in a table with the mesh and sam2_runs_pos from sam2_edge_eval.py on the same frames.
The prompts of each frame are printed beside the Python prompts for the same frame.

Writes <out>/results.csv, summary.txt and a contact sheet per frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402
import sam2_edge_eval as S  # noqa: E402


def parse_times(text: str) -> dict[str, int]:
    pairs = {}
    for item in text.split(","):
        name, index = item.split(":")
        pairs[f"take_{name.strip()}"] = int(index)
    return pairs


def load_labels(dump: Path, file_index: int) -> np.ndarray | None:
    path = dump / f"{file_index:06d}_labels.png"
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def apply_labels(ctx: E.Context, labels_full: np.ndarray, full_size: tuple[int, int]) -> np.ndarray:
    """The prior with the C++ labels applied, in the crop's coordinates."""
    full_w, full_h = full_size
    if labels_full.shape[1] != full_w or labels_full.shape[0] != full_h:
        labels_full = cv2.resize(labels_full, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    x0, y0 = ctx.offset
    h, w = ctx.gt.shape
    labels = labels_full[y0:y0 + h, x0:x0 + w]
    added = labels == 255
    removed = labels == 0
    return (ctx.prior | added) & ~removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--model", default="small", choices=list(S.MODELS))
    parser.add_argument("--times", default="t02:60,t06:180,t10:300,t18:540")
    parser.add_argument("--band", type=int, default=24)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "mikan_matte")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    indices = parse_times(args.times)

    runner = S.Sam2Runner(args.model, args.checkpoints, args.root, full_frame=False)
    python_method = S.make_method(runner, "runs_pos", args.band)

    rows = []
    for name, file_index in indices.items():
        folder = args.root / name
        if not (folder / "gt_mask.png").exists():
            print(f"{name}: not marked, skipped")
            continue
        labels_full = load_labels(args.dump, file_index)
        if labels_full is None:
            print(f"{name}: no dump for file index {file_index}, skipped")
            continue
        ctx = E.load_context(folder)
        full = cv2.imread(str(folder / "video.png"))
        full_size = (full.shape[1], full.shape[0])
        preds = {"mesh": ctx.prior, "sam2_runs_pos": python_method(ctx),
                 "mikan_matte": apply_labels(ctx, labels_full, full_size)}
        for method, pred in preds.items():
            rows.append({"capture": ctx.name, "method": method, "seconds": 0.0, **E.score(ctx, pred)})
        E.contact_sheet(ctx, preds, args.out / f"{ctx.name}.png")

        prompts_path = args.dump / f"{file_index:06d}_prompts.json"
        if prompts_path.exists():
            dumped = json.loads(prompts_path.read_text())
            cpp_points = sum(len(r["positives"]) for r in dumped["runs"])
            py_runs = S.synthesize_run_prompts(ctx.prior, ctx.mesh, args.band)
            py_points = sum(len(r.pos) for r in py_runs)
            print(f"{name}: C++ {len(dumped['runs'])} runs, {cpp_points} points at {dumped['width']}x{dumped['height']}; "
                  f"Python {len(py_runs)} runs, {py_points} points on the crop")

    if not rows:
        raise SystemExit("nothing scored")
    import csv
    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    E.write_summary(rows, ["mesh", "sam2_runs_pos", "mikan_matte"], args.out)


if __name__ == "__main__":
    main()
