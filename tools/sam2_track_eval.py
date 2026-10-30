"""Track the occlusion mesh's silhouette through a take with SAM2's video predictor from one prompt,
and score the tracked mask on the hand-marked frames next to the mesh and the image-mode result.

    .venv_sam2/Scripts/python tools/sam2_track_eval.py <take.trackcamera> [--root out/live_occlusion]
        [--model small] [--prompt mask|box|runs_box|runs_pos|runs_box_pos|runs_box_pos_neg] [--band 24] [--step 1]
        [--frames-dir out/sam2_track/frames] [--checkpoints models/sam2] [--out out/sam2_track]

The marked take_tNN folders under --root name playback seconds of the take. The frames from the first
marked second to the last are decoded from the take's movie (every --step th frame) and written as
JPEGs for the predictor. Each marked folder is mapped to its decoded frame by the track's time_us and
checked against the folder's video.png by mean absolute difference; a mismatch is searched over the
neighbouring decoded frames and reported. The prompt is given once, on the first marked frame, from
that folder's mesh prior: 'mask' hands the prior to add_new_mask, which the shipped configs take as
that frame's output as it is, so refinement starts on the next frame; 'box' is the prior's bounding
box as one object; the runs_* prompts are sam2_edge_eval.py's per occluding run prompts, one tracked
object per run, their masks unioned per frame. propagate_in_video then runs to the last marked frame.

Writes out/<model>_<prompt>/results.csv and summary.txt with three rows per marked frame (the mesh,
the image-mode result of the same runs prompt on that frame alone, the tracked mask), a contact sheet
per marked frame, and trace.csv with the tracked mask's area and its IoU with the previous frame's.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402
import render_take_frames as R  # noqa: E402
import sam2_edge_eval as S  # noqa: E402
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

THUMB = 8   # every decoded frame keeps a 1/8 copy for the mapping check


def marked_take_folders(root: Path) -> list[tuple[float, Path]]:
    """(playback seconds, folder) of every marked take_tNN folder, in time order."""
    found = []
    for p in root.iterdir():
        if p.name.startswith("take_t") and (p / "gt_mask.png").exists():
            found.append((float(p.name[len("take_t"):]), p))
    if not found:
        raise SystemExit(f"no marked take_t* folders under {root}")
    return sorted(found)


def thumb(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (rgb.shape[1] // THUMB, rgb.shape[0] // THUMB), interpolation=cv2.INTER_AREA)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--model", default="small", choices=list(S.MODELS))
    parser.add_argument("--prompt", default="runs_pos",
                        choices=["mask", "box", "runs_box", "runs_pos", "runs_box_pos", "runs_box_pos_neg"])
    parser.add_argument("--band", type=int, default=24)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "out" / "sam2_track" / "frames")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "sam2_track")
    args = parser.parse_args()
    out = args.out / f"{args.model}_{args.prompt}"
    out.mkdir(parents=True, exist_ok=True)

    # The take's frames between the first and last marked second
    archive = tarfile.open(args.take)
    members = json.load(archive.extractfile("capture.json"))["members"]
    track = json.load(archive.extractfile(members["movie_pose"]))
    t0 = min(f["time_us"] for f in track["frames"])
    rel = np.array([(f["time_us"] - t0) / 1e6 for f in track["frames"]])
    marked = marked_take_folders(args.root)
    indices = [int(np.argmin(np.abs(rel - t))) for t, _ in marked]
    wanted = list(range(indices[0], indices[-1] + 1, args.step))
    for i in indices:
        if i not in wanted:
            wanted.append(i)
    wanted = sorted(set(wanted))

    if args.frames_dir.exists():
        shutil.rmtree(args.frames_dir)
    args.frames_dir.mkdir(parents=True)
    position, thumbs = {}, {}
    for pos, (index, rgb) in enumerate(R.decode_frames(archive, members["movie"], set(wanted), args.frames_dir)):
        cv2.imwrite(str(args.frames_dir / f"{pos:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        position[index] = pos
        thumbs[index] = thumb(rgb)
    print(f"decoded {len(position)} frames ({wanted[0]}..{wanted[-1]}, step {args.step}) into {args.frames_dir}")

    # Each marked folder against the frame its time names
    contexts = []
    for (t, folder), index in zip(marked, indices):
        video = thumb(read_rgb(folder / "video.png"))
        d = mad(video, thumbs[index])
        note = ""
        if d > 5.0:
            near = {i: mad(video, thumbs[i]) for i in thumbs if abs(i - index) <= 3 * args.step}
            best = min(near, key=near.get)
            note = f"  MISMATCH: best match is frame {best} at {near[best]:.2f}"
        print(f"{folder.name}: t={t:.1f}s -> frame {index} (pos {position[index]}), video MAD {d:.2f}{note}")
        contexts.append(E.load_context(folder))
    full_h, full_w = read_rgb(marked[0][1] / "video.png").shape[:2]

    # The image-mode result on each marked frame alone, for the same table
    image_variant = args.prompt if args.prompt.startswith("runs_") else "runs_pos"
    runner = S.Sam2Runner(args.model, args.checkpoints, args.root, full_frame=False)
    image_method = S.make_method(runner, image_variant, args.band)
    image_preds = [image_method(ctx) for ctx in contexts]
    del runner
    torch.cuda.empty_cache()

    # One prompt on the first marked frame, then propagate
    cfg, ckpt = S.MODELS[args.model]
    predictor = build_sam2_video_predictor(cfg, str(args.checkpoints / ckpt), device="cuda")
    ctx0 = contexts[0]
    x0, y0 = ctx0.offset
    h0, w0 = ctx0.gt.shape
    tracked, trace, prev = {}, [], None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(args.frames_dir), offload_video_to_cpu=True)
        if args.prompt == "mask":
            full_prior = np.zeros((full_h, full_w), bool)
            full_prior[y0:y0 + h0, x0:x0 + w0] = ctx0.prior
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(full_prior))
        elif args.prompt == "box":
            prompts = S.synthesize_prompts(ctx0.prior, args.band).shifted(x0, y0)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=prompts.box)
        else:
            parts = set(args.prompt[5:].split("_"))
            run_prompts = S.synthesize_run_prompts(ctx0.prior, ctx0.mesh, args.band)
            if not run_prompts:
                raise SystemExit(f"{ctx0.name}: no occluding run to prompt from")
            for k, rp in enumerate(run_prompts):
                rp = rp.shifted(x0, y0)
                coords, labels = [], []
                if "pos" in parts:
                    coords.append(rp.pos)
                    labels += [1] * len(rp.pos)
                if "neg" in parts and len(rp.neg):
                    coords.append(rp.neg)
                    labels += [0] * len(rp.neg)
                kw = {"box": rp.box} if "box" in parts else {}
                if coords:
                    kw["points"] = np.concatenate(coords).astype(np.float32)
                    kw["labels"] = np.array(labels, int)
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=k + 1, **kw)
            print(f"prompted {len(run_prompts)} runs as objects on frame 0")
        marked_positions = {position[i]: i for i in indices}
        for pos, _, logits in predictor.propagate_in_video(state):
            mask = (logits[:, 0] > 0).any(0).cpu().numpy()
            overlap = S.iou(mask, prev) if prev is not None else 1.0
            trace.append({"position": pos, "frame": wanted[pos], "time_s": round(float(rel[wanted[pos]]), 3),
                          "area_px": int(mask.sum()), "iou_prev": round(overlap, 4)})
            prev = mask
            if pos in marked_positions:
                tracked[marked_positions[pos]] = mask

    # Score the marked frames
    track_name = f"sam2_track_{args.prompt}"
    image_name = f"sam2_image_{image_variant}"
    names = ["mesh", image_name, track_name]
    rows = []
    for (t, folder), index, ctx, image_pred in zip(marked, indices, contexts, image_preds):
        cx, cy = ctx.offset
        ch, cw = ctx.gt.shape
        preds = {"mesh": ctx.prior, image_name: image_pred,
                 track_name: tracked[index][cy:cy + ch, cx:cx + cw]}
        for name, pred in preds.items():
            rows.append({"capture": ctx.name, "method": name, "seconds": 0.0, **E.score(ctx, pred)})
        E.contact_sheet(ctx, preds, out / f"{ctx.name}.png")

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "trace.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    E.write_summary(rows, names, out)


if __name__ == "__main__":
    main()
