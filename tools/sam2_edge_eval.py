"""Prompt a promptable segmenter with the occlusion mesh's silhouette on the hand-marked frames and
score its masks against the marked occluding edges, next to the mesh and the best band-local methods.

    .venv_sam2/Scripts/python tools/sam2_edge_eval.py [--root out/live_occlusion] [--family sam2]
        [--models tiny,small,base_plus,large] [--size 0] [--variants ...] [--band 24] [--full-frame]
        [--checkpoints <dir>] [--out out/<family>_edges]

The families are the segmenters with a SAM-style prompt decoder that the live matte could run
instead of SAM2, each in its own venv since two of them are forks of the sam2 package:

    sam2             SAM 2.1 tiny, small, base_plus, large (.venv_sam2, the sam2 1.1.0 PyPI release)
    edgetam          EdgeTAM, RepViT-M1 trunk (.venv_edgetam, third_party/EdgeTAM on sys.path)
    efficienttam     EfficientTAM ti, s and their 512 variants ti_512, s_512 (.venv_efficienttam,
                     third_party/EfficientTAM on sys.path)
    efficientvit_sam EfficientViT-SAM l0, l1 (512 long side) and xl0 (1024) (.venv_efficientvit,
                     third_party/efficientvit installed without its dependencies)

The venvs are built as requirements-sam2.txt, requirements-edgetam.txt, requirements-efficienttam.txt
and requirements-efficientvit.txt describe. The SAM2-API families run under bfloat16 autocast as the
recorded SAM2 rows did; EfficientViT-SAM runs in fp32, since its predictor preprocesses on the device
itself. --size overrides the SAM2-API families' encoder input (512 is the real-time configuration;
the predictor's hard-coded feature sizes are set from the model's image size here, which is what
fails in sam2 1.1.0 otherwise); EfficientViT-SAM's input is fixed by the checkpoint.

Reads the marked folders occlusion_edge_eval.py reads (video.png, mesh_depth.npy, gt_mask.png). The
prior is the mesh silhouette occlusion_edge_eval.load_context extracts, the mesh region nearer than
its threshold, which is the occluder and whatever it stands on. Every prompt is made from the prior
and the mesh depth alone.

The runs_* variants prompt per occluding contour run. The prior's outline is probed PROBE px either
side along its normal, and the stretches where the mesh depth jumps are the occluding contours; they
are grouped into runs by proximity. Each run gets its own SAM2 call: a box around the run grown by
the band (box), positive points just inside it at the thickest spot within two bands (pos), negative
points one and a half bands outside it (neg). Among the three multimask outputs the one with the
highest IoU against the prior inside the box is kept (the _score suffix keeps the model's own top
score instead), _iter2 feeds the chosen logits back with the same prompts, and the runs' masks are
unioned. Negatives are the risk: where the mesh is short by more than the band (the can's missing cap)
they land on the object.

The whole-prior variants are the naive prompts, kept as the record: box is the prior's bounding box,
box_pts adds points sampled over the prior, mask hands the prior to the decoder as 256x256 logits.
SAM2 sees the same crop every method sees (the marked occluder plus 160 px); --full-frame hands it
the whole frame and crops the mask afterwards.

Writes out/<model>/results.csv, summary.txt and one contact sheet per frame, and out/summary_all.txt
with every model's table. The seconds column is decoder time once a frame's embedding is cached (the
first SAM2 method per frame pays the encoder); sam2_timing.py measures the cost properly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402

MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
THIRD_PARTY = ROOT / "third_party"
# Per family: the clone put on sys.path (None for an installed package), the default checkpoint
# directory, and the models as (hydra config or model zoo name, checkpoint file)
FAMILIES = {
    "sam2": (None, ROOT / "models" / "sam2", MODELS),
    "edgetam": (THIRD_PARTY / "EdgeTAM", THIRD_PARTY / "EdgeTAM" / "checkpoints", {
        "edgetam": ("configs/edgetam.yaml", "edgetam.pt"),
    }),
    "efficienttam": (THIRD_PARTY / "EfficientTAM", ROOT / "models" / "efficienttam", {
        "ti": ("configs/efficienttam/efficienttam_ti.yaml", "efficienttam_ti.pt"),
        "ti_512": ("configs/efficienttam/efficienttam_ti_512x512.yaml", "efficienttam_ti_512x512.pt"),
        "s": ("configs/efficienttam/efficienttam_s.yaml", "efficienttam_s.pt"),
        "s_512": ("configs/efficienttam/efficienttam_s_512x512.yaml", "efficienttam_s_512x512.pt"),
    }),
    "efficientvit_sam": (THIRD_PARTY / "efficientvit", ROOT / "models" / "efficientvit_sam", {
        "l0": ("efficientvit-sam-l0", "efficientvit_sam_l0.pt"),
        "l1": ("efficientvit-sam-l1", "efficientvit_sam_l1.pt"),
        "xl0": ("efficientvit-sam-xl0", "efficientvit_sam_xl0.pt"),
    }),
}
CLASSICAL = ["mesh", "trimap_vote_45", "contour_dp_smooth3"]
VARIANTS = ["box", "box_pts", "mask",
            "runs_box", "runs_pos", "runs_box_pos", "runs_box_pos_neg", "runs_box_pos_iter2", "runs_box_pos_score"]
LOW_RES = 256          # the decoder's mask prompt and low-res logits grid
N_POSITIVE = 5         # whole-prior points
N_NEGATIVE = 8
RUN_GAP = 9            # px within which outline stretches join one run
RUN_MIN_POINTS = 20
RUN_SPACING = 40       # px of run per prompt point
RUN_MAX_POINTS = 12


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum()) / max(float(np.logical_or(a, b).sum()), 1.0)


def farthest_points(candidates: np.ndarray, first: int, count: int) -> list[int]:
    """Greedy farthest point sampling over candidate xy rows, returning row indices."""
    chosen = [first]
    if len(candidates) == 0:
        return chosen
    dist = np.linalg.norm(candidates - candidates[first], axis=1)
    while len(chosen) < count:
        i = int(np.argmax(dist))
        if dist[i] <= 0:
            break
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(candidates - candidates[i], axis=1))
    return chosen


# ---- whole-prior prompts ----------------------------------------------------------

@dataclass
class Prompts:
    box: np.ndarray        # (4,) xyxy
    points: np.ndarray     # (N, 2) xy
    labels: np.ndarray     # (N,) 1 inside, 0 outside

    def shifted(self, dx: float, dy: float) -> "Prompts":
        return Prompts(self.box + np.array([dx, dy, dx, dy], np.float32),
                       self.points + np.array([dx, dy], np.float32), self.labels)


def synthesize_prompts(prior: np.ndarray, band: int) -> Prompts:
    """The naive prompts: the prior's box, points deep inside it, points in the ring outside it."""
    h, w = prior.shape
    fg, bg, _ = E.trimap(prior, band)
    if not fg.any():
        fg = prior
    ys, xs = np.nonzero(prior)
    box = np.array([max(xs.min() - band, 0), max(ys.min() - band, 0),
                    min(xs.max() + band, w - 1), min(ys.max() + band, h - 1)], np.float32)

    dt = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    deep = np.argwhere(dt >= 0.5 * dt.max())[:, ::-1].astype(np.float32)
    if len(deep) > 4000:
        deep = deep[np.random.default_rng(0).choice(len(deep), 4000, replace=False)]
    first = int(np.argmax(dt[deep[:, 1].astype(int), deep[:, 0].astype(int)]))
    pos = deep[farthest_points(deep, first, N_POSITIVE)]

    sdf = E.signed_distance(prior)
    ring = np.argwhere((sdf > band) & (sdf <= 2 * band) & bg)[:, ::-1]
    neg = []
    if len(ring):
        cx, cy = xs.mean(), ys.mean()
        ang = np.arctan2(ring[:, 1] - cy, ring[:, 0] - cx)
        bins = np.floor((ang + np.pi) / (2 * np.pi) * N_NEGATIVE).astype(int) % N_NEGATIVE
        for b in range(N_NEGATIVE):
            sel = np.nonzero(bins == b)[0]
            if len(sel) == 0:
                continue
            j = sel[np.argmin(np.abs(sdf[ring[sel, 1], ring[sel, 0]] - 1.5 * band))]
            neg.append(ring[j])
    neg = np.array(neg, np.float32).reshape(-1, 2)
    points = np.concatenate([pos, neg]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    return Prompts(box, points, labels)


# ---- per occluding run prompts ----------------------------------------------------

@dataclass
class RunPrompt:
    box: np.ndarray        # (4,) xyxy
    pos: np.ndarray        # (N, 2) xy inside, on the near side
    neg: np.ndarray        # (M, 2) xy outside, on the far side
    region: np.ndarray     # the prior inside the box, for choosing among the multimask outputs

    def shifted(self, dx: float, dy: float) -> "RunPrompt":
        d = np.array([dx, dy], np.float32)
        return RunPrompt(self.box + np.array([dx, dy, dx, dy], np.float32), self.pos + d, self.neg + d, self.region)


def occluding_outline(prior: np.ndarray, mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outline points of the prior where the mesh depth jumps across: ys, xs, outward normals (N, 2) xy."""
    h, w = prior.shape
    sdf = cv2.GaussianBlur(E.signed_distance(prior), (0, 0), 3)
    gx = cv2.Sobel(sdf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(sdf, cv2.CV_32F, 0, 1)
    norm = np.maximum(np.hypot(gx, gy), 1e-6)
    nx, ny = gx / norm, gy / norm
    ey, ex = np.nonzero(E.outline(prior))

    def probe(sign: float) -> np.ndarray:
        px = np.clip(np.round(ex + sign * E.PROBE * nx[ey, ex]).astype(int), 0, w - 1)
        py = np.clip(np.round(ey + sign * E.PROBE * ny[ey, ex]).astype(int), 0, h - 1)
        return mesh[py, px]

    d_out, d_in = probe(+1.0), probe(-1.0)
    z_out = E.ZNEAR + d_out * (E.ZFAR - E.ZNEAR)
    z_in = E.ZNEAR + d_in * (E.ZFAR - E.ZNEAR)
    interior = (ex > 1) & (ex < w - 2) & (ey > 1) & (ey < h - 2)
    occl = (z_out - z_in > 0.05 * z_out) & (d_in < 1.0) & interior
    normals = np.stack([nx[ey[occl], ex[occl]], ny[ey[occl], ex[occl]]], axis=1)
    return ey[occl], ex[occl], normals


def contour_runs(ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray]:
    """Index sets of the outline points, grouped into runs by proximity."""
    mask = np.zeros(shape, np.uint8)
    mask[ys, xs] = 1
    joined = cv2.dilate(mask, np.ones((RUN_GAP, RUN_GAP), np.uint8))
    count, labels = cv2.connectedComponents(joined)
    ids = labels[ys, xs]
    runs = []
    for k in range(1, count):
        sel = np.nonzero(ids == k)[0]
        if len(sel) >= RUN_MIN_POINTS:
            runs.append(sel)
    return runs


def synthesize_run_prompts(prior: np.ndarray, mesh: np.ndarray, band: int) -> list[RunPrompt]:
    h, w = prior.shape
    ys, xs, normals = occluding_outline(prior, mesh)
    dt = cv2.distanceTransform(prior.astype(np.uint8), cv2.DIST_L2, 5)
    sdf = E.signed_distance(prior)
    prompts = []
    for run in contour_runs(ys, xs, prior.shape):
        pts = np.stack([xs[run], ys[run]], axis=1).astype(np.float32)
        nrm = normals[run]
        count = min(max(2, int(round(len(pts) / RUN_SPACING))), RUN_MAX_POINTS)
        pos, neg = [], []
        for i in farthest_points(pts, 0, count):
            p, n = pts[i], nrm[i]
            best, best_dt = None, 0.0
            for s in range(3, 2 * band + 1, 2):
                q = np.round(p - s * n).astype(int)
                if not (0 <= q[0] < w and 0 <= q[1] < h) or not prior[q[1], q[0]]:
                    break
                if dt[q[1], q[0]] > best_dt:
                    best, best_dt = q, float(dt[q[1], q[0]])
            if best is not None:
                pos.append(best)
            for s in (1.5 * band, band + 4.0, 8.0):
                q = np.round(p + s * n).astype(int)
                if 0 <= q[0] < w and 0 <= q[1] < h and sdf[q[1], q[0]] > min(band, s - 1.0):
                    neg.append(q)
                    break
        if not pos:
            continue
        pos = np.array(pos, np.float32).reshape(-1, 2)
        neg = np.array(neg, np.float32).reshape(-1, 2)
        extent = np.concatenate([pts, pos])
        box = np.array([max(extent[:, 0].min() - band, 0), max(extent[:, 1].min() - band, 0),
                        min(extent[:, 0].max() + band, w - 1), min(extent[:, 1].max() + band, h - 1)], np.float32)
        region = np.zeros_like(prior)
        x0, y0, x1, y1 = box.astype(int)
        region[y0:y1 + 1, x0:x1 + 1] = prior[y0:y1 + 1, x0:x1 + 1]
        prompts.append(RunPrompt(box, pos, neg, region))
    return prompts


# ---- the predictor ----------------------------------------------------------------

def model_paths(family: str, model: str, checkpoints: Path | None) -> tuple[str, Path]:
    """The model's hydra config or zoo name and its checkpoint path, the clone put on sys.path."""
    clone, default_checkpoints, models = FAMILIES[family]
    if model not in models:
        raise ValueError(f"unknown {family} model {model}, one of {list(models)}")
    spec, ckpt = models[model]
    if clone is not None and str(clone) not in sys.path:
        sys.path.insert(0, str(clone))
    return spec, (checkpoints or default_checkpoints) / ckpt


def build_model(family: str, model: str, checkpoints: Path | None, size: int, device: str = "cuda"):
    """The family's model module on the device, in eval mode."""
    spec, ckpt_path = model_paths(family, model, checkpoints)
    if family == "efficientvit_sam":
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        if size:
            print(f"{family} {model}: --size ignored, the input is fixed by the checkpoint")
        return create_efficientvit_sam_model(spec, True, str(ckpt_path)).to(device).eval()
    overrides = [f"++model.image_size={size}"] if size else []
    if family == "efficienttam":
        from efficient_track_anything.build_efficienttam import build_efficienttam
        # The shipped configs torch.compile the encoder, which has no inductor backend on Windows
        overrides = overrides + ["++model.compile_image_encoder=False"]
        return build_efficienttam(spec, str(ckpt_path), device=device, hydra_overrides_extra=overrides)
    from sam2.build_sam import build_sam2
    return build_sam2(spec, str(ckpt_path), device=device, hydra_overrides_extra=overrides)


def build_predictor(family: str, model: str, checkpoints: Path | None, size: int):
    """The family's image predictor for the model, with set_image and the SAM predict signature.

    Returns the predictor, whether it runs under bfloat16 autocast, and the encoder input size.
    """
    sam = build_model(family, model, checkpoints, size, "cuda")
    if family == "efficientvit_sam":
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        return EfficientViTSamPredictor(sam), False, sam.image_size[1]
    if family == "efficienttam":
        from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor
        predictor = EfficientTAMImagePredictor(sam)
    else:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(sam)
    # The predictor hard-codes the feature sizes of a 1024 input; set them from the model's size
    s = int(sam.image_size)
    predictor._bb_feat_sizes = [(s // 4, s // 4), (s // 8, s // 8), (s // 16, s // 16)]
    return predictor, True, s


class Sam2Runner:
    """One image predictor; the embedding is set once per frame and shared by the variants."""

    def __init__(self, model: str, checkpoints: Path | None, root: Path, full_frame: bool,
                 family: str = "sam2", size: int = 0):
        self.predictor, self.autocast, self.input_size = build_predictor(family, model, checkpoints, size)
        self.root, self.full_frame = root, full_frame
        self.embedded = None
        self.image_hw = None

    def prepare(self, ctx: E.Context) -> tuple[int, int]:
        """Set the frame's image once; return the shift from crop coordinates to the image set."""
        if self.embedded != ctx.name:
            if self.full_frame:
                img = cv2.cvtColor(cv2.imread(str(self.root / ctx.name / "video.png")), cv2.COLOR_BGR2RGB)
            else:
                img = ctx.rgb
            self.predictor.set_image(img)
            self.embedded = ctx.name
            self.image_hw = img.shape[:2]
        return ctx.offset if self.full_frame else (0, 0)

    def to_crop(self, mask: np.ndarray, ctx: E.Context) -> np.ndarray:
        if not self.full_frame:
            return mask
        x0, y0 = ctx.offset
        h, w = ctx.gt.shape
        return mask[y0:y0 + h, x0:x0 + w]

    def prior_logits(self, ctx: E.Context) -> np.ndarray:
        """The prior as the decoder's 256x256 mask prompt, +8 inside and -8 outside."""
        canvas = ctx.prior.astype(np.float32)
        if self.full_frame:
            full = np.zeros(self.image_hw, np.float32)
            x0, y0 = ctx.offset
            h, w = ctx.gt.shape
            full[y0:y0 + h, x0:x0 + w] = canvas
            canvas = full
        low = cv2.resize(canvas, (LOW_RES, LOW_RES), interpolation=cv2.INTER_AREA)
        return ((low * 2.0 - 1.0) * 8.0)[None].astype(np.float32)

    def predict(self, **kw):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.predictor.predict(**kw)

    def choose(self, ctx: E.Context, kw: dict, region: np.ndarray, by_score: bool, iterate: bool) -> np.ndarray:
        """One multimask prediction, the chosen mask in crop coordinates, refined once when asked."""
        masks, scores, low_res = self.predict(**kw)
        cropped = [self.to_crop(m > 0.5, ctx) for m in masks]
        if by_score:
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmax([iou(m, region) for m in cropped]))
        if iterate:
            kw = dict(kw, mask_input=low_res[idx:idx + 1], multimask_output=False)
            masks, _, _ = self.predict(**kw)
            return self.to_crop(masks[0] > 0.5, ctx)
        return cropped[idx]


def make_method(runner: Sam2Runner, variant: str, band: int):
    per_run = variant.startswith("runs_")
    rest = variant[5:] if per_run else variant
    by_score = rest.endswith("_score")
    if by_score:
        rest = rest[:-6]
    iterate = rest.endswith("_iter2")
    if iterate:
        rest = rest[:-6]
    parts = set(rest.split("_"))
    allowed = {"box", "pos", "neg"} if per_run else {"box", "pts", "mask"}
    if not parts or not parts <= allowed or (not per_run and parts not in ({"box"}, {"box", "pts"}, {"mask"})):
        raise ValueError(f"unknown variant {variant}")

    def whole(ctx: E.Context) -> np.ndarray:
        dx, dy = runner.prepare(ctx)
        prompts = synthesize_prompts(ctx.prior, band).shifted(dx, dy)
        kw = {"multimask_output": True}
        if "box" in parts:
            kw["box"] = prompts.box
        if "pts" in parts:
            kw["point_coords"], kw["point_labels"] = prompts.points, prompts.labels
        if "mask" in parts:
            kw["mask_input"] = runner.prior_logits(ctx)
        return runner.choose(ctx, kw, ctx.prior, by_score, iterate)

    def runs(ctx: E.Context) -> np.ndarray:
        dx, dy = runner.prepare(ctx)
        run_prompts = synthesize_run_prompts(ctx.prior, ctx.mesh, band)
        if not run_prompts:
            return whole_box(ctx)
        union = np.zeros_like(ctx.prior)
        for rp in run_prompts:
            shifted = rp.shifted(dx, dy)
            kw = {"multimask_output": True}
            if "box" in parts:
                kw["box"] = shifted.box
            coords, labels = [], []
            if "pos" in parts:
                coords.append(shifted.pos)
                labels += [1] * len(shifted.pos)
            if "neg" in parts and len(shifted.neg):
                coords.append(shifted.neg)
                labels += [0] * len(shifted.neg)
            if coords:
                kw["point_coords"] = np.concatenate(coords).astype(np.float32)
                kw["point_labels"] = np.array(labels, int)
            union |= runner.choose(ctx, kw, rp.region, by_score, iterate)
        return union

    def whole_box(ctx: E.Context) -> np.ndarray:
        dx, dy = runner.prepare(ctx)
        prompts = synthesize_prompts(ctx.prior, band).shifted(dx, dy)
        return runner.choose(ctx, {"multimask_output": True, "box": prompts.box}, ctx.prior, False, False)

    return runs if per_run else whole


# ---- main -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--family", default="sam2", choices=list(FAMILIES))
    parser.add_argument("--models", default=None, help="comma separated, default every model of the family")
    parser.add_argument("--size", type=int, default=0, help="encoder input override for the SAM2-API families")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--band", type=int, default=24)
    parser.add_argument("--full-frame", action="store_true")
    parser.add_argument("--checkpoints", type=Path, default=None, help="default per family, see FAMILIES")
    parser.add_argument("--out", type=Path, default=None, help="default out/<family>_edges")
    args = parser.parse_args()
    out = args.out or ROOT / "out" / f"{args.family}_edges"
    out.mkdir(parents=True, exist_ok=True)
    suffix = ("_full" if args.full_frame else "") + (f"_{args.size}" if args.size else "")
    models = args.models.split(",") if args.models else list(FAMILIES[args.family][2])

    summaries = []
    for model in models:
        try:
            runner = Sam2Runner(model, args.checkpoints, args.root, args.full_frame, args.family, args.size)
        except Exception as exc:
            line = f"== {args.family} {model}{suffix}: failed to build: {str(exc).splitlines()[0][:200]}"
            print(line)
            summaries.append(line)
            continue
        methods = {name: E.METHODS[name] for name in CLASSICAL}
        for variant in args.variants.split(","):
            methods[f"sam2_{variant}"] = make_method(runner, variant, args.band)
        model_out = out / (model + suffix)
        print(f"== {args.family} {model}{suffix}, input {runner.input_size}")
        rows = E.evaluate_folders(args.root, model_out, methods)
        summary = E.write_summary(rows, list(methods), model_out)
        summaries.append(f"== {args.family} {model}{suffix}, input {runner.input_size}, band {args.band}\n{summary}")
        del runner
        torch.cuda.empty_cache()

    (out / f"summary_all{suffix}.txt").write_text("\n\n".join(summaries) + "\n")


if __name__ == "__main__":
    main()
