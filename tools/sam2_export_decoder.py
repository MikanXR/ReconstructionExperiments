"""Export SAM2's prompt decoder (the prompt encoder and the mask decoder) to ONNX with a free point
count, the second half of the encoder/decoder pair the MikanXR editor runs under ONNX Runtime.

    .venv_sam2/Scripts/python tools/sam2_export_decoder.py [--models tiny,small] [--opset 17]
        [--checkpoints models/sam2] [--out models/sam2/onnx] [--check] [--frames angle_02,angle_05,take_t10]
        [--no-export]

The wrapper takes the features exactly as sam2_export_encoder.py emits them, which are what
SAM2ImagePredictor.set_image stores: image_embed [1,256,64,64] (no_mem_embed already added),
high_res_feats_0 [1,32,256,256] and high_res_feats_1 [1,64,128,128] (conv_s0 and conv_s1 already
applied), plus point_coords [1,N,2] float32 in the 1024x1024 encoder input pixel space (x then y) and
point_labels [1,N] int32 (1 positive, 0 negative). N is the free dimension num_points: the consumer
passes exactly its points and the graph reproduces SAM2ImagePredictor.predict for any N. It returns
low_res_masks [1,3,256,256], the mask logits clamped to [-32, 32] as the predictor returns them, and
iou_scores [1,3]. Inside it is only what _predict does: the prompt encoder with points only (it appends
its own padding point) and the mask decoder with multimask_output=True and repeat_image=False. Padding
to a fixed count with label -1 is deliberately not offered: the padding tokens take part in the mask
decoder's token self-attention, so N points plus pads is a different prompt from N points and moves
the logits by units where the prompt is ambiguous. The legacy TorchScript exporter is tried first and
the dynamo exporter second.

--check validates the encoder+decoder ONNX chain under onnxruntime CPU against SAM2ImagePredictor
on the CPU in fp32 for the named frames under out/live_occlusion: the crop context from
occlusion_edge_eval.load_context, positive points from sam2_edge_eval.synthesize_run_prompts, the
frame resized to 1024x1024 with torchvision's Resize (bilinear, antialias) the way SAM2Transforms
does. It prints per frame the max abs logit difference, the max abs score difference, and the IoU of
the thresholded top-scoring masks, then the size and sha256 of each model's two ONNX files.

Writes <out>/sam2.1_<model>_decoder_dynpts.onnx and prints the input and output names and shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402
import sam2_edge_eval as S  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

INPUT_SIZE = 1024
FEATURE_SHAPES = [(1, 256, 64, 64), (1, 32, 256, 256), (1, 64, 128, 128)]
FEATURE_NAMES = ["image_embed", "high_res_feats_0", "high_res_feats_1"]
INPUT_NAMES = FEATURE_NAMES + ["point_coords", "point_labels"]
OUTPUT_NAMES = ["low_res_masks", "iou_scores"]
DYNAMIC_AXES = {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}}
LABEL_DTYPE = torch.int32
EXAMPLE_POINTS = 16     # the point count the export is traced with; the axis is free afterwards
MAX_CHECK_POINTS = 16


class DecoderWrapper(torch.nn.Module):
    """(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels) -> (low_res_masks, iou_scores)."""

    def __init__(self, sam):
        super().__init__()
        self.prompt_encoder = sam.sam_prompt_encoder
        self.mask_decoder = sam.sam_mask_decoder
        assert self.mask_decoder.use_high_res_features

    def forward(self, image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels):
        sparse, dense = self.prompt_encoder(points=(point_coords, point_labels), boxes=None, masks=None)
        masks, iou, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[high_res_feats_0, high_res_feats_1],
        )
        return torch.clamp(masks, -32.0, 32.0), iou


def decoder_path(out_dir: Path, model: str) -> Path:
    return out_dir / f"sam2.1_{model}_decoder_dynpts.onnx"


def encoder_path(out_dir: Path, model: str) -> Path:
    return out_dir / f"sam2.1_{model}_encoder_{INPUT_SIZE}.onnx"


def export(model: str, opset: int, checkpoints: Path, out_dir: Path) -> Path:
    cfg, ckpt = S.MODELS[model]
    sam = build_sam2(cfg, str(checkpoints / ckpt), device="cpu")
    wrapper = DecoderWrapper(sam).eval()
    feats = [torch.randn(*shape) for shape in FEATURE_SHAPES]
    coords = torch.rand(1, EXAMPLE_POINTS, 2) * INPUT_SIZE
    labels = torch.ones(1, EXAMPLE_POINTS, dtype=LABEL_DTYPE)
    labels[0, EXAMPLE_POINTS // 4] = 0
    inputs = tuple(feats) + (coords, labels)
    with torch.no_grad():
        reference = wrapper(*inputs)
    path = decoder_path(out_dir, model)
    for dynamo in (False, True):
        try:
            torch.onnx.export(wrapper, inputs, str(path), opset_version=opset, input_names=INPUT_NAMES,
                              output_names=OUTPUT_NAMES, dynamic_axes=DYNAMIC_AXES, dynamo=dynamo)
            print(f"{path.name}: exported with {'dynamo' if dynamo else 'torchscript'} exporter")
            break
        except Exception as exc:
            print(f"{path.name}: {'dynamo' if dynamo else 'torchscript'} export failed: {str(exc).splitlines()[0][:160]}")
    else:
        raise SystemExit("both exporters failed")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  input  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name}: {o.shape} {o.type}")
    got = sess.run(None, dict(zip(INPUT_NAMES, (t.numpy() for t in inputs))))
    for name, ref, g in zip(OUTPUT_NAMES, reference, got):
        print(f"  {name}: {tuple(g.shape)} max abs diff vs torch {float(np.max(np.abs(ref.numpy() - g))):.3e}")
    return path


def frame_prompts(ctx: E.Context, max_points: int) -> np.ndarray:
    runs = S.synthesize_run_prompts(ctx.prior, ctx.mesh, 24)
    pos = np.concatenate([r.pos for r in runs]).astype(np.float32)
    return pos[:max_points]


def resize_like_sam2(rgb: np.ndarray) -> np.ndarray:
    """SAM2Transforms' resize: ToTensor then torchvision Resize (bilinear, antialias), RGB in [0,1]."""
    return Resize((INPUT_SIZE, INPUT_SIZE))(ToTensor()(rgb))[None].numpy()


def check(model: str, checkpoints: Path, out_dir: Path, frames: list[str]) -> None:
    import onnxruntime as ort
    cfg, ckpt = S.MODELS[model]
    predictor = SAM2ImagePredictor(build_sam2(cfg, str(checkpoints / ckpt), device="cpu"))
    enc = ort.InferenceSession(str(encoder_path(out_dir, model)), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(decoder_path(out_dir, model)), providers=["CPUExecutionProvider"])
    enc_names = [o.name for o in enc.get_outputs()]
    assert enc_names == FEATURE_NAMES, enc_names
    print(f"{model}: {'frame':10s} {'pts':>3s} {'logit diff':>10s} {'score diff':>10s} {'iou':>6s}")
    for name in frames:
        ctx = E.load_context(ROOT / "out" / "live_occlusion" / name)
        pos = frame_prompts(ctx, MAX_CHECK_POINTS)
        h, w = ctx.rgb.shape[:2]
        labels = np.ones(len(pos), np.int32)

        with torch.inference_mode():
            predictor.set_image(ctx.rgb)
            _, ref_scores, ref_low = predictor.predict(pos, labels, multimask_output=True, return_logits=True)

        feats = dict(zip(enc_names, enc.run(None, {"image": resize_like_sam2(ctx.rgb)})))
        coords = (pos * np.array([INPUT_SIZE / w, INPUT_SIZE / h], np.float32))[None]
        onnx_low, onnx_scores = dec.run(None, dict(feats, point_coords=coords, point_labels=labels[None]))
        onnx_low, onnx_scores = onnx_low[0], onnx_scores[0]

        top = int(np.argmax(ref_scores))
        print(f"{'':{len(model) + 1}s} {name:10s} {len(pos):3d} {float(np.max(np.abs(ref_low - onnx_low))):10.3e} "
              f"{float(np.max(np.abs(ref_scores - onnx_scores))):10.3e} {S.iou(ref_low[top] > 0, onnx_low[top] > 0):6.4f}")


def report_files(out_dir: Path, models: list[str]) -> None:
    for model in models:
        for path in (encoder_path(out_dir, model), decoder_path(out_dir, model)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{path.name}: {path.stat().st_size} bytes sha256 {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="tiny,small")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "sam2" / "onnx")
    parser.add_argument("--check", action="store_true", help="validate the ONNX chain against the predictor on real frames")
    parser.add_argument("--frames", default="angle_02,angle_05,take_t10")
    parser.add_argument("--no-export", action="store_true", help="skip the export, only check and report")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")
    for model in models:
        if not args.no_export:
            export(model, args.opset, args.checkpoints, args.out)
        if args.check:
            check(model, args.checkpoints, args.out, args.frames.split(","))
    report_files(args.out, models)


if __name__ == "__main__":
    main()
