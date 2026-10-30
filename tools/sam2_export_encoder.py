"""Export SAM2's image encoder to ONNX at a fixed input size, so sam2_dml_timing.py can time it under
ONNX Runtime's DirectML provider, the runtime the MikanXR editor would use.

    .venv_sam2/Scripts/python tools/sam2_export_encoder.py [--models tiny,small] [--size 1024] [--opset 17]
        [--checkpoints models/sam2] [--out models/sam2/onnx]

The graph is wrapped to take one [1,3,size,size] float32 tensor of RGB in [0,1] (the resized frame,
not yet normalized) and return exactly what SAM2ImagePredictor.set_image stores as its features:
image_embed [1,256,size/16,size/16] (the lowest backbone_fpn level with no_mem_embed added),
high_res_feats_0 [1,32,size/4,size/4] and high_res_feats_1 [1,64,size/8,size/8] (the two upper levels
through the mask decoder's conv_s0 and conv_s1, the projection forward_image applies). The ImageNet
normalization (mean 0.485, 0.456, 0.406, std 0.229, 0.224, 0.225) that SAM2Transforms applies is
folded into the graph, so the consumer only resizes and scales to [0,1]. Putting the split here keeps
the prompt decoder's (sam2_export_decoder.py) per-call inputs at 2.5 M floats instead of 22 M. The
position encodings are not emitted; the decoder does not need them. The legacy TorchScript exporter
is tried first and the dynamo exporter second. The export is checked against the PyTorch module on
the same random input when onnxruntime imports in this venv; otherwise the check is left to
sam2_dml_timing.py --check.

Writes <out>/sam2.1_<model>_encoder_<size>.onnx (with an external .data sidecar if the exporter
makes one) and prints the output names and shapes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import sam2_edge_eval as S  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


OUTPUT_NAMES = ["image_embed", "high_res_feats_0", "high_res_feats_1"]


class EncoderWrapper(torch.nn.Module):
    """RGB in [0,1] -> (image_embed, high_res_feats_0, high_res_feats_1), the predictor's _features."""

    def __init__(self, sam):
        super().__init__()
        assert sam.directly_add_no_mem_embed and sam.use_high_res_features_in_sam
        self.encoder = sam.image_encoder
        self.conv_s0 = sam.sam_mask_decoder.conv_s0
        self.conv_s1 = sam.sam_mask_decoder.conv_s1
        self.no_mem_embed = sam.no_mem_embed
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        fpn0, fpn1, fpn2 = self.encoder((image - self.mean) / self.std)["backbone_fpn"]
        image_embed = fpn2 + self.no_mem_embed.view(1, -1, 1, 1)
        return image_embed, self.conv_s0(fpn0), self.conv_s1(fpn1)


def export(model: str, size: int, opset: int, checkpoints: Path, out_dir: Path) -> Path:
    cfg, ckpt = S.MODELS[model]
    overrides = [] if size == 1024 else [f"++model.image_size={size}"]
    sam = build_sam2(cfg, str(checkpoints / ckpt), device="cpu", hydra_overrides_extra=overrides)
    wrapper = EncoderWrapper(sam).eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        reference = wrapper(x)
    names = OUTPUT_NAMES
    path = out_dir / f"sam2.1_{model}_encoder_{size}.onnx"
    for dynamo in (False, True):
        try:
            torch.onnx.export(wrapper, (x,), str(path), opset_version=opset, input_names=["image"],
                              output_names=names, dynamo=dynamo)
            print(f"{path.name}: exported with {'dynamo' if dynamo else 'torchscript'} exporter")
            break
        except Exception as exc:
            print(f"{path.name}: {'dynamo' if dynamo else 'torchscript'} export failed: {str(exc).splitlines()[0][:160]}")
    else:
        raise SystemExit("both exporters failed")

    for name, t in zip(names, reference):
        print(f"  {name}: {tuple(t.shape)}")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not in this venv, numeric check deferred to sam2_dml_timing.py --check")
        return path
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})
    for name, ref, g in zip(names, reference, got):
        diff = float(np.max(np.abs(ref.numpy() - g)))
        print(f"  {name}: max abs diff vs torch {diff:.3e}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="tiny,small")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "sam2" / "onnx")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for model in args.models.split(","):
        export(model, args.size, args.opset, args.checkpoints, args.out)


if __name__ == "__main__":
    main()
