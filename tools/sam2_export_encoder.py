"""Export a SAM2-API image encoder to ONNX at a fixed input size, so sam2_dml_timing.py can time it
under ONNX Runtime's DirectML provider, the runtime the MikanXR editor uses.

    .venv_sam2/Scripts/python tools/sam2_export_encoder.py [--family sam2] [--models tiny,small] [--size 0]
        [--opset 17] [--checkpoints <dir>] [--out models/<family>/onnx]

The families and their venvs are sam2_edge_eval.py's: sam2, edgetam and efficienttam, each run with
its own venv's python. EfficientViT-SAM is not exported here; its authors publish ONNX encoders
(models/efficientvit_sam/onnx, from HuggingFace mit-han-lab/efficientvit-sam).

The graph is wrapped to take one [1,3,size,size] float32 tensor of RGB in [0,1] (the resized frame,
not yet normalized) and return exactly what the image predictor's set_image stores as its features:
image_embed [1,256,size/16,size/16] (the lowest backbone_fpn level with no_mem_embed added), and
for a model with use_high_res_features_in_sam, high_res_feats_0 [1,32,size/4,size/4] and
high_res_feats_1 [1,64,size/8,size/8] (the two upper levels through the mask decoder's conv_s0 and
conv_s1, the projection forward_image applies). EfficientTAM has no high-res features and its graph
returns image_embed alone. The ImageNet normalization (mean 0.485, 0.456, 0.406, std 0.229, 0.224,
0.225) that the transforms apply is folded into the graph, so the consumer only resizes and scales
to [0,1]. Putting the split here keeps the prompt decoder's (sam2_export_decoder.py) per-call inputs
at 2.5 M floats instead of 22 M. The position encodings are not emitted; the decoder does not need
them. --size 0 exports at the model's own input; another size overrides model.image_size through
hydra. EdgeTAM's RepViT trunk is reparameterized (timm's reparameterize_model, the fused inference
form) before the export, checked against the unfused module. The legacy TorchScript exporter is
tried first and the dynamo exporter second. The export is checked against the PyTorch module on
the same random input when onnxruntime imports in this venv; otherwise the check is left to
sam2_dml_timing.py --check.

Writes <out>/sam2.1_<model>_encoder_<size>.onnx for sam2 and <out>/<family>_<model>_encoder_<size>.onnx
otherwise (with an external .data sidecar if the exporter makes one) and prints the output names and
shapes.
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


OUTPUT_NAMES = ["image_embed", "high_res_feats_0", "high_res_feats_1"]


class EncoderWrapper(torch.nn.Module):
    """RGB in [0,1] -> (image_embed, high_res_feats_0, high_res_feats_1), the predictor's _features,
    or image_embed alone for a model without high-res features."""

    def __init__(self, sam):
        super().__init__()
        assert sam.directly_add_no_mem_embed
        self.high_res = bool(sam.use_high_res_features_in_sam)
        self.encoder = sam.image_encoder
        if self.high_res:
            self.conv_s0 = sam.sam_mask_decoder.conv_s0
            self.conv_s1 = sam.sam_mask_decoder.conv_s1
        self.no_mem_embed = sam.no_mem_embed
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        levels = self.encoder((image - self.mean) / self.std)["backbone_fpn"]
        image_embed = levels[-1] + self.no_mem_embed.view(1, -1, 1, 1)
        if not self.high_res:
            return (image_embed,)
        return image_embed, self.conv_s0(levels[0]), self.conv_s1(levels[1])


def export(family: str, model: str, size: int, opset: int, checkpoints: Path | None, out_dir: Path) -> Path:
    sam = S.build_model(family, model, checkpoints, size, device="cpu")
    size = int(sam.image_size)
    wrapper = EncoderWrapper(sam).eval()
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        reference = wrapper(x)
    if family == "edgetam":
        from timm.utils.model import reparameterize_model
        fused = reparameterize_model(wrapper).eval()
        with torch.no_grad():
            fused_out = fused(x)
        print(f"{family} {model}: reparameterized trunk, max abs diff vs unfused "
              f"{max(float((a - b).abs().max()) for a, b in zip(reference, fused_out)):.3e}")
        wrapper, reference = fused, fused_out
    names = OUTPUT_NAMES[:len(reference)]
    stem = f"sam2.1_{model}" if family == "sam2" else f"{family}_{model}"
    path = out_dir / f"{stem}_encoder_{size}.onnx"
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
    parser.add_argument("--family", default="sam2", choices=["sam2", "edgetam", "efficienttam"])
    parser.add_argument("--models", default=None, help="comma separated, default tiny,small for sam2, else all")
    parser.add_argument("--size", type=int, default=0, help="encoder input, 0 for the model's own")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--checkpoints", type=Path, default=None, help="default per family")
    parser.add_argument("--out", type=Path, default=None, help="default models/<family>/onnx")
    args = parser.parse_args()
    out = args.out or ROOT / "models" / args.family / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    default_models = "tiny,small" if args.family == "sam2" else ",".join(S.FAMILIES[args.family][2])
    for model in (args.models or default_models).split(","):
        export(args.family, model, args.size, args.opset, args.checkpoints, out)


if __name__ == "__main__":
    main()
