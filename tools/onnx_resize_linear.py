"""Rewrite every cubic Resize node of an ONNX graph to linear, saving <stem>_linear.onnx beside it.

    python tools/onnx_resize_linear.py models/efficientvit_sam/onnx/l0_encoder.onnx [...]

DirectML has no cubic resize kernel, so ONNX Runtime runs such a node on the CPU with a host round
trip either side of it, which turned EfficientViT-SAM's encoders (three bicubic upsamples in the
neck) from tens of milliseconds into hundreds. The linear graph is for timing: its output differs
from the bicubic one the accuracy rows were scored with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx


def main() -> None:
    for arg in sys.argv[1:]:
        src = Path(arg)
        model = onnx.load(str(src))
        patched = 0
        for node in model.graph.node:
            if node.op_type != "Resize":
                continue
            keep = []
            for attr in node.attribute:
                if attr.name == "mode" and attr.s == b"cubic":
                    attr.s = b"linear"
                    patched += 1
                if attr.name != "cubic_coeff_a":
                    keep.append(attr)
            del node.attribute[:]
            node.attribute.extend(keep)
        dst = src.with_name(src.stem + "_linear" + src.suffix)
        onnx.save(model, str(dst))
        print(f"{src.name}: {patched} Resize nodes switched to linear -> {dst}")


if __name__ == "__main__":
    main()
