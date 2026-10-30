"""Bundle the exported SAM2 graphs into the release layout MikanXR's model download expects.

    .venv_sam2/Scripts/python tools/sam2_release_bundle.py [--models tiny,small] [--onnx models/sam2/onnx]
        [--out models/sam2/release] [--license <path to the SAM2 LICENSE text>]

For each model, writes <out>/sam2.1-<model>-onnx-1/ holding encoder.onnx (the 1024 encoder with the
normalization inside), decoder.onnx (the dynamic-point prompt decoder), manifest.json listing both
with their byte sizes and sha256 as ModelDownloadTask reads it, the LICENSE text, and a NOTICE.md
naming the checkpoints the graphs came from. The folder's name is the release tag the MikanXR model
catalog points at, so publishing is uploading the folder's files as that release's assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "hiera tiny"),
    "small": ("sam2.1_hiera_small.pt", "hiera small"),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle(model: str, onnx_dir: Path, out_dir: Path, license_path: Path | None) -> Path:
    release = out_dir / f"sam2.1-{model}-onnx-1"
    release.mkdir(parents=True, exist_ok=True)
    sources = {
        "encoder.onnx": onnx_dir / f"sam2.1_{model}_encoder_1024.onnx",
        "decoder.onnx": onnx_dir / f"sam2.1_{model}_decoder_dynpts.onnx",
    }
    files = []
    for name, source in sources.items():
        if not source.exists():
            raise SystemExit(f"{source} is missing; run the export scripts first")
        target = release / name
        shutil.copyfile(source, target)
        files.append({"name": name, "size": target.stat().st_size, "sha256": sha256_of(target)})

    checkpoint, flavour = CHECKPOINTS[model]
    notice = (
        f"# SAM 2.1 {flavour} for MikanXR\n\n"
        f"encoder.onnx and decoder.onnx are exported from the `{checkpoint}` checkpoint of\n"
        "facebookresearch/sam2 (SAM 2.1, Apache License 2.0) by the export scripts in\n"
        "MikanXR/ReconstructionExperiments (`tools/sam2_export_encoder.py`, `tools/sam2_export_decoder.py`).\n"
        "The encoder takes the frame resized to 1024x1024 as RGB in [0,1] with the ImageNet normalization\n"
        "inside and returns the image embedding and the two high-resolution feature maps as the predictor\n"
        "keeps them. The decoder takes those with any number of prompt points and returns three candidate\n"
        "masks as 256x256 logits with a score each. The model weights keep their original license, in\n"
        "LICENSE beside this file.\n"
    )
    (release / "NOTICE.md").write_text(notice, encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8") if license_path else ""
    if license_text:
        (release / "LICENSE").write_text(license_text, encoding="utf-8")
    manifest = {"files": files, "license": {"file": "LICENSE", "notice": "NOTICE.md"}}
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="tiny,small")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models" / "sam2" / "onnx")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "sam2" / "release")
    parser.add_argument("--license", type=Path, default=None)
    args = parser.parse_args()
    for model in args.models.split(","):
        release = bundle(model, args.onnx, args.out, args.license)
        manifest = json.loads((release / "manifest.json").read_text())
        print(release)
        for entry in manifest["files"]:
            print(f"  {entry['name']}: {entry['size']} bytes, sha256 {entry['sha256']}")


if __name__ == "__main__":
    main()
