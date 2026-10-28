"""Capture one live camera angle from a running MikanXR editor for offline occlusion edge work.

    python tools/capture_live_occlusion.py <label> [--compositor 2209] [--camera 2205]
           [--port 21120] [--out out/live_occlusion]

The editor's automation server switches the compositor between three capture graphs in the
project (cap_video, cap_depth, cap_refined) and screenshots each, then restores the graph it
found. The depth graphs draw the mesh depth mask through the dbgDepthPack material, which
packs normalized linear depth into an 8-bit sRGB frame: red is depth * 4 (about 5 m), green
repeats every 1/64 of the range, blue is mesh coverage. decode_depth() inverts that.

Writes, per label:
    video.png         the undistorted video as composited, no client layers
    mesh_depth.npy    float32 normalized linear depth of the mesh mask, 1 where no mesh
    refined_depth.npy the same after the editor's DepthEdgeRefineNode (cap_refined's settings)
    overlay_mesh.png  the mesh mask's silhouette edges (red) on the video
    overlay_refined.png  the refined mask's silhouette edges (green) on the video
    meta.json         label, time, graph settings, clip planes, camera pose
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The editor's mono camera defaults: the mask's normalized depth runs between these
ZNEAR = 0.1
ZFAR = 20.0


class Automation:
    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.file = self.sock.makefile("rb")

    def command(self, line: str) -> list[str]:
        self.sock.sendall((line + "\n").encode())
        count = int(self.file.readline().decode().strip())
        lines = [self.file.readline().decode().rstrip("\r\n") for _ in range(count)]
        # An error is one line, "<namespace> <verb>: <message>"
        prefix = " ".join(line.split(" ")[:2]) + ":"
        if len(lines) == 1 and lines[0].startswith(prefix):
            raise RuntimeError(lines[0])
        return lines


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    c = encoded.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def decode_depth(png_rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(png_rgb)
    coarse = lin[..., 0] / 4.0
    fine = lin[..., 1]
    steps = np.round(coarse * 64.0 - fine)
    depth = (steps + fine) / 64.0
    depth[lin[..., 2] < 0.5] = 1.0
    return depth.astype(np.float32)


def silhouette_edges(depth: np.ndarray, jump: float = 0.05) -> np.ndarray:
    """Pixels where the 3x3 neighbourhood's depth spread exceeds jump * max, the node's band test."""
    mn = cv2.erode(depth, np.ones((3, 3), np.uint8))
    mx = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    return (mx - mn) > jump * mx


def screenshot(auto: Automation, compositor: int, graph: str, path: Path, settle: float) -> np.ndarray:
    auto.command(f"property set CompositorObjectSystem {compositor} compositor_graph_path compositors/{graph}.compgraph")
    time.sleep(settle)
    errors = auto.command(f"nodegraph errors {compositor}")
    if errors:
        raise RuntimeError(f"{graph}: " + "; ".join(errors))
    auto.command(f"screenshot compositor {compositor} {path.as_posix()}")
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def capture(auto: Automation, label: str, compositor: int, camera: int, out_root: Path, settle: float = 1.5) -> Path:
    """Capture one view through an open automation connection into out_root/label."""
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)

    original_graph = auto.command(f"property get CompositorObjectSystem {compositor} compositor_graph_path")[0]
    try:
        video = screenshot(auto, compositor, "cap_video", out / "video.png", settle)
        mesh = decode_depth(screenshot(auto, compositor, "cap_depth", out / "mesh_depth_enc.png", settle))
        refined = decode_depth(screenshot(auto, compositor, "cap_refined", out / "refined_depth_enc.png", settle))
        pose = {
            name: auto.command(f"property get CameraObjectSystem {camera} {name}")[0]
            for name in ("relative_position", "relative_quaternion", "parent_transform_id")
        }
    finally:
        auto.command(f"property set CompositorObjectSystem {compositor} compositor_graph_path {original_graph}")

    np.save(out / "mesh_depth.npy", mesh)
    np.save(out / "refined_depth.npy", refined)

    for name, depth, colour in (("mesh", mesh, (0, 0, 255)), ("refined", refined, (0, 255, 0))):
        overlay = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        overlay[silhouette_edges(depth)] = colour
        cv2.imwrite(str(out / f"overlay_{name}.png"), overlay)

    meta = {
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "compositor": compositor,
        "camera": camera,
        "camera_pose": pose,
        "znear": ZNEAR,
        "zfar": ZFAR,
        "size": [int(video.shape[1]), int(video.shape[0])],
        "refined_graph": "cap_refined (band_radius 24, color_sigma 0.1, depth_jump 0.02)",
        "depth_encoding": "red = d * 4, green = fract(d * 64), blue = coverage, all through the sRGB encode",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    covered = mesh < 1.0
    metres = ZNEAR + mesh[covered] * (ZFAR - ZNEAR)
    print(f"{out}: {video.shape[1]}x{video.shape[0]}, mesh covers {covered.mean():.0%}, "
          f"depth {metres.min():.2f} to {metres.max():.2f} m")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label")
    parser.add_argument("--compositor", type=int, default=2209)
    parser.add_argument("--camera", type=int, default=2205)
    parser.add_argument("--port", type=int, default=21120)
    parser.add_argument("--settle", type=float, default=1.5, help="seconds to let a graph switch land")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "live_occlusion")
    args = parser.parse_args()

    capture(Automation(args.port), args.label, args.compositor, args.camera, args.out, args.settle)


if __name__ == "__main__":
    main()
