"""Baseline: the raw ARKit LiDAR depth, fused as-is.

Confidence 0/1/2 becomes fusion weight 0/0.5/1, so low-confidence pixels are
dropped and medium ones count half.
"""

from __future__ import annotations

from recon.capture import Capture, DEPTH_WIDTH, DEPTH_HEIGHT
from recon.depthcache import DepthCache

NAME = "lidar_tsdf"
LICENSE = "n/a (sensor data)"


def predict(capture: Capture, frame_ids, cache: DepthCache, cfg: dict) -> None:
    cache.begin(NAME, DEPTH_WIDTH, DEPTH_HEIGHT)
    for fid in frame_ids:
        cache.write(int(fid), capture.depth(fid), capture.lidar_weight(fid), capture.intrinsics_depth(fid))
    cache.finish()
