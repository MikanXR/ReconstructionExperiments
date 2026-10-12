"""Per-method, per-frame depth cache.

Every method ends by writing one of these; fusion and evaluation read it and
never see the model. Layout inside the cache directory:

    meta.json                  method name, image size, frame ids, per-frame intrinsics
    NNNNNN_depth.npy           float32 metres, 0 or non-finite where invalid
    NNNNNN_weight.npy          float32 fusion weight in [0,1] (0 drops the pixel)

The depth lives at whatever resolution the method predicted. Intrinsics are
stored per frame at that resolution, so readers never rescale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DepthCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / "meta.json"
        self.meta = json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {
            "method": None, "width": None, "height": None, "frames": {}}

    # ---- writing -------------------------------------------------------------

    def begin(self, method: str, width: int, height: int, extra: dict | None = None) -> None:
        self.meta = {"method": method, "width": int(width), "height": int(height), "frames": {}}
        if extra:
            self.meta.update(extra)
        self._save()

    def write(self, frame_id: int, depth: np.ndarray, weight: np.ndarray, K: np.ndarray) -> None:
        h, w = depth.shape
        if (w, h) != (self.meta["width"], self.meta["height"]):
            raise ValueError(f"frame {frame_id}: depth is {w}x{h}, cache is {self.meta['width']}x{self.meta['height']}")
        if weight.shape != depth.shape:
            raise ValueError(f"frame {frame_id}: weight shape {weight.shape} != depth shape {depth.shape}")
        np.save(self.dir / f"{frame_id:06d}_depth.npy", depth.astype(np.float32))
        np.save(self.dir / f"{frame_id:06d}_weight.npy", weight.astype(np.float32))
        self.meta["frames"][str(int(frame_id))] = np.asarray(K, dtype=float).tolist()
        if len(self.meta["frames"]) % 50 == 0:   # a crashed run keeps the frames it finished
            self._save()

    def finish(self) -> None:
        self._save()

    def _save(self) -> None:
        self._meta_path.write_text(json.dumps(self.meta, indent=1))

    # ---- reading -------------------------------------------------------------

    @property
    def method(self) -> str:
        return self.meta["method"]

    @property
    def size(self) -> tuple[int, int]:
        return self.meta["width"], self.meta["height"]

    @property
    def frame_ids(self) -> np.ndarray:
        return np.array(sorted(int(f) for f in self.meta["frames"]), dtype=np.int64)

    def has(self, frame_id: int) -> bool:
        return str(int(frame_id)) in self.meta["frames"]

    def intrinsics(self, frame_id: int) -> np.ndarray:
        return np.array(self.meta["frames"][str(int(frame_id))])

    def read(self, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (depth, weight, K) for one frame."""
        depth = np.load(self.dir / f"{frame_id:06d}_depth.npy")
        weight = np.load(self.dir / f"{frame_id:06d}_weight.npy")
        return depth, weight, self.intrinsics(frame_id)

    def complete_for(self, frame_ids) -> bool:
        return all(self.has(f) for f in frame_ids)
