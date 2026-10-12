"""Stray Scanner capture loader.

A capture directory holds rgb.mp4, depth/NNNNNN.png (uint16 millimetres,
256x192), confidence/NNNNNN.png (uint8 0/1/2, higher is better), and
odometry.csv with one row per frame: timestamp, frame, x y z, qx qy qz qw,
fx fy cx cy (colour-frame intrinsics), distortion centre.

Pose convention: odometry rows are camera-to-world. Stray Scanner already
rotates ARKit's camera frame by 180 degrees about X, so the camera axes are
OpenCV style (+X right, +Y down, +Z forward) and no further flip is needed.
World is ARKit's gravity-aligned right-handed frame (+Y up).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192


@dataclass
class Capture:
    path: Path
    frame_ids: np.ndarray          # (N,) capture frame indices with depth, confidence, and a pose
    timestamps: np.ndarray         # (N,)
    T_WC: np.ndarray               # (N,4,4) camera-to-world, OpenCV camera axes
    K_color: np.ndarray            # (N,3,3) intrinsics of the colour frame
    color_size: tuple[int, int]    # (width, height) of rgb.mp4

    @property
    def name(self) -> str:
        return self.path.name

    def __len__(self) -> int:
        return len(self.frame_ids)

    # ---- lookups -----------------------------------------------------------

    def index_of(self, frame_id: int) -> int:
        i = int(np.searchsorted(self.frame_ids, frame_id))
        if i >= len(self.frame_ids) or self.frame_ids[i] != frame_id:
            raise KeyError(f"frame {frame_id} not in capture")
        return i

    def pose(self, frame_id: int) -> np.ndarray:
        return self.T_WC[self.index_of(frame_id)]

    def intrinsics_color(self, frame_id: int) -> np.ndarray:
        return self.K_color[self.index_of(frame_id)]

    def intrinsics_depth(self, frame_id: int) -> np.ndarray:
        return scale_intrinsics(self.intrinsics_color(frame_id), self.color_size, (DEPTH_WIDTH, DEPTH_HEIGHT))

    # ---- per-frame data ----------------------------------------------------

    def depth(self, frame_id: int) -> np.ndarray:
        """Depth in metres, float32, 0 where the sensor reported nothing."""
        img = cv2.imread(str(self.path / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            raise IOError(f"bad depth image for frame {frame_id}")
        return img.astype(np.float32) / 1000.0

    def confidence(self, frame_id: int) -> np.ndarray:
        """ARKit confidence, uint8 0 (low) / 1 (medium) / 2 (high)."""
        img = cv2.imread(str(self.path / "confidence" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"missing confidence image for frame {frame_id}")
        return img

    def lidar_weight(self, frame_id: int) -> np.ndarray:
        """Confidence mapped to a fusion weight in [0,1]: 0 -> 0, 1 -> 0.5, 2 -> 1."""
        return self.confidence(frame_id).astype(np.float32) * 0.5

    def frames(self, frame_ids) -> Iterator[tuple[int, np.ndarray]]:
        """Decode rgb.mp4 sequentially and yield (frame_id, RGB uint8 HxWx3) for the requested ids."""
        import av

        wanted = set(int(f) for f in frame_ids)
        if not wanted:
            return
        last = max(wanted)
        with av.open(str(self.path / "rgb.mp4")) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    yield index, frame.to_ndarray(format="rgb24")
                if index >= last:
                    break

    def read_frames(self, frame_ids) -> dict[int, np.ndarray]:
        return dict(self.frames(frame_ids))

    # ---- frame selection ---------------------------------------------------

    def split(self, keyframe_stride: int, holdout_every: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_ids, holdout_ids). Keyframes are every keyframe_stride-th
        frame; every holdout_every-th keyframe is held out of fusion."""
        keyframes = self.frame_ids[::keyframe_stride]
        mask = np.zeros(len(keyframes), dtype=bool)
        mask[holdout_every // 2::holdout_every] = True
        return keyframes[~mask], keyframes[mask]

    # ---- geometry helpers --------------------------------------------------

    def lidar_points(self, frame_id: int, min_confidence: int = 2, depth_max: float = 5.0) -> np.ndarray:
        """World-space points from one LiDAR frame, (M,3)."""
        depth = self.depth(frame_id)
        valid = (self.confidence(frame_id) >= min_confidence) & (depth > 0) & (depth < depth_max)
        pts_cam = unproject(depth, self.intrinsics_depth(frame_id))[valid]
        return transform_points(self.pose(frame_id), pts_cam)

    def check_pose_consistency(self, frame_id: int, other_id: int) -> dict:
        """Reproject frame_id's LiDAR into other_id's camera and compare depths.
        With correct poses the median absolute difference is centimetres; a
        wrong convention gives tens of centimetres or metres."""
        pts_w = self.lidar_points(frame_id)
        T_CW = np.linalg.inv(self.pose(other_id))
        pts_c = transform_points(T_CW, pts_w)
        K = self.intrinsics_depth(other_id)
        z = pts_c[:, 2]
        front = z > 0.05
        uv = (K @ pts_c[front].T).T
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
        v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        inside = (u >= 0) & (u < DEPTH_WIDTH) & (v >= 0) & (v < DEPTH_HEIGHT)
        other_depth = self.depth(other_id)
        other_conf = self.confidence(other_id)
        u, v, zf = u[inside], v[inside], z[front][inside]
        ok = (other_conf[v, u] == 2) & (other_depth[v, u] > 0)
        diff = np.abs(other_depth[v, u][ok] - zf[ok])
        return {
            "compared": int(ok.sum()),
            "median_abs_diff_m": float(np.median(diff)) if ok.any() else float("nan"),
            "p90_abs_diff_m": float(np.percentile(diff, 90)) if ok.any() else float("nan"),
        }


# ---- free functions ----------------------------------------------------------

def load_capture(path: str | Path, frame_count_cache: str | Path | None = None) -> Capture:
    """Load a capture. Stray Scanner writes one more depth and odometry frame
    than rgb.mp4 decodes to, so frames past the decodable video length are
    dropped. Counting decodable frames means a full decode (seconds); pass
    frame_count_cache to remember the count in a small text file."""
    path = Path(path)
    rows = _read_odometry(path / "odometry.csv")
    depth_ids = {int(p.stem) for p in (path / "depth").glob("*.png")}
    conf_ids = {int(p.stem) for p in (path / "confidence").glob("*.png")}
    video_frames = _decodable_frame_count(path / "rgb.mp4", frame_count_cache)
    rows = [r for r in rows if r["frame"] in depth_ids and r["frame"] in conf_ids and r["frame"] < video_frames]
    rows.sort(key=lambda r: r["frame"])
    if not rows:
        raise IOError(f"no frames with depth, confidence, and a pose in {path}")

    frame_ids = np.array([r["frame"] for r in rows], dtype=np.int64)
    timestamps = np.array([r["timestamp"] for r in rows])
    T_WC = np.zeros((len(rows), 4, 4))
    K_color = np.zeros((len(rows), 3, 3))
    for i, r in enumerate(rows):
        T_WC[i, :3, :3] = Rotation.from_quat([r["qx"], r["qy"], r["qz"], r["qw"]]).as_matrix()
        T_WC[i, :3, 3] = [r["x"], r["y"], r["z"]]
        T_WC[i, 3, 3] = 1.0
        K_color[i] = [[r["fx"], 0, r["cx"]], [0, r["fy"], r["cy"]], [0, 0, 1]]

    return Capture(path, frame_ids, timestamps, T_WC, K_color, _probe_video_size(path / "rgb.mp4"))


def _read_odometry(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        col = {name: i for i, name in enumerate(header)}
        needed = ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"]
        missing = [n for n in needed if n not in col]
        if missing:
            raise IOError(f"odometry.csv lacks columns {missing}; per-frame intrinsics are required")
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            rows.append({n: (int(raw[col[n]]) if n == "frame" else float(raw[col[n]])) for n in needed})
    return rows


def _decodable_frame_count(video_path: Path, cache_path: str | Path | None) -> int:
    """Number of frames rgb.mp4 actually decodes to. The container's own
    frame count (stream.frames) is one higher than that on Stray Scanner
    recordings, so the video is decoded once and the answer cached."""
    import av

    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        return int(cache.read_text().strip())
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        count = sum(1 for _ in container.decode(stream))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(str(count))
    return count


def _probe_video_size(video_path: Path) -> tuple[int, int]:
    import av

    with av.open(str(video_path)) as container:
        s = container.streams.video[0]
        return int(s.codec_context.width), int(s.codec_context.height)


def scale_intrinsics(K: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
    return S @ K


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-pixel camera-space points, (H,W,3), using pixel-centre convention."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]
