"""Score how long a keyframe's matte stays valid when carried forward through the occlusion mesh.

    .venv_sam2/Scripts/python tools/keyframe_reproject_eval.py <take.trackcamera> <mesh.glb> [--poses <track.pose.json>]
        [--root out/live_occlusion] [--model small] [--band 24] [--correction-band 64] [--depth-tolerance 0.06]
        [--keyframe-stride 5] [--k 1,2,3,5,8,10,15,20,30,45,60] [--checkpoints models/sam2]
        [--out out/keyframe_reproject]

The live matte design segments a keyframe and carries the result to the frames after it, since the
occluders are static and only the camera moves. This scores what a frame t+k would show when the last
segmentation was on frame t, for each keyframe distance k, two ways of carrying it, and two truths.

The marked folders under --root (gt_mask.png beside video.png, named take_tNN or tNN for playback
seconds, or carrying meta.json's frame index when render_take_frames.py wrote them) fix the span: every
movie frame from the first marked index to the last is decoded once into <out>/frames, the mesh is ray
cast from each frame's pose into <out>/depth (float16 metres, inf where no mesh), and each frame gets a
prior and an image-mode mask. The prior needs no marked mask: the marked frames' occluder pixels at the
occluder's mesh depth (within --depth-tolerance of the region's median, which leaves out the mesh
showing through a part the mesh lacks and the desk under the base) are unprojected once into one world
point cloud of the occluder, and a frame's prior is that cloud projected into the frame, splatted,
closed and reduced to its largest component, handed to occlusion_edge_eval.make_context as the marked
region would be (its depth threshold and mesh prior then follow as on the marked frames). The
corrections a keyframe carries are confined to --correction-band px of its prior's occluding contours
(sam2_edge_eval.occluding_outline), the matte's own reach, not the prior's whole outline. The image-mode
mask is sam2_edge_eval.py's runs_pos prompt on the crop around that prior, the design's segmentation
of that frame alone, cached in <out>/image_mask with the context's prior in <out>/prior.

Carried from keyframe t to frame t+k, all full-frame regions:
    hold             the keyframe's image-mode mask as it is
    mesh             the mesh prior of frame t+k, uncorrected
    mask_warp        every keyframe mask pixel unprojected with the keyframe's mesh depth (a pixel the
                     mesh misses takes the nearest mesh pixel's depth), projected into frame t+k,
                     splatted at the nearest pixel and closed 3x3
    correction_warp  the mesh prior of frame t+k plus the keyframe's added pixels (mask outside the
                     prior) and minus its removed pixels (prior outside the mask), both within
                     --correction-band px of the prior's contour, carried as 3D points the same way,
                     a point dropped where it lands more than 2 cm behind frame t+k's mesh
    image            frame t+k's own image-mode mask, the ceiling (only in the marked table)

Re-anchoring (off with --no-anchor): a carried silhouette is aligned to the video's edges at the target
frame before scoring. The evidence is the frame's Canny edges (20 and 60 after a 1 px blur) binned by
gradient orientation into eight bins over 180 degrees, one distance transform per bin truncated at
8 px (an edge counts for its own bin and its neighbours), so a contour point is measured only against
edges whose orientation matches its normal's, which keeps a printed label from attracting a contour
that runs the other way, and a point with no edge near it neither flattens nor drags the search. The
contour aligned is the mask's occluding outline (sam2_edge_eval.occluding_outline against the target's
mesh depth), a translation is searched on the integer grid within --anchor-radius px (the bare mesh
rows within --mesh-anchor-radius, since the mesh sits farther off), refined to sub-pixel by a parabolic
fit, and rejected when it gains less than half a pixel of mean edge distance over no shift. `global`
moves the whole mask by one translation; `runs` gives each occluding contour run
(sam2_edge_eval.contour_runs) its own and substitutes that run's shifted mask within the correction
band plus the radius of its points, the nearest run winning where bands meet.
    correction_anchored       correction_warp re-anchored, global
    correction_anchored_runs  correction_warp re-anchored per run
    mesh_anchored             the mesh prior of t+k re-anchored, global, with no segmenter at all
    mesh_anchored_runs        the same per run
    correction_chained        the live design: carry one frame, re-anchor per run within --chain-radius
                              (one translation cannot serve a can 3 px off and a desk 18 px off in the
                              same contour, so the global scope never fires there), rebuild the keyframe
                              from the anchored mask against that frame's prior and mesh depth, carry one
                              more, up to --chain-max frames

Truths: the proxy is frame t+k's image-mode mask, scored for keyframes every --keyframe-stride frames
over the span with the context built around that mask; the marked truth is each marked frame m with
keyframe m-k, scored as every other table is. Writes results.csv, summary.md (the two tables, the mean
over pairs of the per-pair median, p90 and share within 5 px, as the other summaries average over
captures, with the camera's translation and rotation at each k), prior_check.png (eight frames across
the span with the prior and the image-mode mask outlined) and contact sheets for the marked frames at
k 5, 15 and 30.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import occlusion_edge_eval as E  # noqa: E402
import render_take_frames as R  # noqa: E402
import sam2_edge_eval as S  # noqa: E402
from recon.capture import ARKIT_TO_OPENCV_CAMERA  # noqa: E402
from recon.evaluate import make_scene, raycast  # noqa: E402

THUMB = 8                    # the marked folder's video against the decoded frame, at 1/8
OCCLUSION_TOLERANCE = 0.02   # m behind the frame's mesh at which a carried point is hidden
CLOUD_VOXEL = 0.004          # m, the occluder cloud's decimation
MATCH_MAD = 5.0              # mean absolute difference above which a time-matched frame is wrong
CARRIED = ["hold", "mesh", "mask_warp", "correction_warp"]
ANCHORED = ["correction_anchored", "correction_anchored_runs", "mesh_anchored", "mesh_anchored_runs",
            "correction_chained"]
SHEET_KS = (5, 15, 30)
ANCHOR_BINS = 8              # edge orientation bins over 180 degrees
ANCHOR_CANNY = (20, 60)      # Canny thresholds after a 1 px blur: the segmenter's silhouette has an edge
                             # within 2 px on 74 to 86 percent of its points here, 60 percent at 40/120
ANCHOR_CAP = 8.0             # px at which the edge distance is truncated, so a point with no edge near it
                             # neither flattens the landscape nor drags the shift
ANCHOR_MIN_GAIN = 0.5        # px of mean edge distance a shift must gain to be applied
EDGE_CACHE = 16              # edge fields kept in memory, by frame index


# ---- the take -------------------------------------------------------------------

class Take:
    """The take's pose track: per-frame intrinsics and camera-to-world in the OpenCV camera convention."""

    def __init__(self, path: Path, poses: Path | None):
        self.archive = tarfile.open(path)
        self.members = json.load(self.archive.extractfile("capture.json"))["members"]
        self.track = json.load(self.archive.extractfile(self.members["movie_pose"]))
        self.frames = self.track["frames"]
        self.transforms = (R.override_transforms(self.track, poses) if poses
                           else [f["transform"] for f in self.frames])
        self.track_size = (int(self.track["image_width"]), int(self.track["image_height"]))
        t0 = min(f["time_us"] for f in self.frames)
        self.seconds = np.array([(f["time_us"] - t0) / 1e6 for f in self.frames])
        self.size: tuple[int, int] | None = None   # the decoded size, known after the first frame

    def camera(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        f = self.frames[index]
        w, h = self.size
        sx, sy = w / self.track_size[0], h / self.track_size[1]
        K = np.array([[f["fx"] * sx, 0, f["cx"] * sx], [0, f["fy"] * sy, f["cy"] * sy], [0, 0, 1.0]])
        T_WC = np.array(self.transforms[index], dtype=np.float64).reshape(4, 4) @ ARKIT_TO_OPENCV_CAMERA
        return K, T_WC

    def motion(self, a: int, b: int) -> tuple[float, float]:
        """Translation (m) and rotation (deg) of the camera between two frames."""
        Ta = np.array(self.transforms[a], dtype=np.float64).reshape(4, 4)
        Tb = np.array(self.transforms[b], dtype=np.float64).reshape(4, 4)
        d = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
        cos = (np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1.0) / 2.0
        return d, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def marked_folders(root: Path, take: Take) -> list[tuple[Path, int]]:
    """(folder, track index) of every marked folder under root, in frame order."""
    found = []
    for p in sorted(root.iterdir()):
        if not (p / "gt_mask.png").exists():
            continue
        meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
        if "frame" in meta:
            index = int(meta["frame"])
        else:
            name = p.name[len("take_"):] if p.name.startswith("take_") else p.name
            if not (name.startswith("t") and name[1:].replace(".", "", 1).isdigit()):
                continue
            index = int(np.argmin(np.abs(take.seconds - float(name[1:]))))
        found.append((p, index))
    if not found:
        raise SystemExit(f"no marked folders under {root}")
    return sorted(found, key=lambda x: x[1])


def thumb(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (rgb.shape[1] // THUMB, rgb.shape[0] // THUMB), interpolation=cv2.INTER_AREA)


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# ---- geometry -------------------------------------------------------------------

def unproject(us: np.ndarray, vs: np.ndarray, z: np.ndarray, K: np.ndarray, T_WC: np.ndarray) -> np.ndarray:
    """World points (N, 3) of pixels at camera depth z."""
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    return cam @ T_WC[:3, :3].T + T_WC[:3, 3]


def project(world: np.ndarray, K: np.ndarray, T_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates and camera depth of world points in the frame at T_WC."""
    cam = (world - T_WC[:3, 3]) @ T_WC[:3, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return u, v, z


def splat(u: np.ndarray, v: np.ndarray, z: np.ndarray, shape: tuple[int, int],
          z_mesh: np.ndarray | None = None) -> np.ndarray:
    """The pixels the points land on, closed 3x3; with z_mesh, points behind the mesh there are dropped."""
    h, w = shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    keep = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if z_mesh is not None:
        ok = np.zeros_like(keep)
        ok[keep] = z[keep] <= z_mesh[vi[keep], ui[keep]] + OCCLUSION_TOLERANCE
        keep &= ok
    out = np.zeros(shape, np.uint8)
    out[vi[keep], ui[keep]] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def filled_depth(z: np.ndarray) -> np.ndarray:
    """The mesh depth with every miss taking the nearest hit's depth."""
    hit = np.isfinite(z)
    if hit.all():
        return z
    idx = ndimage.distance_transform_edt(~hit, return_indices=True)[1]
    return z[idx[0], idx[1]]


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros_like(mask, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


# ---- per-frame store ------------------------------------------------------------

class Frames:
    """Decoded frames, mesh depth, prior and image-mode mask per track index, each cached under out."""

    def __init__(self, take: Take, scene, out: Path):
        self.take, self.scene = take, scene
        self.dirs = {name: out / name for name in ("frames", "depth", "prior", "image_mask")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.cloud: np.ndarray | None = None   # the occluder's world points (N, 3)
        self.segment = None     # the image-mode method, set once the predictor exists

    def path(self, kind: str, index: int) -> Path:
        ext = {"frames": "jpg", "depth": "npy"}.get(kind, "png")
        return self.dirs[kind] / f"{index:05d}.{ext}"

    def decode(self, first: int, last: int) -> None:
        wanted = {i for i in range(first, last + 1) if not self.path("frames", i).exists()}
        if wanted:
            print(f"decoding {len(wanted)} frames of {first}..{last}", flush=True)
            for index, rgb in R.decode_frames(self.take.archive, self.take.members["movie"], wanted, self.dirs["frames"]):
                cv2.imwrite(str(self.path("frames", index)), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = self.rgb(first).shape[:2]
        self.take.size = (w, h)

    def rgb(self, index: int) -> np.ndarray:
        return read_rgb(self.path("frames", index))

    def z(self, index: int) -> np.ndarray:
        p = self.path("depth", index)
        if p.exists():
            return np.load(p).astype(np.float32)
        K, T_WC = self.take.camera(index)
        w, h = self.take.size
        z, _, _ = raycast(self.scene, K, T_WC, w, h)
        np.save(p, z.astype(np.float16))
        return z.astype(np.float32)

    def normalized(self, index: int) -> np.ndarray:
        return R.normalized_depth(self.z(index), R.ZNEAR, R.ZFAR)

    def cloud_prior(self, index: int) -> np.ndarray:
        """The convex hull of the occluder cloud projected into the frame (the occluders measured are convex)."""
        K, T_WC = self.take.camera(index)
        w, h = self.take.size
        u, v, z = project(self.cloud, K, T_WC)
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        region = np.zeros((h, w), np.uint8)
        if ok.sum() >= 3:
            hull = cv2.convexHull(np.round(np.stack([u[ok], v[ok]], axis=1)).astype(np.int32))
            cv2.fillConvexPoly(region, hull, 1)
        return region > 0

    def ensure_segmented(self, index: int) -> bool:
        """Compute and cache the frame's prior and image-mode mask; False when the frame has no occluder."""
        if self.path("image_mask", index).exists():
            return cv2.imread(str(self.path("image_mask", index)), cv2.IMREAD_GRAYSCALE).max() > 0
        box = self.cloud_prior(index)
        empty = np.zeros(box.shape, np.uint8)
        if box.sum() < 2000:
            cv2.imwrite(str(self.path("image_mask", index)), empty)
            cv2.imwrite(str(self.path("prior", index)), empty)
            return False
        ctx = E.make_context(f"f{index:05d}", self.rgb(index), self.normalized(index), self.normalized(index), box)
        if not np.isfinite(ctx.tau) or not ctx.prior.any():
            cv2.imwrite(str(self.path("image_mask", index)), empty)
            cv2.imwrite(str(self.path("prior", index)), empty)
            return False
        pred = self.segment(ctx)
        x0, y0 = ctx.offset
        h, w = ctx.gt.shape
        mask, prior = empty.copy(), empty.copy()
        mask[y0:y0 + h, x0:x0 + w] = pred.astype(np.uint8) * 255
        prior[y0:y0 + h, x0:x0 + w] = ctx.prior.astype(np.uint8) * 255
        cv2.imwrite(str(self.path("image_mask", index)), mask)
        cv2.imwrite(str(self.path("prior", index)), prior)
        return True

    def mask(self, kind: str, index: int) -> np.ndarray:
        return cv2.imread(str(self.path(kind, index)), cv2.IMREAD_GRAYSCALE) > 0


# ---- carrying a keyframe --------------------------------------------------------

class Keyframe:
    """A segmented frame with its corrections as world points, ready to carry to another frame."""

    def __init__(self, frames: Frames, index: int, correction_band: int, mask: np.ndarray | None = None):
        self.index = index
        self.mask = frames.mask("image_mask", index) if mask is None else mask
        self.prior = frames.mask("prior", index)
        z = frames.z(index)
        zfill = filled_depth(z)
        self.K, self.T_WC = frames.take.camera(index)
        # The matte's reach: the band around the prior's occluding contours, not its whole outline
        ys, xs, _ = S.occluding_outline(self.prior, R.normalized_depth(z, R.ZNEAR, R.ZFAR))
        contour = np.ones(self.prior.shape, np.uint8)
        contour[ys, xs] = 0
        band = cv2.distanceTransform(contour, cv2.DIST_L2, 5) <= correction_band
        self.world_mask = self.points(self.mask, zfill)
        self.world_added = self.points(self.mask & ~self.prior & band, zfill)
        self.world_removed = self.points(self.prior & ~self.mask & band, zfill)

    def points(self, region: np.ndarray, depth: np.ndarray) -> np.ndarray:
        vs, us = np.nonzero(region)
        return unproject(us.astype(np.float64), vs.astype(np.float64), depth[vs, us].astype(np.float64),
                         self.K, self.T_WC)

    def carry(self, frames: Frames, target: int) -> dict[str, np.ndarray]:
        K, T_WC = frames.take.camera(target)
        z_mesh = frames.z(target)
        prior_k = frames.mask("prior", target)
        shape = prior_k.shape
        warped = splat(*project(self.world_mask, K, T_WC), shape)
        added = splat(*project(self.world_added, K, T_WC), shape, z_mesh)
        removed = splat(*project(self.world_removed, K, T_WC), shape, z_mesh)
        return {"hold": self.mask, "mesh": prior_k, "mask_warp": warped,
                "correction_warp": (prior_k | added) & ~removed}


def crop(mask: np.ndarray, ctx: E.Context) -> np.ndarray:
    x0, y0 = ctx.offset
    h, w = ctx.gt.shape
    return mask[y0:y0 + h, x0:x0 + w]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


# ---- re-anchoring to the video's edges ------------------------------------------

class EdgeField:
    """A frame's Canny edges as one truncated distance transform per gradient orientation bin."""

    def __init__(self, rgb: np.ndarray):
        grey = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        edges = cv2.Canny(grey, *ANCHOR_CANNY) > 0
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
        bins = orientation_bins(gx, gy)
        fields = []
        for b in range(ANCHOR_BINS):
            near = (bins == b) | (bins == (b - 1) % ANCHOR_BINS) | (bins == (b + 1) % ANCHOR_BINS)
            dt = cv2.distanceTransform((~(edges & near)).astype(np.uint8), cv2.DIST_L2, 5)
            fields.append(np.minimum(dt, ANCHOR_CAP).astype(np.float16))
        self.dt = np.stack(fields)
        self.shape = edges.shape

    def costs(self, xs: np.ndarray, ys: np.ndarray, bins: np.ndarray, radius: int) -> np.ndarray:
        """Mean oriented edge distance of the contour under every integer shift, (2r+1, 2r+1) indexed [dy, dx]."""
        h, w = self.shape
        offs = np.arange(-radius, radius + 1)
        v = np.clip(ys[:, None, None] + offs[None, :, None], 0, h - 1)
        u = np.clip(xs[:, None, None] + offs[None, None, :], 0, w - 1)
        return self.dt[bins[:, None, None], v, u].astype(np.float32).mean(axis=0)


def orientation_bins(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    return (angle / np.pi * ANCHOR_BINS).astype(np.int64) % ANCHOR_BINS


def best_shift(field: EdgeField, xs: np.ndarray, ys: np.ndarray, bins: np.ndarray, radius: int) -> tuple[float, float] | None:
    """The sub-pixel translation that best aligns the contour to the edges, None when no shift gains enough."""
    if len(xs) == 0:
        return None
    costs = field.costs(xs, ys, bins, radius)
    i, j = np.unravel_index(int(np.argmin(costs)), costs.shape)
    if costs[radius, radius] - costs[i, j] < ANCHOR_MIN_GAIN:
        return None
    dy, dx = float(i - radius), float(j - radius)

    def refine(c_minus: float, c0: float, c_plus: float) -> float:
        denom = c_minus - 2.0 * c0 + c_plus
        return 0.5 * (c_minus - c_plus) / denom if denom > 1e-6 else 0.0

    if 0 < i < costs.shape[0] - 1:
        dy += refine(costs[i - 1, j], costs[i, j], costs[i + 1, j])
    if 0 < j < costs.shape[1] - 1:
        dx += refine(costs[i, j - 1], costs[i, j], costs[i, j + 1])
    return dx, dy


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask.shape
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return cv2.warpAffine(mask.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0) > 0.5


class Anchor:
    """Re-anchors silhouettes to a frame's edges; keeps the edge fields of recent frames and the shifts found."""

    def __init__(self, frames: Frames, radius: int, mesh_radius: int, correction_band: int):
        self.frames, self.radius, self.mesh_radius, self.correction_band = frames, radius, mesh_radius, correction_band
        self.fields: dict[int, EdgeField] = {}
        self.shifts: dict[str, list[float | None]] = {}

    def field(self, index: int) -> EdgeField:
        if index not in self.fields:
            if len(self.fields) >= EDGE_CACHE:
                del self.fields[next(iter(self.fields))]
            self.fields[index] = EdgeField(self.frames.rgb(index))
        return self.fields[index]

    def contour(self, mask: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The mask's occluding outline at the frame: xs, ys and the orientation bin of each point's normal."""
        if not mask.any():
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
        ys, xs, normals = S.occluding_outline(mask, self.frames.normalized(index))
        return xs, ys, orientation_bins(normals[:, 0], normals[:, 1])

    def record(self, name: str, shift: tuple[float, float] | None) -> None:
        self.shifts.setdefault(name, []).append(None if shift is None else float(np.hypot(*shift)))

    def apply(self, mask: np.ndarray, index: int, name: str, radius: int | None = None) -> np.ndarray:
        """One translation for the whole mask."""
        radius = radius or self.radius
        xs, ys, bins = self.contour(mask, index)
        shift = best_shift(self.field(index), xs, ys, bins, radius)
        self.record(name, shift)
        return mask if shift is None else shift_mask(mask, *shift)

    def apply_runs(self, mask: np.ndarray, index: int, name: str, radius: int | None = None) -> np.ndarray:
        """A translation per occluding contour run, each substituted within its band, the nearest run winning."""
        radius = radius or self.radius
        xs, ys, bins = self.contour(mask, index)
        runs = S.contour_runs(ys, xs, mask.shape) if len(xs) else []
        if not runs:
            self.record(name, None)
            return mask
        field = self.field(index)
        seeds = np.ones(mask.shape, np.uint8)
        run_of_point = np.zeros(len(xs), np.int64)
        for r, run in enumerate(runs):
            seeds[ys[run], xs[run]] = 0
            run_of_point[run] = r + 1
        dist, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        run_of_label = np.zeros(int(labels.max()) + 1, np.int64)
        run_of_label[labels[ys, xs]] = run_of_point
        nearest = run_of_label[labels]
        within = dist <= self.correction_band + radius
        out = mask.copy()
        for r, run in enumerate(runs):
            shift = best_shift(field, xs[run], ys[run], bins[run], radius)
            self.record(name, shift)
            if shift is None:
                continue
            sel = within & (nearest == r + 1)
            out[sel] = shift_mask(mask, *shift)[sel]
        return out

    def anchored(self, preds: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
        """The single-shot anchored rows for a frame's carried predictions."""
        return {"correction_anchored": self.apply(preds["correction_warp"], index, "correction_anchored"),
                "correction_anchored_runs": self.apply_runs(preds["correction_warp"], index, "correction_anchored_runs"),
                "mesh_anchored": self.apply(preds["mesh"], index, "mesh_anchored", self.mesh_radius),
                "mesh_anchored_runs": self.apply_runs(preds["mesh"], index, "mesh_anchored_runs", self.mesh_radius)}

    def chain(self, start: int, steps: int, have: set[int], chain_radius: int) -> dict[int, np.ndarray]:
        """Carry one frame, re-anchor, rebuild, repeat: the anchored mask at each k from 1 to steps."""
        mask = self.frames.mask("image_mask", start)
        out = {}
        for k in range(1, steps + 1):
            index, target = start + k - 1, start + k
            if target not in have:
                break
            kf = Keyframe(self.frames, index, self.correction_band, mask=mask)
            mask = self.apply_runs(kf.carry(self.frames, target)["correction_warp"], target, "correction_chained", chain_radius)
            out[k] = mask
        return out

    def summary(self) -> str:
        lines = ["| method | applications | shift median px | shift p90 px | rejected |", "|---|---|---|---|---|"]
        for name, shifts in self.shifts.items():
            found = [s for s in shifts if s is not None]
            med = f"{np.median(found):.1f}" if found else ""
            p90 = f"{np.percentile(found, 90):.1f}" if found else ""
            lines.append(f"| {name} | {len(shifts)} | {med} | {p90} | {1.0 - len(found) / max(len(shifts), 1):.2f} |")
        return "\n".join(lines)


# ---- reports --------------------------------------------------------------------

def prior_check(frames: Frames, indices: list[int], path: Path) -> None:
    tiles = []
    for index in indices:
        rgb = frames.rgb(index)
        if frames.path("prior", index).exists():
            rgb = rgb.copy()
            rgb[cv2.dilate(E.outline(frames.mask("prior", index)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 64, 64)
            rgb[cv2.dilate(E.outline(frames.mask("image_mask", index)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (255, 255, 255)
        t = cv2.resize(rgb, (rgb.shape[1] // 3, rgb.shape[0] // 3), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"frame {index}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, f"frame {index}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 4
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def table(rows: list[dict], ks: list[int], methods: list[str], motion: dict[int, tuple[float, float, float, float]]) -> str:
    lines = ["| k | translation cm | rotation deg | pairs | " + " | ".join(methods) + " |",
             "|---|---|---|---|" + "---|" * len(methods)]
    for k in ks:
        rs = [r for r in rows if r["k"] == k]
        if not rs:
            continue
        pairs = len({(r["keyframe"], r["frame"]) for r in rs})
        cells = []
        for name in methods:
            m = [r for r in rs if r["method"] == name]
            if not m:
                cells.append("")
                continue
            cells.append(f"{np.mean([r['gt_to_pred_median'] for r in m]):.1f} / "
                         f"{np.mean([r['gt_to_pred_p90'] for r in m]):.1f} / "
                         f"{np.mean([r['within_5px'] for r in m]):.2f}")
        dm, dp, am, ap = motion[k]
        lines.append(f"| {k} | {100 * dm:.1f} ({100 * dp:.1f}) | {am:.1f} ({ap:.1f}) | {pairs} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("take", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--model", default="small", choices=list(S.MODELS))
    parser.add_argument("--band", type=int, default=24)
    parser.add_argument("--correction-band", type=int, default=64)
    parser.add_argument("--depth-tolerance", type=float, default=0.06,
                        help="m around a marked region's median mesh depth that counts as the occluder")
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--k", default="1,2,3,5,8,10,15,20,30,45,60")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models" / "sam2")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "keyframe_reproject")
    parser.add_argument("--no-anchor", action="store_true", help="skip the re-anchored rows")
    parser.add_argument("--anchor-radius", type=int, default=16, help="px, the single-shot search radius")
    parser.add_argument("--mesh-anchor-radius", type=int, default=40,
                        help="px, the search radius for the bare mesh rows, which sit farther off the image")
    parser.add_argument("--chain-radius", type=int, default=8, help="px, the per-step search radius of the chain")
    parser.add_argument("--chain-max", type=int, default=10, help="frames the chained carry is followed for")
    args = parser.parse_args()
    ks = sorted({int(x) for x in args.k.split(",") if x.strip()})
    methods = CARRIED + ([] if args.no_anchor else ANCHORED)
    args.out.mkdir(parents=True, exist_ok=True)

    take = Take(args.take, args.poses)
    marked = marked_folders(args.root, take)
    first, last = marked[0][1], marked[-1][1]
    scene = make_scene(o3d.io.read_triangle_mesh(str(args.mesh)))
    frames = Frames(take, scene, args.out)
    frames.decode(first, last)
    print(f"span {first}..{last} ({last - first + 1} frames), decoded size {take.size}", flush=True)

    # Time-matched marked folders against the decoded frames
    for folder, index in marked:
        if (folder / "meta.json").exists() and "frame" in json.loads((folder / "meta.json").read_text()):
            continue
        a = thumb(read_rgb(folder / "video.png"))
        d = float(np.mean(np.abs(a.astype(np.int16) - thumb(frames.rgb(index)).astype(np.int16))))
        note = "" if d <= MATCH_MAD else "  MISMATCH"
        print(f"{folder.name}: frame {index}, video MAD {d:.2f}{note}")

    # Mesh depth for every frame
    for i, index in enumerate(range(first, last + 1)):
        frames.z(index)
        if i % 100 == 0:
            print(f"depth {index}", flush=True)

    # The occluder cloud from the marked frames: their marked pixels at the occluder's depth, in world
    # space. The mesh behind a part the mesh lacks (the can's cap) shows through the marked region and
    # the desk shows under its base, so only pixels within --depth-tolerance of the region's median
    # mesh depth count.
    world = []
    for folder, index in marked:
        gt = cv2.imread(str(folder / "gt_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        z = frames.z(index)
        covered = gt & np.isfinite(z)
        sel = covered & (np.abs(z - np.median(z[covered])) <= args.depth_tolerance)
        vs, us = np.nonzero(sel)
        K, T_WC = take.camera(index)
        pts = unproject(us.astype(np.float64), vs.astype(np.float64), z[vs, us].astype(np.float64), K, T_WC)
        world.append(pts)
        print(f"{folder.name}: {len(pts)} of {int(covered.sum())} covered marked pixels at the occluder's depth, "
              f"extent {np.round(pts.max(axis=0) - pts.min(axis=0), 3)} m")
    world = np.concatenate(world)
    frames.cloud = np.unique(np.round(world / CLOUD_VOXEL).astype(np.int64), axis=0) * CLOUD_VOXEL
    print(f"occluder cloud {len(frames.cloud)} points, extent {np.round(frames.cloud.max(axis=0) - frames.cloud.min(axis=0), 3)} m")
    for folder, index in marked:
        gt = cv2.imread(str(folder / "gt_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        print(f"{folder.name}: cloud prior IoU {iou(frames.cloud_prior(index), gt):.3f} with the marked region")

    # Image-mode segmentation of every frame
    runner = S.Sam2Runner(args.model, args.checkpoints, args.root, full_frame=False)
    frames.segment = S.make_method(runner, "runs_pos", args.band)
    segmented = []
    for i, index in enumerate(range(first, last + 1)):
        if frames.ensure_segmented(index):
            segmented.append(index)
        if i % 100 == 0:
            print(f"segmented {index}", flush=True)
    del runner
    torch.cuda.empty_cache()
    print(f"{len(segmented)} of {last - first + 1} frames have an occluder and a mask", flush=True)
    picks = [segmented[int(round(x))] for x in np.linspace(0, len(segmented) - 1, 8)]
    prior_check(frames, picks, args.out / "prior_check.png")

    # Identity check: carrying a keyframe to itself, scored against its own image-mode mask
    for index in segmented[:2]:
        kf = Keyframe(frames, index, args.correction_band)
        got = kf.carry(frames, index)
        ctx = E.make_context(f"f{index:05d}", frames.rgb(index), frames.normalized(index), frames.normalized(index), kf.mask)
        print(f"k = 0 at frame {index}, against the image mask: " +
              ", ".join(f"{name} IoU {iou(got[name], kf.mask):.3f} median {E.score(ctx, crop(got[name], ctx))['gt_to_pred_median']:.1f} px"
                        for name in ("hold", "mask_warp", "correction_warp")))

    # Edge agreement check: the image-mode mask re-anchored to its own frame should barely move
    anchor = None if args.no_anchor else Anchor(frames, args.anchor_radius, args.mesh_anchor_radius, args.correction_band)
    if anchor:
        moved = []
        for index in segmented[::max(1, len(segmented) // 50)]:
            anchor.apply(frames.mask("image_mask", index), index, "self_check")
            s = anchor.shifts["self_check"][-1]
            moved.append(0.0 if s is None else s)
        moved = np.array(moved)
        print(f"image mask re-anchored to its own frame on {len(moved)} frames: shift median {np.median(moved):.2f} px, "
              f"p90 {np.percentile(moved, 90):.2f}, under 1 px on {100 * (moved < 1.0).mean():.0f} percent", flush=True)
        del anchor.shifts["self_check"]

    motion = {}
    for k in ks:
        pairs = [(i, i + k) for i in range(first, last + 1 - k)]
        d = np.array([take.motion(a, b) for a, b in pairs]) if pairs else np.zeros((1, 2))
        motion[k] = (float(np.median(d[:, 0])), float(np.percentile(d[:, 0], 90)),
                     float(np.median(d[:, 1])), float(np.percentile(d[:, 1], 90)))

    rows = []
    # Proxy truth: the image-mode mask of frame t+k
    have = set(segmented)
    for t in range(first, last + 1, args.keyframe_stride):
        if t not in have:
            continue
        kf = Keyframe(frames, t, args.correction_band)
        chained = anchor.chain(t, args.chain_max, have, args.chain_radius) if anchor else {}
        for k in ks:
            target = t + k
            if target > last or target not in have:
                continue
            ctx = E.make_context(f"f{target:05d}", frames.rgb(target), frames.normalized(target),
                                 frames.normalized(target), frames.mask("image_mask", target))
            if not np.isfinite(ctx.tau) or not ctx.occluding.any():
                continue
            preds = kf.carry(frames, target)
            if anchor:
                preds.update(anchor.anchored(preds, target))
                if k in chained:
                    preds["correction_chained"] = chained[k]
            for name, pred in preds.items():
                rows.append({"truth": "proxy", "capture": f"f{target:05d}", "keyframe": t, "frame": target, "k": k,
                             "method": name, **E.score(ctx, crop(pred, ctx))})
        print(f"proxy keyframe {t}", flush=True)

    # Marked truth: each marked frame from the keyframe k before it
    for folder, m in marked:
        ctx = E.load_context(folder)
        if m in have:
            rows.append({"truth": "marked", "capture": folder.name, "keyframe": m, "frame": m, "k": 0,
                         "method": "image", **E.score(ctx, crop(frames.mask("image_mask", m), ctx))})
        for k in ks:
            t = m - k
            if t < first or t not in have:
                continue
            kf = Keyframe(frames, t, args.correction_band)
            preds = kf.carry(frames, m)
            if anchor:
                preds.update(anchor.anchored(preds, m))
                if k <= args.chain_max:
                    chained = anchor.chain(t, k, have, args.chain_radius)
                    if k in chained:
                        preds["correction_chained"] = chained[k]
            for name, pred in preds.items():
                rows.append({"truth": "marked", "capture": folder.name, "keyframe": t, "frame": m, "k": k,
                             "method": name, **E.score(ctx, crop(pred, ctx))})
            if k in SHEET_KS:
                E.contact_sheet(ctx, {name: crop(preds[name], ctx) for name in CARRIED},
                                args.out / f"{folder.name}_k{k:02d}.png")
                if anchor:
                    names = ["correction_warp", "correction_anchored", "correction_anchored_runs", "mesh_anchored",
                             "correction_chained" if "correction_chained" in preds else "mesh_anchored_runs"]
                    E.contact_sheet(ctx, {name: crop(preds[name], ctx) for name in names},
                                    args.out / f"{folder.name}_k{k:02d}_anchored.png")
        print(f"marked {folder.name} (frame {m})", flush=True)

    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    proxy = [r for r in rows if r["truth"] == "proxy"]
    marked_rows = [r for r in rows if r["truth"] == "marked"]
    image_rows = [r for r in marked_rows if r["method"] == "image"]
    image_line = ""
    if image_rows:
        image_line = (f"\nThe marked frames' own image-mode mask: "
                      f"{np.mean([r['gt_to_pred_median'] for r in image_rows]):.1f} / "
                      f"{np.mean([r['gt_to_pred_p90'] for r in image_rows]):.1f} / "
                      f"{np.mean([r['within_5px'] for r in image_rows]):.2f} over "
                      + ", ".join(sorted({r['capture'] for r in image_rows})) + ".\n")
    text = (f"# Keyframe reprojection, {args.take.name}, {args.mesh.name}, {args.model}, band {args.band}, "
            f"correction band {args.correction_band}\n\n"
            f"Cells are median / p90 / within 5 px of the occluding edge, averaged over the pairs. "
            f"Motion is the camera's median (p90) over the span's frame pairs k apart.\n\n"
            f"## Against the image-mode mask of frame t+k, keyframes every {args.keyframe_stride} frames\n\n"
            + table(proxy, ks, methods, motion) + "\n\n"
            f"## Against the hand-marked frames, keyframe k frames before each\n\n"
            + table(marked_rows, ks, methods, motion) + "\n" + image_line)
    if anchor:
        text += (f"\n## Re-anchoring shifts, radius {args.anchor_radius} px single-shot and {args.chain_radius} px per "
                 f"chained step, over both truths\n\n" + anchor.summary() + "\n")
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
