"""Hand-mark an occluder's true silhouette on captured live frames, as ground truth for the
occlusion edge prototypes.

    python tools/mark_silhouette.py [--root out/live_occlusion] [angle_01 angle_02 ...]

Opens each capture's video.png (written by capture_live_occlusion.py). Outline the occluder
with clicks; a closed outline is filled into the mask. Several outlines per frame are allowed
(one object in pieces, or several objects). Points are kept in full resolution pixels.

    left click        add a point to the current outline
    backspace / z     remove the last point (of the current outline, else the last outline)
    enter / c         close the current outline
    wheel             zoom at the cursor
    middle drag       pan
    f                 fit the frame to the window
    o                 toggle the mesh mask's silhouette edges (red)
    m                 toggle the filled mask preview
    s                 save
    n / p             save, then next / previous capture
    q / esc           save and quit

Writes gt_mask.png (255 inside the marked occluder, full resolution) and gt_polygons.json
beside the capture, and loads gt_polygons.json again when it exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "mark silhouette"
VIEW_W, VIEW_H = 1280, 960
LOUPE = 41      # source pixels across the magnifier
LOUPE_SCALE = 6


def mesh_edges(depth: np.ndarray, jump: float = 0.05) -> np.ndarray:
    mn = cv2.erode(depth, np.ones((3, 3), np.uint8))
    mx = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    return (mx - mn) > jump * mx


class Marker:
    def __init__(self, capture: Path):
        self.capture = capture
        self.image = cv2.imread(str(capture / "video.png"), cv2.IMREAD_COLOR)
        self.h, self.w = self.image.shape[:2]
        depth_path = capture / "mesh_depth.npy"
        self.edges = mesh_edges(np.load(depth_path)) if depth_path.exists() else None
        self.outlines: list[list[tuple[float, float]]] = []
        self.current: list[tuple[float, float]] = []
        self.show_edges = True
        self.show_mask = True
        self.cursor = (0, 0)
        self.pan_from = None
        self.load()
        self.fit()

    # -- view transform: screen = (image - offset) * scale
    def fit(self) -> None:
        self.scale = min(VIEW_W / self.w, VIEW_H / self.h)
        self.offset = np.array([0.0, 0.0])

    def to_image(self, x: int, y: int) -> tuple[float, float]:
        return (x / self.scale + self.offset[0], y / self.scale + self.offset[1])

    def zoom(self, x: int, y: int, factor: float) -> None:
        ix, iy = self.to_image(x, y)
        self.scale = float(np.clip(self.scale * factor, 0.2, 16.0))
        self.offset = np.array([ix - x / self.scale, iy - y / self.scale])

    # -- persistence
    def gt_paths(self) -> tuple[Path, Path]:
        return self.capture / "gt_polygons.json", self.capture / "gt_mask.png"

    def load(self) -> None:
        polygons, _ = self.gt_paths()
        if polygons.exists():
            data = json.loads(polygons.read_text())
            self.outlines = [[tuple(p) for p in outline] for outline in data["outlines"]]

    def mask(self) -> np.ndarray:
        mask = np.zeros((self.h, self.w), np.uint8)
        for outline in self.outlines:
            if len(outline) >= 3:
                # Fill at 8x subpixel precision so a click's fraction is not rounded away
                pts = np.round(np.array(outline) * 8).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_8, shift=3)
        return mask

    def save(self) -> None:
        if self.current:
            self.close_outline()
        polygons, mask_path = self.gt_paths()
        # An empty mask would read as a frame with no occluder, so an unmarked capture writes nothing
        if not self.outlines and not polygons.exists():
            return
        polygons.write_text(json.dumps({"size": [self.w, self.h], "outlines": self.outlines}, indent=1))
        cv2.imwrite(str(mask_path), self.mask())
        print(f"saved {mask_path} ({len(self.outlines)} outline(s))")

    # -- editing
    def close_outline(self) -> None:
        if len(self.current) >= 3:
            self.outlines.append(self.current)
        self.current = []

    def undo(self) -> None:
        if self.current:
            self.current.pop()
        elif self.outlines:
            self.current = self.outlines.pop()

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append(self.to_image(x, y))
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.pan_from = (x, y, self.offset.copy())
        elif event == cv2.EVENT_MBUTTONUP:
            self.pan_from = None
        elif event == cv2.EVENT_MOUSEMOVE and self.pan_from is not None:
            x0, y0, offset0 = self.pan_from
            self.offset = offset0 - np.array([x - x0, y - y0]) / self.scale
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom(x, y, 1.25 if flags > 0 else 0.8)

    # -- drawing
    def frame(self) -> np.ndarray:
        base = self.image.copy()
        if self.show_mask and self.outlines:
            m = self.mask() > 0
            base[m] = (base[m] * 0.6 + np.array([255, 160, 0]) * 0.4).astype(np.uint8)
        if self.show_edges and self.edges is not None:
            base[self.edges] = (0, 0, 255)

        affine = np.array([[self.scale, 0, -self.offset[0] * self.scale],
                           [0, self.scale, -self.offset[1] * self.scale]], np.float32)
        interp = cv2.INTER_NEAREST if self.scale > 2 else cv2.INTER_LINEAR
        view = cv2.warpAffine(base, affine, (VIEW_W, VIEW_H), flags=interp, borderValue=(40, 40, 40))

        def screen(p):
            return (int(round((p[0] - self.offset[0]) * self.scale)), int(round((p[1] - self.offset[1]) * self.scale)))

        for outline in self.outlines:
            cv2.polylines(view, [np.array([screen(p) for p in outline], np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
        if self.current:
            pts = [screen(p) for p in self.current]
            cv2.polylines(view, [np.array(pts + [self.cursor], np.int32)], False, (0, 255, 0), 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(view, p, 2, (0, 255, 0), -1)

        self.draw_loupe(view, base)
        status = (f"{self.capture.name}  zoom {self.scale:.2f}  outlines {len(self.outlines)}"
                  f"  points {len(self.current)}  [enter close, z undo, s save, n/p next/prev, q quit]")
        cv2.putText(view, status, (10, VIEW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, status, (10, VIEW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def draw_loupe(self, view: np.ndarray, base: np.ndarray) -> None:
        ix, iy = self.to_image(*self.cursor)
        cx, cy, half = int(ix), int(iy), LOUPE // 2
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return
        patch = np.zeros((LOUPE, LOUPE, 3), np.uint8)
        x0, y0 = cx - half, cy - half
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + LOUPE, self.w), min(y0 + LOUPE, self.h)
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = base[sy0:sy1, sx0:sx1]
        big = cv2.resize(patch, (LOUPE * LOUPE_SCALE, LOUPE * LOUPE_SCALE), interpolation=cv2.INTER_NEAREST)
        c = half * LOUPE_SCALE + LOUPE_SCALE // 2
        cv2.line(big, (c, 0), (c, big.shape[0]), (0, 255, 0), 1)
        cv2.line(big, (0, c), (big.shape[1], c), (0, 255, 0), 1)
        view[8:8 + big.shape[0], VIEW_W - 8 - big.shape[1]:VIEW_W - 8] = big


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("captures", nargs="*", help="capture folder names (default: all under --root)")
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    args = parser.parse_args()

    names = args.captures or sorted(p.name for p in args.root.iterdir() if (p / "video.png").exists())
    if not names:
        raise SystemExit(f"no captures under {args.root}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    index = 0
    marker = Marker(args.root / names[index])
    cv2.setMouseCallback(WINDOW, marker.on_mouse)

    while True:
        cv2.imshow(WINDOW, marker.frame())
        key = cv2.waitKeyEx(15)
        if key == -1:
            continue
        ch = key & 0xFF
        if ch in (13, ord("c")):
            marker.close_outline()
        elif ch in (8, ord("z")):
            marker.undo()
        elif ch == ord("f"):
            marker.fit()
        elif ch == ord("o"):
            marker.show_edges = not marker.show_edges
        elif ch == ord("m"):
            marker.show_mask = not marker.show_mask
        elif ch == ord("s"):
            marker.save()
        elif ch in (ord("n"), ord("p")):
            marker.save()
            index = (index + (1 if ch == ord("n") else -1)) % len(names)
            marker = Marker(args.root / names[index])
            cv2.setMouseCallback(WINDOW, marker.on_mouse)
        elif ch in (ord("q"), 27):
            marker.save()
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            marker.save()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
