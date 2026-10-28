"""Score occlusion edge methods against hand-marked occluder silhouettes on live captures.

    python tools/occlusion_edge_eval.py [--root out/live_occlusion] [--out out/live_occlusion/eval]
           [--methods mesh,mikan_refined,...]

Each capture folder holds what capture_live_occlusion.py and mark_silhouette.py wrote: video.png,
mesh_depth.npy (the mesh depth mask Mikan renders), refined_depth.npy (Mikan's DepthEdgeRefineNode
output), and gt_mask.png (the occluder's true silhouette).

Only occluding edges are scored: points of the marked outline where the mesh depth 45 px outside
is farther than 45 px inside by more than 5 percent, so the can's base on the desk and a stretch
hidden behind a nearer object are left out. Every method yields a binary occluder region in a crop
around the marked one. A depth result becomes a region by thresholding at the midpoint between the
occluder's depth and the background's, keeping the component that overlaps the marked occluder.

Per method and capture:
    gt_to_pred   distance from each occluding outline point to the method's nearest boundary
                 (mean, median, p90, share within 2 and 5 px): how well the true edge is found
    pred_to_gt   mean distance from the method's boundary pixels within 40 px of an occluding
                 edge to the marked outline: how much stray or lumpy boundary it adds

Writes results.csv, summary.txt, and one contact sheet per capture (marked outline white,
method boundary coloured, occluding stretches yellow on the first tile).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.edge_refine import refine_depth  # noqa: E402

CROP_MARGIN = 160     # px around the marked occluder that every method sees
PROBE = 45            # px either side of the outline for the occluding-edge test
NEAR_BAND = 40        # px from an occluding edge within which stray boundary counts
ZNEAR, ZFAR = 0.1, 20.0


# ---- capture context ------------------------------------------------------------

@dataclass
class Context:
    name: str
    rgb: np.ndarray          # crop, uint8 sRGB
    mesh: np.ndarray         # crop, normalized linear depth, 1 where no mesh
    refined: np.ndarray      # crop, Mikan's refined depth
    gt: np.ndarray           # crop, bool
    tau: float               # occluder / background depth threshold
    prior: np.ndarray        # crop, bool: the mesh's occluder region
    occluding: np.ndarray    # crop, bool: occluding stretches of the marked outline
    offset: tuple[int, int]  # crop origin in the full frame


def to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    c = rgb_u8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def outline(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Positive outside the mask, negative inside, in pixels."""
    m = mask.astype(np.uint8)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return outside - inside


def region_from_depth(depth: np.ndarray, tau: float, gt: np.ndarray) -> np.ndarray:
    """The component of depth < tau that overlaps the marked occluder most."""
    near = (depth < tau).astype(np.uint8)
    count, labels = cv2.connectedComponents(near, connectivity=4)
    if count <= 1:
        return np.zeros_like(gt)
    overlap = np.bincount(labels[gt], minlength=count)
    overlap[0] = 0
    return labels == int(np.argmax(overlap))


def load_context(folder: Path) -> Context:
    rgb = cv2.cvtColor(cv2.imread(str(folder / "video.png")), cv2.COLOR_BGR2RGB)
    mesh = np.load(folder / "mesh_depth.npy")
    refined = np.load(folder / "refined_depth.npy")
    gt = cv2.imread(str(folder / "gt_mask.png"), cv2.IMREAD_GRAYSCALE) > 0

    ys, xs = np.nonzero(gt)
    h, w = gt.shape
    y0, y1 = max(ys.min() - CROP_MARGIN, 0), min(ys.max() + CROP_MARGIN + 1, h)
    x0, x1 = max(xs.min() - CROP_MARGIN, 0), min(xs.max() + CROP_MARGIN + 1, w)
    crop = (slice(y0, y1), slice(x0, x1))
    rgb, mesh, refined, gt = rgb[crop], mesh[crop], refined[crop], gt[crop]

    # Occluder depth from inside the marked region, background from a ring outside it
    sdf = signed_distance(gt)
    covered = mesh < 1.0
    inner = (sdf < -15) & covered
    ring = (sdf > 40) & (sdf < 70) & covered
    d_occluder = float(np.median(mesh[inner]))
    d_background = float(np.median(mesh[ring])) if ring.any() else 1.0
    tau = 0.5 * (d_occluder + d_background)
    prior = region_from_depth(mesh, tau, gt)

    # Occluding stretches: probe the mesh depth along the outline's normal
    gy = cv2.Sobel(cv2.GaussianBlur(sdf, (0, 0), 3), cv2.CV_32F, 0, 1)
    gx = cv2.Sobel(cv2.GaussianBlur(sdf, (0, 0), 3), cv2.CV_32F, 1, 0)
    norm = np.maximum(np.hypot(gx, gy), 1e-6)
    nx, ny = gx / norm, gy / norm
    edge = outline(gt)
    ey, ex = np.nonzero(edge)
    hh, ww = gt.shape

    def probe(sign: float) -> np.ndarray:
        px = np.clip(np.round(ex + sign * PROBE * nx[ey, ex]).astype(int), 0, ww - 1)
        py = np.clip(np.round(ey + sign * PROBE * ny[ey, ex]).astype(int), 0, hh - 1)
        return mesh[py, px]

    d_out, d_in = probe(+1.0), probe(-1.0)
    z_out = ZNEAR + d_out * (ZFAR - ZNEAR)
    z_in = ZNEAR + d_in * (ZFAR - ZNEAR)
    full_edge = (ex + x0 > 1) & (ex + x0 < w - 2) & (ey + y0 > 1) & (ey + y0 < h - 2)
    occl = (z_out - z_in > 0.05 * z_out) & (d_in < 1.0) & full_edge
    occluding = np.zeros_like(gt)
    occluding[ey[occl], ex[occl]] = True

    return Context(folder.name, rgb, mesh, refined, gt, tau, prior, occluding, (x0, y0))


# ---- scoring --------------------------------------------------------------------

def score(ctx: Context, pred: np.ndarray) -> dict:
    boundary = outline(pred)
    gt_edge = outline(ctx.gt)
    d_to_pred = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_gt = cv2.distanceTransform((~gt_edge).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_occl = cv2.distanceTransform((~ctx.occluding).astype(np.uint8), cv2.DIST_L2, 5)

    g = d_to_pred[ctx.occluding] if boundary.any() else np.full(ctx.occluding.sum(), 999.0)
    near = boundary & (d_to_occl < NEAR_BAND)
    p = d_to_gt[near] if near.any() else np.array([999.0])
    return {
        "gt_to_pred_mean": float(g.mean()),
        "gt_to_pred_median": float(np.median(g)),
        "gt_to_pred_p90": float(np.percentile(g, 90)),
        "within_2px": float((g <= 2.0).mean()),
        "within_5px": float((g <= 5.0).mean()),
        "pred_to_gt_mean": float(p.mean()),
        "occluding_px": int(ctx.occluding.sum()),
    }


# ---- methods --------------------------------------------------------------------

def trimap(prior: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Definite inside, definite outside, unknown band of +/- band px around the prior's edge."""
    sdf = signed_distance(prior)
    unknown = np.abs(sdf) <= band
    return prior & ~unknown, ~prior & ~unknown, unknown


def m_mesh(ctx: Context) -> np.ndarray:
    return ctx.prior


def m_mikan_refined(ctx: Context) -> np.ndarray:
    return region_from_depth(ctx.refined, ctx.tau, ctx.gt)


def m_py_refine(radius: int, sigma: float, jump: float):
    def run(ctx: Context) -> np.ndarray:
        refined = refine_depth(ctx.mesh, to_linear(ctx.rgb), radius, sigma, jump)
        return region_from_depth(refined, ctx.tau, ctx.gt)
    return run


def lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def m_trimap_vote(band: int, smooth: int = 7):
    """The node's two-prototype vote with the prototypes taken from the definite regions only,
    Gaussian weighted, and the band's labels median filtered for spatial coherence."""
    def run(ctx: Context) -> np.ndarray:
        fg, bg, unknown = trimap(ctx.prior, band)
        img = lab(ctx.rgb)
        sigma = float(band)

        def local_mean(region: np.ndarray) -> np.ndarray:
            w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), sigma)
            acc = cv2.GaussianBlur(img * region[..., None], (0, 0), sigma)
            return acc / np.maximum(w, 1e-6)[..., None]

        mu_f, mu_b = local_mean(fg), local_mean(bg)
        pick_f = np.sum((img - mu_f) ** 2, axis=2) < np.sum((img - mu_b) ** 2, axis=2)
        label = cv2.medianBlur((pick_f * 255).astype(np.uint8), smooth) > 127
        return fg | (unknown & label)
    return run


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s colour guided filter, box windows of radius r."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)

    I = guide
    mean_I = box(I)
    mean_p = box(src)
    mean_Ip = box(I * src[..., None])
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            var[..., i, j] = box(I[..., i] * I[..., j]) - mean_I[..., i] * mean_I[..., j]
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mean_p - np.sum(a * mean_I, axis=2)
    return np.sum(box(a) * I, axis=2) + box(b)


def m_guided(radius: int, eps: float, band: int = 45):
    def run(ctx: Context) -> np.ndarray:
        fg, bg, unknown = trimap(ctx.prior, band)
        guide = ctx.rgb.astype(np.float32) / 255.0
        q = guided_filter(guide, ctx.prior.astype(np.float32), radius, eps)
        return fg | (unknown & (q > 0.5))
    return run


def m_grabcut(band: int = 45, iterations: int = 5, scale: float = 0.5):
    def run(ctx: Context) -> np.ndarray:
        fg, bg, unknown = trimap(ctx.prior, band)
        init = np.where(fg, cv2.GC_FGD, np.where(bg, cv2.GC_BGD,
                        np.where(ctx.prior, cv2.GC_PR_FGD, cv2.GC_PR_BGD))).astype(np.uint8)
        h, w = init.shape
        small_img = cv2.resize(cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2BGR), (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        small = cv2.resize(init, (small_img.shape[1], small_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small_img, small, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
        res = np.isin(small, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR) > 0
        return fg | (unknown & res)
    return run


def contour_points(mask: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    samples = np.arange(0, arc[-1], spacing)
    closed = np.vstack([c, c[:1]])
    return np.stack([np.interp(samples, arc, closed[:, 0]), np.interp(samples, arc, closed[:, 1])], axis=1)


def m_contour_dp(band: int = 45, smoothness: float = 1.0, polarity: bool = True):
    """Snap the prior's contour to the image edge: per contour point, an offset along the outward
    normal in [-band, band] chosen by dynamic programming over the closed contour (cut open at one
    point), trading edge strength against offset change between neighbours. With polarity, an
    edge also has to separate colours that look like the definite inside from the definite outside."""
    def run(ctx: Context) -> np.ndarray:
        pts = contour_points(ctx.prior)
        n = len(pts)
        k = 5  # contour smoothing half-width in samples, for the normals only
        sm = np.stack([np.convolve(np.pad(pts[:, i], (k, k), mode="wrap"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
                       for i in range(2)], axis=1)
        tangent = np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        probe = pts + 4 * normal
        h, w = ctx.prior.shape
        px = np.clip(probe[:, 0].round().astype(int), 0, w - 1)
        py = np.clip(probe[:, 1].round().astype(int), 0, h - 1)
        if ctx.prior[py, px].mean() > 0.5:
            normal = -normal

        offsets = np.arange(-band - 3, band + 4, dtype=np.float32)
        mx = (pts[:, 0:1] + offsets[None] * normal[:, 0:1]).astype(np.float32)
        my = (pts[:, 1:2] + offsets[None] * normal[:, 1:2]).astype(np.float32)
        img = lab(ctx.rgb)
        prof = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)   # (n, m, 3)
        grad = np.linalg.norm(prof[:, 2:] - prof[:, :-2], axis=2)[:, 2:-2]               # offsets -band..band
        strength = grad / max(np.percentile(grad, 99), 1e-6)

        unary = -strength
        if polarity:
            fg, bg, _ = trimap(ctx.prior, band)
            def mean_colour(region):
                return img[region].mean(axis=0)
            mu_f, mu_b = mean_colour(fg), mean_colour(bg)
            inner = prof[:, 1:-5]   # samples just inside each candidate (offset - 2)
            outer = prof[:, 5:-1]   # samples just outside (offset + 2)
            side_ok = (np.linalg.norm(inner - mu_f, axis=2) < np.linalg.norm(inner - mu_b, axis=2)).astype(np.float32) \
                + (np.linalg.norm(outer - mu_b, axis=2) < np.linalg.norm(outer - mu_f, axis=2)).astype(np.float32)
            unary = unary - 0.25 * side_ok

        m = unary.shape[1]
        steps = np.arange(m)
        jump = smoothness / 10.0 * np.abs(steps[:, None] - steps[None, :])
        cost = unary[0].copy()
        back = np.zeros((n, m), np.int32)
        for i in range(1, n):
            total = cost[None, :] + jump            # [to, from]
            back[i] = np.argmin(total, axis=1)
            cost = unary[i] + total[steps, back[i]]
        choice = np.empty(n, np.int32)
        choice[-1] = int(np.argmin(cost))
        for i in range(n - 1, 0, -1):
            choice[i - 1] = back[i, choice[i]]
        chosen = (choice - band).astype(np.float32)
        snapped = pts + chosen[:, None] * normal

        out = np.zeros(ctx.prior.shape, np.uint8)
        cv2.fillPoly(out, [np.round(snapped * 8).astype(np.int32)], 1, shift=3)
        return out > 0
    return run


METHODS = {
    "mesh": m_mesh,
    "mikan_refined": m_mikan_refined,
    "py_refine_default": m_py_refine(16, 0.2, 0.05),
    "py_refine_24": m_py_refine(24, 0.1, 0.02),
    "trimap_vote_30": m_trimap_vote(30),
    "trimap_vote_45": m_trimap_vote(45),
    "guided_r8_e1e-3": m_guided(8, 1e-3),
    "guided_r16_e1e-3": m_guided(16, 1e-3),
    "guided_r16_e1e-2": m_guided(16, 1e-2),
    "grabcut": m_grabcut(),
    "contour_dp": m_contour_dp(polarity=False),
    "contour_dp_polarity": m_contour_dp(polarity=True),
    "contour_dp_smooth3": m_contour_dp(smoothness=3.0),
}

COLOURS = [(255, 64, 64), (64, 220, 64), (64, 160, 255), (255, 200, 0), (255, 64, 255), (0, 220, 220)]


def contact_sheet(ctx: Context, preds: dict[str, np.ndarray], path: Path) -> None:
    tiles = []
    base = ctx.rgb.copy()
    base[ctx.occluding] = (255, 255, 0)
    tiles.append(("occluding edges", base))
    for i, (name, pred) in enumerate(preds.items()):
        t = ctx.rgb.copy()
        t[outline(ctx.gt)] = (255, 255, 255)
        t[cv2.dilate(outline(pred).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = COLOURS[i % len(COLOURS)]
        tiles.append((name, t))
    th = 360
    resized = []
    for name, t in tiles:
        tw = int(t.shape[1] * th / t.shape[0])
        r = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(r, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(r, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(r)
    cols = 5
    width = max(r.shape[1] for r in resized)
    rows = []
    for i in range(0, len(resized), cols):
        row = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in resized[i:i + cols]]
        row += [np.zeros((th, width, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "live_occlusion")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    out = args.out or args.root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    names = args.methods.split(",")

    rows = []
    for folder in sorted(p for p in args.root.iterdir() if (p / "gt_mask.png").exists()):
        ctx = load_context(folder)
        preds = {}
        for name in names:
            t0 = time.perf_counter()
            pred = METHODS[name](ctx)
            elapsed = time.perf_counter() - t0
            preds[name] = pred
            rows.append({"capture": ctx.name, "method": name, "seconds": round(elapsed, 2), **score(ctx, pred)})
        contact_sheet(ctx, preds, out / f"{ctx.name}.png")
        print(f"{ctx.name}: occluding {int(ctx.occluding.sum())} px, tau {ctx.tau:.4f}")

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"{'method':24s} {'gt>pred mean':>12s} {'median':>7s} {'p90':>6s} {'<=2px':>6s} {'<=5px':>6s}"
             f" {'pred>gt':>8s}   per-capture gt>pred mean"]
    captures = sorted({r["capture"] for r in rows})
    for name in names:
        rs = [r for r in rows if r["method"] == name]
        mean = lambda k: np.mean([r[k] for r in rs])
        per = " ".join(f"{r['gt_to_pred_mean']:5.1f}" for r in sorted(rs, key=lambda r: r["capture"]))
        lines.append(f"{name:24s} {mean('gt_to_pred_mean'):12.2f} {mean('gt_to_pred_median'):7.2f}"
                     f" {mean('gt_to_pred_p90'):6.1f} {mean('within_2px'):6.2f} {mean('within_5px'):6.2f}"
                     f" {mean('pred_to_gt_mean'):8.2f}   {per}")
    lines.append("captures: " + " ".join(captures))
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
