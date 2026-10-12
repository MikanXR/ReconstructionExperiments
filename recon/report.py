"""Collect every method's metrics.json under out/<capture>/ into report.md."""

from __future__ import annotations

import json
from pathlib import Path

COLUMNS = [
    ("edge mean px", lambda m: m["eval"]["summary"]["edge_symmetric_px"], "{:.2f}"),
    ("edge median px", lambda m: m["eval"]["summary"]["edge_median_px"], "{:.2f}"),
    ("edges within 2px", lambda m: m["eval"]["summary"]["edge_matched_fraction"] * 100, "{:.1f}%"),
    ("edge mean px (colour res)", lambda m: m["eval"]["summary"]["edge_symmetric_px_color"], "{:.1f}"),
    ("MAE cm", lambda m: m["eval"]["summary"]["mae_m"] * 100, "{:.2f}"),
    ("RMSE cm", lambda m: m["eval"]["summary"]["rmse_m"] * 100, "{:.2f}"),
    ("within 2cm", lambda m: m["eval"]["summary"]["within_2cm"] * 100, "{:.1f}%"),
    ("within 5cm", lambda m: m["eval"]["summary"]["within_5cm"] * 100, "{:.1f}%"),
    ("completeness", lambda m: m["eval"]["summary"]["completeness"] * 100, "{:.1f}%"),
    ("floater hits", lambda m: m["eval"]["summary"]["floater_fraction"] * 100, "{:.2f}%"),
    ("unsupported surface", lambda m: m["eval"]["summary"]["unsupported_surface_fraction"] * 100, "{:.2f}%"),
    ("chamfer vs LiDAR cm", lambda m: m["eval"].get("vs_baseline", {}).get("chamfer_m", float("nan")) * 100, "{:.2f}"),
    ("F-score vs LiDAR", lambda m: m["eval"].get("vs_baseline", {}).get("fscore", float("nan")), "{:.3f}"),
    ("triangles", lambda m: m["fusion"]["triangles_clean"], "{:,}"),
    ("islands removed", lambda m: m["fusion"]["triangles_removed_as_islands"], "{:,}"),
    ("depth res", lambda m: f"{m['depth']['width']}x{m['depth']['height']}", "{}"),
    ("predict s", lambda m: m["depth"]["seconds"], "{:.0f}"),
    ("fuse s", lambda m: m["fusion"]["fuse_seconds"], "{:.0f}"),
    ("peak VRAM GB", lambda m: m["depth"].get("peak_vram_gb", float("nan")), "{:.1f}"),
    ("license", lambda m: m["method"].get("license", ""), "{}"),
]


def _fmt(fmt: str, value) -> str:
    try:
        if isinstance(value, float) and value != value:
            return "n/a"
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def write_report(capture_out: Path) -> Path:
    runs = []
    for metrics_path in sorted(capture_out.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        if "eval" in m:   # runs that only exported or only fused have nothing to compare yet
            runs.append((metrics_path.parent.name, m))
    lines = [f"# Reconstruction comparison: {capture_out.name}", ""]
    if not runs:
        lines.append("No runs yet.")
    else:
        cfg = runs[0][1].get("config", {})
        lines.append(f"Keyframe stride {cfg.get('keyframe_stride')}, holdout every {cfg.get('holdout_every')} keyframes, "
                     f"{runs[0][1].get('frames', {}).get('train')} training frames, "
                     f"{runs[0][1].get('frames', {}).get('holdout')} held-out frames. "
                     f"Voxel {cfg.get('fusion', {}).get('voxel_size')} m.")
        lines.append("")
        lines.append("Edge px is the nearest-pixel distance between silhouette pixels in the held-out LiDAR depth and in the mesh rendered from the same pose, both directions, at LiDAR resolution (256x192), clipped at 20 px. Lower is better everywhere except within/matched/completeness/F-score. Unsupported surface is relative to the LiDAR cloud, so the baseline's own value is the noise floor.")
        lines.append("")
        lines.append("| method | " + " | ".join(c[0] for c in COLUMNS) + " |")
        lines.append("|---|" + "---|" * len(COLUMNS))
        for name, m in runs:
            cells = []
            for _, getter, fmt in COLUMNS:
                try:
                    cells.append(_fmt(fmt, getter(m)))
                except (KeyError, TypeError):
                    cells.append("n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("## Views")
        lines.append("")
        lines.append("Each strip: video frame, mesh shaded from the same pose, rendered depth over the frame (red near, blue far).")
        lines.append("")
        view_ids = sorted({int(v) for _, m in runs for v in m.get("renders", {}).get("views", {})})
        for vid in view_ids:
            lines.append(f"### Frame {vid:06d}")
            lines.append("")
            for name, m in runs:
                rel = m.get("renders", {}).get("views", {}).get(str(vid))
                if rel:
                    lines.append(f"**{name}**")
                    lines.append("")
                    lines.append(f"![{name} {vid}]({name}/renders/{rel})")
                    lines.append("")
        lines.append("### Top-down")
        lines.append("")
        for name, m in runs:
            rel = m.get("renders", {}).get("topdown")
            if rel:
                lines.append(f"**{name}**")
                lines.append("")
                lines.append(f"![{name} top-down]({name}/renders/{rel})")
                lines.append("")
    path = capture_out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
