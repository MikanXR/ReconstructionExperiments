"""Run one reconstruction method on a capture and score it.

    python run.py <method> [--capture ID] [--stage depth|fuse|eval|render|report|all] [--config path]

Outputs land in out/<capture>/<method>/: the depth cache, mesh_raw.ply,
mesh.ply, renders/, and metrics.json. The report stage rebuilds
out/<capture>/report.md from every method present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from recon.capture import load_capture  # noqa: E402
from recon.depthcache import DepthCache  # noqa: E402
from recon import fusion, evaluate, render, report  # noqa: E402
import methods  # noqa: E402

STAGES = ["depth", "fuse", "eval", "render", "report"]


def pick_capture(capture_root: Path, capture_id: str | None) -> Path:
    if capture_id:
        return capture_root / capture_id
    dirs = sorted(p for p in capture_root.iterdir() if (p / "odometry.csv").exists())
    if len(dirs) != 1:
        raise SystemExit(f"pass --capture; found {len(dirs)} captures under {capture_root}")
    return dirs[0]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_metrics(path: Path, metrics: dict, stage: str | None = None) -> None:
    """Write metrics.json, stamping the stage that just completed so status.py
    can tell a fresh result from one that predates a code change."""
    if stage:
        metrics.setdefault("stages", {})[stage] = time.time()
    path.write_text(json.dumps(metrics, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(methods.METHODS))
    ap.add_argument("--capture")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--stage", default="all", choices=STAGES + ["all"])
    ap.add_argument("--tag", default="", help="suffix for the output directory, for parameter variants")
    ap.add_argument("--force", action="store_true", help="recompute the depth cache even if complete")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value by dotted key, e.g. --opt da3.model=depth-anything/DA3-LARGE-1.1")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.opt:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    capture_dir = pick_capture(ROOT / cfg["capture_root"], args.capture)
    capture = load_capture(capture_dir, ROOT / cfg["out_root"] / capture_dir.name / "video_frames.txt")
    train_ids, holdout_ids = capture.split(cfg["keyframe_stride"], cfg["holdout_every"])
    method = methods.load(args.method)
    run_name = method.NAME + (f"_{args.tag}" if args.tag else "")
    out_dir = ROOT / cfg["out_root"] / capture.name / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics = load_metrics(metrics_path)
    metrics.update({
        "method": {"name": method.NAME, "license": getattr(method, "LICENSE", ""), "run": run_name},
        "capture": capture.name,
        "config": cfg,
        "frames": {"train": int(len(train_ids)), "holdout": int(len(holdout_ids))},
    })
    stages = STAGES if args.stage == "all" else [args.stage]
    print(f"{run_name}: capture {capture.name}, {len(capture)} frames, {len(train_ids)} train / {len(holdout_ids)} held out")

    cache = DepthCache(out_dir / "depth")
    method_cfg = cfg.get(method.NAME, {}) or {}
    if args.tag and not cache.complete_for(train_ids):
        # A tagged run that only varies fusion or evaluation reads the untagged
        # run's depth cache, provided that run used the same method settings.
        shared = DepthCache(ROOT / cfg["out_root"] / capture.name / method.NAME / "depth")
        if shared.complete_for(train_ids) and shared.method == method.NAME \
                and (shared.meta.get("method_cfg") or {}) == method_cfg:
            cache = shared
    produces_mesh = getattr(method, "PRODUCES_MESH", False)

    if "depth" in stages and produces_mesh:
        # Mesh-producing methods (Gaussian splatting) train outside the
        # harness; this stage only exports their input.
        info = method.export(capture, train_ids, out_dir, cfg)
        metrics["depth"] = {"width": info["width"], "height": info["height"], "frames": info["images"],
                            "seconds": float("nan"), "peak_vram_gb": float("nan"), "export": info}
        save_metrics(metrics_path, metrics, "depth")
        print(f"depth: exported {info['images']} images and {info['points']:,} points to {out_dir / 'colmap'}")
    elif "depth" in stages:
        if cache.complete_for(train_ids) and cache.method == method.NAME and not args.force:
            print("depth: cache complete, skipping (use --force to redo)")
        else:
            peak = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                torch = None
            t0 = time.perf_counter()
            method.predict(capture, train_ids, cache, cfg)
            seconds = time.perf_counter() - t0
            cache.meta["method_cfg"] = method_cfg
            cache.finish()
            if torch is not None and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
            metrics["depth"] = {"width": cache.size[0], "height": cache.size[1], "frames": int(len(train_ids)),
                                "seconds": seconds, "peak_vram_gb": peak}
            save_metrics(metrics_path, metrics, "depth")
            print(f"depth: {len(train_ids)} frames at {cache.size[0]}x{cache.size[1]} in {seconds:.0f}s")

    if "fuse" in stages and produces_mesh:
        t0 = time.perf_counter()
        mesh_raw = method.produce_mesh(capture, train_ids, out_dir, cfg)
        mesh_clean, removed = fusion.clean_mesh(mesh_raw, cfg["fusion"]["min_component_triangles"])
        lo, hi = fusion.training_bounds(capture, train_ids, cfg["fusion"])
        o3d.io.write_triangle_mesh(str(out_dir / "mesh_raw.ply"), mesh_raw)
        o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), mesh_clean)
        metrics["fusion"] = {
            "voxel_size": None, "grid_dims": None,
            "bounds_lo": [float(x) for x in lo], "bounds_hi": [float(x) for x in hi],
            "frames_integrated": int(len(train_ids)), "integrate_seconds": 0.0,
            "fuse_seconds": time.perf_counter() - t0,
            "triangles_raw": len(mesh_raw.triangles), "triangles_clean": len(mesh_clean.triangles),
            "triangles_removed_as_islands": removed,
        }
        save_metrics(metrics_path, metrics, "fuse")
        print(f"fuse: imported mesh, {len(mesh_clean.triangles):,} triangles ({removed:,} island triangles removed)")
    elif "fuse" in stages:
        result = fusion.fuse(cache, capture, train_ids, cfg["fusion"])
        o3d.io.write_triangle_mesh(str(out_dir / "mesh_raw.ply"), result.mesh_raw)
        o3d.io.write_triangle_mesh(str(out_dir / "mesh.ply"), result.mesh)
        metrics["fusion"] = result.stats
        save_metrics(metrics_path, metrics, "fuse")
        print(f"fuse: {result.stats['triangles_clean']:,} triangles ({result.stats['triangles_removed_as_islands']:,} island triangles removed) in {result.stats['fuse_seconds']:.0f}s")

    if "eval" in stages or "render" in stages:
        mesh = o3d.io.read_triangle_mesh(str(out_dir / "mesh.ply"))
        if not len(mesh.triangles):
            raise SystemExit("mesh.ply is empty; run the fuse stage first")

    if "eval" in stages:
        # The reference is the LiDAR baseline fused with the same tag when one
        # exists (fusion variants), else the default baseline.
        baseline = None
        for baseline_name in (["lidar_tsdf_" + args.tag] if args.tag else []) + ["lidar_tsdf"]:
            baseline_path = ROOT / cfg["out_root"] / capture.name / baseline_name / "mesh.ply"
            if baseline_path.exists() and run_name != baseline_name:
                baseline = o3d.io.read_triangle_mesh(str(baseline_path))
                break
        t0 = time.perf_counter()
        metrics["eval"] = evaluate.evaluate(mesh, capture, train_ids, holdout_ids, {**cfg["eval"], "depth_max": cfg["fusion"]["depth_max"]}, baseline)
        metrics["eval"]["seconds"] = time.perf_counter() - t0
        save_metrics(metrics_path, metrics, "eval")
        s = metrics["eval"]["summary"]
        print(f"eval: edge {s['edge_symmetric_px']:.2f}px  MAE {s['mae_m']*100:.2f}cm  completeness {s['completeness']*100:.1f}%  "
              f"floaters {s['floater_fraction']*100:.2f}%  unsupported {s['unsupported_surface_fraction']*100:.2f}%")

    if "render" in stages:
        n = cfg["eval"]["render_views"]
        view_ids = [int(v) for v in np.array(holdout_ids)[np.linspace(0, len(holdout_ids) - 1, n).round().astype(int)]]
        bounds = (np.array(metrics["fusion"]["bounds_lo"]), np.array(metrics["fusion"]["bounds_hi"]))
        metrics["renders"] = render.render_sheet(mesh, capture, view_ids, out_dir / "renders", bounds,
                                                 cfg["fusion"]["depth_max"], cfg["eval"]["topdown_pixels"])
        save_metrics(metrics_path, metrics, "render")
        print(f"render: {len(view_ids)} views + top-down")

    if "report" in stages:
        path = report.write_report(ROOT / cfg["out_root"] / capture.name)
        print(f"report: {path}")


if __name__ == "__main__":
    main()
