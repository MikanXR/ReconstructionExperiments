"""MILo: Mesh-In-the-Loop Gaussian splatting (https://github.com/Anttwo/MILo).

MILo produces a mesh directly, so this module does not fill a depth cache.
Instead the `depth` stage exports the training frames as a COLMAP model and
the `fuse` stage imports the mesh that MILo wrote. Training runs outside the
harness in its own environment:

    cd third_party\\MILo\\milo
    ..\\..\\..\\.venv_milo\\Scripts\\python train.py -s <repo>\\out\\<capture>\\milo\\colmap -m <repo>\\out\\<capture>\\milo\\train --imp_metric indoor --rasterizer radegs
    ..\\..\\..\\.venv_milo\\Scripts\\python mesh_extract_sdf.py -s <repo>\\out\\<capture>\\milo\\colmap -m <repo>\\out\\<capture>\\milo\\train
    copy <repo>\\out\\<capture>\\milo\\train\\mesh_learnable_sdf.ply <repo>\\out\\<capture>\\milo\\mesh_input.ply

The scripts use relative imports and a relative default config path, so they
run from `third_party/MILo/milo`. Building the environment on Windows needs
CUDA Toolkit 12.8 (torch cu128), the VS 2022 toolset, and CGAL with GMP and
MPFR from vcpkg; `.venv_milo/build_ext.cmd` does all of it. MILo's reader
loads `sparse/0/points3D.bin` unconditionally, which is why the exporter
writes the binary file alongside the text one.

If MILo does not build on this machine, 2DGS (https://github.com/hbb1/2d-gaussian-splatting)
reads the same COLMAP export and its render.py writes a TSDF mesh; copy that
to mesh_input.ply instead and set the `--tag 2dgs` run name.

License: MILo and 2DGS inherit the Gaussian Splatting license (Inria/MPII),
research use only. This entry is a quality ceiling, not a shipping candidate.
"""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

from recon.capture import Capture
from recon.colmap_export import export_colmap

NAME = "milo"
LICENSE = "Gaussian Splatting license, non-commercial"
PRODUCES_MESH = True


def export(capture: Capture, frame_ids, out_dir: Path, cfg: dict) -> dict:
    """Stage `depth`: write the COLMAP model under out_dir/colmap."""
    root = Path(out_dir) / "colmap"
    return export_colmap(capture, frame_ids, root, depth_max=cfg["fusion"]["depth_max"])


def produce_mesh(capture: Capture, frame_ids, out_dir: Path, cfg: dict) -> o3d.geometry.TriangleMesh:
    """Stage `fuse`: load the externally trained mesh from out_dir/mesh_input.ply."""
    path = Path(out_dir) / "mesh_input.ply"
    if not path.exists():
        raise SystemExit(f"{path} not found. Train MILo on {Path(out_dir) / 'colmap'} and copy its mesh there (see methods/milo.py).")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.triangles):
        raise SystemExit(f"{path} holds no triangles")
    mesh.compute_vertex_normals()
    return mesh
