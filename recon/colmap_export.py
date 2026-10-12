"""Write a capture as a COLMAP text model so Gaussian splatting trainers
(MILo, 2DGS, vanilla 3DGS) can consume the ARKit poses without running SfM.

Layout written under `root`:

    images/NNNNNN.jpg          colour frames
    sparse/0/cameras.txt       one PINHOLE camera per image (intrinsics vary per frame)
    sparse/0/images.txt        world-to-camera rotation (quaternion wxyz) and translation
    sparse/0/points3D.txt      coloured LiDAR points, the splat initialisation

COLMAP's camera frame matches OpenCV's, which is what the capture poses
already use, so the only conversion is camera-to-world -> world-to-camera.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .capture import Capture, unproject, transform_points


def export_colmap(capture: Capture, frame_ids, root: Path, depth_max: float,
                  point_voxel: float = 0.01, image_scale: float = 1.0) -> dict:
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    cam_lines = ["# Camera list with one line of data per camera:", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    img_lines = ["# Image list with two lines of data per image:",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "#   POINTS2D[] as (X, Y, POINT3D_ID)"]
    points = []
    colours = []
    width = height = None

    for i, (fid, rgb) in enumerate(capture.frames(frame_ids), start=1):
        if image_scale != 1.0:
            rgb = cv2.resize(rgb, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
        height, width = rgb.shape[:2]
        name = f"{fid:06d}.jpg"
        cv2.imwrite(str(root / "images" / name), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

        K = capture.intrinsics_color(fid) * np.array([[image_scale], [image_scale], [1.0]])
        cam_lines.append(f"{i} PINHOLE {width} {height} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}")

        T_CW = np.linalg.inv(capture.pose(fid))
        q = Rotation.from_matrix(T_CW[:3, :3]).as_quat()   # xyzw
        t = T_CW[:3, 3]
        img_lines.append(f"{i} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {i} {name}")
        img_lines.append("")

        # Coloured LiDAR points for the initial Gaussians.
        depth = capture.depth(fid)
        valid = (capture.confidence(fid) == 2) & (depth > 0) & (depth < depth_max)
        Kd = capture.intrinsics_depth(fid)
        pts_cam = unproject(depth, Kd)[valid]
        small = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        points.append(transform_points(capture.pose(fid), pts_cam))
        colours.append(small[valid])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colours).astype(np.float64) / 255.0)
    pcd = pcd.voxel_down_sample(point_voxel)
    xyz = np.asarray(pcd.points)
    rgb = (np.asarray(pcd.colors) * 255).round().astype(int)
    pt_lines = ["# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"]
    for j, (p, c) in enumerate(zip(xyz, rgb), start=1):
        pt_lines.append(f"{j} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0.0")

    (sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (sparse / "images.txt").write_text("\n".join(img_lines) + "\n")
    (sparse / "points3D.txt").write_text("\n".join(pt_lines) + "\n")
    write_points3d_binary(sparse / "points3D.bin", xyz, rgb)
    o3d.io.write_point_cloud(str(sparse / "points3D.ply"), pcd)
    return {"images": len(cam_lines) - 2, "points": int(len(xyz)), "width": width, "height": height}


def write_points3d_binary(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """COLMAP points3D.bin with empty tracks. The 3DGS family of loaders reads
    the binary file unconditionally (cameras and images fall back to text)."""
    import struct

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for j, (p, c) in enumerate(zip(xyz, rgb), start=1):
            f.write(struct.pack("<Q3d3BdQ", j, float(p[0]), float(p[1]), float(p[2]),
                                int(c[0]), int(c[1]), int(c[2]), 0.0, 0))
