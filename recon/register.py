"""Frame-to-model registration: projective point-to-plane ICP of one LiDAR
frame against a depth and normal map of the model so far.

The model map is a ray cast of the TSDF from the frame's predicted pose, so
data association is a projection: each frame point, transformed by the
current estimate, lands on one model pixel. Correspondences beyond a distance
or a normal angle are dropped, the gate tightens as the iterations go, and
the six-parameter update is solved in closed form each round. A solve that
moves the pose further than a bound, or that has too few correspondences,
is rejected and the prior stands: the ARKit pose is already close, and a
confident wrong answer is worse than no correction.

A frame that sees one wall or the floor alone constrains nothing along that
plane, and point-to-plane ICP then slides freely. A prior term on the
deviation from the ARKit pose (prior_sigma_m, prior_sigma_deg) holds those
directions where the data does not, and costs nothing where it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import unproject


@dataclass
class RegistrationResult:
    T_WC: np.ndarray           # the refined camera-to-world pose, or the prior when rejected
    accepted: bool
    iterations: int
    correspondences: int
    rms_before_m: float
    rms_after_m: float
    translation_m: float       # size of the correction against the prior
    rotation_deg: float


def frame_normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel normals of a camera-space point map from neighbouring
    differences, pointing toward the camera. Pixels whose neighbours are
    missing or lie across a depth jump are marked invalid."""
    h, w = valid.shape
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]
    ok = np.zeros_like(valid)
    ok[1:-1, 1:-1] = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    # A neighbour step longer than a tenth of the depth is a silhouette, not a surface.
    depth = np.maximum(points[..., 2], 1e-6)
    ok &= (np.linalg.norm(dx, axis=-1) < 0.1 * depth) & (np.linalg.norm(dy, axis=-1) < 0.1 * depth)
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    ok &= length[..., 0] > 1e-12
    n = n / np.maximum(length, 1e-12)
    # Toward the camera: the camera looks down +Z, so a normal facing it has negative Z.
    flip = n[..., 2] > 0
    n[flip] *= -1
    return n, ok


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _exp_se3(xi: np.ndarray) -> np.ndarray:
    """Small-motion exponential: xi = (omega, t) to a 4x4 transform."""
    omega, t = xi[:3], xi[3:]
    angle = np.linalg.norm(omega)
    K = _skew(omega)
    if angle < 1e-9:
        R = np.eye(3) + K
        V = np.eye(3)
    else:
        R = np.eye(3) + np.sin(angle) / angle * K + (1 - np.cos(angle)) / angle ** 2 * K @ K
        V = np.eye(3) + (1 - np.cos(angle)) / angle ** 2 * K + (angle - np.sin(angle)) / angle ** 3 * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ t
    return T


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a small rotation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def register_frame(depth: np.ndarray, weight: np.ndarray, K: np.ndarray, T_prior: np.ndarray,
                   model_points: np.ndarray, model_normals: np.ndarray, model_hit: np.ndarray,
                   T_model: np.ndarray, cfg: dict) -> RegistrationResult:
    """Refine T_prior (camera-to-world) so the frame's points lie on the
    model. The model map was ray cast from T_model, which is normally the
    prior itself; it is passed separately so the association stays right if
    a caller casts once and refines twice."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > cfg["depth_min"]) & (depth < cfg["depth_max"]) & (weight > 0)
    points_c = unproject(depth, K)
    normals_c, normal_ok = frame_normals(points_c, valid)
    use = valid & normal_ok
    if cfg.get("subsample", 1) > 1:
        s = int(cfg["subsample"])
        grid = np.zeros_like(use)
        grid[::s, ::s] = True
        use &= grid
    p_c = points_c[use]
    n_c = normals_c[use]

    T_model_inv = np.linalg.inv(T_model)
    Km = K
    T = T_prior.copy()
    max_dist = float(cfg["max_correspondence_m"])
    min_dist = max_dist / 3
    cos_limit = np.cos(np.radians(float(cfg["max_normal_angle_deg"])))
    iterations = int(cfg["iterations"])
    min_corr = int(cfg["min_correspondences"])
    # The prior enters the normal equations as a diagonal block on the
    # deviation from T_prior, so each round pulls toward the prior by
    # exactly as much as the data leaves unconstrained.
    sigma_r = np.radians(float(cfg["prior_sigma_deg"]))
    sigma_t = float(cfg["prior_sigma_m"])
    prior_weight = np.r_[np.full(3, 1 / sigma_r ** 2), np.full(3, 1 / sigma_t ** 2)]
    # The data term is weighted by its own noise so the prior reads in the
    # same units: thousands of points at data_sigma outweigh the prior along
    # any direction they constrain, and leave it in charge of the rest.
    data_weight = 1 / float(cfg["data_sigma_m"]) ** 2

    rms_before = float("nan")
    rms_after = float("nan")
    count = 0
    it = 0
    for it in range(1, iterations + 1):
        gate = max_dist - (max_dist - min_dist) * (it - 1) / max(iterations - 1, 1)
        R, t = T[:3, :3], T[:3, 3]
        p_w = p_c @ R.T + t
        n_w = n_c @ R.T
        # Associate by projecting into the model camera.
        q = p_w @ T_model_inv[:3, :3].T + T_model_inv[:3, 3]
        z = q[:, 2]
        front = z > 1e-3
        u = np.round(q[:, 0] / np.where(front, z, 1) * Km[0, 0] + Km[0, 2]).astype(int)
        v = np.round(q[:, 1] / np.where(front, z, 1) * Km[1, 1] + Km[1, 2]).astype(int)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = np.clip(u, 0, w - 1)
        v = np.clip(v, 0, h - 1)
        m = model_points[v, u]
        mn = model_normals[v, u]
        ok = inside & model_hit[v, u]
        diff = p_w - m
        dist = np.linalg.norm(diff, axis=1)
        ok &= dist < gate
        ok &= np.einsum("ij,ij->i", n_w, mn) > cos_limit
        count = int(ok.sum())
        if count < min_corr:
            break
        r = np.einsum("ij,ij->i", diff[ok], mn[ok])
        rms = float(np.sqrt(np.mean(r ** 2)))
        if it == 1:
            rms_before = rms
        rms_after = rms
        # Point-to-plane Jacobian for a left-multiplied twist on T:
        # d(n . (exp(xi) p - m)) / d xi = [ (p x n), n ].
        J = np.concatenate([np.cross(p_w[ok], mn[ok]), mn[ok]], axis=1)
        # Huber weighting keeps a few bad associations from steering the solve.
        huber = float(cfg["huber_m"])
        wgt = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-9)) * data_weight
        JtW = J.T * wgt
        A = JtW @ J + np.eye(6) * 1e-9
        b = -JtW @ r
        # Deviation of the current estimate from the prior as a twist in the
        # same left-multiplied parameterisation as the update.
        dev = T @ np.linalg.inv(T_prior)
        dev_xi = np.r_[_log_so3(dev[:3, :3]), dev[:3, 3]]
        A += np.diag(prior_weight)
        b -= prior_weight * dev_xi
        try:
            xi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        T = _exp_se3(xi) @ T
        if np.linalg.norm(xi[3:]) < 1e-5 and np.linalg.norm(xi[:3]) < 1e-5:
            break

    delta = T @ np.linalg.inv(T_prior)
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = rotation_angle_deg(delta[:3, :3])
    accepted = (count >= min_corr and np.isfinite(rms_after)
                and translation <= float(cfg["max_correction_m"])
                and rotation <= float(cfg["max_correction_deg"]))
    return RegistrationResult(
        T_WC=T if accepted else T_prior.copy(),
        accepted=accepted,
        iterations=it,
        correspondences=count,
        rms_before_m=rms_before,
        rms_after_m=rms_after,
        translation_m=translation,
        rotation_deg=rotation,
    )
