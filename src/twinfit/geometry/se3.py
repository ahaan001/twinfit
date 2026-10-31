"""SO(3) and SE(3) helpers. Poses are 4x4 matrices T_A_B (see contracts/conventions.py)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def inverse(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, p)
    out[..., 3, 3] = 1.0
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    out = Ts[0]
    for T in Ts[1:]:
        out = out @ T
    return out


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: R = I + sin(θ)/θ [w]_x + (1 - cos θ)/θ² [w]_x²."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = np.linalg.norm(w)
    W = skew(w)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + a * W + b * W @ W


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R, with |w| in [0, π]. The θ = π case uses the symmetric part."""
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-6:
        # Near π the antisymmetric part r = 2 sin θ a is tiny; take the axis from the symmetric
        # part R + I = (1 + cos θ) I + (1 - cos θ) a aᵀ and its sign from r.
        S = R + np.eye(3)
        axis = S[:, int(np.argmax(np.diag(S)))]
        axis = axis / np.linalg.norm(axis)
        if np.dot(axis, r) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * r


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    """V(w) with p = V(w) v for the SE(3) exponential (Barfoot, State Estimation, eq. 7.33)."""
    theta = np.linalg.norm(w)
    W = skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * W + W @ W / 6.0
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * W + b * W @ W


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential of a twist xi = (w, v): rotation vector first, then the translational part."""
    xi = np.asarray(xi, dtype=float).reshape(6)
    w, v = xi[:3], xi[3:]
    return make_T(so3_exp(w), _left_jacobian(w) @ v)


def se3_log(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    w = so3_log(T[:3, :3])
    v = np.linalg.solve(_left_jacobian(w), T[:3, 3])
    return np.concatenate([w, v])


def chordal_mean_rotation(Rs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Chordal L2 rotation average: the projection of the weighted mean matrix onto SO(3) by SVD
    (Hartley, Trumpf, Dai, Li, "Rotation averaging", IJCV 2013, section 5.3)."""
    Rs = np.asarray(Rs, dtype=float).reshape(-1, 3, 3)
    w = np.ones(len(Rs)) if weights is None else np.asarray(weights, dtype=float)
    M = np.einsum("n,nij->ij", w / w.sum(), Rs)
    return project_to_so3(M)


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of R, with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(np.asarray(R_a).T @ np.asarray(R_b)))))
