"""Pinhole camera with OpenCV distortion, and the board-parallel plane homography.

Camera frames follow OpenCV (x right, y down, z forward). T_W_C is the camera pose in W.
"""

from __future__ import annotations

import cv2
import numpy as np

from twinfit.contracts.conventions import NORMAL_TOWARD_CAMERA_W, R_CV_FROM_MJ
from twinfit.geometry.se3 import inverse, make_T


def K_from_fovy(fovy_rad: float, width: int, height: int) -> np.ndarray:
    """MuJoCo-style intrinsics: fy = (H/2)/tan(fovy/2), fx = fy, principal point at the centre."""
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def T_W_C_from_mujoco(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """OpenCV camera pose from MuJoCo's cam_xpos and cam_xmat (R_cv = R_mj · diag(1, -1, -1))."""
    R_mj = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    return make_T(R_mj @ R_CV_FROM_MJ, cam_xpos)


def project_points(
    X_W: np.ndarray, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray
) -> np.ndarray:
    """Pixel coordinates (N, 2) of world points (N, 3), through cv2.projectPoints."""
    X_W = np.asarray(X_W, dtype=float).reshape(-1, 3)
    T_C_W = inverse(np.asarray(T_W_C, dtype=float))
    rvec, _ = cv2.Rodrigues(T_C_W[:3, :3])
    uv, _ = cv2.projectPoints(
        X_W, rvec, T_C_W[:3, 3], np.asarray(K, float), np.asarray(dist, float)
    )
    return uv.reshape(-1, 2)


def points_in_front(X_W: np.ndarray, T_W_C: np.ndarray) -> np.ndarray:
    """Boolean mask of points with positive depth in the camera frame."""
    T_C_W = inverse(np.asarray(T_W_C, dtype=float))
    Xc = np.asarray(X_W, float).reshape(-1, 3) @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    return Xc[:, 2] > 0


def plane_frame(d: float) -> np.ndarray:
    """T_W_Q of the plane frame Q: W axes, origin shifted by d along normal_toward_camera_W.

    Points on the plane are X_W = T_W_Q (x, y, 0, 1)."""
    return make_T(np.eye(3), d * NORMAL_TOWARD_CAMERA_W)


def plane_homography(K: np.ndarray, T_W_C: np.ndarray, d: float) -> np.ndarray:
    """H with undistorted pixel ~ H (x, y, 1) for plane coordinates (x, y) in the frame of
    `plane_frame(d)`. H = K [r1 r2 t] of T_C_Q."""
    T_C_Q = inverse(np.asarray(T_W_C, float)) @ plane_frame(d)
    return np.asarray(K, float) @ T_C_Q[:3, [0, 1, 3]]


def undistort_pixels(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistorted pixel coordinates (same K), (N, 2)."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 1, 2)
    out = cv2.undistortPoints(
        uv, np.asarray(K, float), np.asarray(dist, float), P=np.asarray(K, float)
    )
    return out.reshape(-1, 2)


def pixels_to_plane(
    uv: np.ndarray, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray, d: float
) -> np.ndarray:
    """Back-project pixels onto the plane at offset d. Returns world points (N, 3)."""
    uv_u = undistort_pixels(uv, K, dist)
    H = plane_homography(K, T_W_C, d)
    xy1 = np.linalg.solve(H, np.hstack([uv_u, np.ones((len(uv_u), 1))]).T).T
    xy = xy1[:, :2] / xy1[:, 2:3]
    T_W_Q = plane_frame(d)
    return np.hstack([xy, np.zeros((len(xy), 1))]) @ T_W_Q[:3, :3].T + T_W_Q[:3, 3]


def plane_to_pixels(
    xy: np.ndarray, K: np.ndarray, dist: np.ndarray, T_W_C: np.ndarray, d: float
) -> np.ndarray:
    """Project plane coordinates (N, 2) to distorted pixels, through the full camera model."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    T_W_Q = plane_frame(d)
    X_W = np.hstack([xy, np.zeros((len(xy), 1))]) @ T_W_Q[:3, :3].T + T_W_Q[:3, 3]
    return project_points(X_W, K, dist, T_W_C)
