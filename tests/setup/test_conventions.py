"""The frame conventions, each backed by a test that fails if a sign flips or axes swap."""

import cv2
import numpy as np

from twinfit.contracts import NORMAL_TOWARD_CAMERA_W, UP_W, charuco_board, gravity_for_orientation
from twinfit.contracts.conventions import R_CV_FROM_MJ, board_axes_in_image


def test_board_image_axes_match_detection_at_full_size():
    """Independent computation: detect corners in a large zero-margin image and compare the
    mapping with the import-time result at a different resolution."""
    small = board_axes_in_image()
    board = charuco_board()
    pps = 140
    img = board.generateImage((7 * pps, 5 * pps), marginSize=0, borderBits=1)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    xy_true = board.getChessboardCorners()[ids.ravel(), :2]
    # Map detected pixels through the small-image map scaled to this resolution.
    scale = small.image_shape[1] / (7 * pps)
    xy_mapped = small.pixel_to_w(corners.reshape(-1, 2) * scale + 0.5 * scale - 0.5)
    assert np.abs(xy_mapped - xy_true).max() < 1e-4


def test_printed_axes_x_right_y_down():
    m = board_axes_in_image()
    A = np.asarray(m.A)
    assert A[0, 0] > 0 and abs(A[0, 1]) < 1e-9, "+x must run to the right of the printed page"
    assert A[1, 1] > 0 and abs(A[1, 0]) < 1e-9, "+y must run down the printed page"
    assert abs(A[0, 0] - A[1, 1]) < 1e-9, "isotropic pixel scale"
    assert abs(A[0, 0] - 0.035 / 100) < 1e-9


def test_up_and_normal_constants():
    np.testing.assert_allclose(UP_W, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(NORMAL_TOWARD_CAMERA_W, [0.0, 0.0, -1.0])
    # Right-handed: with x right and y down on the page, z goes into the page (OpenCV style).
    x_dir = np.array([1.0, 0.0, 0.0])
    y_dir = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(np.cross(x_dir, y_dir), -NORMAL_TOWARD_CAMERA_W)


def test_gravity_from_orientation():
    np.testing.assert_allclose(gravity_for_orientation("vertical_arrow_up"), [0.0, 9.81, 0.0])
    np.testing.assert_allclose(gravity_for_orientation("horizontal_face_up"), [0.0, 0.0, 9.81])
    assert np.dot(gravity_for_orientation("vertical_arrow_up"), UP_W) < 0
    assert np.dot(gravity_for_orientation("horizontal_face_up"), NORMAL_TOWARD_CAMERA_W) < 0


def test_opencv_from_mujoco_camera():
    """A MuJoCo camera looks along -z with y up; the OpenCV camera looks along +z with y down."""
    R_mj = np.eye(3)
    R_cv = R_mj @ R_CV_FROM_MJ
    assert np.allclose(R_cv[:, 2], -R_mj[:, 2])
    assert np.allclose(R_cv[:, 1], -R_mj[:, 1])
    assert np.allclose(R_cv[:, 0], R_mj[:, 0])
    assert np.isclose(np.linalg.det(R_cv), 1.0)
