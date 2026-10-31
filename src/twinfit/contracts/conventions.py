"""Frames, units and ids shared by every Twinfit contract.

Conventions:
- Units are SI: m, kg, s, rad, N, N·m. Reports may show degrees and mm; files never do.
- `T_A_B` is the 4×4 homogeneous pose of frame B expressed in frame A, so p_A = T_A_B p_B.
  Same for R_A_B and p_A_B.
- Rotations are stored as matrices. Quaternions appear only at the MuJoCo boundary,
  scalar-first (w, x, y, z).
- Camera frames follow OpenCV: x right, y down, z forward along the optical axis. MuJoCo
  cameras look along -z with y up. Convert with R_cv = R_mj · diag(1, -1, -1).
- The world frame W is the ChArUco board frame, exactly as defined by the coordinates from the
  installed OpenCV's `CharucoBoard.getChessboardCorners()`, with z = 0 on the printed face.
  - Determine empirically, and document, which printed direction is +x and which is +y, and
    whether +z points out of the printed face toward the viewer or into it.
  - Derive `up_W`: the unit vector pointing up when the board is mounted vertically with the
    printed UP arrow on top.
  - Derive `normal_toward_camera_W`: the unit normal pointing from the printed face toward the
    viewer.
  - Both are constants computed at import from the board definition, with a test.
- Gravity: `gravity_W = -9.81 · up_W` for `vertical_arrow_up`, and
  `gravity_W = -9.81 · normal_toward_camera_W` for `horizontal_face_up`.
- `plane_offset_m` is the signed distance from the printed face to the plane containing the
  moving parts' camera-facing surfaces. It is positive toward the camera, along
  `normal_toward_camera_W`.
- Part ids match `p[0-9]+` and joint ids match `j[0-9]+`. The id `"world"` is reserved for W
  treated as a fixed link, and never appears in tracks.
- A part frame P is any frame rigidly attached to the part. Consumers must not assume its origin
  or orientation.
- Time is in seconds from the first decoded frame, taken from container timestamps. Variable
  frame rate is tolerated.
- The joint coordinate q is 0 at the reference frame index. Revolute q follows the right-hand
  rule about `axis_W`; prismatic q is the displacement along `axis_W`.

Empirical result for the installed OpenCV (see `board_axes_in_image`): printed +x runs to the
right and +y runs down the page when the board is viewed with the UP arrow on top, so +z points
into the printed face, away from the viewer. Hence up_W = (0, -1, 0) and
normal_toward_camera_W = (0, 0, -1). The test in tests/setup/test_conventions.py fails if either
sign flips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

GRAVITY_MPS2 = 9.81

CHARUCO_DICTIONARY = "DICT_5X5_100"
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.035
CHARUCO_MARKER_LENGTH_M = 0.026

PART_MARKER_DICTIONARY = "DICT_4X4_50"
PART_MARKER_IDS = tuple(range(12))
PART_MARKER_LENGTH_M = 0.030

PART_ID_RE = re.compile(r"^p[0-9]+$")
JOINT_ID_RE = re.compile(r"^j[0-9]+$")
WORLD_ID = "world"

# R_cv = R_mj · R_CV_FROM_MJ: OpenCV camera axes expressed in MuJoCo camera axes.
R_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])


def aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown aruco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def charuco_board(
    dictionary: str = CHARUCO_DICTIONARY,
    squares_x: int = CHARUCO_SQUARES_X,
    squares_y: int = CHARUCO_SQUARES_Y,
    square_length_m: float = CHARUCO_SQUARE_LENGTH_M,
    marker_length_m: float = CHARUCO_MARKER_LENGTH_M,
) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length_m, marker_length_m, aruco_dictionary(dictionary)
    )


def board_size_m(
    squares_x: int = CHARUCO_SQUARES_X,
    squares_y: int = CHARUCO_SQUARES_Y,
    square_length_m: float = CHARUCO_SQUARE_LENGTH_M,
) -> tuple[float, float]:
    return squares_x * square_length_m, squares_y * square_length_m


@dataclass(frozen=True)
class BoardImageMap:
    """Affine map from pixel coordinates of a zero-margin board image to W (x, y) in metres.

    xy_W = A @ (u, v) + b. `px_per_m` is the uniform scale and `image_shape` is (height, width).
    """

    A: tuple[tuple[float, float], tuple[float, float]]
    b: tuple[float, float]
    image_shape: tuple[int, int]

    def pixel_to_w(self, uv: np.ndarray) -> np.ndarray:
        return np.asarray(uv, float) @ np.asarray(self.A).T + np.asarray(self.b)

    def w_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        A_inv = np.linalg.inv(np.asarray(self.A))
        return (np.asarray(xy, float) - np.asarray(self.b)) @ A_inv.T


@lru_cache(maxsize=4)
def board_axes_in_image(px_per_square: int = 100) -> BoardImageMap:
    """Fit the pixel-to-W affine map by detecting corners in the generated board image.

    This is the empirical step that pins down the printed directions of +x and +y. The image
    is generated with zero margin so its edges coincide with the physical board edges. OpenCV 5.0
    rejects some sizes (60 and 200 px per square fail an internal ROI assertion); 100 works.
    """
    board = charuco_board()
    w_px = CHARUCO_SQUARES_X * px_per_square
    h_px = CHARUCO_SQUARES_Y * px_per_square
    img = board.generateImage((w_px, h_px), marginSize=0, borderBits=1)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
    if corners is None or len(corners) < 6:
        raise RuntimeError("could not detect the generated ChArUco board")
    uv = corners.reshape(-1, 2).astype(float)
    xy = board.getChessboardCorners()[ids.ravel(), :2].astype(float)
    design = np.hstack([uv, np.ones((len(uv), 1))])
    coef, _, _, _ = np.linalg.lstsq(design, xy, rcond=None)
    A = coef[:2].T
    b = coef[2]
    residual = np.abs(design @ coef - xy).max()
    if residual > 1e-6:
        raise RuntimeError(f"board image map is not affine (residual {residual:.2e} m)")
    return BoardImageMap(
        A=((float(A[0, 0]), float(A[0, 1])), (float(A[1, 0]), float(A[1, 1]))),
        b=(float(b[0]), float(b[1])),
        image_shape=(h_px, w_px),
    )


def _derive_frame_constants() -> tuple[np.ndarray, np.ndarray]:
    m = board_axes_in_image()
    A = np.asarray(m.A)
    # W-directions of the pixel axes: u (right on the printed page) and v (down the page).
    x_dir_u = np.append(A @ np.array([1.0, 0.0]), 0.0)
    y_dir_v = np.append(A @ np.array([0.0, 1.0]), 0.0)
    # The UP arrow is printed above the image's top edge, so "up" is the -v pixel direction.
    up = -y_dir_v / np.linalg.norm(y_dir_v)
    # A viewer facing the printed page sees (right, down, into-page) as a right-handed triad, so
    # right × down points into the page and the normal toward the viewer is its negative.
    into_page = np.cross(x_dir_u, y_dir_v)
    normal = -into_page / np.linalg.norm(into_page)
    up = np.round(up, 12) + 0.0
    normal = np.round(normal, 12) + 0.0
    return up, normal


UP_W, NORMAL_TOWARD_CAMERA_W = _derive_frame_constants()
UP_W.setflags(write=False)
NORMAL_TOWARD_CAMERA_W.setflags(write=False)

BoardOrientation = str


def gravity_for_orientation(orientation: str) -> np.ndarray:
    if orientation == "vertical_arrow_up":
        return -GRAVITY_MPS2 * UP_W
    if orientation == "horizontal_face_up":
        return -GRAVITY_MPS2 * NORMAL_TOWARD_CAMERA_W
    raise ValueError(f"unknown board orientation {orientation!r}")
