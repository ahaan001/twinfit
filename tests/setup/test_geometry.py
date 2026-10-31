import numpy as np
import pytest

from twinfit.geometry import camera, lines, se3

rng = np.random.default_rng(0)


@pytest.mark.parametrize("scale", [1e-6, 0.3, 2.0, np.pi - 1e-7])
def test_so3_exp_log_round_trip(scale):
    for _ in range(20):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * scale
        R = se3.so3_exp(w)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(se3.so3_log(R), w, atol=1e-6)


def test_so3_log_at_pi():
    R = np.diag([1.0, -1.0, -1.0])
    w = se3.so3_log(R)
    assert np.isclose(np.linalg.norm(w), np.pi)
    np.testing.assert_allclose(se3.so3_exp(w), R, atol=1e-12)


def test_se3_exp_log_round_trip():
    for _ in range(50):
        xi = rng.normal(size=6)
        xi[:3] *= 0.9 * np.pi / np.linalg.norm(xi[:3]) * rng.uniform(0.0, 1.0)
        T = se3.se3_exp(xi)
        np.testing.assert_allclose(se3.se3_log(T), xi, atol=1e-9)
        np.testing.assert_allclose(se3.inverse(T) @ T, np.eye(4), atol=1e-12)


def test_compose_and_batched_inverse():
    Ts = np.stack([se3.se3_exp(rng.normal(size=6)) for _ in range(5)])
    inv = se3.inverse(Ts)
    for T, Ti in zip(Ts, inv, strict=True):
        np.testing.assert_allclose(se3.compose(T, Ti), np.eye(4), atol=1e-12)


def test_chordal_mean_recovers_centre():
    """Symmetric perturbation pairs R0 exp(+w), R0 exp(-w) average exactly to R0."""
    R0 = se3.so3_exp([0.3, -0.2, 0.5])
    Rs = []
    for _ in range(50):
        w = rng.normal(scale=0.2, size=3)
        Rs.append(R0 @ se3.so3_exp(w))
        Rs.append(R0 @ se3.so3_exp(-w))
    R_mean = se3.chordal_mean_rotation(np.stack(Rs))
    assert se3.rotation_angle_deg(R_mean, R0) < 1e-6
    noisy = np.stack([R0 @ se3.so3_exp(rng.normal(scale=0.02, size=3)) for _ in range(200)])
    assert se3.rotation_angle_deg(se3.chordal_mean_rotation(noisy), R0) < 0.5


def test_quaternion_conversion_scalar_first():
    R = se3.so3_exp([0.0, 0.0, np.pi / 2])
    q = se3.quat_from_matrix(R)
    np.testing.assert_allclose(q, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-12)
    for _ in range(50):
        R = se3.so3_exp(rng.normal(size=3))
        np.testing.assert_allclose(se3.matrix_from_quat(se3.quat_from_matrix(R)), R, atol=1e-12)


def test_skew():
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(se3.skew(a) @ b, np.cross(a, b))


def test_projection_hand_computed():
    """An upright head-on view of the board: the OpenCV camera axes coincide with W (x right,
    y down the page, z into the page), so R_W_C = I and the camera sits at z = -1 m. A point
    0.1 m right and 0.05 m down the page lands at the principal point plus f·(0.1, 0.05)."""
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    T_W_C = se3.make_T(np.eye(3), [0.0, 0.0, -1.0])
    assert camera.points_in_front([[0.1, 0.05, 0.0]], T_W_C).all()
    uv = camera.project_points([[0.1, 0.05, 0.0]], K, np.zeros(5), T_W_C)[0]
    np.testing.assert_allclose(uv, [400.0, 280.0], atol=1e-9)


def test_projection_matches_pinhole_formula_with_distortion():
    K = np.array([[700.0, 0.0, 330.0], [0.0, 710.0, 250.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.1, -0.05, 0.001, -0.002, 0.01])
    T_W_C = se3.se3_exp([0.1, 0.2, -0.1, 0.3, 0.1, -1.2])
    X = rng.normal(size=(20, 3)) * 0.1
    uv = camera.project_points(X, K, dist, T_W_C)
    T_C_W = se3.inverse(T_W_C)
    Xc = X @ T_C_W[:3, :3].T + T_C_W[:3, 3]
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = dist
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    expected = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)
    np.testing.assert_allclose(uv, expected, atol=1e-9)


def test_homography_against_direct_projection():
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    T_W_C = se3.make_T(se3.so3_exp([0.1, -0.15, 0.05]), [0.15, 0.1, -0.9])
    d = 0.012
    xy = rng.uniform(-0.1, 0.3, size=(30, 2))
    X_W = np.hstack([xy, np.full((30, 1), -d)])  # plane at offset d toward the camera is z = -d
    assert camera.points_in_front(X_W, T_W_C).all()
    direct = camera.project_points(X_W, K, np.zeros(5), T_W_C)
    H = camera.plane_homography(K, T_W_C, d)
    h = (H @ np.hstack([xy, np.ones((30, 1))]).T).T
    np.testing.assert_allclose(h[:, :2] / h[:, 2:3], direct, atol=1e-9)
    back = camera.pixels_to_plane(direct, K, np.zeros(5), T_W_C, d)
    np.testing.assert_allclose(back, X_W, atol=1e-9)


def test_back_projection_with_distortion_round_trip():
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.05, -0.01, 0.0005, -0.0003, 0.0])
    T_W_C = se3.make_T(se3.so3_exp([0.05, 0.1, 0.0]), [0.1, 0.1, -0.8])
    xy = rng.uniform(-0.05, 0.25, size=(20, 2))
    uv = camera.plane_to_pixels(xy, K, dist, T_W_C, -0.02)
    back = camera.pixels_to_plane(uv, K, dist, T_W_C, -0.02)
    np.testing.assert_allclose(back[:, :2], xy, atol=1e-7)
    np.testing.assert_allclose(back[:, 2], 0.02, atol=1e-7)


def test_K_from_fovy():
    K = camera.K_from_fovy(np.radians(90.0), 640, 480)
    np.testing.assert_allclose(K, [[240.0, 0, 320.0], [0, 240.0, 240.0], [0, 0, 1]], atol=1e-9)


def test_line_distance_skew_and_parallel():
    assert np.isclose(lines.line_distance([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]), 1.0)
    assert np.isclose(lines.line_distance([0, 0, 0], [1, 0, 0], [5, 3, 4], [2, 0, 0]), 5.0)
    assert np.isclose(lines.line_distance([0, 0, 0], [1, 1, 0], [1, 1, 0], [-1, -1, 0]), 0.0)


def test_closest_point_to_origin():
    p = lines.closest_point_to_origin([2.0, 1.0, 0.0], [0.0, 0.0, 3.0])
    np.testing.assert_allclose(p, [2.0, 1.0, 0.0])
    p = lines.closest_point_to_origin([1.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(p, [0.0, 0.0, 0.0], atol=1e-12)
