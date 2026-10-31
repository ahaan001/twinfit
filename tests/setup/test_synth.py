"""Synthetic suite checks: board frame, projection into masks, GT self-consistency, determinism.

These read the shared synth_small suite (generated on first use, which needs offscreen rendering),
so they carry the render marker."""

import cv2
import imageio
import numpy as np
import pytest

from twinfit.contracts import (
    charuco_board,
    load_dynamics,
    load_kinematics,
    load_metadata,
    load_tracks,
)
from twinfit.contracts.check import check_path
from twinfit.contracts.io import load_npz, load_pose_json
from twinfit.eval.metrics import motion_point_rms_m
from twinfit.geometry import se3
from twinfit.geometry.camera import project_points
from twinfit.sim import replay
from twinfit.synth import SCENE_NAMES, SUITES, generate_clip, suite_dir
from twinfit.synth.scenes import build_scene

pytestmark = pytest.mark.render

CLIPS = [f"{scene}__s{seed}" for scene in SCENE_NAMES for seed in SUITES["synth_small"].seeds]


@pytest.fixture(scope="module")
def small():
    return suite_dir("synth_small")


def test_suite_has_twelve_valid_clips(small):
    folders = sorted(p.name for p in small.iterdir() if p.is_dir())
    assert folders == sorted(CLIPS)
    for name in CLIPS:
        lines = check_path(small / name)
        bad = [(x.file, x.message) for x in lines if not x.ok]
        assert not bad, (name, bad)


def _frame(clip, index):
    reader = imageio.get_reader(str(clip / "video.mp4"))
    try:
        return reader.get_data(index)
    finally:
        reader.close()


@pytest.mark.parametrize("name", CLIPS)
def test_board_pose_recovered_from_render(small, name):
    """ChArUco detection + solvePnP with the GT intrinsics recovers the GT T_W_C within 0.5 deg
    and 5 mm, with reprojection RMS <= 0.5 px. The camera is static, so chessboard corners
    detected in a few frames (the moving part uncovers different corners) are pooled into one
    PnP problem."""
    clip = small / name
    meta = load_metadata(clip / "metadata.yaml")
    T_W_C = load_pose_json(clip / "gt" / "camera_extrinsics.json")
    tracks, _ = load_tracks(clip / "gt" / "tracks.json")
    K = np.asarray(tracks.camera.intrinsics.K)
    dist = np.asarray(tracks.camera.intrinsics.dist)
    board = charuco_board(
        meta.board.dictionary,
        meta.board.squares_x,
        meta.board.squares_y,
        meta.board.square_length_m,
        meta.board.marker_length_m,
    )
    detector = cv2.aruco.CharucoDetector(board)
    obj_all, img_all = [], []
    for k in np.linspace(0, tracks.n_frames - 1, 6).astype(int):
        gray = cv2.cvtColor(_frame(clip, int(k)), cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is None or len(corners) < 6:
            continue
        obj, img = board.matchImagePoints(corners, ids)
        obj_all.append(obj.reshape(-1, 3))
        img_all.append(img.reshape(-1, 2))
    assert obj_all, "board not detected in any sampled frame"
    obj = np.concatenate(obj_all)
    img = np.concatenate(img_all)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R_C_W, _ = cv2.Rodrigues(rvec)
    T_est = se3.inverse(se3.make_T(R_C_W, tvec.ravel()))
    proj = project_points(obj, K, dist, T_est)
    rms = np.sqrt(((proj - img) ** 2).sum(axis=1).mean())
    angle = se3.rotation_angle_deg(T_est[:3, :3], T_W_C[:3, :3])
    pos = np.linalg.norm(T_est[:3, 3] - T_W_C[:3, 3])
    assert rms <= 0.5, rms
    assert angle <= 0.5, angle
    assert pos <= 0.005, pos


def _ray_hits_box(origin, points, T_W_B, half, center):
    """For each point, whether the segment camera -> point crosses the box (slab test in the box
    frame). Points on the box's own surface are excluded by the caller."""
    R, p = T_W_B[:3, :3], T_W_B[:3, 3]
    o = R.T @ (origin - p) - center
    d = (points - p) @ R - center - o  # segment direction per point, box frame
    t0 = np.zeros(len(points))
    t1 = np.ones(len(points))
    for a in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d[:, a]
            ta = (-half[a] - o[a]) * inv
            tb = (half[a] - o[a]) * inv
        lo = np.where(
            np.isfinite(inv), np.minimum(ta, tb), np.where(abs(o[a]) <= half[a], -np.inf, np.inf)
        )
        hi = np.where(
            np.isfinite(inv), np.maximum(ta, tb), np.where(abs(o[a]) <= half[a], np.inf, -np.inf)
        )
        t0 = np.maximum(t0, lo)
        t1 = np.minimum(t1, hi)
    return t0 < t1 - 1e-9


@pytest.mark.parametrize("name", CLIPS)
def test_part_surface_points_fall_in_masks(small, name):
    """Points sampled on each part's camera-facing geom surface, projected with the GT camera,
    land inside that part's GT mask at least 95 percent of the time. Points whose line of sight
    is blocked by another part's box (ray-box test, an independent computation) are excluded."""
    clip = small / name
    tracks, arrays = load_tracks(clip / "gt" / "tracks.json")
    K = np.asarray(tracks.camera.intrinsics.K)
    T_W_C = np.asarray(tracks.camera.T_W_C)
    cam = T_W_C[:3, 3]
    spec = build_scene(name.split("__s")[0], int(name.split("__s")[1]))
    rng = np.random.default_rng(0)
    total = hit = 0
    for k in range(0, tracks.n_frames, max(1, tracks.n_frames // 12)):
        mask = cv2.imread(str(clip / "gt" / "masks" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
        poses = {p.part_id: arrays[f"T_W_P__{p.part_id}"][k] for p in spec.parts}
        for i, part in enumerate(spec.parts):
            lo, hi = np.asarray(part.extent_P)
            # Sample strictly inside the face so edge pixels do not count against the test.
            u = rng.uniform(0.1, 0.9, size=(60, 2))
            pts_P = np.column_stack(
                [
                    lo[0] + u[:, 0] * (hi[0] - lo[0]),
                    lo[1] + u[:, 1] * (hi[1] - lo[1]),
                    np.full(60, lo[2]),
                ]
            )
            T = poses[part.part_id]
            pts_W = pts_P @ T[:3, :3].T + T[:3, 3]
            visible = np.ones(len(pts_W), bool)
            for other in spec.parts:
                if other is part:
                    continue
                visible &= ~_ray_hits_box(
                    cam, pts_W, poses[other.part_id], other.half, np.asarray(other.geom_pos)
                )
            pts_W = pts_W[visible]
            uv = np.round(project_points(pts_W, K, np.zeros(5), T_W_C)).astype(int)
            inside = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < mask.shape[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < mask.shape[0])
            )
            labels = mask[uv[inside, 1], uv[inside, 0]]
            total += len(uv)
            hit += int((labels == i + 1).sum())
    assert hit / total >= 0.95, hit / total


@pytest.mark.parametrize("name", CLIPS)
def test_forward_kinematics_reproduces_tracks(small, name):
    """FK from the GT kinematics (axes, points, q) rebuilds every part's motion to < 1e-6 m."""
    clip = small / name
    tracks, arrays = load_tracks(clip / "gt" / "tracks.json")
    kin, karr = load_kinematics(clip / "gt" / "kinematics.json")
    ref = kin.reference_frame_index
    n = tracks.n_frames
    poses = {kin.base_part_id: arrays[f"T_W_P__{kin.base_part_id}"]}
    joints = {j.child_part_id: j for j in kin.joints}
    pending = [link.part_id for link in kin.links if link.part_id not in poses]
    while pending:
        pid = pending.pop(0)
        j = joints[pid]
        if j.parent_part_id not in poses:
            pending.append(pid)
            continue
        T_parent = poses[j.parent_part_id]
        T_child_ref = arrays[f"T_W_P__{pid}"][ref]
        T_parent_ref = T_parent[ref]
        q = karr[f"q__{j.joint_id}"] - karr[f"q__{j.joint_id}"][ref]
        axis_p = np.asarray(j.axis_parent)
        point_p = np.zeros(3) if j.point_parent is None else np.asarray(j.point_parent)
        T_fk = np.zeros((n, 4, 4))
        rel_ref = se3.inverse(T_parent_ref) @ T_child_ref  # child in parent at the reference
        for k in range(n):
            if j.type == "revolute":
                R = se3.so3_exp(axis_p * q[k])
                motion = se3.make_T(R, point_p - R @ point_p)
            else:
                motion = se3.make_T(np.eye(3), axis_p * q[k])
            T_fk[k] = T_parent[k] @ motion @ rel_ref
        poses[pid] = T_fk
        rms = motion_point_rms_m(
            T_fk, arrays[f"T_W_P__{pid}"], arrays[f"valid__{pid}"], ref_index=ref
        )
        assert rms < 1e-6, (pid, rms)


@pytest.mark.parametrize("name", [c for c in CLIPS if not c.startswith("arm_3r")])
def test_simulate_reproduces_gt_q_over_free_segments(small, name):
    clip = small / name
    meta = load_metadata(clip / "metadata.yaml")
    assert meta.dynamics_gt
    dyn, xml_path = load_dynamics(clip / "gt" / "dynamics.json")
    kin, karr = load_kinematics(clip / "gt" / "kinematics.json")
    names = replay.joint_names(replay.load_model(xml_path))
    assert names == [j.joint_id for j in kin.joints]
    t = karr["t"]
    q = np.column_stack([karr[f"q__{j}"] for j in names])
    qd = np.column_stack([karr[f"qd__{j}"] for j in names])
    assert meta.free_motion_segments_s
    for t0, t1 in meta.free_motion_segments_s:
        sel = (t >= t0 - 1e-9) & (t <= t1 + 1e-9)
        q_sim, _ = replay.simulate(xml_path, q[sel][0], qd[sel][0], t[sel])
        assert np.abs(q_sim - q[sel]).max() < 1e-6
    # The hold phase is at rest at the hold pose.
    held = t < meta.extras["t_release"] - 1e-9
    assert np.all(qd[held] == 0.0)


def test_generation_is_deterministic(tmp_path):
    settings = SUITES["synth_tiny"]
    a = generate_clip("double_pendulum_2r", 1, tmp_path / "a", settings)
    b = generate_clip("double_pendulum_2r", 1, tmp_path / "b", settings)
    for rel in (
        "metadata.yaml",
        "camera_intrinsics.json",
        "gt/camera_extrinsics.json",
        "gt/tracks.json",
        "gt/tracks.npz",
        "gt/kinematics.json",
        "gt/kinematics.npz",
        "gt/dynamics.json",
        "gt/model.xml",
    ):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel
    assert load_npz(a / "gt" / "tracks.npz")["t"].shape == (60,)


def test_lamp_rest_pose_is_an_equilibrium(small):
    clip = small / "lamp_spring_2r__s0"
    dyn, xml_path = load_dynamics(clip / "gt" / "dynamics.json")
    spec = build_scene("lamp_spring_2r", 0)
    q_rest = np.asarray(spec.extras["q_rest"])
    t = np.linspace(0, 2.0, 121)
    q, _ = replay.simulate(xml_path, q_rest, np.zeros(2), t)
    assert np.abs(q - q_rest).max() < 1e-6
    assert all(abs(j.stiffness) > 0 for j in dyn.joints)
