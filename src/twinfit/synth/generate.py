"""Generation pipeline for synthetic ground-truth clips.

Per clip: build the scene, compute the joint trajectories (hold, then passive simulation with
`twinfit.sim.replay.simulate`, or prescribed motion), compute part poses with MuJoCo forward
kinematics, sample a camera that keeps the board and the whole motion in view, render RGB and
segmentation frames, and write the clip folder with full ground truth. The json and npz outputs
are byte-identical across runs for the same scene, seed and settings.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
from filelock import FileLock

from twinfit.contracts import (
    BoardSpec,
    CameraIntrinsics,
    ClipMetadata,
    DynamicsFit,
    DynamicsModel,
    DynJoint,
    DynLink,
    Identifiability,
    Joint,
    JointFit,
    KinematicModel,
    Link,
    PartHint,
    PartTracks,
    Plane,
    TrackCamera,
    TrackPart,
    TypeScores,
    charuco_board,
    gravity_for_orientation,
    save_dynamics,
    save_intrinsics,
    save_kinematics,
    save_metadata,
    save_tracks,
)
from twinfit.contracts.conventions import (
    CHARUCO_DICTIONARY,
    CHARUCO_MARKER_LENGTH_M,
    CHARUCO_SQUARE_LENGTH_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    NORMAL_TOWARD_CAMERA_W,
    UP_W,
)
from twinfit.contracts.io import save_pose_json
from twinfit.geometry import se3
from twinfit.geometry.camera import K_from_fovy, T_W_C_from_mujoco, project_points
from twinfit.geometry.lines import closest_point_to_origin
from twinfit.geometry.se3 import matrix_from_quat
from twinfit.paths import data_dir
from twinfit.sim import replay
from twinfit.synth.mjcf import mechanism_xml, scene_xml
from twinfit.synth.scenes import BOARD_H, BOARD_W, SCENE_NAMES, SceneSpec, build_scene

FOVY_DEG = 45.0
BOARD_TEXTURE_PX_PER_SQUARE = 140
VISIBLE_FRACTION = 0.98
CAMERA_ATTEMPTS = 200


@dataclass(frozen=True)
class SuiteSettings:
    seeds: tuple[int, ...]
    width: int
    height: int
    fps: float
    duration_s: float


SUITES: dict[str, SuiteSettings] = {
    "synth_tiny": SuiteSettings((0,), 320, 180, 30.0, 2.0),
    "synth_small": SuiteSettings((0, 1), 1280, 720, 60.0, 8.0),
}


def suite_dir(name: str, ensure: bool = True) -> Path:
    """Folder of a synthetic suite in the shared data dir, generated if absent.

    Every milestone's tests call this to get shared fixtures. A file lock serialises generation
    across worktrees."""
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; choose from {sorted(SUITES)}")
    root = data_dir() / "synth" / name
    if ensure:
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root) + ".lock"):
            if not _suite_complete(root, SUITES[name]):
                generate_suite(name, root)
    return root


def clip_folder_name(scene: str, seed: int) -> str:
    return f"{scene}__s{seed}"


def _suite_complete(root: Path, settings: SuiteSettings) -> bool:
    for scene in SCENE_NAMES:
        for seed in settings.seeds:
            if not (root / clip_folder_name(scene, seed) / "gt" / "kinematics.json").exists():
                return False
    return True


def generate_suite(name: str, out_root: Path | None = None) -> list[Path]:
    settings = SUITES[name]
    root = Path(out_root) if out_root is not None else data_dir() / "synth" / name
    out = []
    for scene in SCENE_NAMES:
        for seed in settings.seeds:
            out.append(generate_clip(scene, seed, root / clip_folder_name(scene, seed), settings))
    return out


# Trajectories


def _lamp_springrefs(spec: SceneSpec, gravity_W) -> None:
    """Set springref on each spring joint so the requested rest pose is a static equilibrium:
    with qdot = 0, M qddot + g(q) = -k (q - ref), so ref = q + g(q) / k."""
    q_rest = np.asarray(spec.extras["q_rest"], dtype=float)
    model = replay.load_model(mechanism_xml(spec, gravity_W))
    data = mujoco.MjData(model)
    data.qpos[:] = q_rest
    mujoco.mj_forward(model, data)
    for i, joint in enumerate(spec.joints):
        joint.springref = float(q_rest[i] + data.qfrc_bias[i] / joint.stiffness)
    spec.q_hold = q_rest + spec.extras["hold_offset_rad"]


def _accelerations(model: mujoco.MjModel, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    qdd = np.zeros_like(q)
    for k in range(len(q)):
        data.qpos[:] = q[k]
        data.qvel[:] = qd[k]
        mujoco.mj_forward(model, data)
        qdd[k] = data.qacc
    return qdd


def trajectories(
    spec: SceneSpec, mech_xml: str, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    """q, qd, qdd at the frame times, and the free-motion segments.

    Dynamics scenes hold q_hold with zero velocity until t_release (equivalent to resetting the
    state at every physics step) and then run `simulate` from (q_hold, 0) at t_release, so the
    ground truth is reproducible by the same function from the released state."""
    nq = len(spec.joints)
    if spec.prescribed is not None:
        q, qd, qdd = spec.prescribed(t)
        return q, qd, qdd, []
    q = np.tile(spec.q_hold, (len(t), 1))
    qd = np.zeros((len(t), nq))
    qdd = np.zeros((len(t), nq))
    released = t >= spec.t_release - 1e-9
    t_free = t[released]
    if len(t_free) == 0:
        return q, qd, qdd, []
    if abs(t_free[0] - spec.t_release) > 1e-9:
        t_free = np.concatenate([[spec.t_release], t_free])
        q_free, qd_free = replay.simulate(mech_xml, spec.q_hold, np.zeros(nq), t_free)
        q_free, qd_free = q_free[1:], qd_free[1:]
    else:
        q_free, qd_free = replay.simulate(mech_xml, spec.q_hold, np.zeros(nq), t_free)
    q[released] = q_free
    qd[released] = qd_free
    model = replay.load_model(mech_xml)
    qdd[released] = _accelerations(model, q_free, qd_free)
    t_end = float(t[-1])
    if spec.free_segment_rule == "before_limit":
        limit = spec.joints[0].range[1]
        hit = np.nonzero(q[:, 0] >= limit - 2e-3)[0]
        if len(hit):
            t_end = min(t_end, float(t[hit[0]]) - 0.1)
    segments = [[float(spec.t_release), t_end]] if t_end > spec.t_release else []
    return q, qd, qdd, segments


def part_poses(model: mujoco.MjModel, part_ids: list[str], q: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    body_ids = {pid: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pid) for pid in part_ids}
    poses = {pid: np.zeros((len(q), 4, 4)) for pid in part_ids}
    for k in range(len(q)):
        data.qpos[:] = q[k]
        mujoco.mj_forward(model, data)
        for pid, b in body_ids.items():
            poses[pid][k] = se3.make_T(data.xmat[b].reshape(3, 3), data.xpos[b])
    return poses


# Camera


def _corners_W(spec: SceneSpec, poses: dict[str, np.ndarray]) -> np.ndarray:
    pts = []
    for part in spec.parts:
        lo, hi = np.asarray(part.extent_P)
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        T = poses[part.part_id]
        pts.append(np.einsum("nij,kj->nki", T[:, :3, :3], corners) + T[:, None, :3, 3])
    return np.concatenate(pts, axis=0).reshape(-1, 3)


def _board_corners() -> np.ndarray:
    return np.array([[0, 0, 0], [BOARD_W, 0, 0], [BOARD_W, BOARD_H, 0], [0, BOARD_H, 0]], float)


def _look_at(cam_pos: np.ndarray, aim: np.ndarray) -> np.ndarray:
    """OpenCV camera rotation R_W_C looking from cam_pos at aim with image-down along +y_W."""
    z = aim - cam_pos
    z = z / np.linalg.norm(z)
    down = -UP_W
    y = down - np.dot(down, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def _camera_from_angles(aim, dist, tilt_rad, azimuth_rad) -> np.ndarray:
    """Camera pose at `dist` from `aim`, tilted from the board normal by `tilt_rad` toward the
    in-plane direction with angle `azimuth_rad` measured from +x_W."""
    toward_camera = NORMAL_TOWARD_CAMERA_W
    in_plane = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
    direction = np.cos(tilt_rad) * toward_camera + np.sin(tilt_rad) * in_plane
    cam_pos = aim + dist * direction
    return se3.make_T(_look_at(cam_pos, aim), cam_pos)


def _visible_fraction(points_W, K, T_W_C, width, height) -> float:
    uv = project_points(points_W, K, np.zeros(5), T_W_C)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    return float(inside.mean())


def sample_camera(
    spec: SceneSpec,
    poses: dict[str, np.ndarray],
    K: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """T_W_C such that the whole board and at least 98 percent of all part corners over all frames
    project inside the image. Seed 0 uses the nominal camera (the scene's nominal tilt toward +x_W
    at the nominal distance); other seeds draw tilt (5 to 15 degrees for planar scenes, 35 +/- 5
    degrees for the spatial scene), azimuth and distance (0.7 to 1.1 m) until the view is valid."""
    part_pts = _corners_W(spec, poses)
    board_pts = _board_corners()
    all_pts = np.concatenate([part_pts, board_pts])
    aim = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    aim[2] = 0.0
    nominal_tilt = np.radians(spec.camera_tilt_deg)
    candidates = []
    if spec.seed == 0:
        d0 = spec.camera_distance_m
        candidates = [(dist, nominal_tilt, 0.0) for dist in (d0, d0 + 0.1, d0 + 0.2, d0 + 0.3, 1.4)]
    else:
        for _ in range(CAMERA_ATTEMPTS):
            dist = float(rng.uniform(0.7, 1.1))
            if spec.motion_model == "spatial":
                tilt = nominal_tilt + np.radians(rng.uniform(-5.0, 5.0))
                azimuth = np.radians(rng.uniform(-20.0, 20.0))
            else:
                tilt = np.radians(rng.uniform(spec.camera_tilt_min_deg, spec.camera_tilt_max_deg))
                azimuth = float(rng.uniform(0.0, 2 * np.pi))
            candidates.append((dist, tilt, azimuth))
        candidates += [(1.2, nominal_tilt, 0.0), (1.4, nominal_tilt, 0.0)]
    for dist, tilt, azimuth in candidates:
        T_W_C = _camera_from_angles(aim, dist, tilt, azimuth)
        if _visible_fraction(board_pts, K, T_W_C, width, height) < 1.0:
            continue
        if _visible_fraction(part_pts, K, T_W_C, width, height) >= VISIBLE_FRACTION:
            return T_W_C
    raise RuntimeError(f"{spec.name} seed {spec.seed}: no camera keeps the scene in view")


def _light_dir(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    d = np.array([0.2, 0.3, 1.0])
    if spec.seed != 0:
        d = d + np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.0])
    return d / np.linalg.norm(d)


# Rendering


def _write_board_texture(path: Path) -> None:
    board = charuco_board()
    pps = BOARD_TEXTURE_PX_PER_SQUARE
    img = board.generateImage(
        (CHARUCO_SQUARES_X * pps, CHARUCO_SQUARES_Y * pps), marginSize=0, borderBits=1
    )
    cv2.imwrite(str(path), img)


def _video_writer(path: Path, width: int, height: int, fps: float):
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        quality=None,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-preset", "veryfast", "-crf", "18", "-r", repr(float(fps))],
    )
    writer.send(None)
    return writer


def render_clip(
    scene_model: mujoco.MjModel,
    q: np.ndarray,
    part_ids: list[str],
    width: int,
    height: int,
    fps: float,
    video_path: Path,
    masks_dir: Path,
) -> None:
    data = mujoco.MjData(scene_model)
    lut = np.zeros(scene_model.ngeom + 1, dtype=np.uint8)  # index -1 (background) maps to 0
    for g in range(scene_model.ngeom):
        body = mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_BODY, scene_model.geom_bodyid[g])
        if body in part_ids:
            lut[g] = part_ids.index(body) + 1
    masks_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(scene_model, height, width)
    writer = _video_writer(video_path, width, height, fps)
    try:
        for k in range(len(q)):
            data.qpos[:] = q[k]
            mujoco.mj_forward(scene_model, data)
            renderer.disable_segmentation_rendering()
            renderer.update_scene(data, camera="cam")
            rgb = renderer.render()
            writer.send(np.ascontiguousarray(rgb))
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="cam")
            seg = renderer.render()
            cv2.imwrite(str(masks_dir / f"{k:06d}.png"), lut[seg[..., 0]])
    finally:
        writer.close()
        renderer.close()


# Ground truth files


def _board_spec() -> BoardSpec:
    return BoardSpec(
        dictionary=CHARUCO_DICTIONARY,
        squares_x=CHARUCO_SQUARES_X,
        squares_y=CHARUCO_SQUARES_Y,
        square_length_m=CHARUCO_SQUARE_LENGTH_M,
        marker_length_m=CHARUCO_MARKER_LENGTH_M,
    )


def _joint_gt(
    spec: SceneSpec, model: mujoco.MjModel, poses: dict[str, np.ndarray], q: np.ndarray, ref: int
) -> list[Joint]:
    data = mujoco.MjData(model)
    data.qpos[:] = q[ref]
    mujoco.mj_forward(model, data)
    joints = []
    for part in spec.moving_parts:
        j = part.joint
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.joint_id)
        axis_W = data.xaxis[jid] / np.linalg.norm(data.xaxis[jid])
        T_W_parent = poses[part.parent][ref]
        axis_parent = T_W_parent[:3, :3].T @ axis_W
        point_W = point_parent = None
        if j.type == "revolute":
            anchor_W = data.xanchor[jid]
            point_W = closest_point_to_origin(anchor_W, axis_W)
            anchor_P = T_W_parent[:3, :3].T @ (anchor_W - T_W_parent[:3, 3])
            point_parent = closest_point_to_origin(anchor_P, axis_parent)
        col = spec.joints.index(j)
        joints.append(
            Joint(
                joint_id=j.joint_id,
                parent_part_id=part.parent,
                child_part_id=part.part_id,
                type=j.type,
                axis_W=axis_W.tolist(),
                point_W=None if point_W is None else point_W.tolist(),
                axis_parent=axis_parent.tolist(),
                point_parent=None if point_parent is None else point_parent.tolist(),
                q_range_observed=[float(q[:, col].min()), float(q[:, col].max())],
                fit=JointFit(rms_point_m=0.0, n_frames=len(q)),
                type_scores=TypeScores(**{j.type: 0.0}),
                confidence=1.0,
                extras={"method": "synthetic_gt"},
            )
        )
    return joints


def _dynamics_gt(
    spec: SceneSpec, model: mujoco.MjModel, gravity_W, segments: list[list[float]], clip_id: str
) -> DynamicsModel:
    links = []
    for part in spec.parts:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part.part_id)
        R = matrix_from_quat(model.body_iquat[b])
        inertia = R @ np.diag(model.body_inertia[b]) @ R.T
        links.append(
            DynLink(
                part_id=part.part_id,
                mass_kg=float(model.body_mass[b]),
                com_P=model.body_ipos[b].tolist(),
                inertia_com_P=inertia.tolist(),
            )
        )
    joints = []
    for j in spec.joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.joint_id)
        dof = model.jnt_dofadr[jid]
        joints.append(
            DynJoint(
                joint_id=j.joint_id,
                damping=float(model.dof_damping[dof]),
                frictionloss=float(model.dof_frictionloss[dof]),
                armature=float(model.dof_armature[dof]),
                stiffness=float(model.jnt_stiffness[jid]),
                springref=float(model.qpos_spring[model.jnt_qposadr[jid]]),
            )
        )
    n_params = 10 * len(spec.moving_parts) + 5 * len(spec.joints)
    return DynamicsModel(
        clip_id=clip_id,
        source="synthetic_gt",
        gravity_W=np.asarray(gravity_W).tolist(),
        mass_scale_known=True,
        links=links,
        joints=joints,
        identifiability=Identifiability(rank=n_params, n_params=n_params, condition_number=1.0),
        fit=DynamicsFit(
            segments_s=segments,
            replay_rms={j.joint_id: 0.0 for j in spec.joints},
            window_s=0.5,
            method="synthetic_gt",
        ),
    )


def generate_clip(scene: str, seed: int, out_dir: Path, settings: SuiteSettings) -> Path:
    out_dir = Path(out_dir)
    spec = build_scene(scene, seed)
    gravity_W = gravity_for_orientation(spec.board_orientation)
    if "q_rest" in spec.extras:
        _lamp_springrefs(spec, gravity_W)
    mech_xml = mechanism_xml(spec, gravity_W)
    mech_model = replay.load_model(mech_xml)
    part_ids = [p.part_id for p in spec.parts]
    n_frames = int(round(settings.duration_s * settings.fps))
    t = np.arange(n_frames, dtype=np.float64) / settings.fps
    q, qd, qdd, segments = trajectories(spec, mech_xml, t)
    poses = part_poses(mech_model, part_ids, q)

    rng = np.random.default_rng([seed, SCENE_NAMES.index(scene)])
    K = K_from_fovy(np.radians(FOVY_DEG), settings.width, settings.height)
    T_W_C = sample_camera(spec, poses, K, settings.width, settings.height, rng)
    light_dir = _light_dir(spec, rng)
    R_mj = T_W_C[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    cam_xyaxes = np.concatenate([R_mj[:, 0], R_mj[:, 1]])

    clip_id = clip_folder_name(scene, seed)
    gt = out_dir / "gt"
    gt.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        board_png = Path(tmp) / "board.png"
        _write_board_texture(board_png)
        xml = scene_xml(
            spec,
            gravity_W,
            str(board_png),
            T_W_C[:3, 3],
            cam_xyaxes,
            FOVY_DEG,
            light_dir,
            settings.width,
            settings.height,
        )
        scene_model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(scene_model)
        mujoco.mj_forward(scene_model, data)
        T_W_C_render = T_W_C_from_mujoco(data.cam_xpos[0], data.cam_xmat[0])
        if not np.allclose(T_W_C_render, T_W_C, atol=1e-9):
            raise RuntimeError("camera pose in the compiled scene differs from the sampled pose")
        render_clip(
            scene_model,
            q,
            part_ids,
            settings.width,
            settings.height,
            settings.fps,
            out_dir / "video.mp4",
            gt / "masks",
        )

    intrinsics = CameraIntrinsics(
        camera_id="synthetic",
        width=settings.width,
        height=settings.height,
        K=K.tolist(),
        dist=[0.0] * 5,
        rms_reprojection_px=None,
        n_frames_used=None,
        source="synthetic",
    )
    save_intrinsics(out_dir / "camera_intrinsics.json", intrinsics)
    save_pose_json(gt / "camera_extrinsics.json", T_W_C)

    metadata = ClipMetadata(
        clip_id=clip_id,
        source="synthetic",
        mechanism=scene,
        video="video.mp4",
        camera_id="synthetic",
        intrinsics="camera_intrinsics.json",
        board=_board_spec(),
        board_orientation=spec.board_orientation,
        motion_model=spec.motion_model,
        plane_offset_m=spec.plane_offset_m,
        gravity_W=None,
        parts_hint=[PartHint(label=p.label, fixed=p.joint is None) for p in spec.parts],
        part_markers=None,
        free_motion_segments_s=segments,
        dynamics_gt=spec.dynamics_gt,
        notes=spec.notes,
        extras={
            "seed": seed,
            "fps": settings.fps,
            "duration_s": settings.duration_s,
            "t_release": spec.t_release if spec.prescribed is None else None,
        },
    )
    save_metadata(out_dir / "metadata.yaml", metadata)

    plane = None
    if spec.motion_model == "planar":
        plane = Plane(
            point_W=(spec.plane_offset_m * NORMAL_TOWARD_CAMERA_W).tolist(),
            normal_W=NORMAL_TOWARD_CAMERA_W.tolist(),
        )
    tracks = PartTracks(
        clip_id=clip_id,
        source="synthetic_gt",
        fps_nominal=settings.fps,
        n_frames=n_frames,
        motion_model=spec.motion_model,
        plane=plane,
        gravity_W=gravity_W.tolist(),
        camera=TrackCamera(intrinsics=intrinsics, static=True, T_W_C=T_W_C.tolist()),
        parts=[
            TrackPart(
                part_id=p.part_id, label=p.label, is_base_hint=p.joint is None, extent_P=p.extent_P
            )
            for p in spec.parts
        ],
        free_motion_segments_s=segments,
        free_motion_source="metadata" if segments else "none",
    )
    arrays: dict[str, np.ndarray] = {"t": t}
    for pid in part_ids:
        arrays[f"T_W_P__{pid}"] = poses[pid]
        arrays[f"valid__{pid}"] = np.ones(n_frames, dtype=bool)
        arrays[f"quality__{pid}"] = np.zeros(n_frames)
    save_tracks(gt / "tracks.json", tracks, arrays)

    ref = 0
    kin = KinematicModel(
        clip_id=clip_id,
        source="synthetic_gt",
        base_part_id="p0",
        reference_frame_index=ref,
        links=[
            Link(part_id=p.part_id, parent_joint_id=None if p.joint is None else p.joint.joint_id)
            for p in spec.parts
        ],
        joints=_joint_gt(spec, mech_model, poses, q, ref),
    )
    kin_arrays: dict[str, np.ndarray] = {"t": t}
    for col, j in enumerate(spec.joints):
        kin_arrays[f"q__{j.joint_id}"] = q[:, col].copy()
        kin_arrays[f"qd__{j.joint_id}"] = qd[:, col].copy()
        kin_arrays[f"qdd__{j.joint_id}"] = qdd[:, col].copy()
        kin_arrays[f"valid__{j.joint_id}"] = np.ones(n_frames, dtype=bool)
    save_kinematics(gt / "kinematics.json", kin, kin_arrays)

    if spec.dynamics_gt:
        save_dynamics(
            gt / "dynamics.json",
            _dynamics_gt(spec, mech_model, gravity_W, segments, clip_id),
            mech_xml,
        )
    return out_dir
