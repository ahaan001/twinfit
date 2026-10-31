"""Hand-built contract objects shared by the setup tests."""

from __future__ import annotations

import numpy as np

from twinfit.contracts import (
    BoardSpec,
    CameraIntrinsics,
    ClipMetadata,
    Joint,
    JointFit,
    KinematicModel,
    Link,
    PartHint,
    PartTracks,
    Plane,
    TrackCamera,
    TrackPart,
)


def board_spec() -> BoardSpec:
    return BoardSpec(
        dictionary="DICT_5X5_100",
        squares_x=7,
        squares_y=5,
        square_length_m=0.035,
        marker_length_m=0.026,
    )


def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_id="synthetic",
        width=640,
        height=480,
        K=[[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dist=[0.0] * 5,
        source="synthetic",
    )


def metadata() -> ClipMetadata:
    return ClipMetadata(
        clip_id="pendulum_1r__s0",
        source="synthetic",
        mechanism="pendulum_1r",
        video="video.mp4",
        camera_id="synthetic",
        intrinsics="camera_intrinsics.json",
        board=board_spec(),
        board_orientation="vertical_arrow_up",
        motion_model="planar",
        plane_offset_m=0.01,
        parts_hint=[PartHint(label="base", fixed=True), PartHint(label="rod", fixed=False)],
        free_motion_segments_s=[[0.5, 2.0]],
        dynamics_gt=True,
    )


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def tracks(n: int = 10, static_camera: bool = True) -> tuple[PartTracks, dict[str, np.ndarray]]:
    t = np.arange(n, dtype=np.float64) / 30.0
    T_W_C = pose(np.eye(3), [0.1, 0.1, -1.0])
    model = PartTracks(
        clip_id="pendulum_1r__s0",
        source="synthetic_gt",
        fps_nominal=30.0,
        n_frames=n,
        motion_model="planar",
        plane=Plane(point_W=[0.0, 0.0, -0.01], normal_W=[0.0, 0.0, -1.0]),
        gravity_W=[0.0, 9.81, 0.0],
        camera=TrackCamera(
            intrinsics=intrinsics(),
            static=static_camera,
            T_W_C=T_W_C.tolist() if static_camera else None,
        ),
        parts=[
            TrackPart(part_id="p0", label="base", is_base_hint=True),
            TrackPart(
                part_id="p1",
                label="rod",
                extent_P=[[-0.015, -0.02, -0.002], [0.015, 0.28, 0.002]],
            ),
        ],
        free_motion_segments_s=[[0.1, 0.3]],
        free_motion_source="metadata",
    )
    arrays: dict[str, np.ndarray] = {"t": t}
    if not static_camera:
        arrays["T_W_C"] = np.repeat(T_W_C[None], n, axis=0)
    for pid in ("p0", "p1"):
        Ts = np.stack(
            [pose(rot_z(0.1 * k if pid == "p1" else 0.0), [0.1, 0.05, -0.01]) for k in range(n)]
        )
        arrays[f"T_W_P__{pid}"] = Ts
        arrays[f"valid__{pid}"] = np.ones(n, bool)
        arrays[f"quality__{pid}"] = np.zeros(n)
    return model, arrays


def kinematics(n: int = 10) -> tuple[KinematicModel, dict[str, np.ndarray]]:
    model = KinematicModel(
        clip_id="pendulum_1r__s0",
        source="synthetic_gt",
        base_part_id="p0",
        reference_frame_index=0,
        links=[Link(part_id="p0"), Link(part_id="p1", parent_joint_id="j0")],
        joints=[
            Joint(
                joint_id="j0",
                parent_part_id="p0",
                child_part_id="p1",
                type="revolute",
                axis_W=[0.0, 0.0, -1.0],
                point_W=[0.1, 0.05, 0.0],
                axis_parent=[0.0, 0.0, -1.0],
                point_parent=[0.0, 0.0, 0.01],
                q_range_observed=[0.0, 0.9],
                fit=JointFit(rms_point_m=0.0, n_frames=n),
                confidence=1.0,
            )
        ],
    )
    t = np.arange(n, dtype=np.float64) / 30.0
    arrays = {
        "t": t,
        "q__j0": 0.1 * np.arange(n, dtype=np.float64),
        "qd__j0": np.full(n, 3.0),
        "qdd__j0": np.zeros(n),
        "valid__j0": np.ones(n, bool),
    }
    return model, arrays
