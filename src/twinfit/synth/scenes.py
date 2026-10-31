"""Scene definitions for the synthetic suites.

Frames: the MuJoCo world frame equals W (x right, y down the page, z into the board). Parts are
box geoms. A part's body frame is its part frame P; the geom centre is `geom_pos` in P. Moving
parts have their camera-facing face (minimum z) at z = -plane_offset_m. In every scene p0 is the
fixed, visible base welded to the world.

Seed 0 uses the nominal values. Other seeds scale lengths, masses, damping, friction and
stiffness by U(0.85, 1.15), shift initial angles by up to 10 degrees, and randomise the camera
pose, light direction and part colours (see generate.py for the camera and light).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from twinfit.contracts.conventions import (
    CHARUCO_SQUARE_LENGTH_M,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    NORMAL_TOWARD_CAMERA_W,
)

BOARD_W = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M
BOARD_H = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M
BOARD_CENTER = np.array([BOARD_W / 2, BOARD_H / 2, 0.0])
AXIS_NORMAL = tuple(float(v) for v in NORMAL_TOWARD_CAMERA_W)  # (0, 0, -1)

Vec = tuple[float, float, float]


@dataclass
class JointSpec:
    joint_id: str
    type: str  # "revolute" or "prismatic"
    axis: Vec  # in the child body frame
    damping: float = 0.0
    frictionloss: float = 0.0
    armature: float = 0.0
    stiffness: float = 0.0
    springref: float = 0.0
    range: tuple[float, float] | None = None


@dataclass
class PartSpec:
    part_id: str
    label: str
    size: Vec  # full box dimensions in P
    mass_kg: float
    geom_pos: Vec = (0.0, 0.0, 0.0)
    color: Vec = (0.8, 0.8, 0.8)
    speckle: bool = True
    parent: str | None = None  # None: welded to the world
    body_pos: Vec = (0.0, 0.0, 0.0)  # body origin in the parent frame (W for the base)
    body_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    joint: JointSpec | None = None

    @property
    def half(self) -> np.ndarray:
        return np.asarray(self.size) / 2.0

    @property
    def extent_P(self) -> list[list[float]]:
        c = np.asarray(self.geom_pos)
        return [(c - self.half).tolist(), (c + self.half).tolist()]


Trajectory = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class SceneSpec:
    name: str
    seed: int
    board_orientation: str
    motion_model: str
    plane_offset_m: float
    dynamics_gt: bool
    parts: list[PartSpec]
    q_hold: np.ndarray | None = None
    t_release: float = 0.5
    free_segment_rule: str = "to_end"  # or "before_limit"
    prescribed: Trajectory | None = None
    # Nominal camera tilt from fronto-parallel, used by seed 0; other seeds draw a tilt between
    # camera_tilt_min_deg and camera_tilt_max_deg. The planar nominal is not 0: an exactly
    # fronto-parallel view of a planar target makes PnP ill-conditioned (tilt trades off against
    # lateral translation), and measured errors reached 1 degree even on an ideal image warp.
    # See docs/ledger/setup.md, Phase 6.
    camera_tilt_deg: float = 12.0
    camera_tilt_min_deg: float = 5.0
    camera_tilt_max_deg: float = 15.0
    camera_distance_m: float = 0.8  # nominal, used by seed 0; other seeds draw 0.7 to 1.1 m
    notes: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def moving_parts(self) -> list[PartSpec]:
        return [p for p in self.parts if p.joint is not None]

    @property
    def joints(self) -> list[JointSpec]:
        return [p.joint for p in self.parts if p.joint is not None]


class _Rand:
    """Seeded randomisation helpers. Seed 0 returns nominal values."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def scale(self) -> float:
        return 1.0 if self.seed == 0 else float(self.rng.uniform(0.85, 1.15))

    def angle_shift(self) -> float:
        return 0.0 if self.seed == 0 else float(self.rng.uniform(-np.radians(10), np.radians(10)))

    def color(self, nominal: Vec) -> Vec:
        if self.seed == 0:
            return nominal
        c = np.clip(np.asarray(nominal) + self.rng.uniform(-0.15, 0.15, size=3), 0.05, 0.95)
        return tuple(float(v) for v in c)


def quat_z(angle: float) -> tuple[float, float, float, float]:
    return (float(np.cos(angle / 2)), 0.0, 0.0, float(np.sin(angle / 2)))


def _front_z(plane_offset: float, thickness: float) -> float:
    """Body-frame z of a geom whose camera-facing face lies at -plane_offset."""
    return -plane_offset + thickness / 2


def pendulum_1r(seed: int) -> SceneSpec:
    r = _Rand(seed)
    L, w, h = 0.30 * r.scale(), 0.03, 0.004
    mass = 0.06 * r.scale()
    offset = 0.010
    pivot_from_top = 0.02
    pivot = (BOARD_W / 2, -0.045, 0.0)
    base = PartSpec(
        "p0",
        "base",
        (0.05, 0.04, 0.006),
        0.05,
        geom_pos=(0.0, 0.0, _front_z(offset - h, 0.006)),
        color=r.color((0.2, 0.3, 0.8)),
        speckle=False,
        body_pos=pivot,
    )
    rod = PartSpec(
        "p1",
        "rod",
        (w, L, h),
        mass,
        geom_pos=(0.0, L / 2 - pivot_from_top, _front_z(offset, h)),
        color=r.color((0.85, 0.75, 0.3)),
        parent="p0",
        body_pos=(0.0, 0.0, 0.0),
        joint=JointSpec(
            "j0", "revolute", AXIS_NORMAL, damping=2e-4 * r.scale(), frictionloss=5e-4 * r.scale()
        ),
    )
    return SceneSpec(
        "pendulum_1r",
        seed,
        "vertical_arrow_up",
        "planar",
        offset,
        True,
        [base, rod],
        q_hold=np.array([np.radians(60.0) + r.angle_shift()]),
        notes="Rod pendulum released from 60 degrees.",
    )


def double_pendulum_2r(seed: int) -> SceneSpec:
    r = _Rand(seed)
    L1, L2 = 0.20 * r.scale(), 0.15 * r.scale()
    w, h = 0.025, 0.003
    m1, m2 = 0.025 * r.scale(), 0.020 * r.scale()
    offset = 0.008
    end = 0.01  # pin distance from each strip end
    pivot = (BOARD_W / 2, -0.04, 0.0)
    base = PartSpec(
        "p0",
        "base",
        (0.05, 0.04, 0.006),
        0.05,
        geom_pos=(0.0, 0.0, _front_z(offset - 2 * h, 0.006)),
        color=r.color((0.2, 0.3, 0.8)),
        speckle=False,
        body_pos=pivot,
    )
    # p2 lies on the motion plane; p1 sits one thickness behind it.
    p1 = PartSpec(
        "p1",
        "upper_strip",
        (w, L1, h),
        m1,
        geom_pos=(0.0, L1 / 2 - end, _front_z(offset - h, h)),
        color=r.color((0.85, 0.75, 0.3)),
        parent="p0",
        joint=JointSpec(
            "j0", "revolute", AXIS_NORMAL, damping=1e-4 * r.scale(), frictionloss=2e-4 * r.scale()
        ),
    )
    p2 = PartSpec(
        "p2",
        "lower_strip",
        (w, L2, h),
        m2,
        geom_pos=(0.0, L2 / 2 - end, _front_z(offset, h)),
        color=r.color((0.4, 0.8, 0.5)),
        parent="p1",
        body_pos=(0.0, L1 - 2 * end, 0.0),
        joint=JointSpec(
            "j1", "revolute", AXIS_NORMAL, damping=1e-4 * r.scale(), frictionloss=2e-4 * r.scale()
        ),
    )
    return SceneSpec(
        "double_pendulum_2r",
        seed,
        "vertical_arrow_up",
        "planar",
        offset,
        True,
        [base, p1, p2],
        q_hold=np.array([np.radians(90.0) + r.angle_shift(), 0.0 + r.angle_shift()]),
        notes="Two strips joined by a pin, released from (90, 0) degrees.",
    )


def lamp_spring_2r(seed: int) -> SceneSpec:
    r = _Rand(seed)
    L1, L2 = 0.25 * r.scale(), 0.20 * r.scale()
    w, h = 0.03, 0.01
    m1, m2 = 0.12 * r.scale(), 0.10 * r.scale()
    offset = 0.020
    shoulder = (0.06, BOARD_H + 0.02, 0.0)
    base = PartSpec(
        "p0",
        "lamp_base",
        (0.06, 0.04, 0.012),
        0.3,
        geom_pos=(0.0, 0.02, _front_z(offset - 2 * h, 0.012)),
        color=r.color((0.25, 0.25, 0.3)),
        speckle=False,
        body_pos=shoulder,
    )
    # Arms point along -y (up the page) at q = 0; p2 is on the plane, p1 one thickness behind.
    p1 = PartSpec(
        "p1",
        "lower_arm",
        (w, L1, h),
        m1,
        geom_pos=(0.0, -L1 / 2 + 0.015, _front_z(offset - h, h)),
        color=r.color((0.8, 0.5, 0.2)),
        parent="p0",
        joint=JointSpec(
            "j0",
            "revolute",
            AXIS_NORMAL,
            damping=5e-3 * r.scale(),
            frictionloss=2e-3 * r.scale(),
            stiffness=0.6 * r.scale(),
        ),
    )
    p2 = PartSpec(
        "p2",
        "upper_arm",
        (w, L2, h),
        m2,
        geom_pos=(0.0, -L2 / 2 + 0.015, _front_z(offset, h)),
        color=r.color((0.3, 0.6, 0.85)),
        parent="p1",
        body_pos=(0.0, -(L1 - 0.03), 0.0),
        joint=JointSpec(
            "j1",
            "revolute",
            AXIS_NORMAL,
            damping=5e-3 * r.scale(),
            frictionloss=2e-3 * r.scale(),
            stiffness=0.4 * r.scale(),
        ),
    )
    # The rest pose (static equilibrium) is visibly bent; generate.py sets springref so that this
    # pose is the equilibrium, then holds 25 degrees away from it.
    q_rest = np.array([np.radians(-35.0) + r.angle_shift(), np.radians(75.0) + r.angle_shift()])
    spec = SceneSpec(
        "lamp_spring_2r",
        seed,
        "vertical_arrow_up",
        "planar",
        offset,
        True,
        [base, p1, p2],
        q_hold=None,
        notes="Two spring-loaded arms released 25 degrees from their rest pose.",
        extras={"q_rest": q_rest, "hold_offset_rad": np.radians(25.0)},
    )
    return spec


def slider_incline_1p(seed: int) -> SceneSpec:
    r = _Rand(seed)
    rail_len = 0.35
    incline = np.radians(25.0) + r.angle_shift()
    offset = 0.015
    block = (0.04 * r.scale(), 0.03, 0.02)
    mass = 0.05 * r.scale()
    rail_center = (BOARD_W / 2, BOARD_H / 2 - 0.02, 0.0)
    rail_h = 0.006
    base = PartSpec(
        "p0",
        "rail",
        (rail_len, 0.02, rail_h),
        0.2,
        geom_pos=(0.0, 0.0, -offset + 0.006 + rail_h / 2),
        color=r.color((0.2, 0.3, 0.8)),
        speckle=False,
        body_pos=rail_center,
        body_quat=quat_z(incline),
    )
    travel = rail_len - block[0]
    # q = 0 at the top end of the rail; +x in the rail frame runs downhill (y grows down the page).
    p1 = PartSpec(
        "p1",
        "block",
        block,
        mass,
        geom_pos=(0.0, 0.0, _front_z(offset, block[2])),
        color=r.color((0.85, 0.75, 0.3)),
        parent="p0",
        body_pos=(-travel / 2, 0.0, 0.0),
        joint=JointSpec(
            "j0",
            "prismatic",
            (1.0, 0.0, 0.0),
            damping=1.0 * r.scale(),
            frictionloss=0.02 * r.scale(),
            range=(0.0, travel),
        ),
    )
    return SceneSpec(
        "slider_incline_1p",
        seed,
        "vertical_arrow_up",
        "planar",
        offset,
        True,
        [base, p1],
        q_hold=np.array([0.0]),
        free_segment_rule="before_limit",
        notes="Block sliding down a rail inclined 25 degrees; free segment ends before the stop.",
    )


def door_spring_1r_h(seed: int) -> SceneSpec:
    r = _Rand(seed)
    L = 0.25 * r.scale()
    panel = (L, 0.02, 0.03)
    mass = 0.15 * r.scale()
    offset = 0.020
    hinge = (0.04, BOARD_H / 2 + 0.03, 0.0)
    base = PartSpec(
        "p0",
        "door_post",
        (0.03, 0.03, 0.03),
        0.2,
        geom_pos=(-0.02, 0.0, _front_z(offset, 0.03)),
        color=r.color((0.25, 0.25, 0.3)),
        speckle=False,
        body_pos=hinge,
    )
    # Panel extends along +x from the hinge at q = 0.
    p1 = PartSpec(
        "p1",
        "panel",
        panel,
        mass,
        geom_pos=(L / 2, 0.0, _front_z(offset, panel[2])),
        color=r.color((0.8, 0.55, 0.3)),
        parent="p0",
        joint=JointSpec(
            "j0",
            "revolute",
            AXIS_NORMAL,
            damping=0.01 * r.scale(),
            frictionloss=1e-3 * r.scale(),
            stiffness=0.05 * r.scale(),
            springref=0.0,
        ),
    )
    return SceneSpec(
        "door_spring_1r_h",
        seed,
        "horizontal_face_up",
        "planar",
        offset,
        True,
        [base, p1],
        q_hold=np.array([np.radians(70.0) + r.angle_shift()]),
        notes="Spring-return door seen from above, released from 70 degrees.",
    )


def _bump(t: np.ndarray, t0: float, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth out-and-back profile b = (1 - cos(2 pi s)) / 2 on s = (t - t0) / T, zero elsewhere."""
    s = (t - t0) / T
    active = (s >= 0.0) & (s <= 1.0)
    b = np.where(active, 0.5 * (1.0 - np.cos(2 * np.pi * s)), 0.0)
    bd = np.where(active, np.pi / T * np.sin(2 * np.pi * s), 0.0)
    bdd = np.where(active, 2 * np.pi**2 / T**2 * np.cos(2 * np.pi * s), 0.0)
    return b, bd, bdd


def arm_3r_spatial(seed: int) -> SceneSpec:
    r = _Rand(seed)
    L1, L2, L3 = 0.15 * r.scale(), 0.15 * r.scale(), 0.10 * r.scale()
    s = 0.02
    base_z = -0.10
    base_pos = (BOARD_W / 2, BOARD_H + 0.04, base_z)
    base = PartSpec(
        "p0",
        "arm_base",
        (0.04, 0.04, 0.04),
        0.5,
        color=r.color((0.25, 0.25, 0.3)),
        speckle=False,
        body_pos=base_pos,
    )
    # Links point along -y (up the page) at q = 0. j0 is about the board normal, j1 and j2 about
    # +x (in the board plane, perpendicular to p1); positive j1, j2 tilt the arm toward the camera.
    p1 = PartSpec(
        "p1",
        "link1",
        (s, L1, s),
        0.1,
        geom_pos=(0.0, -L1 / 2, 0.0),
        color=r.color((0.85, 0.75, 0.3)),
        parent="p0",
        joint=JointSpec("j0", "revolute", AXIS_NORMAL),
    )
    p2 = PartSpec(
        "p2",
        "link2",
        (s, L2, s),
        0.1,
        geom_pos=(0.0, -L2 / 2, 0.0),
        color=r.color((0.4, 0.8, 0.5)),
        parent="p1",
        body_pos=(0.0, -L1, 0.0),
        joint=JointSpec("j1", "revolute", (1.0, 0.0, 0.0)),
    )
    p3 = PartSpec(
        "p3",
        "link3",
        (s, L3, s),
        0.06,
        geom_pos=(0.0, -L3 / 2, 0.0),
        color=r.color((0.3, 0.6, 0.85)),
        parent="p2",
        body_pos=(0.0, -L2, 0.0),
        joint=JointSpec("j2", "revolute", (1.0, 0.0, 0.0)),
    )
    q0 = np.array([r.angle_shift(), abs(r.angle_shift()), abs(r.angle_shift())])
    amp = np.radians([50.0, 35.0, 45.0])

    def prescribed(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.tile(q0, (len(t), 1))
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        phases = [(0, 0.0), (1, 2.0), (2, 4.0)]
        for j, t0 in phases:
            b, bd, bdd = _bump(t, t0, 2.0)
            q[:, j] += amp[j] * b
            qd[:, j] += amp[j] * bd
            qdd[:, j] += amp[j] * bdd
        b, bd, bdd = _bump(t, 6.0, 2.0)
        for j in range(3):
            q[:, j] += amp[j] * b
            qd[:, j] += amp[j] * bd
            qdd[:, j] += amp[j] * bdd
        return q, qd, qdd

    return SceneSpec(
        "arm_3r_spatial",
        seed,
        "vertical_arrow_up",
        "spatial",
        0.10,
        False,
        [base, p1, p2, p3],
        prescribed=prescribed,
        camera_tilt_deg=35.0,
        notes="3R arm with prescribed joint motion; each joint alone for 2 s, then all together.",
    )


SCENE_BUILDERS = {
    "pendulum_1r": pendulum_1r,
    "double_pendulum_2r": double_pendulum_2r,
    "lamp_spring_2r": lamp_spring_2r,
    "slider_incline_1p": slider_incline_1p,
    "door_spring_1r_h": door_spring_1r_h,
    "arm_3r_spatial": arm_3r_spatial,
}
SCENE_NAMES = tuple(SCENE_BUILDERS)


def build_scene(name: str, seed: int) -> SceneSpec:
    if name not in SCENE_BUILDERS:
        raise ValueError(f"unknown scene {name!r}; choose from {SCENE_NAMES}")
    return SCENE_BUILDERS[name](seed)
