"""Frozen pydantic v2 models for every Twinfit file. See conventions.py for frames and units."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from twinfit.contracts.conventions import JOINT_ID_RE, PART_ID_RE, WORLD_ID

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Vec5 = Annotated[list[float], Field(min_length=5, max_length=5)]
Row3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Row4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Mat3 = Annotated[list[Row3], Field(min_length=3, max_length=3)]
Mat4 = Annotated[list[Row4], Field(min_length=4, max_length=4)]
Segment = Annotated[list[float], Field(min_length=2, max_length=2)]
Extent = Annotated[list[Vec3], Field(min_length=2, max_length=2)]
ContractVersion = Literal["1.0"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_part_id(value: str) -> str:
    if not PART_ID_RE.match(value):
        raise ValueError(f"part id {value!r} must match p[0-9]+")
    return value


def _check_joint_id(value: str) -> str:
    if not JOINT_ID_RE.match(value):
        raise ValueError(f"joint id {value!r} must match j[0-9]+")
    return value


def _check_unit(value: list[float], name: str) -> list[float]:
    if not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must be finite")
    norm = math.sqrt(sum(v * v for v in value))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a unit vector (norm {norm:.9f})")
    return value


def _check_finite(value: list[float] | None, name: str) -> list[float] | None:
    if value is not None and not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must be finite")
    return value


# 1. ClipMetadata (metadata.yaml)


class BoardSpec(Strict):
    type: Literal["charuco"] = "charuco"
    dictionary: str
    squares_x: int = Field(ge=2)
    squares_y: int = Field(ge=2)
    square_length_m: float = Field(gt=0)
    marker_length_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _marker_fits(self) -> BoardSpec:
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m must be smaller than square_length_m")
        return self


class PartHint(Strict):
    label: str
    fixed: bool | None = None
    mass_kg: float | None = Field(default=None, gt=0)


class PartMarkers(Strict):
    dictionary: str
    marker_length_m: float = Field(gt=0)
    assignments: dict[str, int] = {}


class ClipMetadata(Strict):
    contract_version: ContractVersion = "1.0"
    clip_id: str = Field(min_length=1)
    source: Literal["real", "synthetic"]
    mechanism: str
    video: str
    camera_id: str
    intrinsics: str
    board: BoardSpec
    board_orientation: Literal["vertical_arrow_up", "horizontal_face_up"]
    motion_model: Literal["planar", "spatial"]
    plane_offset_m: float
    gravity_W: Vec3 | None = None
    parts_hint: list[PartHint] = []
    part_markers: PartMarkers | None = None
    free_motion_segments_s: list[Segment] = []
    dynamics_gt: bool = False
    notes: str = ""
    extras: dict[str, Any] = {}

    @field_validator("gravity_W")
    @classmethod
    def _gravity_finite(cls, v: list[float] | None) -> list[float] | None:
        return _check_finite(v, "gravity_W")

    @field_validator("free_motion_segments_s")
    @classmethod
    def _segments_ordered(cls, v: list[list[float]]) -> list[list[float]]:
        for t0, t1 in v:
            if not (t0 < t1):
                raise ValueError(f"segment [{t0}, {t1}] must have t0 < t1")
        return v


# 2. CameraIntrinsics (camera_intrinsics.json)


class CameraIntrinsics(Strict):
    contract_version: ContractVersion = "1.0"
    camera_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    K: Mat3
    dist: Vec5
    rms_reprojection_px: float | None = None
    n_frames_used: int | None = None
    source: Literal["calibration", "synthetic"]
    extras: dict[str, Any] = {}

    @model_validator(mode="after")
    def _k_shape(self) -> CameraIntrinsics:
        K = self.K
        flat = [x for row in K for x in row] + list(self.dist)
        if not all(math.isfinite(x) for x in flat):
            raise ValueError("K and dist must be finite")
        if K[2] != [0.0, 0.0, 1.0] or K[1][0] != 0.0:
            raise ValueError("K must be [[fx, s, cx], [0, fy, cy], [0, 0, 1]]")
        if K[0][0] <= 0 or K[1][1] <= 0:
            raise ValueError("focal lengths must be positive")
        return self


# 3. PartTracks (tracks.json + tracks.npz)


class Plane(Strict):
    point_W: Vec3
    normal_W: Vec3

    @field_validator("normal_W")
    @classmethod
    def _unit(cls, v: list[float]) -> list[float]:
        return _check_unit(v, "normal_W")


class TrackCamera(Strict):
    intrinsics: CameraIntrinsics
    static: bool
    T_W_C: Mat4 | None = None

    @model_validator(mode="after")
    def _static_pose(self) -> TrackCamera:
        if self.static and self.T_W_C is None:
            raise ValueError("a static camera needs T_W_C")
        if not self.static and self.T_W_C is not None:
            raise ValueError("T_W_C must be null when the camera is not static")
        return self


class TrackPart(Strict):
    part_id: str
    label: str
    is_base_hint: bool | None = None
    extent_P: Extent | None = None

    @field_validator("part_id")
    @classmethod
    def _pid(cls, v: str) -> str:
        return _check_part_id(v)

    @field_validator("extent_P")
    @classmethod
    def _extent(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is not None:
            lo, hi = v
            _check_finite(lo + hi, "extent_P")
            if any(h < lo_i for lo_i, h in zip(lo, hi, strict=True)):
                raise ValueError("extent_P must be [[min], [max]] with min <= max")
        return v


class PartTracks(Strict):
    contract_version: ContractVersion = "1.0"
    clip_id: str
    source: Literal["perception", "synthetic_gt", "marker_gt"]
    fps_nominal: float = Field(gt=0)
    n_frames: int = Field(ge=1)
    motion_model: Literal["planar", "spatial"]
    plane: Plane | None = None
    gravity_W: Vec3
    camera: TrackCamera
    parts: list[TrackPart]
    free_motion_segments_s: list[Segment] = []
    free_motion_source: Literal["metadata", "llm", "none"] = "none"
    arrays: str = "tracks.npz"
    extras: dict[str, Any] = {}

    @field_validator("gravity_W")
    @classmethod
    def _gravity(cls, v: list[float]) -> list[float]:
        return _check_finite(v, "gravity_W")

    @model_validator(mode="after")
    def _unique_parts(self) -> PartTracks:
        ids = [p.part_id for p in self.parts]
        if len(set(ids)) != len(ids):
            raise ValueError("part ids must be unique")
        return self

    @property
    def part_ids(self) -> list[str]:
        return [p.part_id for p in self.parts]


# 4. KinematicModel (kinematics.json + kinematics.npz)


class Link(Strict):
    part_id: str
    parent_joint_id: str | None = None

    @field_validator("part_id")
    @classmethod
    def _pid(cls, v: str) -> str:
        return _check_part_id(v)

    @field_validator("parent_joint_id")
    @classmethod
    def _jid(cls, v: str | None) -> str | None:
        return None if v is None else _check_joint_id(v)


class JointFit(Strict):
    rms_point_m: float = Field(ge=0)
    n_frames: int = Field(ge=0)


class TypeScores(Strict):
    revolute: float | None = None
    prismatic: float | None = None
    fixed: float | None = None


JointType = Literal["revolute", "prismatic", "fixed"]


class Joint(Strict):
    joint_id: str
    parent_part_id: str
    child_part_id: str
    type: JointType
    axis_W: Vec3
    point_W: Vec3 | None = None
    axis_parent: Vec3
    point_parent: Vec3 | None = None
    q_range_observed: Segment
    fit: JointFit
    type_scores: TypeScores = TypeScores()
    confidence: float = Field(ge=0, le=1)
    extras: dict[str, Any] = {}

    @field_validator("joint_id")
    @classmethod
    def _jid(cls, v: str) -> str:
        return _check_joint_id(v)

    @field_validator("parent_part_id")
    @classmethod
    def _parent(cls, v: str) -> str:
        return v if v == WORLD_ID else _check_part_id(v)

    @field_validator("child_part_id")
    @classmethod
    def _child(cls, v: str) -> str:
        return _check_part_id(v)

    @field_validator("axis_W", "axis_parent")
    @classmethod
    def _axes(cls, v: list[float]) -> list[float]:
        return _check_unit(v, "axis")

    @field_validator("point_W", "point_parent")
    @classmethod
    def _points(cls, v: list[float] | None) -> list[float] | None:
        return _check_finite(v, "point")

    @model_validator(mode="after")
    def _revolute_point(self) -> Joint:
        has_point = self.point_W is not None and self.point_parent is not None
        if self.type == "revolute" and not has_point:
            raise ValueError("revolute joints need point_W and point_parent")
        if self.type != "revolute" and (self.point_W is not None or self.point_parent is not None):
            raise ValueError("point_W and point_parent must be null for non-revolute joints")
        lo, hi = self.q_range_observed
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError("q_range_observed must be finite with lo <= hi")
        return self


class KinematicModel(Strict):
    contract_version: ContractVersion = "1.0"
    clip_id: str
    source: Literal["identify", "synthetic_gt"]
    base_part_id: str
    reference_frame_index: int = Field(ge=0)
    links: list[Link]
    joints: list[Joint]
    arrays: str = "kinematics.npz"
    extras: dict[str, Any] = {}

    @field_validator("base_part_id")
    @classmethod
    def _base(cls, v: str) -> str:
        return v if v == WORLD_ID else _check_part_id(v)

    @model_validator(mode="after")
    def _consistent(self) -> KinematicModel:
        joint_ids = [j.joint_id for j in self.joints]
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("joint ids must be unique")
        link_ids = [link.part_id for link in self.links]
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("link part ids must be unique")
        for link in self.links:
            if link.parent_joint_id is not None and link.parent_joint_id not in joint_ids:
                raise ValueError(f"link {link.part_id} references unknown joint")
        for j in self.joints:
            if j.child_part_id not in link_ids:
                raise ValueError(f"joint {j.joint_id} child {j.child_part_id} is not a link")
            if j.parent_part_id != WORLD_ID and j.parent_part_id not in link_ids:
                raise ValueError(f"joint {j.joint_id} parent {j.parent_part_id} is not a link")
        return self

    @property
    def joint_ids(self) -> list[str]:
        return [j.joint_id for j in self.joints]


# 5. DynamicsModel (dynamics.json + model.xml)


class DynLink(Strict):
    part_id: str
    mass_kg: float
    com_P: Vec3
    inertia_com_P: Mat3
    prior_dominated: list[str] = []

    @field_validator("part_id")
    @classmethod
    def _pid(cls, v: str) -> str:
        return _check_part_id(v)


class DynJoint(Strict):
    joint_id: str
    damping: float
    frictionloss: float
    armature: float
    stiffness: float
    springref: float

    @field_validator("joint_id")
    @classmethod
    def _jid(cls, v: str) -> str:
        return _check_joint_id(v)


class Unidentifiable(Strict):
    combination: str
    reason: str


class Identifiability(Strict):
    rank: int = Field(ge=0)
    n_params: int = Field(ge=0)
    condition_number: float
    unidentifiable: list[Unidentifiable] = []


class DynamicsFit(Strict):
    segments_s: list[Segment]
    replay_rms: dict[str, float]
    window_s: float = Field(gt=0)
    method: str


class DynamicsModel(Strict):
    contract_version: ContractVersion = "1.0"
    clip_id: str
    source: Literal["identify", "synthetic_gt"]
    mjcf: str = "model.xml"
    gravity_W: Vec3
    mass_scale_known: bool
    links: list[DynLink]
    joints: list[DynJoint]
    identifiability: Identifiability
    fit: DynamicsFit
    extras: dict[str, Any] = {}


# 6. Certificate (certificate.json + certificate.md)

CheckStatus = Literal["pass", "warn", "fail", "not_applicable", "not_run"]
Verdict = Literal["VERIFIED", "VERIFIED_WITH_CAVEATS", "REJECTED", "NOT_RUN"]


class Check(Strict):
    check_id: str
    title: str
    subject: str
    status: CheckStatus
    value: float | None = None
    threshold: float | None = None
    unit: str = ""
    evidence: str = ""


class Certificate(Strict):
    contract_version: ContractVersion = "1.0"
    clip_id: str
    verdict: Verdict
    checks: list[Check]
    caveats: list[str] = []
    generated_by: str
    extras: dict[str, Any] = {}


def compute_verdict(checks: list[Check], caveats: list[str]) -> Verdict:
    """Verdict rules, applied in order. Milestone 3 must use this function.

    1. Any fail gives REJECTED.
    2. Otherwise, any warn or any caveat gives VERIFIED_WITH_CAVEATS.
    3. Otherwise, at least one pass gives VERIFIED.
    4. Otherwise, NOT_RUN.
    """
    statuses = [c.status for c in checks]
    if "fail" in statuses:
        return "REJECTED"
    if "warn" in statuses or caveats:
        return "VERIFIED_WITH_CAVEATS"
    if "pass" in statuses:
        return "VERIFIED"
    return "NOT_RUN"
