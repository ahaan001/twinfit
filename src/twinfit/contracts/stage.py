"""Stage interface: names, folders, context, result and options.

Every stage implementation exposes `run(ctx: StageContext, options) -> StageResult` and writes
only inside `ctx.stage_dir`. Real implementations must never read `ctx.clip_dir / "gt"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from twinfit.contracts.models import ClipMetadata, Segment

StageName = Literal["perceive", "kinematics", "dynamics", "certify", "report"]
STAGES: tuple[StageName, ...] = ("perceive", "kinematics", "dynamics", "certify", "report")
STAGE_FOLDERS: dict[StageName, str] = {
    "perceive": "01_perceive",
    "kinematics": "02_kinematics",
    "dynamics": "03_dynamics",
    "certify": "04_certify",
    "report": "05_report",
}
REQUIRED_OUTPUT_KEYS: dict[StageName, tuple[str, ...]] = {
    "perceive": ("tracks_json", "tracks_npz"),
    "kinematics": ("kinematics_json", "kinematics_npz"),
    "dynamics": ("dynamics_json", "model_xml"),
    "certify": ("certificate_json", "certificate_md"),
    "report": ("report_html",),
}
OPTIONAL_OUTPUT_KEYS: dict[StageName, tuple[str, ...]] = {
    "perceive": ("masks_dir", "overlay_mp4", "perception_report"),
    "kinematics": (),
    "dynamics": (),
    "certify": (),
    "report": (),
}


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: StageName
    status: Literal["ok", "failed", "skipped"]
    impl: Literal["real", "stub"]
    outputs: dict[str, str] = {}
    metrics: dict[str, float] = {}
    messages: list[str] = []
    cost_usd: float = 0.0
    duration_s: float = 0.0


@dataclass(frozen=True)
class StageContext:
    clip_dir: Path
    run_dir: Path
    stage_dir: Path
    metadata: ClipMetadata
    upstream: dict[StageName, StageResult]
    cost_tag: str

    def upstream_path(self, stage: StageName, key: str) -> Path:
        """Absolute path of an upstream output. Raises KeyError when it is absent."""
        return self.run_dir / self.upstream[stage].outputs[key]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoxPrompt(Options):
    label: str
    frame_index: int = Field(ge=0)
    box_xyxy_px: Annotated[list[float], Field(min_length=4, max_length=4)]


class OracleNoise(Options):
    trans_m: float = Field(ge=0)
    rot_rad: float = Field(ge=0)
    seed: int = 0


class PerceptionOptions(Options):
    part_prompts: list[str] | None = None
    box_prompts: list[BoxPrompt] | None = None
    frame_stride: int = Field(default=1, ge=1)
    max_frames: int | None = Field(default=None, ge=1)
    exclude_labels: list[str] = []
    use_part_markers: bool = False
    llm_part_proposal: bool = True
    oracle_noise: OracleNoise | None = None


class KinematicsOptions(Options):
    force_joint_types: dict[str, Literal["revolute", "prismatic", "fixed"]] = {}
    force_base_part: str | None = None
    exclude_time_ranges_s: list[Segment] = []
    planar_constraint: bool | None = None


class DynamicsOptions(Options):
    segments_s: list[Segment] | None = None
    fit_stiffness: bool = False
    fit_coulomb: bool = True
    fix_params: dict[str, float] = {}
    mass_kg: dict[str, float] = {}
    window_s: float = Field(default=0.5, gt=0)


class CertifyOptions(Options):
    thresholds_override: dict[str, float] = {}


class RunOptions(Options):
    perceive: PerceptionOptions | None = None
    kinematics: KinematicsOptions | None = None
    dynamics: DynamicsOptions | None = None
    certify: CertifyOptions | None = None

    def for_stage(self, stage: StageName) -> Any:
        """The options object passed to a stage; a default instance when unset, None for report."""
        defaults = {
            "perceive": PerceptionOptions,
            "kinematics": KinematicsOptions,
            "dynamics": DynamicsOptions,
            "certify": CertifyOptions,
        }
        if stage == "report":
            return None
        value = getattr(self, stage)
        return value if value is not None else defaults[stage]()
