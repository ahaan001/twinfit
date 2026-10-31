"""`twinfit contracts-check PATH`: validate a clip folder or a run folder, one line per file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from twinfit.contracts import io
from twinfit.contracts.stage import REQUIRED_OUTPUT_KEYS, STAGE_FOLDERS, StageResult


@dataclass
class CheckLine:
    file: str
    ok: bool
    message: str


def _try(lines: list[CheckLine], label: str, fn) -> bool:
    try:
        fn()
    except Exception as e:  # report every failure kind as a line instead of aborting
        lines.append(CheckLine(label, False, str(e).splitlines()[0][:300]))
        return False
    lines.append(CheckLine(label, True, "ok"))
    return True


def check_clip_dir(clip_dir: Path) -> list[CheckLine]:
    clip_dir = Path(clip_dir)
    lines: list[CheckLine] = []
    meta_path = clip_dir / "metadata.yaml"
    holder: dict = {}

    def load_meta():
        holder["meta"] = io.load_metadata(meta_path)

    if not _try(lines, "metadata.yaml", load_meta):
        return lines
    meta = holder["meta"]
    video = clip_dir / meta.video
    lines.append(
        CheckLine(meta.video, video.exists(), "ok" if video.exists() else "video file missing")
    )
    intr = clip_dir / meta.intrinsics
    if intr.exists():
        _try(lines, meta.intrinsics, lambda: io.load_intrinsics(intr))
    else:
        lines.append(CheckLine(meta.intrinsics, False, "intrinsics file missing (calibrate first)"))
    gt = clip_dir / "gt"
    if gt.is_dir():
        if (gt / "camera_extrinsics.json").exists():
            _try(
                lines,
                "gt/camera_extrinsics.json",
                lambda: io.load_pose_json(gt / "camera_extrinsics.json"),
            )
        if (gt / "tracks.json").exists():
            _try(lines, "gt/tracks.json", lambda: io.load_tracks(gt / "tracks.json"))
        if (gt / "kinematics.json").exists():
            _try(lines, "gt/kinematics.json", lambda: io.load_kinematics(gt / "kinematics.json"))
        if (gt / "dynamics.json").exists():
            _try(lines, "gt/dynamics.json", lambda: io.load_dynamics(gt / "dynamics.json"))
        masks = sorted((gt / "masks").glob("*.png")) if (gt / "masks").is_dir() else []
        if masks:
            lines.append(CheckLine("gt/masks", True, f"{len(masks)} label images"))
    return lines


_OUTPUT_LOADERS = {
    "tracks_json": io.load_tracks,
    "kinematics_json": io.load_kinematics,
    "dynamics_json": io.load_dynamics,
    "certificate_json": io.load_certificate,
}


def check_run_dir(run_dir: Path) -> list[CheckLine]:
    run_dir = Path(run_dir)
    lines: list[CheckLine] = []
    run_json = run_dir / "run.json"
    if not _try(lines, "run.json", lambda: json.loads(run_json.read_text())):
        return lines
    for stage, folder in STAGE_FOLDERS.items():
        result_path = run_dir / folder / "stage_result.json"
        if not result_path.exists():
            continue
        try:
            result = StageResult.model_validate_json(result_path.read_text())
        except Exception as e:
            lines.append(CheckLine(f"{folder}/stage_result.json", False, str(e).splitlines()[0]))
            continue
        lines.append(CheckLine(f"{folder}/stage_result.json", True, "ok"))
        if result.status != "ok":
            lines.append(CheckLine(folder, True, f"status {result.status}, outputs not checked"))
            continue
        for key in REQUIRED_OUTPUT_KEYS[stage]:
            if key not in result.outputs:
                lines.append(CheckLine(f"{folder}:{key}", False, "required output key missing"))
        for key, rel in result.outputs.items():
            path = run_dir / rel
            if not path.exists():
                lines.append(CheckLine(rel, False, "output path missing"))
                continue
            loader = _OUTPUT_LOADERS.get(key)
            if loader is not None:
                _try(lines, rel, lambda p=path, f=loader: f(p))
            else:
                lines.append(CheckLine(rel, True, "exists"))
    return lines


def check_path(path: Path) -> list[CheckLine]:
    path = Path(path)
    if (path / "metadata.yaml").exists():
        return check_clip_dir(path)
    if (path / "run.json").exists():
        return check_run_dir(path)
    return [
        CheckLine(
            str(path), False, "neither a clip folder (metadata.yaml) nor a run folder (run.json)"
        )
    ]
