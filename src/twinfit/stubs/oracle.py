"""Oracle stage stubs with the same signatures as the real stages.

Each stub reads the clip's `gt/` folder and writes the stage's contract files into
`ctx.stage_dir`. Real stages never read `gt/`; these stubs exist so that every milestone can
develop and test against perfect upstream outputs before the other milestones land.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import numpy as np

from twinfit.contracts import (
    Certificate,
    CertifyOptions,
    Check,
    DynamicsOptions,
    KinematicsOptions,
    PerceptionOptions,
    StageContext,
    StageResult,
    compute_verdict,
    load_tracks,
    save_certificate,
    save_tracks,
)
from twinfit.contracts.stage import STAGE_FOLDERS
from twinfit.geometry.se3 import so3_exp

CHECK_IDS = {
    "C01": "schema_valid",
    "C02": "tree_structure",
    "C03": "axis_wellformed",
    "C04": "kinematic_reconstruction",
    "C05": "track_quality",
    "C06": "mass_positive",
    "C07": "inertia_pd",
    "C08": "triangle_inequality",
    "C09": "com_inside_extent",
    "C10": "replay_windowed",
    "C11": "passivity_energy",
    "C12": "sim_stability",
    "C13": "identifiability",
    "C14": "singularity_margin",
    "C15": "cross_engine",
    "C16": "rest_pose",
}


def _rel(ctx: StageContext, path: Path) -> str:
    return str(Path(path).relative_to(ctx.run_dir))


def _gt(ctx: StageContext) -> Path | None:
    gt = ctx.clip_dir / "gt"
    return gt if gt.is_dir() else None


def _failed(stage, message: str) -> StageResult:
    return StageResult(stage=stage, status="failed", impl="stub", messages=[message])


def _missing_gt(stage, ctx: StageContext) -> StageResult:
    return _failed(stage, f"oracle stub needs {ctx.clip_dir / 'gt'}; synthetic clips only")


def perceive(ctx: StageContext, options: PerceptionOptions | None) -> StageResult:
    """Copies gt/tracks.*; with `oracle_noise` set, perturbs every part pose independently per
    frame: translation N(0, trans_m) per W axis and a rotation by an angle N(0, rot_rad) about a
    uniformly random axis, applied on the left (in W). Quality is set to trans_m on noisy copies."""
    gt = _gt(ctx)
    if gt is None or not (gt / "tracks.json").exists():
        return _missing_gt("perceive", ctx)
    tracks, arrays = load_tracks(gt / "tracks.json")
    messages = []
    noise = options.oracle_noise if options is not None else None
    if noise is not None and (noise.trans_m > 0 or noise.rot_rad > 0):
        rng = np.random.default_rng(noise.seed)
        for part in tracks.parts:
            T = arrays[f"T_W_P__{part.part_id}"].copy()
            n = len(T)
            trans = (
                rng.normal(scale=noise.trans_m, size=(n, 3))
                if noise.trans_m > 0
                else np.zeros((n, 3))
            )
            axes = rng.normal(size=(n, 3))
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            angles = rng.normal(scale=noise.rot_rad, size=n) if noise.rot_rad > 0 else np.zeros(n)
            for k in range(n):
                T[k, :3, :3] = so3_exp(angles[k] * axes[k]) @ T[k, :3, :3]
                T[k, :3, 3] += trans[k]
            arrays[f"T_W_P__{part.part_id}"] = T
            q = arrays[f"quality__{part.part_id}"].copy()
            q[arrays[f"valid__{part.part_id}"]] = noise.trans_m
            arrays[f"quality__{part.part_id}"] = q
        tracks = tracks.model_copy(
            update={"extras": {**tracks.extras, "oracle_noise": noise.model_dump()}}
        )
        messages.append(
            f"oracle noise trans {noise.trans_m} m, rot {noise.rot_rad} rad, seed {noise.seed}"
        )
    out_json = ctx.stage_dir / "tracks.json"
    save_tracks(out_json, tracks, arrays)
    return StageResult(
        stage="perceive",
        status="ok",
        impl="stub",
        outputs={
            "tracks_json": _rel(ctx, out_json),
            "tracks_npz": _rel(ctx, ctx.stage_dir / tracks.arrays),
        },
        metrics={"n_parts": float(len(tracks.parts)), "n_frames": float(tracks.n_frames)},
        messages=messages,
    )


def _copy(ctx: StageContext, names: list[str]) -> dict[str, Path]:
    gt = ctx.clip_dir / "gt"
    out = {}
    for name in names:
        shutil.copyfile(gt / name, ctx.stage_dir / name)
        out[name] = ctx.stage_dir / name
    return out


def kinematics(ctx: StageContext, options: KinematicsOptions | None) -> StageResult:
    gt = _gt(ctx)
    if gt is None or not (gt / "kinematics.json").exists():
        return _missing_gt("kinematics", ctx)
    files = _copy(ctx, ["kinematics.json", "kinematics.npz"])
    return StageResult(
        stage="kinematics",
        status="ok",
        impl="stub",
        outputs={
            "kinematics_json": _rel(ctx, files["kinematics.json"]),
            "kinematics_npz": _rel(ctx, files["kinematics.npz"]),
        },
    )


def dynamics(ctx: StageContext, options: DynamicsOptions | None) -> StageResult:
    gt = _gt(ctx)
    if gt is None or not (gt / "dynamics.json").exists():
        if gt is not None and not ctx.metadata.dynamics_gt:
            return StageResult(
                stage="dynamics",
                status="skipped",
                impl="stub",
                messages=["clip has no dynamics ground truth (dynamics_gt false)"],
            )
        return _missing_gt("dynamics", ctx)
    files = _copy(ctx, ["dynamics.json", "model.xml"])
    return StageResult(
        stage="dynamics",
        status="ok",
        impl="stub",
        outputs={
            "dynamics_json": _rel(ctx, files["dynamics.json"]),
            "model_xml": _rel(ctx, files["model.xml"]),
        },
    )


def certify(ctx: StageContext, options: CertifyOptions | None) -> StageResult:
    checks = [
        Check(check_id=cid, title=title, subject="run", status="not_run")
        for cid, title in CHECK_IDS.items()
    ]
    cert = Certificate(
        clip_id=ctx.metadata.clip_id,
        verdict=compute_verdict(checks, []),
        checks=checks,
        caveats=[],
        generated_by="twinfit.stubs.oracle.certify",
    )
    json_path = ctx.stage_dir / "certificate.json"
    md_path = ctx.stage_dir / "certificate.md"
    save_certificate(json_path, cert)
    lines = [
        f"# Certificate for {cert.clip_id}",
        "",
        f"Verdict: {cert.verdict}",
        "",
        "| Check | Title | Status |",
        "|---|---|---|",
    ]
    lines += [f"| {c.check_id} | {c.title} | {c.status} |" for c in checks]
    md_path.write_text("\n".join(lines) + "\n")
    return StageResult(
        stage="certify",
        status="ok",
        impl="stub",
        outputs={"certificate_json": _rel(ctx, json_path), "certificate_md": _rel(ctx, md_path)},
        messages=["oracle stub: all checks not_run"],
    )


def report(ctx: StageContext, options=None) -> StageResult:
    rows = []
    for stage, result in ctx.upstream.items():
        outputs = ", ".join(f"{k}: {v}" for k, v in result.outputs.items())
        rows.append(
            f"<tr><td>{stage}</td><td>{result.status}</td><td>{result.impl}</td>"
            f"<td>{html.escape(outputs)}</td>"
            f"<td>{html.escape('; '.join(result.messages))}</td></tr>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Twinfit run</title>"
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #999;padding:4px 8px;text-align:left}</style></head><body>"
        f"<h1>Twinfit run: {html.escape(ctx.metadata.clip_id)}</h1>"
        "<p>Oracle stub report. Stage results:</p>"
        "<table><tr><th>Stage</th><th>Status</th><th>Impl</th><th>Outputs</th><th>Messages</th></tr>"
        + "".join(rows)
        + "</table></body></html>\n"
    )
    out = ctx.stage_dir / "index.html"
    out.write_text(page)
    return StageResult(
        stage="report", status="ok", impl="stub", outputs={"report_html": _rel(ctx, out)}
    )


STUBS = {
    "perceive": perceive,
    "kinematics": kinematics,
    "dynamics": dynamics,
    "certify": certify,
    "report": report,
}
__all__ = ["STUBS", "STAGE_FOLDERS", "certify", "dynamics", "kinematics", "perceive", "report"]
