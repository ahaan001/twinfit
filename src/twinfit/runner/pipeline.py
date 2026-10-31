"""Runs the five stages in order and writes `runs/<run_id>/`."""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from twinfit.contracts import RunOptions, StageContext, StageResult, load_metadata
from twinfit.contracts.stage import REQUIRED_OUTPUT_KEYS, STAGE_FOLDERS, STAGES, StageName
from twinfit.paths import repo_root, runs_dir
from twinfit.runner.registry import resolve
from twinfit.session import load_session


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def default_run_id(clip_id: str) -> str:
    return f"{clip_id}__{utc_stamp()}"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_stage_result(stage_dir: Path, result: StageResult) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage_result.json").write_text(result.model_dump_json(indent=2) + "\n")


def _check_outputs(run_dir: Path, result: StageResult) -> StageResult:
    if result.status != "ok":
        return result
    missing = [k for k in REQUIRED_OUTPUT_KEYS[result.stage] if k not in result.outputs]
    absent = [v for v in result.outputs.values() if not (run_dir / v).exists()]
    if missing or absent:
        msg = f"stage reported ok but required outputs are missing: keys {missing}, paths {absent}"
        return result.model_copy(update={"status": "failed", "messages": [*result.messages, msg]})
    return result


def run_pipeline(
    clip_dir: Path,
    run_id: str | None = None,
    impl_map: dict[StageName, str] | None = None,
    options: RunOptions | None = None,
    stop_after: StageName | None = None,
    runs_root: Path | None = None,
) -> list[StageResult]:
    """Runs perceive, kinematics, dynamics, certify and report on one clip.

    `impl_map` gives auto, real or stub per stage (default auto). A failed stage marks all later
    stages as skipped. Each stage writes only inside its own folder; `run.json` records the clip,
    the implementation used per stage, the git commit, start and end times and the total cost."""
    clip_dir = Path(clip_dir).resolve()
    metadata = load_metadata(clip_dir / "metadata.yaml")
    options = options or RunOptions()
    impl_map = {s: (impl_map or {}).get(s, "auto") for s in STAGES}
    run_id = run_id or default_run_id(metadata.clip_id)
    run_dir = (Path(runs_root) if runs_root is not None else runs_dir()) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = load_session().tag
    started = datetime.now(UTC).isoformat()
    results: list[StageResult] = []
    upstream: dict[StageName, StageResult] = {}
    impls_used: dict[str, str] = {}
    failed = False
    for stage in STAGES:
        stage_dir = run_dir / STAGE_FOLDERS[stage]
        stage_dir.mkdir(parents=True, exist_ok=True)
        fn, impl = resolve(stage, impl_map[stage])
        impls_used[stage] = impl
        if failed:
            result = StageResult(
                stage=stage, status="skipped", impl=impl, messages=["an earlier stage failed"]
            )
        else:
            ctx = StageContext(
                clip_dir=clip_dir,
                run_dir=run_dir,
                stage_dir=stage_dir,
                metadata=metadata,
                upstream=dict(upstream),
                cost_tag=tag,
            )
            t0 = time.perf_counter()
            try:
                result = fn(ctx, options.for_stage(stage))
                if result.stage != stage:
                    raise ValueError(f"stage {stage} returned a result for {result.stage}")
            except Exception as e:  # a crashing stage becomes a failed result
                result = StageResult(
                    stage=stage,
                    status="failed",
                    impl=impl,
                    messages=[f"{type(e).__name__}: {e}", traceback.format_exc()],
                )
            if result.duration_s == 0.0:
                result = result.model_copy(update={"duration_s": time.perf_counter() - t0})
            result = _check_outputs(run_dir, result)
            if result.status == "failed":
                failed = True
        write_stage_result(stage_dir, result)
        results.append(result)
        upstream[stage] = result
        if stage == stop_after:
            break
    run_json = {
        "run_id": run_id,
        "clip_dir": str(clip_dir),
        "clip_id": metadata.clip_id,
        "impl": impls_used,
        "git_commit": git_commit(),
        "cost_tag": tag,
        "started": started,
        "ended": datetime.now(UTC).isoformat(),
        "total_cost_usd": float(sum(r.cost_usd for r in results)),
        "stop_after": stop_after,
        "options": options.model_dump(mode="json"),
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2) + "\n")
    return results
