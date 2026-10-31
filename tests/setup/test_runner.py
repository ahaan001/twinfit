import json
from pathlib import Path

import numpy as np
import pytest

from twinfit.contracts import (
    OracleNoise,
    PerceptionOptions,
    RunOptions,
    load_certificate,
    load_tracks,
)
from twinfit.contracts.check import check_path
from twinfit.runner import registry, run_pipeline
from twinfit.synth.generate import SuiteSettings, generate_clip

TINY = SuiteSettings((0,), 64, 36, 10.0, 1.0)


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clip") / "pendulum_1r__s0"
    return generate_clip("pendulum_1r", 0, out, TINY)


def test_resolve_auto_falls_back_to_stub_and_real_raises_when_absent():
    fn, impl = registry.resolve("perceive", "auto")
    assert impl == "stub" and fn is registry.oracle.perceive
    fn, impl = registry.resolve("certify", "stub")
    assert impl == "stub"
    with pytest.raises(ModuleNotFoundError):
        registry.resolve("kinematics", "real")
    with pytest.raises(ValueError):
        registry.resolve("kinematics", "maybe")


def test_resolve_propagates_other_import_errors(monkeypatch):
    monkeypatch.setitem(registry.REAL_IMPLS, "report", "tests.setup.broken_stage:run")
    with pytest.raises(ModuleNotFoundError) as e:
        registry.resolve("report", "auto")
    assert e.value.name == "nonexistent_dependency_xyz"


def test_stub_pipeline_completes_and_passes_contracts_check(tiny_clip, tmp_path):
    results = run_pipeline(
        tiny_clip,
        "run1",
        {s: "stub" for s in registry.REAL_IMPLS},
        RunOptions(),
        runs_root=tmp_path,
    )
    assert [r.status for r in results] == ["ok"] * 5
    assert [r.impl for r in results] == ["stub"] * 5
    run_dir = tmp_path / "run1"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["clip_id"] == "pendulum_1r__s0" and set(run["impl"]) == set(registry.REAL_IMPLS)
    assert run["started"] <= run["ended"] and run["total_cost_usd"] == 0.0
    lines = check_path(run_dir)
    assert lines and all(line.ok for line in lines), [
        (x.file, x.message) for x in lines if not x.ok
    ]
    for folder in ("01_perceive", "02_kinematics", "03_dynamics", "04_certify", "05_report"):
        assert (run_dir / folder / "stage_result.json").exists()
    cert = load_certificate(run_dir / "04_certify" / "certificate.json")
    assert cert.verdict == "NOT_RUN" and all(c.status == "not_run" for c in cert.checks)
    assert "pendulum_1r__s0" in (run_dir / "05_report" / "index.html").read_text()


def test_oracle_noise_perturbs_tracks_deterministically(tiny_clip, tmp_path):
    noise = OracleNoise(trans_m=0.002, rot_rad=0.01, seed=3)
    opts = RunOptions(perceive=PerceptionOptions(oracle_noise=noise))
    run_pipeline(
        tiny_clip, "a", {"perceive": "stub"}, opts, stop_after="perceive", runs_root=tmp_path
    )
    run_pipeline(
        tiny_clip, "b", {"perceive": "stub"}, opts, stop_after="perceive", runs_root=tmp_path
    )
    _, gt = load_tracks(tiny_clip / "gt" / "tracks.json")
    ta, a = load_tracks(tmp_path / "a" / "01_perceive" / "tracks.json")
    _, b = load_tracks(tmp_path / "b" / "01_perceive" / "tracks.json")
    d = a["T_W_P__p1"][:, :3, 3] - gt["T_W_P__p1"][:, :3, 3]
    assert 0.0005 < np.sqrt((d**2).mean()) < 0.005
    np.testing.assert_array_equal(a["T_W_P__p1"], b["T_W_P__p1"])
    assert ta.extras["oracle_noise"]["seed"] == 3
    assert not (tmp_path / "a" / "02_kinematics" / "stage_result.json").exists()


def test_missing_gt_fails_and_skips_downstream(tmp_path, tiny_clip):
    import shutil

    clip = tmp_path / "clip"
    shutil.copytree(tiny_clip, clip)
    shutil.rmtree(clip / "gt")
    results = run_pipeline(
        clip,
        "r",
        {s: "stub" for s in registry.REAL_IMPLS},
        RunOptions(),
        runs_root=tmp_path / "runs",
    )
    assert results[0].status == "failed" and "gt" in results[0].messages[0]
    assert [r.status for r in results[1:]] == ["skipped"] * 4


def test_crashing_stage_becomes_failed_result(tiny_clip, tmp_path, monkeypatch):
    def boom(ctx, options):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(registry.oracle.STUBS, "kinematics", boom)
    results = run_pipeline(
        tiny_clip, "r", {s: "stub" for s in registry.REAL_IMPLS}, RunOptions(), runs_root=tmp_path
    )
    assert results[1].status == "failed" and "kaboom" in results[1].messages[0]
    assert results[2].status == "skipped"


def test_default_run_id_format():
    from twinfit.runner import default_run_id

    rid = default_run_id("clip")
    assert rid.startswith("clip__") and len(rid.split("__")[1]) == 15 and "T" in rid
    assert Path(rid).name == rid
