import numpy as np
import pytest
from pydantic import ValidationError

from twinfit.contracts import (
    Certificate,
    Check,
    ContractError,
    PartTracks,
    RunOptions,
    compute_verdict,
    load_intrinsics,
    load_kinematics,
    load_metadata,
    load_tracks,
    save_intrinsics,
    save_kinematics,
    save_metadata,
    save_tracks,
)
from twinfit.contracts.io import load_npz, save_npz

from .helpers import intrinsics, kinematics, metadata, tracks


def test_metadata_round_trip(tmp_path):
    meta = metadata()
    save_metadata(tmp_path / "metadata.yaml", meta)
    assert load_metadata(tmp_path / "metadata.yaml") == meta


def test_intrinsics_round_trip(tmp_path):
    intr = intrinsics()
    save_intrinsics(tmp_path / "camera_intrinsics.json", intr)
    assert load_intrinsics(tmp_path / "camera_intrinsics.json") == intr


@pytest.mark.parametrize("static", [True, False])
def test_tracks_round_trip(tmp_path, static):
    model, arrays = tracks(static_camera=static)
    save_tracks(tmp_path / "tracks.json", model, arrays)
    loaded, loaded_arrays = load_tracks(tmp_path / "tracks.json")
    assert loaded == model
    assert set(loaded_arrays) == set(arrays)
    for k in arrays:
        np.testing.assert_array_equal(loaded_arrays[k], arrays[k])


def test_kinematics_round_trip(tmp_path):
    model, arrays = kinematics()
    save_kinematics(tmp_path / "kinematics.json", model, arrays)
    loaded, loaded_arrays = load_kinematics(tmp_path / "kinematics.json")
    assert loaded == model
    for k in arrays:
        np.testing.assert_array_equal(loaded_arrays[k], arrays[k])


def test_extra_fields_are_rejected():
    data = metadata().model_dump()
    data["unknown"] = 1
    with pytest.raises(ValidationError):
        metadata().__class__.model_validate(data)


def test_extras_dict_is_free_form():
    meta = metadata().model_copy(update={"extras": {"anything": [1, 2, {"x": "y"}]}})
    assert meta.extras["anything"][2]["x"] == "y"


def _corrupt(arrays, key, fn):
    out = {k: v.copy() for k, v in arrays.items()}
    out[key] = fn(out[key])
    return out


@pytest.mark.parametrize(
    "key, mutate, reason",
    [
        ("T_W_P__p1", lambda a: a[:-1], "wrong length"),
        ("T_W_P__p1", lambda a: a.astype(np.float32), "wrong dtype"),
        ("T_W_P__p1", lambda a: _set(a, (3, 0, 0), np.nan), "non-finite"),
        ("T_W_P__p1", lambda a: _set(a, (3, 3, 0), 1.0), "bad last row"),
        ("T_W_P__p1", lambda a: _scale_rot(a, 1.001), "not orthonormal"),
        ("T_W_P__p1", lambda a: _flip_det(a), "negative determinant"),
        ("valid__p1", lambda a: a.astype(np.int64), "valid not bool"),
        ("quality__p1", lambda a: _set(a, (2,), np.nan), "quality NaN on a valid frame"),
        ("t", lambda a: a[::-1].copy(), "time not increasing"),
    ],
)
def test_malformed_tracks_arrays_rejected(tmp_path, key, mutate, reason):
    model, arrays = tracks()
    bad = _corrupt(arrays, key, mutate)
    with pytest.raises(ContractError):
        save_tracks(tmp_path / "tracks.json", model, bad)


def test_unexpected_array_rejected(tmp_path):
    model, arrays = tracks()
    arrays["T_W_C"] = arrays["T_W_P__p0"].copy()
    with pytest.raises(ContractError):
        save_tracks(tmp_path / "tracks.json", model, arrays)


def test_invalid_frames_may_hold_non_orthonormal_but_finite_pose(tmp_path):
    model, arrays = tracks()
    arrays["valid__p1"][3] = False
    arrays["T_W_P__p1"][3, :3, :3] *= 2.0
    arrays["quality__p1"][3] = np.nan
    save_tracks(tmp_path / "tracks.json", model, arrays)
    load_tracks(tmp_path / "tracks.json")


def test_malformed_kinematics_arrays_rejected(tmp_path):
    model, arrays = kinematics()
    bad = _corrupt(arrays, "q__j0", lambda a: a[:-2])
    with pytest.raises(ContractError):
        save_kinematics(tmp_path / "kinematics.json", model, bad)
    bad = _corrupt(arrays, "qd__j0", lambda a: _set(a, (1,), np.inf))
    with pytest.raises(ContractError):
        save_kinematics(tmp_path / "kinematics.json", model, bad)


def test_joint_validation():
    model, _ = kinematics()
    j = model.joints[0]
    with pytest.raises(ValidationError):
        j.model_copy(update={"axis_W": [0.0, 0.0, -2.0]}).model_validate(
            j.model_copy(update={"axis_W": [0.0, 0.0, -2.0]}).model_dump()
        )
    with pytest.raises(ValidationError):
        j.__class__.model_validate({**j.model_dump(), "type": "prismatic"})
    with pytest.raises(ValidationError):
        j.__class__.model_validate({**j.model_dump(), "joint_id": "joint0"})


def test_static_camera_requires_pose():
    model, _ = tracks()
    data = model.model_dump()
    data["camera"]["T_W_C"] = None
    with pytest.raises(ValidationError):
        PartTracks.model_validate(data)


def test_compute_verdict_rules():
    def chk(status):
        return Check(check_id="C", title="t", subject="s", status=status)

    assert compute_verdict([chk("pass"), chk("fail")], []) == "REJECTED"
    assert compute_verdict([chk("pass"), chk("warn")], []) == "VERIFIED_WITH_CAVEATS"
    assert compute_verdict([chk("pass")], ["c"]) == "VERIFIED_WITH_CAVEATS"
    assert compute_verdict([chk("pass"), chk("not_applicable")], []) == "VERIFIED"
    assert compute_verdict([chk("not_run"), chk("not_applicable")], []) == "NOT_RUN"
    assert compute_verdict([], []) == "NOT_RUN"
    Certificate(clip_id="c", verdict="NOT_RUN", checks=[chk("not_run")], generated_by="test")


def test_run_options_defaults():
    opts = RunOptions.model_validate({"dynamics": {"fit_stiffness": True}})
    assert opts.for_stage("dynamics").fit_stiffness is True
    assert opts.for_stage("perceive").frame_stride == 1
    assert opts.for_stage("report") is None
    with pytest.raises(ValidationError):
        RunOptions.model_validate({"dynamics": {"unknown": 1}})


def test_npz_is_byte_identical(tmp_path):
    arrays = {"a": np.arange(5.0), "b": np.ones(3, bool)}
    save_npz(tmp_path / "x.npz", arrays)
    first = (tmp_path / "x.npz").read_bytes()
    save_npz(tmp_path / "x.npz", arrays)
    assert (tmp_path / "x.npz").read_bytes() == first
    loaded = load_npz(tmp_path / "x.npz")
    np.testing.assert_array_equal(loaded["a"], arrays["a"])


def _set(a, idx, value):
    a = a.copy()
    a[idx] = value
    return a


def _scale_rot(a, s):
    a = a.copy()
    a[2, :3, :3] *= s
    return a


def _flip_det(a):
    a = a.copy()
    a[2, :3, 0] *= -1
    return a
