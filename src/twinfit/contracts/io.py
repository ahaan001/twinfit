"""Validating load and save functions for every contract file.

Arrays live in .npz files next to the JSON. On both read and write the arrays are checked for
shape and dtype (float64 or bool), finite values on valid frames, rotation orthonormality
|R^T R - I| < 1e-6 with det(R) > 0, and homogeneous last row [0, 0, 0, 1].
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TypeVar

import numpy as np
import yaml
from pydantic import BaseModel, ValidationError

from twinfit.contracts.models import (
    CameraIntrinsics,
    Certificate,
    ClipMetadata,
    DynamicsModel,
    KinematicModel,
    PartTracks,
)

ORTHO_TOL = 1e-6
M = TypeVar("M", bound=BaseModel)


class ContractError(ValueError):
    """A contract file or array violates the frozen contract."""


# Generic helpers


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an .npz whose bytes depend only on the array contents.

    numpy.savez stamps each member with the current time, so two identical saves differ. This
    writer fixes the zip timestamps so synthetic ground truth is byte-identical across runs.
    """
    path = Path(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, buf.getvalue())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _write_text(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def save_json(path: Path, model: BaseModel) -> None:
    _write_text(path, model.model_dump_json(indent=2) + "\n")


def load_json(path: Path, model_type: type[M]) -> M:
    try:
        return model_type.model_validate_json(Path(path).read_text())
    except ValidationError as e:
        raise ContractError(f"{path}: {e}") from e


def save_yaml(path: Path, model: BaseModel) -> None:
    _write_text(path, yaml.safe_dump(model.model_dump(mode="json"), sort_keys=False))


def load_yaml(path: Path, model_type: type[M]) -> M:
    data = yaml.safe_load(Path(path).read_text())
    try:
        return model_type.model_validate(data)
    except ValidationError as e:
        raise ContractError(f"{path}: {e}") from e


# Array checks


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def check_array(
    arrays: dict[str, np.ndarray], key: str, shape: tuple[int, ...], dtype: type
) -> np.ndarray:
    _require(key in arrays, f"missing array {key!r}")
    a = arrays[key]
    _require(a.shape == shape, f"{key}: shape {a.shape} != {shape}")
    _require(a.dtype == np.dtype(dtype), f"{key}: dtype {a.dtype} != {np.dtype(dtype)}")
    return a


def check_pose_array(T: np.ndarray, valid: np.ndarray | None, name: str) -> None:
    """Checks a (N, 4, 4) float64 pose array; rotation checks apply on valid frames only."""
    _require(T.ndim == 3 and T.shape[1:] == (4, 4), f"{name}: expected (N, 4, 4), got {T.shape}")
    _require(T.dtype == np.float64, f"{name}: dtype must be float64")
    _require(bool(np.all(np.isfinite(T))), f"{name}: poses must be finite")
    last = T[:, 3, :]
    _require(
        bool(np.allclose(last, [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0)),
        f"{name}: last row must be [0, 0, 0, 1]",
    )
    mask = np.ones(len(T), bool) if valid is None else valid
    if not mask.any():
        return
    R = T[mask, :3, :3]
    err = np.abs(np.einsum("nij,nik->njk", R, R) - np.eye(3)).max()
    _require(err < ORTHO_TOL, f"{name}: |R^T R - I| = {err:.2e} exceeds {ORTHO_TOL}")
    _require(bool(np.all(np.linalg.det(R) > 0)), f"{name}: det(R) must be positive")


def check_pose(T: np.ndarray, name: str) -> None:
    check_pose_array(np.asarray(T, dtype=np.float64)[None], None, name)


def _check_time(arrays: dict[str, np.ndarray], n: int) -> None:
    t = check_array(arrays, "t", (n,), np.float64)
    _require(bool(np.all(np.isfinite(t))), "t must be finite")
    _require(bool(np.all(np.diff(t) > 0)), "t must be strictly increasing")


def validate_tracks_arrays(tracks: PartTracks, arrays: dict[str, np.ndarray]) -> None:
    n = tracks.n_frames
    expected = {"t"}
    _check_time(arrays, n)
    if not tracks.camera.static:
        expected.add("T_W_C")
        check_pose_array(check_array(arrays, "T_W_C", (n, 4, 4), np.float64), None, "T_W_C")
    else:
        _require("T_W_C" not in arrays, "T_W_C array is only present for a moving camera")
        check_pose(np.asarray(tracks.camera.T_W_C), "camera.T_W_C")
    for part in tracks.parts:
        pid = part.part_id
        keys = (f"T_W_P__{pid}", f"valid__{pid}", f"quality__{pid}")
        expected.update(keys)
        valid = check_array(arrays, keys[1], (n,), np.bool_)
        check_pose_array(check_array(arrays, keys[0], (n, 4, 4), np.float64), valid, keys[0])
        quality = check_array(arrays, keys[2], (n,), np.float64)
        _require(
            bool(np.all(np.isfinite(quality[valid]))), f"{keys[2]}: must be finite on valid frames"
        )
    extra = set(arrays) - expected
    _require(not extra, f"unexpected arrays in tracks.npz: {sorted(extra)}")


def validate_kinematics_arrays(kin: KinematicModel, arrays: dict[str, np.ndarray]) -> None:
    _require("t" in arrays, "missing array 't'")
    n = int(arrays["t"].shape[0])
    _check_time(arrays, n)
    _require(kin.reference_frame_index < n, "reference_frame_index out of range")
    expected = {"t"}
    for joint in kin.joints:
        jid = joint.joint_id
        valid = check_array(arrays, f"valid__{jid}", (n,), np.bool_)
        expected.add(f"valid__{jid}")
        for prefix in ("q", "qd", "qdd"):
            key = f"{prefix}__{jid}"
            expected.add(key)
            a = check_array(arrays, key, (n,), np.float64)
            _require(bool(np.all(np.isfinite(a[valid]))), f"{key}: must be finite on valid frames")
    extra = set(arrays) - expected
    _require(not extra, f"unexpected arrays in kinematics.npz: {sorted(extra)}")


# Per-file API


def load_metadata(path: Path) -> ClipMetadata:
    return load_yaml(path, ClipMetadata)


def save_metadata(path: Path, metadata: ClipMetadata) -> None:
    save_yaml(path, metadata)


def load_intrinsics(path: Path) -> CameraIntrinsics:
    return load_json(path, CameraIntrinsics)


def save_intrinsics(path: Path, intrinsics: CameraIntrinsics) -> None:
    save_json(path, intrinsics)


def load_tracks(json_path: Path) -> tuple[PartTracks, dict[str, np.ndarray]]:
    json_path = Path(json_path)
    tracks = load_json(json_path, PartTracks)
    arrays = load_npz(json_path.parent / tracks.arrays)
    validate_tracks_arrays(tracks, arrays)
    return tracks, arrays


def save_tracks(json_path: Path, tracks: PartTracks, arrays: dict[str, np.ndarray]) -> None:
    validate_tracks_arrays(tracks, arrays)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(json_path.parent / tracks.arrays, arrays)
    save_json(json_path, tracks)


def load_kinematics(json_path: Path) -> tuple[KinematicModel, dict[str, np.ndarray]]:
    json_path = Path(json_path)
    kin = load_json(json_path, KinematicModel)
    arrays = load_npz(json_path.parent / kin.arrays)
    validate_kinematics_arrays(kin, arrays)
    return kin, arrays


def save_kinematics(json_path: Path, kin: KinematicModel, arrays: dict[str, np.ndarray]) -> None:
    validate_kinematics_arrays(kin, arrays)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(json_path.parent / kin.arrays, arrays)
    save_json(json_path, kin)


def validate_mjcf(dyn: DynamicsModel, xml_path: Path) -> None:
    """Checks model.xml against the DynamicsModel rules in the contract."""
    import mujoco

    xml_path = Path(xml_path)
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:  # mujoco raises ValueError with a parser message
        raise ContractError(f"{xml_path}: MuJoCo cannot load it: {e}") from e
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
    ]
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    link_ids = {link.part_id for link in dyn.links}
    _require(
        link_ids <= set(body_names), f"links {sorted(link_ids - set(body_names))} have no body"
    )
    _require(
        set(joint_names) == {j.joint_id for j in dyn.joints},
        f"joint names {sorted(joint_names)} != dynamics joints",
    )
    for i, name in enumerate(body_names, start=1):
        if name not in link_ids:
            _require(
                model.body_jntnum[i] == 0,
                f"body {name} has joints but is not in links; only a welded base may be absent",
            )
            _require(model.body_parentid[i] == 0, f"welded base {name} must be a child of world")
    _require(
        bool(np.all(model.geom_contype == 0)) and bool(np.all(model.geom_conaffinity == 0)),
        "geoms must have contype=0 and conaffinity=0",
    )
    _require(
        bool(np.allclose(model.opt.gravity, dyn.gravity_W, atol=1e-9)),
        f"<option gravity> {model.opt.gravity} != gravity_W {dyn.gravity_W}",
    )
    root = ET.parse(xml_path).getroot()
    for body in root.iter("body"):
        name = body.get("name")
        if name in link_ids:
            inertial = body.find("inertial")
            _require(
                inertial is not None and inertial.get("fullinertia") is not None,
                f"body {name} needs an explicit <inertial fullinertia=...>",
            )


def load_dynamics(json_path: Path) -> tuple[DynamicsModel, Path]:
    json_path = Path(json_path)
    dyn = load_json(json_path, DynamicsModel)
    xml_path = json_path.parent / dyn.mjcf
    _require(xml_path.exists(), f"{xml_path} is missing")
    validate_mjcf(dyn, xml_path)
    return dyn, xml_path


def save_dynamics(json_path: Path, dyn: DynamicsModel, model_xml: str) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path = json_path.parent / dyn.mjcf
    xml_path.write_text(model_xml)
    validate_mjcf(dyn, xml_path)
    save_json(json_path, dyn)


def load_certificate(json_path: Path) -> Certificate:
    return load_json(json_path, Certificate)


def save_certificate(json_path: Path, cert: Certificate) -> None:
    save_json(json_path, cert)


def load_pose_json(path: Path, key: str = "T_W_C") -> np.ndarray:
    """Reads a {"T_W_C": 4x4} style file and validates the pose."""
    data = json.loads(Path(path).read_text())
    _require(isinstance(data, dict) and key in data, f"{path}: expected a {key!r} entry")
    T = np.asarray(data[key], dtype=np.float64)
    _require(T.shape == (4, 4), f"{path}: {key} must be 4x4")
    check_pose(T, key)
    return T


def save_pose_json(path: Path, T: np.ndarray, key: str = "T_W_C") -> None:
    T = np.asarray(T, dtype=np.float64)
    check_pose(T, key)
    _write_text(path, json.dumps({key: T.tolist()}, indent=2) + "\n")
