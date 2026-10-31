"""Evaluation metrics shared by all milestones. Each function states its exact definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from twinfit.contracts import KinematicModel, PartTracks
from twinfit.contracts.conventions import WORLD_ID
from twinfit.geometry.lines import line_distance
from twinfit.geometry.se3 import inverse

Tracks = tuple[PartTracks, dict[str, np.ndarray]]
Kinematics = tuple[KinematicModel, dict[str, np.ndarray]]


def axis_angle_error_deg(a_est, a_gt) -> float:
    """degrees(arccos(|a_est · a_gt|)) with both vectors normalised; sign-agnostic."""
    a = np.asarray(a_est, float).reshape(3)
    b = np.asarray(a_gt, float).reshape(3)
    c = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def line_distance_m(p1, a1, p2, a2) -> float:
    """Shortest distance between the lines (p1, a1) and (p2, a2); see geometry.lines."""
    return line_distance(p1, a1, p2, a2)


def extent_corners(extent_P) -> np.ndarray:
    lo, hi = np.asarray(extent_P, float)
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def default_points_W_ref(T_gt_ref: np.ndarray, extent_P_gt=None) -> np.ndarray:
    """The 8 corners of the GT extent at the reference frame mapped to W, or the corners of a
    0.1 m cube centred on the GT frame origin when no extent is known."""
    corners = (
        extent_corners(extent_P_gt)
        if extent_P_gt is not None
        else extent_corners([[-0.05] * 3, [0.05] * 3])
    )
    return corners @ T_gt_ref[:3, :3].T + T_gt_ref[:3, 3]


def motion_point_rms_m(
    T_est: np.ndarray,
    T_gt: np.ndarray,
    valid: np.ndarray,
    points_W_ref: np.ndarray | None = None,
    ref_index: int | None = None,
    extent_P_gt=None,
) -> float:
    """RMS distance between points moved by the two pose sequences.

    Each sequence moves the reference points with its own displacement relative to its reference
    pose: X(t) = T(t) T(ref)^-1 X(ref). The RMS is over all points and all valid frames. `ref_index`
    defaults to the first valid frame. `points_W_ref` defaults to `default_points_W_ref`."""
    valid = np.asarray(valid, bool)
    if not valid.any():
        return float("nan")
    if ref_index is None:
        ref_index = int(np.argmax(valid))
    if points_W_ref is None:
        points_W_ref = default_points_W_ref(T_gt[ref_index], extent_P_gt)
    X = np.hstack([points_W_ref, np.ones((len(points_W_ref), 1))])
    D_est = T_est[valid] @ inverse(T_est[ref_index])
    D_gt = T_gt[valid] @ inverse(T_gt[ref_index])
    X_est = np.einsum("nij,kj->nki", D_est, X)[..., :3]
    X_gt = np.einsum("nij,kj->nki", D_gt, X)[..., :3]
    return float(np.sqrt(((X_est - X_gt) ** 2).sum(axis=-1).mean()))


@dataclass
class PartMatch:
    part_map: dict[str, str]  # est part id -> gt part id
    costs: dict[str, float]  # est part id -> motion point RMS of the match
    unmatched_est: list[str]
    unmatched_gt: list[str]


def _part_arrays(tracks: Tracks, pid: str):
    model, arrays = tracks
    return arrays[f"T_W_P__{pid}"], arrays[f"valid__{pid}"]


def match_parts(tracks_est: Tracks, tracks_gt: Tracks) -> PartMatch:
    """Hungarian matching of estimated to GT parts on `motion_point_rms_m` (GT extent points,
    frames valid in both). Pairs with no common valid frame get an infinite cost and stay
    unmatched; extra parts on either side are reported."""
    est_ids = tracks_est[0].part_ids
    gt_parts = {p.part_id: p for p in tracks_gt[0].parts}
    gt_ids = list(gt_parts)
    cost = np.full((len(est_ids), len(gt_ids)), np.inf)
    for i, e in enumerate(est_ids):
        T_e, v_e = _part_arrays(tracks_est, e)
        for j, g in enumerate(gt_ids):
            T_g, v_g = _part_arrays(tracks_gt, g)
            n = min(len(T_e), len(T_g))
            valid = v_e[:n] & v_g[:n]
            if valid.any():
                cost[i, j] = motion_point_rms_m(
                    T_e[:n], T_g[:n], valid, extent_P_gt=gt_parts[g].extent_P
                )
    finite = np.where(np.isfinite(cost), cost, 1e6)
    rows, cols = linear_sum_assignment(finite)
    part_map, costs = {}, {}
    for i, j in zip(rows, cols, strict=True):
        if np.isfinite(cost[i, j]):
            part_map[est_ids[i]] = gt_ids[j]
            costs[est_ids[i]] = float(cost[i, j])
    return PartMatch(
        part_map=part_map,
        costs=costs,
        unmatched_est=[e for e in est_ids if e not in part_map],
        unmatched_gt=[g for g in gt_ids if g not in part_map.values()],
    )


@dataclass
class JointComparison:
    joint_id_est: str
    joint_id_gt: str
    type_est: str
    type_gt: str
    type_match: bool
    axis_error_deg: float
    line_distance_m: float | None
    q_rms: float | None
    q_sign: float


@dataclass
class KinematicsComparison:
    joints: list[JointComparison] = field(default_factory=list)
    unmatched_est: list[str] = field(default_factory=list)
    unmatched_gt: list[str] = field(default_factory=list)
    base_match: bool = False

    @property
    def joint_type_accuracy(self) -> float:
        return float(np.mean([j.type_match for j in self.joints])) if self.joints else float("nan")


def _static_parts(tracks: Tracks) -> set[str]:
    model, arrays = tracks
    static = set()
    for p in model.parts:
        T, v = arrays[f"T_W_P__{p.part_id}"], arrays[f"valid__{p.part_id}"]
        if v.sum() < 2:
            continue
        Tv = T[v]
        d = Tv @ inverse(Tv[0])
        if np.abs(d[:, :3, 3]).max() < 1e-6 and np.abs(d[:, :3, :3] - np.eye(3)).max() < 1e-6:
            static.add(p.part_id)
    return static


def _gt_joint_axis_in_W(joint, tracks_gt: Tracks, t_ref: float):
    """GT axis and point in W at time t_ref, from the GT parent pose at that time."""
    if joint.parent_part_id == WORLD_ID:
        return np.asarray(joint.axis_parent), None if joint.point_parent is None else np.asarray(
            joint.point_parent
        )
    model, arrays = tracks_gt
    k = int(np.argmin(np.abs(arrays["t"] - t_ref)))
    T = arrays[f"T_W_P__{joint.parent_part_id}"][k]
    axis = T[:3, :3] @ np.asarray(joint.axis_parent)
    point = (
        None
        if joint.point_parent is None
        else T[:3, :3] @ np.asarray(joint.point_parent) + T[:3, 3]
    )
    return axis, point


def compare_kinematics(
    kin_est: Kinematics, kin_gt: Kinematics, part_map: dict[str, str], tracks_gt: Tracks
) -> KinematicsComparison:
    """Compares joints connecting the same (parent, child) pair under `part_map`.

    - Joint type agreement.
    - Axis angle and line distance in W at the estimate's reference time; the GT axis_parent and
      point_parent are transformed with the GT parent pose at that time (parent "world" unchanged).
    - q compared as q_est(t) against s (q_gt(t) - q_gt(t_ref)), s = sign(a_est · a_gt), over frames
      valid in both (GT q is interpolated to the estimate's times when the time bases differ).
    - A static GT base part is treated as equivalent to "world" when matching parents.
    """
    model_est, arr_est = kin_est
    model_gt, arr_gt = kin_gt
    t_est = arr_est["t"]
    t_ref = float(t_est[model_est.reference_frame_index])
    world_like = _static_parts(tracks_gt) | {WORLD_ID}
    gt_base_like = model_gt.base_part_id in world_like

    def norm(pid: str) -> str:
        return WORLD_ID if pid in world_like else pid

    def map_est(pid: str) -> str:
        return WORLD_ID if pid == WORLD_ID else part_map.get(pid, pid)

    gt_by_pair = {(norm(j.parent_part_id), norm(j.child_part_id)): j for j in model_gt.joints}
    out = KinematicsComparison()
    used = set()
    for je in model_est.joints:
        key = (norm(map_est(je.parent_part_id)), norm(map_est(je.child_part_id)))
        jg = gt_by_pair.get(key)
        if jg is None:
            out.unmatched_est.append(je.joint_id)
            continue
        used.add(jg.joint_id)
        a_gt, p_gt = _gt_joint_axis_in_W(jg, tracks_gt, t_ref)
        a_est = np.asarray(je.axis_W)
        s = float(np.sign(np.dot(a_est, a_gt))) or 1.0
        line = None
        if je.point_W is not None and p_gt is not None:
            line = line_distance_m(je.point_W, a_est, p_gt, a_gt)
        q_rms = None
        if je.joint_id in {j.joint_id for j in model_est.joints} and f"q__{jg.joint_id}" in arr_gt:
            q_e, v_e = arr_est[f"q__{je.joint_id}"], arr_est[f"valid__{je.joint_id}"]
            t_gt = arr_gt["t"]
            q_g_full = arr_gt[f"q__{jg.joint_id}"]
            v_g_full = arr_gt[f"valid__{jg.joint_id}"]
            q_g = np.interp(t_est, t_gt, q_g_full)
            v_g = np.interp(t_est, t_gt, v_g_full.astype(float)) > 0.999
            q_ref = np.interp(t_ref, t_gt, q_g_full)
            valid = v_e & v_g
            if valid.any():
                q_rms = float(np.sqrt(np.mean((q_e[valid] - s * (q_g[valid] - q_ref)) ** 2)))
        out.joints.append(
            JointComparison(
                joint_id_est=je.joint_id,
                joint_id_gt=jg.joint_id,
                type_est=je.type,
                type_gt=jg.type,
                type_match=je.type == jg.type,
                axis_error_deg=axis_angle_error_deg(a_est, a_gt),
                line_distance_m=line,
                q_rms=q_rms,
                q_sign=s,
            )
        )
    out.unmatched_gt = [j.joint_id for j in model_gt.joints if j.joint_id not in used]
    est_base = norm(map_est(model_est.base_part_id))
    out.base_match = (est_base == WORLD_ID and gt_base_like) or est_base == norm(
        model_gt.base_part_id
    )
    return out


def mask_iou(mask_est: np.ndarray, mask_gt: np.ndarray) -> float:
    """Intersection over union of two boolean masks; NaN when both are empty."""
    a = np.asarray(mask_est, bool)
    b = np.asarray(mask_gt, bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def label_image_ious(labels_est: np.ndarray, labels_gt: np.ndarray, n_parts: int) -> list[float]:
    """Per-part IoU for label images where part index i has label i + 1."""
    return [mask_iou(labels_est == i + 1, labels_gt == i + 1) for i in range(n_parts)]
