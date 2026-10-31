import numpy as np
import pytest

from twinfit.contracts import Joint, JointFit, KinematicModel, Link
from twinfit.eval import metrics
from twinfit.geometry.se3 import make_T, so3_exp

from .helpers import kinematics, tracks


def test_axis_angle_error_sign_agnostic():
    assert metrics.axis_angle_error_deg([0, 0, 1], [0, 0, -1]) == pytest.approx(0.0)
    assert metrics.axis_angle_error_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert metrics.axis_angle_error_deg([1, 0, 0], [np.cos(0.1), np.sin(0.1), 0]) == pytest.approx(
        np.degrees(0.1)
    )


def test_line_distance_m():
    assert metrics.line_distance_m(
        [0, 0, 0], [0, 0, 1], [0.003, 0, 5], [0, 0, -1]
    ) == pytest.approx(0.003)


def _rotating_poses(n, axis, point, rate, offset=(0.0, 0.0, 0.0)):
    out = np.zeros((n, 4, 4))
    for k in range(n):
        R = so3_exp(np.asarray(axis) * rate * k)
        out[k] = make_T(R, point - R @ point + offset)
    return out


def test_motion_point_rms_zero_for_identical_displacements_and_hand_value():
    n = 20
    T_gt = _rotating_poses(n, [0, 0, 1], np.array([0.1, 0.0, 0.0]), 0.05)
    # A different part frame rigidly attached to the same motion must give zero.
    offset = make_T(so3_exp([0.3, 0.1, -0.2]), [0.02, 0.01, 0.0])
    T_est = T_gt @ offset
    valid = np.ones(n, bool)
    assert metrics.motion_point_rms_m(T_est, T_gt, valid) < 1e-12
    # A 2 mm translation error on every frame but the reference gives RMS 2 mm sqrt((n-1)/n).
    T_bad = T_gt.copy()
    T_bad[1:, 0, 3] += 0.002
    expected = 0.002 * np.sqrt((n - 1) / n)
    assert metrics.motion_point_rms_m(T_bad, T_gt, valid, ref_index=0) == pytest.approx(expected)
    # Invalid frames are ignored.
    valid[1:] = False
    assert metrics.motion_point_rms_m(T_bad, T_gt, valid) == pytest.approx(0.0)


def test_motion_point_rms_uses_extent_points():
    n = 10
    T_gt = _rotating_poses(n, [0, 0, 1], np.zeros(3), 0.0)
    T_est = _rotating_poses(n, [0, 0, 1], np.zeros(3), 0.01)  # rotates about the origin
    small = metrics.motion_point_rms_m(
        T_est, T_gt, np.ones(n, bool), extent_P_gt=[[-0.01] * 3, [0.01] * 3]
    )
    large = metrics.motion_point_rms_m(
        T_est, T_gt, np.ones(n, bool), extent_P_gt=[[-0.3] * 3, [0.3] * 3]
    )
    assert large > 10 * small > 0


def test_match_parts_hungarian_and_unmatched():
    gt_model, gt_arrays = tracks(n=10)
    est_model, est_arrays = tracks(n=10)
    # Swap the estimated part ids: est p0 moves like gt p1 and vice versa.
    est_arrays = dict(est_arrays)
    est_arrays["T_W_P__p0"], est_arrays["T_W_P__p1"] = (
        gt_arrays["T_W_P__p1"],
        gt_arrays["T_W_P__p0"],
    )
    match = metrics.match_parts((est_model, est_arrays), (gt_model, gt_arrays))
    assert match.part_map == {"p0": "p1", "p1": "p0"}
    assert all(c < 1e-9 for c in match.costs.values())
    # Drop a GT part: it must be reported unmatched.
    gt_one = gt_model.model_copy(update={"parts": gt_model.parts[:1]})
    gt_one_arrays = {k: v for k, v in gt_arrays.items() if "p1" not in k}
    match = metrics.match_parts((est_model, est_arrays), (gt_one, gt_one_arrays))
    assert (
        match.part_map == {"p1": "p0"}
        and match.unmatched_est == ["p0"]
        and match.unmatched_gt == []
    )


def test_compare_kinematics_hand_case():
    gt_model, gt_arrays = kinematics(n=10)
    tr = tracks(n=10)
    est_model, est_arrays = kinematics(n=10)
    # Estimate with flipped axis sign, a 10 mm shifted point, and q with the matching sign flip.
    j = est_model.joints[0]
    j_est = j.model_copy(update={"axis_W": [0.0, 0.0, 1.0], "point_W": [0.11, 0.05, 0.0]})
    est_model = est_model.model_copy(update={"joints": [j_est]})
    est_arrays = dict(est_arrays)
    est_arrays["q__j0"] = -(gt_arrays["q__j0"] - gt_arrays["q__j0"][0])
    cmp = metrics.compare_kinematics(
        (est_model, est_arrays), (gt_model, gt_arrays), {"p0": "p0", "p1": "p0"}, tr
    )
    cmp = metrics.compare_kinematics(
        (est_model, est_arrays), (gt_model, gt_arrays), {"p0": "p0", "p1": "p1"}, tr
    )
    assert len(cmp.joints) == 1 and cmp.joints[0].type_match and cmp.base_match
    assert cmp.joints[0].axis_error_deg == pytest.approx(0.0, abs=1e-9)
    assert cmp.joints[0].line_distance_m == pytest.approx(0.01)
    assert cmp.joints[0].q_sign == -1.0 and cmp.joints[0].q_rms == pytest.approx(0.0, abs=1e-12)
    assert cmp.joint_type_accuracy == 1.0


def test_compare_kinematics_static_base_equals_world_and_type_mismatch():
    gt_model, gt_arrays = kinematics(n=10)
    tr = tracks(n=10)  # p0 is static in the helper tracks
    joint = Joint(
        joint_id="j0",
        parent_part_id="world",
        child_part_id="p1",
        type="prismatic",
        axis_W=[0.0, 0.0, 1.0],
        axis_parent=[0.0, 0.0, 1.0],
        q_range_observed=[0.0, 1.0],
        fit=JointFit(rms_point_m=0.0, n_frames=10),
        confidence=0.5,
    )
    est_model = KinematicModel(
        clip_id="c",
        source="identify",
        base_part_id="world",
        reference_frame_index=0,
        links=[Link(part_id="p1", parent_joint_id="j0")],
        joints=[joint],
    )
    est_arrays = {k: v for k, v in gt_arrays.items()}
    cmp = metrics.compare_kinematics(
        (est_model, est_arrays), (gt_model, gt_arrays), {"p1": "p1"}, tr
    )
    assert len(cmp.joints) == 1 and not cmp.joints[0].type_match and cmp.base_match
    assert cmp.joints[0].joint_id_gt == "j0" and cmp.unmatched_gt == []


def test_mask_iou():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2, :2] = True
    b[:2, :] = True
    assert metrics.mask_iou(a, b) == pytest.approx(0.5)
    assert np.isnan(metrics.mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)))
    labels = np.array([[1, 1, 2], [0, 2, 2]])
    assert metrics.label_image_ious(labels, labels, 2) == [1.0, 1.0]
