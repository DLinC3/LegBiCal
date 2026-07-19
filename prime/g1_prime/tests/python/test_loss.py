"""SE(3)-log trajectory loss contract."""

from __future__ import annotations

import math

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from g1cal.loss import (
    NORMALIZERS,
    SE3_LOG_SCHEMA,
    floating_base_log_residual,
    per_knot_loss_contribution,
    trajectory_loss_arrays,
)


def _identity_states(knots: int) -> np.ndarray:
    states = np.zeros((knots, 71))
    states[:, 6] = 1.0
    return states


def _row_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    row = np.zeros(71)
    row[:3] = translation
    row[3:7] = Rotation.from_matrix(rotation).as_quat()
    return row


def _apply_left_transform(row: np.ndarray, transform: pin.SE3) -> np.ndarray:
    rotation = Rotation.from_quat(row[3:7]).as_matrix()
    moved = np.array(row, copy=True)
    moved[:3] = transform.rotation @ row[:3] + transform.translation
    moved[3:7] = Rotation.from_matrix(transform.rotation @ rotation).as_quat()
    return moved


def test_equality_is_zero_with_full_branch_margin():
    truth = _identity_states(4)
    result = trajectory_loss_arrays(truth.copy(), truth)
    assert result.value == 0.0
    assert result.schema == SE3_LOG_SCHEMA
    assert result.branch_margin_rad == pytest.approx(math.pi)


def test_pure_translation_and_rotation_residuals():
    truth = _identity_states(1)
    displacement = np.array([0.05, -0.02, 0.01])
    estimate = truth.copy()
    estimate[0, :3] = displacement
    residual = floating_base_log_residual(estimate[0, :36], truth[0, :36])
    np.testing.assert_allclose(residual[:3], displacement, rtol=0, atol=1e-15)
    np.testing.assert_allclose(residual[3:], 0.0, atol=1e-15)

    angle = 0.3
    rotation = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    estimate = np.vstack([_row_from_pose(rotation, np.zeros(3))])
    residual = floating_base_log_residual(estimate[0, :36], truth[0, :36])
    np.testing.assert_allclose(residual[:3], 0.0, atol=1e-15)
    np.testing.assert_allclose(residual[3:], [0, 0, angle], atol=1e-14)


def test_exp_log_roundtrip_recovers_exact_residual():
    xi = np.array([0.04, -0.03, 0.02, 0.20, -0.10, 0.15])
    base = pin.SE3(
        Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix(),
        np.array([1.0, 2.0, 0.5]),
    )
    perturbed = base * pin.exp6(pin.Motion(xi[:3], xi[3:]))
    truth_row = _row_from_pose(base.rotation, base.translation)
    estimate_row = _row_from_pose(perturbed.rotation, perturbed.translation)
    residual = floating_base_log_residual(estimate_row[:36], truth_row[:36])
    np.testing.assert_allclose(residual, xi, rtol=1e-12, atol=1e-12)


def test_quaternion_sign_invariance():
    truth = _identity_states(3)
    estimate = truth.copy()
    estimate[:, :3] = 0.02
    estimate[1, 3:7] *= -1.0
    flipped_truth = truth.copy()
    flipped_truth[2, 3:7] *= -1.0
    baseline = trajectory_loss_arrays(estimate, truth)
    flipped = trajectory_loss_arrays(estimate, flipped_truth)
    assert flipped.value == pytest.approx(baseline.value, abs=1e-15)


def test_common_left_transform_invariance():
    rng = np.random.default_rng(20260718)
    truth = _identity_states(5)
    estimate = truth.copy()
    for index in range(5):
        truth[index] = _row_from_pose(
            Rotation.from_rotvec(0.3 * rng.normal(size=3)).as_matrix(),
            rng.normal(size=3),
        )
        base = pin.SE3(
            Rotation.from_quat(truth[index, 3:7]).as_matrix(),
            truth[index, :3].copy(),
        )
        moved = base * pin.exp6(
            pin.Motion(0.05 * rng.normal(size=3), 0.2 * rng.normal(size=3))
        )
        estimate[index] = _row_from_pose(moved.rotation, moved.translation)
        estimate[index, 36:] = truth[index, 36:] + 0.1 * rng.normal(size=35)
    transform = pin.SE3(
        Rotation.from_rotvec([0.7, -0.4, 1.1]).as_matrix(),
        np.array([5.0, -3.0, 2.0]),
    )
    truth_moved = np.vstack(
        [_apply_left_transform(row, transform) for row in truth]
    )
    estimate_moved = np.vstack(
        [_apply_left_transform(row, transform) for row in estimate]
    )
    baseline = trajectory_loss_arrays(estimate, truth)
    moved = trajectory_loss_arrays(estimate_moved, truth_moved)
    assert moved.value == pytest.approx(baseline.value, rel=1e-12)


def test_fixed_normalizers_and_equal_groups():
    assert NORMALIZERS == {
        "se3_log_translation": 0.10,
        "se3_log_rotation": 0.20,
        "joint_position": 0.20,
        "base_linear_velocity": 1.0,
        "base_angular_velocity": 1.0,
        "joint_velocity": 2.0,
    }
    truth = _identity_states(2)
    estimate = truth.copy()
    estimate[:, 0] = 0.10
    result = trajectory_loss_arrays(estimate, truth)
    assert result.value == pytest.approx((1.0 / 3.0) / 6.0)


def test_per_knot_contributions_sum_to_total():
    rng = np.random.default_rng(7)
    knots = 6
    truth = _identity_states(knots)
    estimate = truth.copy()
    for index in range(knots):
        rotation = Rotation.from_rotvec(0.2 * rng.normal(size=3)).as_matrix()
        estimate[index] = _row_from_pose(rotation, 0.05 * rng.normal(size=3))
        estimate[index, 7:36] = 0.05 * rng.normal(size=29)
        estimate[index, 36:] = 0.3 * rng.normal(size=35)
    total = trajectory_loss_arrays(estimate, truth)
    knotwise = sum(
        per_knot_loss_contribution(estimate[index], truth[index], knots)
        for index in range(knots)
    )
    assert knotwise == pytest.approx(total.value, rel=1e-12)


def test_branch_margin_gate_and_quaternion_norm_rejection():
    truth = _identity_states(1)
    near_pi = Rotation.from_rotvec([0.0, 0.0, math.pi - 1e-3]).as_matrix()
    estimate = np.vstack([_row_from_pose(near_pi, np.zeros(3))])
    result = trajectory_loss_arrays(estimate, truth)
    assert 0.0 < result.branch_margin_rad == pytest.approx(1e-3, rel=1e-6)

    bad = truth.copy()
    bad[0, 3:7] = [0.0, 0.0, 0.0, 1.5]
    with pytest.raises(ValueError, match="quaternion norm"):
        trajectory_loss_arrays(bad, truth)


def test_pinocchio_motion_vector_is_linear_then_angular():
    motion = pin.Motion(
        np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    )
    np.testing.assert_allclose(motion.vector, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(motion.linear, [1, 2, 3])
    np.testing.assert_allclose(motion.angular, [4, 5, 6])


def test_scipy_quaternion_contract_is_xyzw():
    quaternion = Rotation.from_rotvec([0.0, 0.0, math.pi / 2]).as_quat()
    np.testing.assert_allclose(
        quaternion,
        [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
        atol=1e-15,
    )
