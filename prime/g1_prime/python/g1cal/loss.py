"""Manifold-aware supervised trajectory loss on ``SE(3)``.

The floating-base pose residual is the left-invariant
``Log(T_GT.inverse() * T_est).vee = [translation; rotation]`` on the
principal branch; four Euclidean groups (joint position, base linear/angular
velocity, joint velocity) complete six equally weighted normalized group
MSEs. Normalizers are fixed and independent of the calibrated covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .backend import MotionFieResult
from .paths import resolve_inside_root


SE3_LOG_SCHEMA = "se3_log_v2"

# The Pinocchio free-flyer configuration stores the base quaternion in
# coefficients q[3:7] with xyzw ordering.
QUATERNION_NORM_TOLERANCE = 1e-6

NORMALIZERS = {
    "se3_log_translation": 0.10,
    "se3_log_rotation": 0.20,
    "joint_position": 0.20,
    "base_linear_velocity": 1.0,
    "base_angular_velocity": 1.0,
    "joint_velocity": 2.0,
}

_GROUP_DIMENSIONS = {
    "se3_log_translation": 3,
    "se3_log_rotation": 3,
    "joint_position": 29,
    "base_linear_velocity": 3,
    "base_angular_velocity": 3,
    "joint_velocity": 29,
}


@dataclass(frozen=True)
class LossResult:
    value: float
    group_normalized_mse: dict[str, float]
    group_rmse: dict[str, float]
    knots: int
    schema: str = SE3_LOG_SCHEMA
    max_rotation_angle_rad: float = float("nan")
    branch_margin_rad: float = float("nan")


def _base_pose(q: np.ndarray, label: str) -> pin.SE3:
    quaternion = np.asarray(q[3:7], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            f"{label} base quaternion norm {norm} exceeds the declared "
            f"roundoff tolerance {QUATERNION_NORM_TOLERANCE}"
        )
    rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    return pin.SE3(rotation, np.asarray(q[:3], dtype=float))


def floating_base_log_residual(
    q_est: np.ndarray, q_truth: np.ndarray
) -> np.ndarray:
    """Left-invariant ``Log(T_GT.inverse()*T_est).vee``, translation first.

    ``q[0:3]`` is the world base position and ``q[3:7]`` the xyzw base
    quaternion of a Pinocchio free-flyer configuration. The named ``linear``
    (translation) and ``angular`` (rotation) accessors make the
    translation-before-rotation export explicit.
    """
    T_est = _base_pose(np.asarray(q_est, dtype=float), "estimate")
    T_gt = _base_pose(np.asarray(q_truth, dtype=float), "truth")
    motion = pin.log6(T_gt.inverse() * T_est)
    return np.concatenate((motion.linear, motion.angular))


def _group_errors(
    estimated: np.ndarray, truth: np.ndarray
) -> tuple[dict[str, np.ndarray], float, float]:
    if estimated.shape != truth.shape or estimated.ndim != 2 or estimated.shape[1] != 71:
        raise ValueError(
            f"estimate/truth shape mismatch: {estimated.shape} vs {truth.shape}"
        )
    residuals = np.empty((estimated.shape[0], 6))
    for index in range(estimated.shape[0]):
        residuals[index] = floating_base_log_residual(
            estimated[index, :36], truth[index, :36]
        )
    rotation_angles = np.linalg.norm(residuals[:, 3:], axis=1)
    max_angle = float(rotation_angles.max()) if rotation_angles.size else 0.0
    branch_margin = math.pi - max_angle
    if not branch_margin > 0.0:
        raise ValueError(
            "principal-branch gate failed: relative rotation angle "
            f"{max_angle} rad has non-positive margin to pi"
        )
    v_est, v_true = estimated[:, 36:], truth[:, 36:]
    errors = {
        "se3_log_translation": residuals[:, :3],
        "se3_log_rotation": residuals[:, 3:],
        "joint_position": estimated[:, 7:36] - truth[:, 7:36],
        "base_linear_velocity": v_est[:, :3] - v_true[:, :3],
        "base_angular_velocity": v_est[:, 3:6] - v_true[:, 3:6],
        "joint_velocity": v_est[:, 6:] - v_true[:, 6:],
    }
    return errors, max_angle, branch_margin


def trajectory_loss_arrays(
    estimated: np.ndarray, truth: np.ndarray
) -> LossResult:
    """The primary loss over physical states (shooting anchor removed)."""
    errors, max_angle, branch_margin = _group_errors(estimated, truth)
    group_rmse = {
        name: float(np.sqrt(np.mean(np.square(value))))
        for name, value in errors.items()
    }
    normalized = {
        name: float(np.mean(np.square(value / NORMALIZERS[name])))
        for name, value in errors.items()
    }
    value = float(np.mean(tuple(normalized.values())))
    return LossResult(
        value,
        normalized,
        group_rmse,
        truth.shape[0],
        schema=SE3_LOG_SCHEMA,
        max_rotation_angle_rad=max_angle,
        branch_margin_rad=branch_margin,
    )


def per_knot_loss_contribution(
    estimated_row: np.ndarray, truth_row: np.ndarray, knots: int
) -> float:
    """Exact single-knot contribution: summing over knots gives the total."""
    if knots <= 0:
        raise ValueError("knots must be positive")
    estimated_row = np.asarray(estimated_row, dtype=float)
    truth_row = np.asarray(truth_row, dtype=float)
    if estimated_row.shape != (71,) or truth_row.shape != (71,):
        raise ValueError("per-knot rows must have dimension 71")
    residual = floating_base_log_residual(
        estimated_row[:36], truth_row[:36]
    )
    errors = {
        "se3_log_translation": residual[:3],
        "se3_log_rotation": residual[3:],
        "joint_position": estimated_row[7:36] - truth_row[7:36],
        "base_linear_velocity": estimated_row[36:39] - truth_row[36:39],
        "base_angular_velocity": estimated_row[39:42] - truth_row[39:42],
        "joint_velocity": estimated_row[42:] - truth_row[42:],
    }
    total = 0.0
    for name, error in errors.items():
        total += float(np.sum(np.square(error / NORMALIZERS[name]))) / (
            knots * _GROUP_DIMENSIONS[name]
        )
    return total / 6.0


def trajectory_loss(result: MotionFieResult, truth_xs: str) -> LossResult:
    estimated_all = np.loadtxt(resolve_inside_root(result.xs_path), delimiter=",")
    truth = np.loadtxt(resolve_inside_root(truth_xs), delimiter=",")
    estimated = estimated_all[1:]  # drop fixed shooting anchor before arrival
    return trajectory_loss_arrays(estimated, truth)
