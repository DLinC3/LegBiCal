"""Pure contact-force geometry shared by Meshcat and MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrowTransforms:
    """Numerical arrow geometry consumed by either renderer."""

    origin: np.ndarray
    endpoint: np.ndarray
    shaft_transform: np.ndarray
    head_transform: np.ndarray
    force_magnitude_n: float
    length_m: float
    visible: bool


def tangent_basis_from_normal(normal: np.ndarray) -> np.ndarray:
    """Match MotionAnitescuSimulator's deterministic tangent basis."""
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) == 0.0:
        raise ValueError("normal must be a finite nonzero 3-vector")
    n = n / np.linalg.norm(n)
    reference = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array(
        [1.0, 0.0, 0.0]
    )
    t1 = reference - n * np.dot(n, reference)
    norm_t1 = np.linalg.norm(t1)
    if norm_t1 < 1e-12:
        reference = np.array([0.0, 1.0, 0.0])
        t1 = reference - n * np.dot(n, reference)
        norm_t1 = np.linalg.norm(t1)
    t1 /= norm_t1
    t2 = np.cross(n, t1)
    return np.column_stack((t1, t2))


def force_components_to_world(
    force_t1_t2_n: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Convert source `[t1,t2,n]` force rows into world coordinates."""
    source = np.asarray(force_t1_t2_n, dtype=float)
    if source.shape[-1] != 3 or not np.isfinite(source).all():
        raise ValueError("force source must be finite with final dimension 3")
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    tangent = tangent_basis_from_normal(n)
    basis = np.column_stack((tangent, n))
    return np.einsum("...j,ij->...i", source, basis)


def aggregate_foot_forces(corner_forces_world: np.ndarray) -> np.ndarray:
    """Sum official contact ordering `[left four,right four]` by foot."""
    forces = np.asarray(corner_forces_world, dtype=float)
    if forces.shape[-2:] != (8, 3) or not np.isfinite(forces).all():
        raise ValueError("corner forces must end in shape [8,3]")
    return np.stack((forces[..., :4, :].sum(axis=-2),
                     forces[..., 4:, :].sum(axis=-2)), axis=-2)


def friction_utilization(force_t1_t2_n: np.ndarray, mu: float) -> np.ndarray:
    """Return `||f_t||/(mu*f_n)` with finite values for barrier tails."""
    source = np.asarray(force_t1_t2_n, dtype=float)
    if source.shape[-1] != 3 or not np.isfinite(source).all() or mu <= 0.0:
        raise ValueError("invalid force source or friction coefficient")
    tangent = np.linalg.norm(source[..., :2], axis=-1)
    normal = source[..., 2]
    if np.any(normal < -1e-10):
        raise ValueError("contact force has a materially negative normal component")
    denominator = mu * np.maximum(normal, np.finfo(float).tiny)
    return tangent / denominator


def support_polygon_xy(points_world: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Return a deterministic 2D convex hull for active contact points."""
    points = np.asarray(points_world, dtype=float)
    mask = np.asarray(active, dtype=bool)
    if points.shape != (8, 3) or mask.shape != (8,):
        raise ValueError("support polygon expects eight positions and flags")
    unique = sorted(set(map(tuple, points[mask, :2].tolist())))
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float).reshape((-1, 2))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (
            a[1] - o[1]
        ) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def center_of_pressure_world(
    points_world: np.ndarray,
    force_t1_t2_n: np.ndarray,
    *,
    minimum_normal_force_n: float,
    plane_height_m: float,
) -> np.ndarray | None:
    """Compute point-force CoP from positive normal forces only."""
    points = np.asarray(points_world, dtype=float)
    source = np.asarray(force_t1_t2_n, dtype=float)
    if points.shape != (8, 3) or source.shape != (8, 3):
        raise ValueError("CoP expects eight positions and force rows")
    normal = np.maximum(source[:, 2], 0.0)
    total = float(normal.sum())
    if total < minimum_normal_force_n:
        return None
    cop = np.empty(3)
    cop[:2] = (normal[:, None] * points[:, :2]).sum(axis=0) / total
    cop[2] = plane_height_m
    return cop


def align_mujoco_post_step_foot_forces(force_rows: np.ndarray) -> np.ndarray:
    """Map GT post-step rows to preceding PRIME transition intervals."""
    forces = np.asarray(force_rows, dtype=float)
    if forces.ndim != 3 or forces.shape[1:] != (2, 3) or forces.shape[0] < 2:
        raise ValueError("MuJoCo foot forces must have shape [N>=2,2,3]")
    if not np.isfinite(forces).all():
        raise ValueError("MuJoCo foot forces contain non-finite values")
    return forces[1:].copy()


def arrow_transforms(
    origin_world: np.ndarray,
    force_world: np.ndarray,
    *,
    scale_m_per_n: float,
    display_threshold_n: float,
    shaft_radius_m: float = 0.004,
    head_radius_m: float = 0.012,
    head_fraction: float = 0.25,
) -> ArrowTransforms:
    """Build transforms for unit Y-axis cylinders used as shaft and cone."""
    origin = np.asarray(origin_world, dtype=float)
    force = np.asarray(force_world, dtype=float)
    if origin.shape != (3,) or force.shape != (3,):
        raise ValueError("arrow origin and force must be 3-vectors")
    if not np.isfinite(origin).all() or not np.isfinite(force).all():
        raise ValueError("arrow geometry must be finite")
    if scale_m_per_n <= 0.0 or display_threshold_n < 0.0:
        raise ValueError("invalid arrow scale/threshold")
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must lie in (0,1)")
    magnitude = float(np.linalg.norm(force))
    length = scale_m_per_n * magnitude
    endpoint = origin + scale_m_per_n * force
    visible = magnitude >= display_threshold_n and length > 0.0
    if not visible:
        zero = np.eye(4)
        zero[:3, :3] = 0.0
        zero[:3, 3] = origin
        return ArrowTransforms(
            origin=origin.copy(), endpoint=endpoint, shaft_transform=zero.copy(),
            head_transform=zero.copy(), force_magnitude_n=magnitude,
            length_m=length, visible=False,
        )

    direction = force / magnitude
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(direction, reference)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, direction)
    rotation = np.column_stack((x_axis, direction, z_axis))
    if np.linalg.det(rotation) < 0.0:
        raise RuntimeError("arrow rotation is not right-handed")

    head_length = head_fraction * length
    shaft_length = length - head_length
    shaft = np.eye(4)
    shaft[:3, :3] = rotation @ np.diag(
        [shaft_radius_m, shaft_length, shaft_radius_m]
    )
    shaft[:3, 3] = origin + 0.5 * shaft_length * direction
    head = np.eye(4)
    head[:3, :3] = rotation @ np.diag(
        [head_radius_m, head_length, head_radius_m]
    )
    head[:3, 3] = origin + (shaft_length + 0.5 * head_length) * direction
    return ArrowTransforms(
        origin=origin.copy(), endpoint=endpoint, shaft_transform=shaft,
        head_transform=head, force_magnitude_n=magnitude, length_m=length,
        visible=True,
    )
