"""Typed, immutable data shared by every visualization backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _readonly_float(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _readonly_bool(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class VisualizationStyle:
    """Cross-renderer physical scale, geometry, and color contract (v2).

    Force arrows keep the physical ``force_scale_m_per_n`` length contract
    but render unlit by default so their colors stay saturated regardless of
    scene lighting. Every color used by the Meshcat renderer is style-driven.
    """

    force_scale_m_per_n: float = 5e-4
    force_display_threshold_n: float = 5.0
    cop_min_normal_force_n: float = 5.0
    friction_warning: float = 0.80
    friction_critical: float = 0.95
    # Arrow prominence (v2): unlit materials plus thicker shafts/heads.
    arrow_unlit: bool = True
    corner_shaft_radius_m: float = 0.007
    corner_head_radius_m: float = 0.018
    foot_sum_radius_scale: float = 2.0
    # Markers (v2): the CoP is a colored sphere, the support polygon a light
    # translucent outline, so neither competes with the force arrows.
    cop_radius_m: float = 0.018
    cop_color: str = "#009E73"
    support_polygon_color: str = "#8899AA"
    support_polygon_opacity: float = 0.6
    support_polygon_radius_m: float = 0.004
    # Floor (v2): checker texture matching the PRIME experiment MJCF
    # groundplane (rgb1/rgb2/markrgb), with thanks to the PRIME authors.
    floor_checker: bool = True
    floor_rgb1: str = "#334D66"
    floor_rgb2: str = "#1A334D"
    floor_mark_rgb: str = "#CCCCCC"
    floor_cell_m: float = 0.4
    robot_gt_opacity: float = 0.25
    colors: dict[str, str] = field(default_factory=lambda: {
        "robot_estimated": "#B8B8B8",
        "robot_gt": "#56B4E9",
        "prime_left": "#0072B2",
        "prime_right": "#E69F00",
        "mujoco_gt": "#CC79A7",
        "normal": "#009E73",
        "tangential": "#D55E00",
        "warning": "#F0E442",
        "invalid": "#D55E00",
        "barrier_tail": "#666666",
    })

    def __post_init__(self) -> None:
        if not self.force_scale_m_per_n > 0.0:
            raise ValueError("force scale must be positive")
        if self.force_display_threshold_n < 0.0:
            raise ValueError("force display threshold must be non-negative")
        if self.cop_min_normal_force_n < 0.0:
            raise ValueError("CoP force threshold must be non-negative")
        if not 0.0 < self.friction_warning < self.friction_critical <= 1.0:
            raise ValueError("invalid friction warning thresholds")
        if not (self.corner_shaft_radius_m > 0.0
                and self.corner_head_radius_m > self.corner_shaft_radius_m):
            raise ValueError("arrow head radius must exceed shaft radius")
        if not 0.0 < self.support_polygon_opacity <= 1.0:
            raise ValueError("support polygon opacity must be in (0,1]")
        if not 0.0 < self.robot_gt_opacity <= 1.0:
            raise ValueError("ghost robot opacity must be in (0,1]")
        if not self.floor_cell_m > 0.0:
            raise ValueError("floor cell size must be positive")


@dataclass(frozen=True)
class ContactDiagnostics:
    """One aligned row for each physical contact transition."""

    knot: np.ndarray
    newton_converged: np.ndarray
    newton_termination: tuple[str, ...]
    newton_iterations: np.ndarray
    newton_grad_norm: np.ndarray
    newton_relative_grad_norm: np.ndarray
    feasible_init_used: np.ndarray
    min_cone_margin: np.ndarray
    min_alpha: np.ndarray
    force_norm: np.ndarray

    def __post_init__(self) -> None:
        numeric = {
            "knot": np.asarray(self.knot, dtype=int),
            "newton_iterations": np.asarray(self.newton_iterations, dtype=int),
            "newton_grad_norm": _readonly_float(
                self.newton_grad_norm, name="newton_grad_norm"
            ),
            "newton_relative_grad_norm": _readonly_float(
                self.newton_relative_grad_norm,
                name="newton_relative_grad_norm",
            ),
            "min_cone_margin": _readonly_float(
                self.min_cone_margin, name="min_cone_margin"
            ),
            "min_alpha": _readonly_float(self.min_alpha, name="min_alpha"),
            "force_norm": _readonly_float(self.force_norm, name="force_norm"),
        }
        boolean = {
            "newton_converged": _readonly_bool(self.newton_converged),
            "feasible_init_used": _readonly_bool(self.feasible_init_used),
        }
        length = len(numeric["knot"])
        if len(self.newton_termination) != length:
            raise ValueError("diagnostic termination length mismatch")
        for name, array in numeric.items():
            if array.shape != (length,):
                raise ValueError(f"diagnostic {name} must have shape ({length},)")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name, array in boolean.items():
            if array.shape != (length,):
                raise ValueError(f"diagnostic {name} must have shape ({length},)")
            object.__setattr__(self, name, array)
        if not np.array_equal(numeric["knot"], np.arange(length)):
            raise ValueError("contact diagnostics are not contiguous zero-based knots")

    def __len__(self) -> int:
        return int(self.knot.size)

    def row(self, index: int) -> dict[str, Any]:
        return {
            "knot": int(self.knot[index]),
            "newton_converged": bool(self.newton_converged[index]),
            "newton_termination": self.newton_termination[index],
            "newton_iterations": int(self.newton_iterations[index]),
            "newton_grad_norm": float(self.newton_grad_norm[index]),
            "newton_relative_grad_norm": float(
                self.newton_relative_grad_norm[index]
            ),
            "feasible_init_used": bool(self.feasible_init_used[index]),
            "min_cone_margin": float(self.min_cone_margin[index]),
            "min_alpha": float(self.min_alpha[index]),
            "force_norm": float(self.force_norm[index]),
        }


@dataclass(frozen=True)
class MotionForceSequence:
    """A validated sequence with no renderer-specific state."""

    run_id: str
    source_result_dir: str
    source_config: str
    profile_id: str
    profile_key: str
    urdf_sha256: str
    mjcf_sha256: str
    total_mass_kg: float
    dt: float
    mu: float
    kappa: float
    contact_names: tuple[str, ...]
    plane_normal_world: np.ndarray
    plane_height_m: float
    physical_states: np.ndarray
    interval_forces_contact: np.ndarray
    interval_forces_world: np.ndarray
    contact_positions_world: np.ndarray
    contact_gaps_m: np.ndarray
    friction_utilization: np.ndarray
    active_display_mask: np.ndarray
    diagnostics: ContactDiagnostics
    solve_summary: dict[str, Any]
    source_hashes: dict[str, str]
    style: VisualizationStyle = field(default_factory=VisualizationStyle)
    gt_states: np.ndarray | None = None
    gt_interval_foot_forces_world: np.ndarray | None = None
    gt_interval_foot_positions_world: np.ndarray | None = None
    gt_source_rows: np.ndarray | None = None
    gt_motion: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not self.dt > 0.0 or not self.mu > 0.0 or not self.kappa > 0.0:
            raise ValueError("dt, mu and kappa must be positive")
        if not self.total_mass_kg > 0.0:
            raise ValueError("total mass must be positive")
        if len(self.contact_names) != 8 or len(set(self.contact_names)) != 8:
            raise ValueError("exactly eight unique contact frames are required")

        states = _readonly_float(self.physical_states, name="physical_states")
        if states.ndim != 2 or states.shape[1] != 71 or states.shape[0] < 2:
            raise ValueError("physical_states must have shape [N>=2,71]")
        intervals = states.shape[0] - 1
        expected_corner = (intervals, 8, 3)
        expected_scalar = (intervals, 8)
        arrays = {
            "interval_forces_contact": (
                _readonly_float(
                    self.interval_forces_contact,
                    name="interval_forces_contact",
                ),
                expected_corner,
            ),
            "interval_forces_world": (
                _readonly_float(
                    self.interval_forces_world, name="interval_forces_world"
                ),
                expected_corner,
            ),
            "contact_positions_world": (
                _readonly_float(
                    self.contact_positions_world,
                    name="contact_positions_world",
                ),
                expected_corner,
            ),
            "contact_gaps_m": (
                _readonly_float(self.contact_gaps_m, name="contact_gaps_m"),
                expected_scalar,
            ),
            "friction_utilization": (
                _readonly_float(
                    self.friction_utilization,
                    name="friction_utilization",
                ),
                expected_scalar,
            ),
        }
        object.__setattr__(self, "physical_states", states)
        for name, (array, expected) in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
            object.__setattr__(self, name, array)
        mask = _readonly_bool(self.active_display_mask)
        if mask.shape != expected_scalar:
            raise ValueError(f"active_display_mask must have shape {expected_scalar}")
        object.__setattr__(self, "active_display_mask", mask)
        normal = _readonly_float(
            self.plane_normal_world, name="plane_normal_world"
        )
        if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1.0):
            raise ValueError("plane normal must be a unit 3-vector")
        object.__setattr__(self, "plane_normal_world", normal)
        if len(self.diagnostics) != intervals:
            raise ValueError("diagnostics must have one row per force interval")

        if self.gt_states is None:
            if self.gt_interval_foot_forces_world is not None:
                raise ValueError("GT forces require GT states")
            if self.gt_interval_foot_positions_world is not None:
                raise ValueError("GT foot positions require GT states")
            if self.gt_source_rows is not None:
                raise ValueError("GT source rows require GT states")
        else:
            gt_states = _readonly_float(self.gt_states, name="gt_states")
            if gt_states.shape != states.shape:
                raise ValueError("GT states must match physical state shape")
            object.__setattr__(self, "gt_states", gt_states)
            if self.gt_interval_foot_forces_world is not None:
                gt_force = _readonly_float(
                    self.gt_interval_foot_forces_world,
                    name="gt_interval_foot_forces_world",
                )
                if gt_force.shape != (intervals, 2, 3):
                    raise ValueError("GT foot forces must have shape [N-1,2,3]")
                object.__setattr__(
                    self, "gt_interval_foot_forces_world", gt_force
                )
            if self.gt_interval_foot_positions_world is not None:
                gt_position = _readonly_float(
                    self.gt_interval_foot_positions_world,
                    name="gt_interval_foot_positions_world",
                )
                if gt_position.shape != (intervals, 2, 3):
                    raise ValueError("GT foot positions must have shape [N-1,2,3]")
                object.__setattr__(
                    self, "gt_interval_foot_positions_world", gt_position
                )
            if self.gt_source_rows is not None:
                rows = np.asarray(self.gt_source_rows, dtype=int)
                if rows.shape != (states.shape[0],):
                    raise ValueError("GT source rows must have shape [N]")
                rows.setflags(write=False)
                object.__setattr__(self, "gt_source_rows", rows)

    @property
    def number_of_states(self) -> int:
        return int(self.physical_states.shape[0])

    @property
    def number_of_intervals(self) -> int:
        return self.number_of_states - 1


@dataclass(frozen=True)
class VisualizationFrame:
    """One scientific-knot frame consumed identically by both viewers."""

    time_seconds: float
    state_index: int
    transition_index: int
    q_estimated: np.ndarray
    v_estimated: np.ndarray
    corner_positions_world: np.ndarray
    corner_forces_world: np.ndarray
    corner_friction_utilization: np.ndarray
    active_display_mask: np.ndarray
    prime_foot_forces_world: np.ndarray
    support_polygon_xy: np.ndarray
    center_of_pressure_world: np.ndarray | None
    diagnostics: dict[str, Any]
    hidden_barrier_tail_count: int
    max_friction_utilization: float
    q_gt: np.ndarray | None = None
    v_gt: np.ndarray | None = None
    gt_foot_forces_world: np.ndarray | None = None
    gt_foot_positions_world: np.ndarray | None = None
