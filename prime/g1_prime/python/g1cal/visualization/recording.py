"""Reproducible numerical sidecars for visualization artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..paths import resolve_inside_root
from .force_geometry import aggregate_foot_forces
from .types import MotionForceSequence


def sampled_transition_indices(
    number_of_intervals: int,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Return deterministic render indices and always include the last one.

    ``stop`` is exclusive, matching Python ranges.  The final transition is
    appended when the stride does not land on it so a decimated recording
    still covers the requested complete timeline.
    """
    stop = number_of_intervals if stop is None else stop
    if (
        number_of_intervals <= 0
        or start < 0
        or stop <= start
        or stop > number_of_intervals
        or stride <= 0
    ):
        raise ValueError("invalid sampled transition range")
    indices = np.arange(start, stop, stride, dtype=int)
    final = stop - 1
    if indices[-1] != final:
        indices = np.append(indices, final)
    return indices


def sampled_playback_fps(indices: np.ndarray, dt: float) -> float:
    """Choose constant FPS whose first/last frame times match source time."""
    values = np.asarray(indices, dtype=int)
    if (
        values.ndim != 1
        or values.size < 1
        or (values.size > 1 and np.any(np.diff(values) <= 0))
        or not dt > 0.0
    ):
        raise ValueError("playback FPS requires increasing indices and dt")
    if values.size == 1:
        return 1.0 / dt
    return float((values.size - 1) / ((values[-1] - values[0]) * dt))


def write_frame_metrics(sequence: MotionForceSequence, output: str) -> Path:
    path = resolve_inside_root(output, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "transition", "time_seconds", "active_display_corners",
        "hidden_barrier_tails", "prime_left_fx_n", "prime_left_fy_n",
        "prime_left_fz_n", "prime_right_fx_n", "prime_right_fy_n",
        "prime_right_fz_n", "prime_total_fz_n", "prime_fz_over_body_weight",
        "max_friction_utilization", "newton_converged",
        "newton_termination", "newton_iterations",
        "newton_relative_grad_norm", "min_cone_margin", "min_alpha",
        "gt_left_fz_n", "gt_right_fz_n", "gt_force_error_norm_n",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(sequence.number_of_intervals):
            prime = aggregate_foot_forces(
                sequence.interval_forces_world[index]
            )
            gt = (
                sequence.gt_interval_foot_forces_world[index]
                if sequence.gt_interval_foot_forces_world is not None else None
            )
            row = {
                "transition": index,
                "time_seconds": index * sequence.dt,
                "active_display_corners": int(
                    sequence.active_display_mask[index].sum()
                ),
                "hidden_barrier_tails": int(
                    (~sequence.active_display_mask[index]).sum()
                ),
                "prime_left_fx_n": prime[0, 0],
                "prime_left_fy_n": prime[0, 1],
                "prime_left_fz_n": prime[0, 2],
                "prime_right_fx_n": prime[1, 0],
                "prime_right_fy_n": prime[1, 1],
                "prime_right_fz_n": prime[1, 2],
                "prime_total_fz_n": prime[:, 2].sum(),
                "prime_fz_over_body_weight": (
                    prime[:, 2].sum() / (sequence.total_mass_kg * 9.81)
                ),
                "max_friction_utilization": np.max(
                    sequence.friction_utilization[index]
                ),
                "newton_converged": int(
                    sequence.diagnostics.newton_converged[index]
                ),
                "newton_termination": (
                    sequence.diagnostics.newton_termination[index]
                ),
                "newton_iterations": (
                    sequence.diagnostics.newton_iterations[index]
                ),
                "newton_relative_grad_norm": (
                    sequence.diagnostics.newton_relative_grad_norm[index]
                ),
                "min_cone_margin": (
                    sequence.diagnostics.min_cone_margin[index]
                ),
                "min_alpha": sequence.diagnostics.min_alpha[index],
                "gt_left_fz_n": gt[0, 2] if gt is not None else "",
                "gt_right_fz_n": gt[1, 2] if gt is not None else "",
                "gt_force_error_norm_n": (
                    np.linalg.norm(prime - gt) if gt is not None else ""
                ),
            }
            writer.writerow(row)
    return path
