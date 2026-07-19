"""Whole-estimator central-difference gradient in the released direction.

The production upper gradient differentiates the complete converged
lower-level estimate, never a frozen trajectory. Centered inside the frozen
``[-0.5, 2.0]`` theta interval; near a bound the stencil shrinks to the
available room, and at a bound it degrades to an explicitly labeled
one-sided difference into the feasible interior.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .attempts import atomic_write_json
from .calibration import (
    CalibrationOracle,
    RELEASED_INDEX,
    THETA13_LOWER,
    THETA13_UPPER,
    theta_hash_bytes,
)

FD_RELATIVE_STEP = 1e-3


def whole_estimator_fd_gradient(
    oracle: CalibrationOracle, theta, *, label: str
) -> tuple[float, dict]:
    theta = oracle.canonicalize_theta(theta)
    theta13 = float(theta[RELEASED_INDEX])
    nominal = FD_RELATIVE_STEP * max(1.0, abs(theta13))
    room_lower = theta13 - THETA13_LOWER
    room_upper = THETA13_UPPER - theta13
    centered_step = min(nominal, room_lower, room_upper)
    started = time.perf_counter()
    base = oracle.evaluate(theta, label=f"{label}_fd_base")
    if centered_step >= 1e-2 * nominal:
        method = "whole_estimator_central_fd"
        step = centered_step
        plus = theta.copy()
        plus[RELEASED_INDEX] += step
        minus = theta.copy()
        minus[RELEASED_INDEX] -= step
        plus_eval = oracle.evaluate(plus, label=f"{label}_fd_plus")
        minus_eval = oracle.evaluate(minus, label=f"{label}_fd_minus")
        slope = (plus_eval.loss - minus_eval.loss) / (2.0 * step)
    else:
        step = min(nominal, max(room_lower, room_upper))
        direction = 1.0 if room_upper >= room_lower else -1.0
        method = (
            "whole_estimator_one_sided_forward_fd_at_active_bound"
            if direction > 0
            else "whole_estimator_one_sided_backward_fd_at_active_bound"
        )
        shifted = theta.copy()
        shifted[RELEASED_INDEX] += direction * step
        side_label = "plus" if direction > 0 else "minus"
        side_eval = oracle.evaluate(shifted, label=f"{label}_fd_{side_label}")
        slope = direction * (side_eval.loss - base.loss) / step
        plus_eval = side_eval if direction > 0 else base
        minus_eval = base if direction > 0 else side_eval
    record = {
        "method": method,
        "step": step,
        "nominal_step": nominal,
        "base_theta13": theta13,
        "base_loss": base.loss,
        "plus_loss": plus_eval.loss,
        "minus_loss": minus_eval.loss,
        "plus_component_losses": {
            component.clip: component.loss.value
            for component in plus_eval.components
        },
        "minus_component_losses": {
            component.clip: component.loss.value
            for component in minus_eval.components
        },
        "plus_cache_hit": plus_eval.cache_hit,
        "minus_cache_hit": minus_eval.cache_hit,
        "wall_seconds": time.perf_counter() - started,
    }
    fd_dir = oracle.root / "gradients/fd"
    fd_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(fd_dir / f"{theta_hash_bytes(theta)[:16]}.json", record)
    return float(slope), record


def cached_gradient_fn(oracle: CalibrationOracle):
    """Per-theta persistent gradient cache over the FD production method."""

    def gradient(theta, *, label: str):
        theta = oracle.canonicalize_theta(theta)
        evaluation_hash = oracle.evaluation_hash(theta)
        cache_dir = oracle.root / "gradients/cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (
            f"{oracle.problem_hash[:16]}_{evaluation_hash[:16]}.json"
        )
        if cache_file.is_file():
            payload = json.loads(cache_file.read_text())
            if (
                payload.get("problem_hash") != oracle.problem_hash
                or payload.get("evaluation_hash") != evaluation_hash
            ):
                raise RuntimeError(
                    f"gradient cache identity mismatch: {cache_file}"
                )
            return payload["dJ_dtheta13"], {
                **payload["meta"], "gradient_cache_hit": True,
            }
        value, meta = whole_estimator_fd_gradient(oracle, theta, label=label)
        atomic_write_json(cache_file, {
            "schema": "g1cal_gradient_cache_v1",
            "problem_hash": oracle.problem_hash,
            "evaluation_hash": evaluation_hash,
            "theta_bytes_hash": theta_hash_bytes(theta),
            "dJ_dtheta13": value,
            "meta": meta,
        })
        return value, {**meta, "gradient_cache_hit": False}

    return gradient
