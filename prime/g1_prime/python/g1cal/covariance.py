"""Version-1 block-isotropic covariance parameterization.

Theta contains log standard-deviation scale factors.  Every physical block has
its own reference standard deviation and floor, so no epsilon mixes units.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CovarianceBlock:
    name: str
    matrix: str
    dim: int
    sigma_ref: float
    unit: str


BLOCKS: tuple[CovarianceBlock, ...] = (
    CovarianceBlock("p0_base_xy", "P0", 2, 0.05, "m"),
    CovarianceBlock("p0_base_z", "P0", 1, 0.03, "m"),
    CovarianceBlock("p0_base_orientation", "P0", 3, 0.05, "rad"),
    CovarianceBlock("p0_joint_position", "P0", 29, 0.03, "rad"),
    CovarianceBlock("p0_base_linear_velocity", "P0", 3, 0.10, "m/s"),
    CovarianceBlock("p0_base_angular_velocity", "P0", 3, 0.10, "rad/s"),
    CovarianceBlock("p0_joint_velocity", "P0", 29, 0.20, "rad/s"),
    CovarianceBlock("q_base_force", "Q", 3, 20.0, "N"),
    CovarianceBlock("q_base_torque", "Q", 3, 5.0, "N*m"),
    CovarianceBlock("q_joint_torque", "Q", 29, 3.0, "N*m"),
    CovarianceBlock("r_base_xy", "R", 2, 0.01, "m"),
    CovarianceBlock("r_base_z", "R", 1, 0.01, "m"),
    CovarianceBlock("r_base_orientation", "R", 3, 0.02, "rad"),
    CovarianceBlock("r_joint_position", "R", 29, 0.01, "rad"),
    CovarianceBlock("r_base_linear_velocity", "R", 3, 0.05, "m/s"),
    CovarianceBlock("r_base_angular_velocity", "R", 3, 0.05, "rad/s"),
    CovarianceBlock("r_joint_velocity", "R", 29, 0.10, "rad/s"),
)

THETA_LOWER = -3.0
THETA_UPPER = 3.0
FLOOR_RATIO = 1e-3


@dataclass(frozen=True)
class CovarianceSet:
    theta: np.ndarray
    sigma_by_block: dict[str, float]
    variance_diag: dict[str, np.ndarray]
    precision_diag: dict[str, np.ndarray]
    whitening_diag: dict[str, np.ndarray]
    precision_jacobian: dict[str, np.ndarray]
    config_hash: str


class CovarianceParameterization:
    blocks = BLOCKS

    @property
    def size(self) -> int:
        return len(self.blocks)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(self.size, THETA_LOWER),
            np.full(self.size, THETA_UPPER),
        )

    def evaluate(self, theta: Iterable[float]) -> CovarianceSet:
        theta = np.asarray(tuple(theta), dtype=float)
        if theta.shape != (self.size,):
            raise ValueError(f"theta must have shape ({self.size},)")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite")
        lower, upper = self.bounds
        if np.any(theta < lower) or np.any(theta > upper):
            raise ValueError("theta outside locked [-3,3] bounds")

        variances: dict[str, list[np.ndarray]] = {"P0": [], "Q": [], "R": []}
        precisions: dict[str, list[np.ndarray]] = {"P0": [], "Q": [], "R": []}
        whitening: dict[str, list[np.ndarray]] = {"P0": [], "Q": [], "R": []}
        jacobians: dict[str, list[np.ndarray]] = {"P0": [], "Q": [], "R": []}
        sigma_by_block: dict[str, float] = {}

        for index, (value, block) in enumerate(zip(theta, self.blocks, strict=True)):
            nominal_variance = block.sigma_ref**2 * np.exp(2.0 * value)
            floor_variance = (FLOOR_RATIO * block.sigma_ref) ** 2
            variance = nominal_variance + floor_variance
            precision = 1.0 / variance
            sigma_by_block[block.name] = float(np.sqrt(variance))
            variances[block.matrix].append(np.full(block.dim, variance))
            precisions[block.matrix].append(np.full(block.dim, precision))
            whitening[block.matrix].append(np.full(block.dim, np.sqrt(precision)))
            jac = np.zeros((block.dim, self.size))
            jac[:, index] = -2.0 * nominal_variance / (variance * variance)
            jacobians[block.matrix].append(jac)

        variance_diag = {key: np.concatenate(value) for key, value in variances.items()}
        precision_diag = {key: np.concatenate(value) for key, value in precisions.items()}
        whitening_diag = {key: np.concatenate(value) for key, value in whitening.items()}
        precision_jacobian = {
            key: np.concatenate(value, axis=0) for key, value in jacobians.items()
        }
        canonical = {
            "schema": "g1cal_covariance_v1",
            "theta": theta.tolist(),
            "blocks": [block.__dict__ for block in self.blocks],
            "floor_ratio": FLOOR_RATIO,
        }
        config_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CovarianceSet(
            theta=theta.copy(),
            sigma_by_block=sigma_by_block,
            variance_diag=variance_diag,
            precision_diag=precision_diag,
            whitening_diag=whitening_diag,
            precision_jacobian=precision_jacobian,
            config_hash=config_hash,
        )

    def write_precision_file(self, covariance: CovarianceSet, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [f"# config_hash={covariance.config_hash}"]
        for label, matrix in (("p0", "P0"), ("q", "Q"), ("r", "R")):
            values = ",".join(f"{v:.17g}" for v in covariance.precision_diag[matrix])
            rows.append(f"{label},{values}")
        path.write_text("\n".join(rows) + "\n")

    def truth_theta(self) -> np.ndarray:
        return np.zeros(self.size)

    def initial_theta(self) -> np.ndarray:
        # P0/R start 2x too large in sigma; Q starts 0.5x.
        return np.array([np.log(2.0)] * 7 + [np.log(0.5)] * 3 + [np.log(2.0)] * 7)
