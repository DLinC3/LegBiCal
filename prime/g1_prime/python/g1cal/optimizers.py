"""The two upper calibration methods over one variance-coordinate oracle.

Only SQP--BFGS (SciPy SLSQP in a scaled scalar Cholesky coordinate) and
Frank--Wolfe with an actually executed SDP linear-minimization oracle are
exposed. Both share one released-variance first-order oracle, exact
persistent cache, production gradient, and bounds.

For the single released block-isotropic scalar the SDP LMO is algebraically
equivalent to an interval endpoint; that degeneracy is disclosed in logs and
results, tested against the analytic endpoint, and never presented as a
nontrivial dense-SPD program.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from .attempts import atomic_write_json
from .calibration import (
    CALIBRATION_STATEMENT,
    CalibrationOracle,
    LowerSolveFailure,
    RELEASED_INDEX,
    THETA13_LOWER,
    THETA13_UPPER,
)
from .covariance import BLOCKS, FLOOR_RATIO
from .gradient import cached_gradient_fn
from .paths import resolve_inside_root


OBJECTIVE_SCALE = 1000.0
SDP_DISCLOSURE = (
    "SDP-formulated block-isotropic scalar LMO; algebraically an interval "
    "endpoint for the one released 29D block"
)
RELEASED_DIM = 29
MINIMUM_PROMOTION_IMPROVEMENT = 1e-6


@dataclass(frozen=True)
class VarianceCoordinate:
    """Exact ``s <-> theta[13]`` map for the released 29D isotropic block."""

    sigma_ref: float = BLOCKS[RELEASED_INDEX].sigma_ref
    floor_ratio: float = FLOOR_RATIO
    theta_lower: float = THETA13_LOWER
    theta_upper: float = THETA13_UPPER

    @property
    def sigma_floor_sq(self) -> float:
        return (self.floor_ratio * self.sigma_ref) ** 2

    def variance(self, theta13: float) -> float:
        return (
            self.sigma_ref**2 * math.exp(2.0 * theta13) + self.sigma_floor_sq
        )

    def theta13(self, s: float) -> float:
        nominal = s - self.sigma_floor_sq
        if nominal <= 0.0:
            raise ValueError(f"variance {s} is at/below the floor")
        return 0.5 * math.log(nominal / self.sigma_ref**2)

    def ds_dtheta(self, s: float) -> float:
        return 2.0 * (s - self.sigma_floor_sq)

    def gradient_to_variance(self, dJ_dtheta13: float, s: float) -> float:
        return dJ_dtheta13 / self.ds_dtheta(s)

    @property
    def s_bounds(self) -> tuple[float, float]:
        return (self.variance(self.theta_lower), self.variance(self.theta_upper))

    # Scaled scalar Cholesky coordinate for SQP--BFGS.
    @property
    def l_bounds(self) -> tuple[float, float]:
        s_min, s_max = self.s_bounds
        return (
            math.sqrt(s_min - self.sigma_floor_sq),
            math.sqrt(s_max - self.sigma_floor_sq),
        )

    def s_from_eta(self, eta: float) -> float:
        l_min, l_max = self.l_bounds
        factor = l_min + eta * (l_max - l_min)
        return self.sigma_floor_sq + factor * factor

    def eta_from_s(self, s: float) -> float:
        l_min, l_max = self.l_bounds
        factor = math.sqrt(s - self.sigma_floor_sq)
        return (factor - l_min) / (l_max - l_min)

    def ds_deta(self, eta: float) -> float:
        l_min, l_max = self.l_bounds
        factor = l_min + eta * (l_max - l_min)
        return 2.0 * factor * (l_max - l_min)


class VarianceOracle:
    """Common first-order oracle ``(J(s), dJ/ds)`` shared by both methods."""

    def __init__(
        self,
        base_oracle,
        gradient_fn,
        *,
        coordinate: VarianceCoordinate | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.base = base_oracle
        self.gradient_fn = gradient_fn
        self.coordinate = coordinate or VarianceCoordinate()
        self.log_path = log_path
        self.call_count = 0
        self.gradient_count = 0

    def expand_theta(self, s: float) -> np.ndarray:
        theta = np.asarray(self.base.theta0, dtype=float).copy()
        theta[RELEASED_INDEX] = self.coordinate.theta13(s)
        return theta

    def _log(self, payload: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def value(self, s: float, *, label: str) -> float:
        theta = self.expand_theta(s)
        self.call_count += 1
        evaluation = self.base.evaluate(theta, label=label)
        self._log(
            {
                "event": "value",
                "label": label,
                "theta13": float(theta[RELEASED_INDEX]),
                "sigma_rad": math.sqrt(s),
                "variance_s": s,
                "precision": 1.0 / s,
                "loss": evaluation.loss,
                "loss_scaled": OBJECTIVE_SCALE * evaluation.loss,
                "cache_hit": evaluation.cache_hit,
                "component_losses": {
                    component.clip: component.loss.value
                    for component in evaluation.components
                },
            }
        )
        return evaluation.loss

    def value_and_gradient(
        self, s: float, *, label: str
    ) -> tuple[float, float]:
        loss = self.value(s, label=label)
        theta = self.expand_theta(s)
        self.gradient_count += 1
        dJ_dtheta13, gradient_meta = self.gradient_fn(theta, label=label)
        dJ_ds = self.coordinate.gradient_to_variance(dJ_dtheta13, s)
        self._log(
            {
                "event": "gradient",
                "label": label,
                "theta13": float(theta[RELEASED_INDEX]),
                "variance_s": s,
                "dJ_dtheta13": dJ_dtheta13,
                "dJ_ds": dJ_ds,
                "chain_ds_dtheta": self.coordinate.ds_dtheta(s),
                "dJ_ds_scaled": OBJECTIVE_SCALE * dJ_ds,
                "gradient_meta": gradient_meta,
            }
        )
        return loss, dJ_ds


def solve_sdp_lmo(
    dJ_ds: float,
    *,
    coordinate: VarianceCoordinate,
    solver: str = "SCS",
) -> dict:
    """Actually executed scalar-isotropic SDP LMO plus analytic parity gate.

    Pre-conditioning only: the conic program is posed in the dimensionless
    ``u = s_vertex / s_max`` with a unit-magnitude objective coefficient so
    the first-order solver resolves the tiny rad^2 variance scale. The
    feasible set and minimizer are the exact images of the natural
    covariance-space formulation; the analytic-endpoint parity gate runs on
    the mapped-back variance.
    """
    import cvxpy as cp

    s_min, s_max = coordinate.s_bounds
    identity = np.eye(RELEASED_DIM)
    coefficient_s = dJ_ds * s_max  # d objective / d u
    normalizer = max(abs(coefficient_s), 1e-16)
    gradient_matrix = (coefficient_s / normalizer / RELEASED_DIM) * identity
    u = cp.Variable()
    covariance_vertex_u = u * identity
    constraints = [
        u >= s_min / s_max,
        u <= 1.0,
        s_max * covariance_vertex_u
        - coordinate.sigma_floor_sq * identity >> 0,
    ]
    problem = cp.Problem(
        cp.Minimize(cp.trace(gradient_matrix @ covariance_vertex_u)),
        constraints,
    )
    started = time.perf_counter()
    problem.solve(solver=getattr(cp, solver), eps=1e-10, max_iters=200000)
    wall = time.perf_counter() - started
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP LMO failed with status {problem.status}")
    solved_vertex = float(u.value) * s_max
    sdp_objective = float(problem.value) * normalizer
    analytic_vertex = s_min if dJ_ds > 0.0 else s_max
    analytic_objective = dJ_ds * analytic_vertex
    flat_objective = dJ_ds == 0.0
    tolerance = max(1e-12, 1e-6 * s_max)
    parity = abs(solved_vertex - analytic_vertex) <= tolerance
    objective_parity = abs(sdp_objective - analytic_objective) <= max(
        1e-12, 1e-5 * abs(analytic_objective) + 1e-12
    )
    if flat_objective:
        # Zero gradient: every feasible vertex is optimal; the endpoint
        # convention (else-branch s_max) stands in and the FW gap is zero.
        parity = s_min - tolerance <= solved_vertex <= s_max + tolerance
        objective_parity = abs(sdp_objective) <= 1e-12
    if not (parity and objective_parity):
        raise RuntimeError(
            "SDP LMO does not match the analytic interval endpoint: "
            f"solved {solved_vertex} vs analytic {analytic_vertex}, "
            f"objective {sdp_objective} vs {analytic_objective}"
        )
    import cvxpy

    return {
        "disclosure": SDP_DISCLOSURE,
        "solver": solver,
        "cvxpy_version": cvxpy.__version__,
        "status": problem.status,
        "preconditioning": (
            "dimensionless u=s/s_max with unit-magnitude objective; exact "
            "image of the covariance-space SDP"
        ),
        "flat_objective": flat_objective,
        "sdp_vertex": solved_vertex,
        "analytic_vertex": analytic_vertex,
        # After the parity gate the exact endpoint is used for the step; the
        # conic solve is executed evidence, not the numerical authority.
        "vertex_used": analytic_vertex,
        "sdp_objective": sdp_objective,
        "analytic_objective": analytic_objective,
        "parity_tolerance": tolerance,
        "wall_seconds": wall,
    }


class OptimizerRunner:
    """SQP--BFGS and Frank--Wolfe--SDP from one shared baseline."""

    def __init__(
        self,
        oracle: VarianceOracle,
        *,
        output_root: Path,
        sqp_options: dict | None = None,
        fw_options: dict | None = None,
    ) -> None:
        self.oracle = oracle
        self.root = Path(output_root)
        self.sqp_options = {"ftol": 1e-9, "maxiter": 2, **(sqp_options or {})}
        self.fw_options = {
            "maxiter": 2,
            "armijo_beta": 0.5,
            "armijo_rho": 1e-4,
            "armijo_max_steps": 4,
            **(fw_options or {}),
        }

    def _algorithm_root(self, name: str) -> Path:
        path = self.root / "optimizers" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _trial_log(self, name: str, payload: dict) -> None:
        path = self._algorithm_root(name) / "trials.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def run_sqp_bfgs(self, *, max_iterations: int = 2) -> dict:
        from scipy.optimize import minimize

        name = "sqp_bfgs"
        coordinate = self.oracle.coordinate
        started = time.perf_counter()
        s0 = coordinate.variance(float(self.oracle.base.theta0[RELEASED_INDEX]))
        baseline_loss = self.oracle.value(s0, label=f"{name}_baseline")
        best = {"s": s0, "loss": baseline_loss, "label": f"{name}_baseline"}
        accepted: list[dict] = []
        calls = {"objective": 0}
        stop_reason = ""

        class _OptimizerAbort(Exception):
            pass

        def objective(eta_array):
            """Value only; line-search trials never pay for a gradient."""
            calls["objective"] += 1
            eta = float(np.clip(eta_array[0], 0.0, 1.0))
            s = coordinate.s_from_eta(eta)
            label = f"{name}_trial_{calls['objective']:02d}"
            try:
                loss = self.oracle.value(s, label=label)
            except LowerSolveFailure as error:
                raise _OptimizerAbort(repr(error)) from error
            self._trial_log(name, {
                "call": calls["objective"],
                "kind": "value",
                "eta": eta,
                "variance_s": s,
                "theta13": coordinate.theta13(s),
                "sigma_rad": math.sqrt(s),
                "loss": loss,
                "loss_scaled": OBJECTIVE_SCALE * loss,
            })
            if loss < best["loss"]:
                best.update({"s": s, "loss": loss, "label": label})
            return OBJECTIVE_SCALE * loss

        def jacobian(eta_array):
            calls.setdefault("jacobian", 0)
            calls["jacobian"] += 1
            eta = float(np.clip(eta_array[0], 0.0, 1.0))
            s = coordinate.s_from_eta(eta)
            label = f"{name}_jac_{calls['jacobian']:02d}"
            try:
                loss, dJ_ds = self.oracle.value_and_gradient(s, label=label)
            except LowerSolveFailure as error:
                raise _OptimizerAbort(repr(error)) from error
            dJ_deta = dJ_ds * coordinate.ds_deta(eta)
            self._trial_log(name, {
                "call": calls["jacobian"],
                "kind": "gradient",
                "eta": eta,
                "variance_s": s,
                "theta13": coordinate.theta13(s),
                "loss": loss,
                "dJ_ds": dJ_ds,
                "dJ_deta_scaled": OBJECTIVE_SCALE * dJ_deta,
            })
            if loss < best["loss"]:
                best.update({"s": s, "loss": loss, "label": label})
            return np.array([OBJECTIVE_SCALE * dJ_deta])

        def callback(eta_array):
            eta = float(eta_array[0])
            s = coordinate.s_from_eta(eta)
            accepted.append({
                "iteration": len(accepted) + 1,
                "eta": eta,
                "variance_s": s,
                "theta13": coordinate.theta13(s),
            })

        eta0 = coordinate.eta_from_s(s0)
        result_payload: dict = {}
        try:
            solution = minimize(
                objective,
                np.array([eta0]),
                method="SLSQP",
                jac=jacobian,
                bounds=[(0.0, 1.0)],
                callback=callback,
                options={
                    "maxiter": int(min(max_iterations,
                                       self.sqp_options["maxiter"])),
                    "ftol": self.sqp_options["ftol"],
                },
            )
            final_jacobian = getattr(solution, "jac", None)
            result_payload = {
                "success": bool(solution.success),
                "message": str(solution.message),
                "iterations": int(solution.nit),
                "function_calls": int(solution.nfev),
                "optimality_residual": float(
                    np.linalg.norm(np.atleast_1d(final_jacobian))
                ) if final_jacobian is not None else float("nan"),
                "final_eta": float(solution.x[0]),
                "final_variance_s": coordinate.s_from_eta(
                    float(solution.x[0])
                ),
            }
            stop_reason = "slsqp_terminated"
        except _OptimizerAbort as error:
            result_payload = {"success": False, "message": str(error)}
            stop_reason = "lower_solve_failure"

        record = {
            "schema": "g1cal_sqp_bfgs_result_v1",
            "algorithm": "sqp-bfgs",
            "method_note": (
                "SciPy SLSQP constrained SQP; its BFGS matrix approximates "
                "the upper Lagrangian Hessian and is not an exact reduced "
                "Hessian"
            ),
            "coordinate": "scaled scalar Cholesky eta in [0,1]",
            "objective_scale": OBJECTIVE_SCALE,
            "baseline": {"variance_s": s0, "loss": baseline_loss},
            "options": self.sqp_options,
            "accepted_iterations": accepted,
            "optimizer": result_payload,
            "stop_reason": stop_reason,
            "best_feasible": {
                **best,
                "theta13": coordinate.theta13(best["s"]),
                "sigma_rad": math.sqrt(best["s"]),
            },
            "oracle_calls": self.oracle.call_count,
            "wall_seconds": time.perf_counter() - started,
            "statement": CALIBRATION_STATEMENT,
        }
        atomic_write_json(self._algorithm_root(name) / "result.json", record)
        return record

    def run_frank_wolfe_sdp(self, *, max_iterations: int = 2) -> dict:
        name = "frank_wolfe_sdp"
        coordinate = self.oracle.coordinate
        options = self.fw_options
        started = time.perf_counter()
        s0 = coordinate.variance(float(self.oracle.base.theta0[RELEASED_INDEX]))
        baseline_loss = self.oracle.value(s0, label=f"{name}_baseline")
        best = {"s": s0, "loss": baseline_loss, "label": f"{name}_baseline"}
        iterations: list[dict] = []
        stop_reason = ""
        s_current = s0
        loss_current = baseline_loss
        try:
            for iteration in range(1, int(min(max_iterations,
                                              options["maxiter"])) + 1):
                _, dJ_ds = self.oracle.value_and_gradient(
                    s_current, label=f"{name}_grad_{iteration:02d}"
                )
                lmo = solve_sdp_lmo(dJ_ds, coordinate=coordinate)
                s_vertex = lmo["vertex_used"]
                direction = s_vertex - s_current
                gap = dJ_ds * (s_current - s_vertex)
                iteration_record = {
                    "iteration": iteration,
                    "variance_s": s_current,
                    "loss": loss_current,
                    "dJ_ds": dJ_ds,
                    "sdp_lmo": lmo,
                    "direction": direction,
                    "frank_wolfe_gap": gap,
                }
                if gap <= 0.0:
                    iteration_record["decision"] = "stationary_gap"
                    iterations.append(iteration_record)
                    stop_reason = "nonpositive_frank_wolfe_gap"
                    break
                gamma = 1.0
                accepted_step = None
                armijo_trials = []
                for _ in range(int(options["armijo_max_steps"])):
                    s_trial = s_current + gamma * direction
                    label = (
                        f"{name}_armijo_{iteration:02d}_gamma"
                        f"{gamma:.4f}".replace(".", "p")
                    )
                    try:
                        trial_loss = self.oracle.value(s_trial, label=label)
                    except LowerSolveFailure as error:
                        # A strict lower failure is a rejected trial, never a
                        # fabricated objective: backtrack instead of aborting.
                        armijo_trials.append({
                            "gamma": gamma,
                            "variance_s": s_trial,
                            "loss": None,
                            "lower_failed": True,
                            "error": repr(error),
                            "sufficient_decrease": False,
                        })
                        gamma *= options["armijo_beta"]
                        continue
                    sufficient = trial_loss <= (
                        loss_current
                        + options["armijo_rho"] * gamma * dJ_ds * direction
                    )
                    armijo_trials.append({
                        "gamma": gamma,
                        "variance_s": s_trial,
                        "loss": trial_loss,
                        "sufficient_decrease": sufficient,
                    })
                    if trial_loss < best["loss"]:
                        best.update(
                            {"s": s_trial, "loss": trial_loss, "label": label}
                        )
                    if sufficient:
                        accepted_step = {"gamma": gamma, "s": s_trial,
                                         "loss": trial_loss}
                        break
                    gamma *= options["armijo_beta"]
                iteration_record["armijo_trials"] = armijo_trials
                if accepted_step is None:
                    iteration_record["decision"] = "line_search_exhausted"
                    iterations.append(iteration_record)
                    stop_reason = "armijo_backtracking_exhausted"
                    break
                iteration_record["decision"] = "accepted"
                iteration_record["accepted"] = accepted_step
                iterations.append(iteration_record)
                self._trial_log(name, iteration_record)
                s_current = accepted_step["s"]
                loss_current = accepted_step["loss"]
            else:
                stop_reason = "iteration_budget_reached"
        except LowerSolveFailure as error:
            stop_reason = "lower_solve_failure"
            iterations.append({"error": repr(error)})

        record = {
            "schema": "g1cal_frank_wolfe_sdp_result_v1",
            "algorithm": "frank-wolfe-sdp",
            "sdp_disclosure": SDP_DISCLOSURE,
            "objective_scale": OBJECTIVE_SCALE,
            "baseline": {"variance_s": s0, "loss": baseline_loss},
            "options": options,
            "iterations": iterations,
            "stop_reason": stop_reason,
            "best_feasible": {
                **best,
                "theta13": coordinate.theta13(best["s"]),
                "sigma_rad": math.sqrt(best["s"]),
            },
            "oracle_calls": self.oracle.call_count,
            "wall_seconds": time.perf_counter() - started,
            "statement": CALIBRATION_STATEMENT,
        }
        atomic_write_json(self._algorithm_root(name) / "result.json", record)
        return record


def run_optimizer(
    *,
    algorithm: str,
    max_iterations: int = 2,
    output_root: str = "out/calibration",
    stream_output: bool = True,
) -> dict:
    if algorithm not in ("sqp_bfgs", "frank_wolfe_sdp"):
        raise ValueError(
            "only sqp_bfgs and frank_wolfe_sdp are supported upper methods"
        )
    base = CalibrationOracle(
        output_root=output_root, stream_output=stream_output
    )
    oracle = VarianceOracle(
        base,
        cached_gradient_fn(base),
        log_path=base.root / "optimizers/oracle_calls.jsonl",
    )
    runner = OptimizerRunner(oracle, output_root=base.root)
    if algorithm == "sqp_bfgs":
        return runner.run_sqp_bfgs(max_iterations=max_iterations)
    return runner.run_frank_wolfe_sdp(max_iterations=max_iterations)


def select_best_feasible(*, output_root: str = "out/calibration") -> dict:
    """Lowest strict aggregate loss across every evaluated theta."""
    base = CalibrationOracle(output_root=output_root)
    evaluations = base.strict_evaluations()
    if not evaluations:
        raise RuntimeError("no strict aggregate evaluations exist")
    baseline_hash = base.evaluation_hash(base.theta0)
    baseline = next(
        (payload for payload in evaluations
         if payload["theta_hash"] == baseline_hash),
        None,
    )
    if baseline is None:
        raise RuntimeError("baseline theta evaluation is missing")
    ranked = sorted(evaluations, key=lambda payload: payload["loss"])
    best = ranked[0]
    improvement = baseline["loss"] - best["loss"]
    promoted = improvement >= MINIMUM_PROMOTION_IMPROVEMENT and (
        best["theta_hash"] != baseline_hash
    )
    final = best if promoted else baseline
    component_improvements = {}
    for clip, entry in final["components"].items():
        component_improvements[clip] = {
            "baseline_loss": baseline["components"][clip]["loss"]["value"],
            "final_loss": entry["loss"]["value"],
            "improved": (
                entry["loss"]["value"]
                < baseline["components"][clip]["loss"]["value"]
            ),
        }
    optimizer_provenance = {}
    root = resolve_inside_root(output_root)
    for name in ("sqp_bfgs", "frank_wolfe_sdp"):
        result_path = root / "optimizers" / name / "result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text())
            optimizer_provenance[name] = {
                "stop_reason": payload.get("stop_reason"),
                "best_feasible": payload.get("best_feasible"),
                "success": payload.get("optimizer", {}).get("success"),
            }
    record = {
        "schema": "g1cal_final_selection_v1",
        "statement": CALIBRATION_STATEMENT,
        "baseline": {
            "theta_hash": baseline["theta_hash"],
            "theta13": baseline["theta13"],
            "loss": baseline["loss"],
        },
        "best_evaluated": {
            "theta_hash": best["theta_hash"],
            "theta13": best["theta13"],
            "loss": best["loss"],
            "label": best.get("label"),
        },
        "improvement_absolute": improvement,
        "minimum_promotion_improvement": MINIMUM_PROMOTION_IMPROVEMENT,
        "promoted": promoted,
        "final_variant": "calibrated" if promoted else "baseline",
        "final": {
            "theta": final["theta"],
            "theta_hash": final["theta_hash"],
            "theta13": final["theta13"],
            "loss": final["loss"],
            "components": {
                clip: entry["selected_attempt"]
                for clip, entry in final["components"].items()
            },
        },
        "component_improvements": component_improvements,
        "optimizer_provenance": optimizer_provenance,
        "all_evaluations": [
            {
                "theta_hash": payload["theta_hash"],
                "theta13": payload["theta13"],
                "loss": payload["loss"],
                "label": payload.get("label"),
            }
            for payload in ranked
        ],
    }
    selection_dir = root / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(selection_dir / "final_theta.json", record)
    return record
