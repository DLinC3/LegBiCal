"""Two-clip covariance-calibration objective with strict promotion.

``CalibrationOracle`` is the exact persistent objective shared by both upper
optimizers: it canonicalizes theta, resolves an on-disk content-addressed
cache first, and for a new theta runs both clips sequentially through
immutable attempts and bounded regularization-preserving continuation. One
atomic aggregate evaluation is written only after both clips pass the full
strict gate; a failed lower solve is recorded and never converted into a
fabricated objective value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .attempts import (
    atomic_write_json,
    resolve_selected,
    strict_gate_verdict,
    try_resolve_selected,
)
from .backend import PrimeMotionFieBackend
from .covariance import CovarianceParameterization
from .horizon_solver import solve_horizon_attempt, validate_warm_start_dir
from .loss import LossResult, SE3_LOG_SCHEMA, trajectory_loss_arrays
from .paths import project_root, resolve_inside_root


HORIZON_STATES = 501
HORIZON_TRANSITIONS = 500
DT_SECONDS = 0.02

DEFAULT_OUTPUT_ROOT = "out/calibration"

RELEASED_INDEX = 13
RELEASED_NAME = "r_joint_position"
THETA13_LOWER = -0.5
THETA13_UPPER = 2.0

MAX_ROUNDS_PER_COMPONENT = 4

CALIBRATION_STATEMENT = (
    "The covariance was calibrated by directly minimizing the SE(3)-log "
    "trajectory loss on the two released 501-state running clips; no "
    "accuracy beyond these two clips is claimed."
)

V3_GATE_SCHEMA = {
    "contact_certification_mode": "action_stationarity_plus_shooting_defect_v3",
    "inner_relative_grad_tolerance": 1e-7,
    "defect_gate": 1e-6,
}

# Baseline theta before calibration: unit P0/other-R scales, inflated Q and
# joint-position measurement scales (log 4).
INITIAL_THETA = tuple(
    [0.0] * 7 + [float(np.log(4.0))] * 3 + [0.0] * 3
    + [float(np.log(4.0))] + [0.0] * 3
)


@dataclass(frozen=True)
class ComponentSpec:
    clip: str
    motion: str
    lower_root: str
    lower_config_template: str
    prior_state: str
    upper_truth: str
    profile_id: str = "g1"


@dataclass(frozen=True)
class SolverPolicy:
    horizon: int = HORIZON_STATES
    max_rounds: int = MAX_ROUNDS_PER_COMPONENT
    max_iter_per_round: int = 100
    n_thread: int = 8
    checkpoint_interval: int = 10
    newton_max_iters: int = 1000
    initial_regularization: float = 1.0
    defect_gate: float = 1e-6
    stale_round_stop: int = 2


@dataclass(frozen=True)
class ComponentEvaluation:
    theta_hash: str
    clip: str
    selected_attempt: str
    loss: LossResult
    attempts: tuple[str, ...]
    lower_attempts: int
    cache_hit: bool
    wall_seconds: float


@dataclass(frozen=True)
class AggregateEvaluation:
    evaluation_id: str
    theta: np.ndarray
    theta_hash: str
    loss: float
    components: tuple[ComponentEvaluation, ...]
    cache_hit: bool
    wall_seconds: float


class LowerSolveFailure(RuntimeError):
    """A clip failed its bounded continuation at this theta; the trial is
    rejected, never converted into a fabricated objective value."""


def component_specs() -> dict[str, ComponentSpec]:
    """The two released clips (repo-relative paths)."""
    specs = {}
    for clip, motion in (("run1", "run1_subject2"), ("run2", "run2_subject1")):
        root = f"data/clips/{clip}"
        specs[clip] = ComponentSpec(
            clip=clip,
            motion=motion,
            lower_root=root,
            lower_config_template="configs/lower/h501_template.xml",
            prior_state=f"{root}/prior_state.csv",
            upper_truth=f"{root}/upper_truth_h501.csv",
        )
    return specs


def initial_theta() -> np.ndarray:
    return np.asarray(INITIAL_THETA, dtype=float)


def calibrated_theta() -> np.ndarray:
    payload = json.loads(
        resolve_inside_root("data/calibrated/theta.json").read_text()
    )
    theta = np.asarray(payload["theta"], dtype=float)
    if theta.shape != (17,):
        raise ValueError("calibrated theta must have dimension 17")
    return theta


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    """Hashes of every released source that can change a lower objective.

    Caches deliberately follow source identity rather than a platform-specific
    shared-library hash, so a clean rebuild of identical sources reuses the
    same namespace while any solver/calibration implementation edit does not.
    """
    python_sources = (
        "attempts.py",
        "backend.py",
        "calibration.py",
        "covariance.py",
        "horizon_solver.py",
        "loss.py",
        "paths.py",
        "profiles.py",
    )
    roots = (
        resolve_inside_root("cpp/include/g1cal"),
        resolve_inside_root("cpp/apps"),
        resolve_inside_root("cpp/bindings"),
        resolve_inside_root("third_party/PRIME/include"),
        resolve_inside_root("third_party/PRIME/src"),
        resolve_inside_root("third_party/PRIME/experiments/common"),
    )
    files = [
        resolve_inside_root(f"python/g1cal/{name}")
        for name in python_sources
    ]
    for root in roots:
        files.extend(
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".hpp", ".hxx", ".cpp"}
        )
    files.extend((
        resolve_inside_root("CMakeLists.txt"),
        resolve_inside_root("cpp/CMakeLists.txt"),
        resolve_inside_root("third_party/PRIME/CMakeLists.txt"),
        resolve_inside_root("third_party/PRIME/VENDORED.md"),
    ))
    return {
        str(path.relative_to(project_root())): _sha256_file(path)
        for path in sorted(set(files))
    }


def theta_hash_bytes(theta: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(theta, dtype="<f8").tobytes()
    ).hexdigest()


def assert_no_truth_leakage(request_payload: dict, specs=None) -> None:
    """No lower request field may reference ground truth."""
    specs = specs or component_specs()
    forbidden = [spec.upper_truth for spec in specs.values()]
    forbidden += ["upper_truth", "gt_clip", "gt__qpos", "contact_debug"]
    flattened = json.dumps(request_payload, sort_keys=True)
    for token in forbidden:
        if token in flattened:
            raise RuntimeError(
                f"lower request leaks privileged ground-truth data: {token}"
            )


def _meaningful_continuation_improvement(previous: dict, current: dict) -> bool:
    """Predeclared numerical progress rule for the bounded continuation."""
    previous_stop = float(previous.get("stop", float("inf")))
    current_stop = float(current.get("stop", float("inf")))
    previous_cost = float(previous.get("final_cost", float("inf")))
    current_cost = float(current.get("final_cost", float("inf")))
    previous_defect = float(previous.get("defect_max", float("inf")))
    current_defect = float(current.get("defect_max", float("inf")))
    stop_improved = (
        np.isfinite(previous_stop) and np.isfinite(current_stop)
        and current_stop <= 0.9 * previous_stop
    )
    cost_scale = max(abs(previous_cost), 1.0)
    cost_improved = (
        np.isfinite(previous_cost) and np.isfinite(current_cost)
        and current_cost <= previous_cost - 1e-6 * cost_scale
    )
    defect_improved = (
        np.isfinite(previous_defect) and np.isfinite(current_defect)
        and current_defect <= 0.5 * previous_defect
    )
    return bool(stop_improved or cost_improved or defect_improved)


class CalibrationOracle:
    """Exact persistent two-clip objective with strict promotion."""

    def __init__(
        self,
        *,
        backend: PrimeMotionFieBackend | None = None,
        stream_output: bool = True,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        policy: SolverPolicy | None = None,
    ) -> None:
        self.specs = component_specs()
        self.policy = policy or SolverPolicy()
        self.theta0 = initial_theta()
        self.parameterization = CovarianceParameterization()
        self.backend = backend or PrimeMotionFieBackend()
        self.stream_output = stream_output
        self.root = resolve_inside_root(output_root, must_exist=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.problem_hash = self._problem_hash()
        self.cache_hit_count = 0
        self.evaluation_count = 0
        self._truth: dict[str, np.ndarray] = {}

    def _problem_hash(self) -> str:
        """Content identity of the frozen problem: inputs, model, loss,
        solver policy. Any change starts a fresh cache namespace."""
        from .profiles import load_model_profile

        lower_hashes = {}
        truth_hashes = {}
        for clip, spec in self.specs.items():
            root = resolve_inside_root(spec.lower_root)
            lower_hashes[clip] = {
                name: _sha256_file(root / name)
                for name in (
                    "q_sense.csv", "v_sense.csv", "tau_sense.csv",
                    "prior_state.csv",
                )
            }
            truth_hashes[clip] = _sha256_file(
                resolve_inside_root(spec.upper_truth)
            )
        profile = load_model_profile("g1")
        return _canonical_hash(
            {
                "schema": "g1cal_calibration_problem_v2",
                "template": _sha256_file(
                    resolve_inside_root(
                        next(iter(self.specs.values())).lower_config_template
                    )
                ),
                "lower_inputs": lower_hashes,
                "upper_truth": truth_hashes,
                "profile_key": profile.cache_key,
                "covariance_baseline_config_hash": (
                    self.parameterization.evaluate(self.theta0).config_hash
                ),
                "implementation_sources": _implementation_hashes(),
                "loss_schema": SE3_LOG_SCHEMA,
                "solver_policy": asdict(self.policy),
                "released_index": RELEASED_INDEX,
                "theta13_bounds": [THETA13_LOWER, THETA13_UPPER],
                "v3_gate": V3_GATE_SCHEMA,
            }
        )

    # -- identity ---------------------------------------------------------

    def canonicalize_theta(self, theta) -> np.ndarray:
        array = np.asarray(tuple(theta), dtype=float)
        if array.shape != (17,):
            raise ValueError("theta must have dimension 17")
        if not np.all(np.isfinite(array)):
            raise ValueError("theta must be finite")
        frozen = [index for index in range(17) if index != RELEASED_INDEX]
        if not np.array_equal(array[frozen], self.theta0[frozen]):
            raise ValueError(
                "only theta[13] (R/r_joint_position) is released; all other "
                "indices must remain bitwise at the frozen baseline theta"
            )
        if not (THETA13_LOWER <= array[RELEASED_INDEX] <= THETA13_UPPER):
            raise ValueError(
                f"theta[13]={array[RELEASED_INDEX]} outside frozen "
                f"[{THETA13_LOWER},{THETA13_UPPER}]"
            )
        return array

    def evaluation_hash(self, theta: np.ndarray) -> str:
        return _canonical_hash(
            {
                "problem_hash": self.problem_hash,
                "theta_bytes": theta_hash_bytes(theta),
            }
        )

    def evaluation_dir(self, theta: np.ndarray) -> Path:
        return self.root / "evaluations" / (
            f"theta_{self.evaluation_hash(theta)[:16]}"
        )

    def _ledger_line(self, payload: dict) -> None:
        path = self.root / "evaluations/ledger.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **payload,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def truth_states(self, clip: str) -> np.ndarray:
        if clip not in self._truth:
            spec = self.specs[clip]
            truth = np.loadtxt(
                resolve_inside_root(spec.upper_truth), delimiter=",", ndmin=2
            )
            if truth.shape != (HORIZON_STATES, 71):
                raise RuntimeError(f"upper truth shape {truth.shape}")
            self._truth[clip] = truth
        return self._truth[clip]

    # -- warm sources -----------------------------------------------------

    def _strict_selected_components(self, clip: str) -> list[dict]:
        sources = []
        evaluations = self.root / "evaluations"
        if evaluations.is_dir():
            for theta_dir in sorted(evaluations.glob("theta_*")):
                theta_file = theta_dir / "theta.json"
                if not theta_file.is_file():
                    continue
                payload = json.loads(theta_file.read_text())
                if payload.get("problem_hash") != self.problem_hash:
                    continue
                theta = np.asarray(payload["theta"], dtype=float)
                selected = try_resolve_selected(theta_dir / clip)
                if selected is None:
                    continue
                sources.append(
                    {
                        "theta13": float(theta[RELEASED_INDEX]),
                        "theta_hash": payload["evaluation_hash"],
                        "path": str(selected.relative_to(project_root())),
                    }
                )
        return sources

    def nearest_warm_source(self, clip: str, theta13: float) -> dict | None:
        sources = self._strict_selected_components(clip)
        reference = resolve_inside_root(
            f"data/clips/{clip}/reference_solution", must_exist=False
        )
        if (reference / "reference_manifest.json").is_file():
            relative_reference = str(reference.relative_to(project_root()))
            validate_warm_start_dir(
                relative_reference,
                expected_profile_id=self.specs[clip].profile_id,
            )
            # The shipped converged estimate at the calibrated covariance is
            # a permanently eligible initializer.
            sources.append(
                {
                    "theta13": float(calibrated_theta()[RELEASED_INDEX]),
                    "theta_hash": "shipped_reference",
                    "path": relative_reference,
                    "reference": True,
                }
            )
        if not sources:
            return None
        sources.sort(
            key=lambda item: (abs(item["theta13"] - theta13), item["theta_hash"])
        )
        return sources[0]

    # -- component execution ---------------------------------------------

    def _generate_component_config(
        self, spec: ComponentSpec, component_dir: Path
    ) -> str:
        source = resolve_inside_root(spec.lower_config_template)
        tree = ET.parse(source)
        config_root = tree.getroot()
        solver = config_root.find("solver")
        data = config_root.find("data")
        if solver is None or data is None:
            raise RuntimeError("lower template lacks solver/data elements")
        solver.set("horizon", str(self.policy.horizon))
        solver.set("start_idx", "0")
        solver.set("max_iter", str(self.policy.max_iter_per_round))
        solver.set("n_thread", str(self.policy.n_thread))
        solver.set("callbacks", "true")
        lower_root = resolve_inside_root(spec.lower_root)
        data.set("q", str(lower_root / "q_sense.csv"))
        data.set("v", str(lower_root / "v_sense.csv"))
        data.set("u", str(lower_root / "tau_sense.csv"))
        output = component_dir / "configs" / "solve.xml"
        output.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return str(output.relative_to(project_root()))

    def _write_theta_inputs(self, theta: np.ndarray, eval_dir: Path) -> str:
        eval_dir.mkdir(parents=True, exist_ok=True)
        precision = eval_dir / "precision.csv"
        covariance = self.parameterization.evaluate(theta)
        if not precision.is_file():
            self.parameterization.write_precision_file(covariance, precision)
        theta_file = eval_dir / "theta.json"
        if not theta_file.is_file():
            atomic_write_json(
                theta_file,
                {
                    "schema": "g1cal_theta_v1",
                    "theta": theta.tolist(),
                    "theta_bytes_hash": theta_hash_bytes(theta),
                    "evaluation_hash": self.evaluation_hash(theta),
                    "problem_hash": self.problem_hash,
                    "covariance_config_hash": covariance.config_hash,
                    "sigma_r_joint_position_rad": float(
                        covariance.sigma_by_block[RELEASED_NAME]
                    ),
                    "loss_schema": SE3_LOG_SCHEMA,
                },
            )
        return covariance.config_hash

    def _write_component_failure(
        self,
        eval_dir: Path,
        clip: str,
        theta_hash: str,
        *,
        reason: str,
        attempts: list[str],
    ) -> None:
        """Honest negative cache: this exact theta failed under the frozen
        solver policy. Future evaluations fail fast instead of replaying a
        bit-identical bounded continuation; the failed attempts themselves
        remain preserved and are never reused."""
        atomic_write_json(
            eval_dir / clip / "component_failed.json",
            {
                "schema": "g1cal_component_failure_v1",
                "clip": clip,
                "theta_hash": theta_hash,
                "reason": reason,
                "attempts": attempts,
                "solver_policy": asdict(self.policy),
            },
        )

    def _solve_component(
        self,
        theta: np.ndarray,
        clip: str,
        eval_dir: Path,
        theta_hash: str,
    ) -> tuple[str, list[str], int]:
        """Bounded regularization-preserving continuation to a strict pass."""
        spec = self.specs[clip]
        component_dir = eval_dir / clip
        component_dir.mkdir(parents=True, exist_ok=True)
        config = self._generate_component_config(spec, component_dir)
        precision = str(
            (eval_dir / "precision.csv").relative_to(project_root())
        )
        warm = self.nearest_warm_source(clip, float(theta[RELEASED_INDEX]))
        warm_dir = warm["path"] if warm else ""
        regularization = self.policy.initial_regularization
        previous_summary: dict | None = None
        stale_rounds = 0
        attempts: list[str] = []
        continuation = {
            "schema": "g1cal_component_continuation_v1",
            "clip": clip,
            "theta_hash": theta_hash,
            "warm_source": warm,
            "rounds": [],
            "status": "running",
        }
        record_path = component_dir / "continuation.json"
        atomic_write_json(record_path, continuation)
        for round_index in range(self.policy.max_rounds):
            request_payload = {
                "config": config,
                "config_xml": resolve_inside_root(config).read_text(),
                "precision": precision,
                "prior_state": spec.prior_state,
                "warm_start_dir": warm_dir,
            }
            assert_no_truth_leakage(request_payload, self.specs)
            try:
                result = solve_horizon_attempt(
                    parent=component_dir,
                    request_id=(
                        f"g1cal_{clip}_{theta_hash[:12]}"
                        f"_round{round_index + 1}"
                    ),
                    config=config,
                    profile_id=spec.profile_id,
                    covariance_precision_file=precision,
                    prior_state_file=spec.prior_state,
                    warm_start_dir=warm_dir,
                    checkpoint_interval=self.policy.checkpoint_interval,
                    newton_max_iters=self.policy.newton_max_iters,
                    initial_regularization=regularization,
                    stream_output=self.stream_output,
                    gate="strict",
                    attempt_label=f"g1cal_{clip}_theta_{theta_hash[:12]}",
                    attempt_metadata={
                        "theta_hash": theta_hash,
                        "theta13": float(theta[RELEASED_INDEX]),
                        "round": round_index + 1,
                        "warm_start_dir": warm_dir,
                        "initial_regularization": regularization,
                    },
                    execution_record={
                        "schema": "g1cal_solve_execution_v1",
                        "clip": clip,
                        "theta_hash": theta_hash,
                        "round": round_index + 1,
                        "horizon": self.policy.horizon,
                        "max_iter": self.policy.max_iter_per_round,
                        "n_thread": self.policy.n_thread,
                        "initial_regularization": regularization,
                        "warm_start_dir": warm_dir,
                    },
                    backend=self.backend,
                )
            except Exception as error:
                continuation["status"] = "failed"
                continuation["error"] = repr(error)
                atomic_write_json(record_path, continuation)
                self._write_component_failure(
                    eval_dir, clip, theta_hash,
                    reason=f"round {round_index + 1} raised {error!r}",
                    attempts=attempts,
                )
                raise LowerSolveFailure(
                    f"clip {clip} round {round_index + 1} raised: {error!r}"
                ) from error
            attempts.append(result.output_dir)
            summary = result.summary
            gates = strict_gate_verdict(resolve_inside_root(result.output_dir))
            meaningful = (
                True if previous_summary is None
                else _meaningful_continuation_improvement(
                    previous_summary, summary
                )
            )
            stale_rounds = 0 if meaningful else stale_rounds + 1
            continuation["rounds"].append(
                {
                    "round": round_index + 1,
                    "attempt": result.output_dir,
                    "request_hash": result.request_hash,
                    "warm_start_dir": warm_dir,
                    "initial_regularization": regularization,
                    "final_regularization": summary.get("final_preg"),
                    "iterations": summary.get("iterations"),
                    "solved": summary.get("solved"),
                    "stop": summary.get("stop"),
                    "final_cost": summary.get("final_cost"),
                    "defect_max": summary.get("defect_max"),
                    "inner_stationarity_rejected": summary.get(
                        "inner_stationarity_rejected"
                    ),
                    "meaningful_improvement": meaningful,
                    "consecutive_stale_rounds": stale_rounds,
                    "strict_gate_passed": gates["all_passed"],
                    "wall_seconds": result.wall_seconds,
                }
            )
            atomic_write_json(record_path, continuation)
            if gates["all_passed"]:
                continuation["status"] = "passed"
                continuation["selected_attempt"] = result.output_dir
                atomic_write_json(record_path, continuation)
                selected = resolve_selected(component_dir)
                return (
                    str(selected.relative_to(project_root())),
                    attempts,
                    len(attempts),
                )
            if stale_rounds >= self.policy.stale_round_stop:
                continuation["status"] = "stopped_stale"
                atomic_write_json(record_path, continuation)
                self._write_component_failure(
                    eval_dir, clip, theta_hash,
                    reason=f"{stale_rounds} consecutive stale rounds",
                    attempts=attempts,
                )
                raise LowerSolveFailure(
                    f"clip {clip} stopped after {stale_rounds} stale rounds "
                    f"at theta {theta_hash[:12]}"
                )
            final_regularization = float(
                summary.get("final_preg", regularization)
            )
            if not np.isfinite(final_regularization) or (
                final_regularization <= 0.0
            ):
                continuation["status"] = "failed_invariant"
                atomic_write_json(record_path, continuation)
                self._write_component_failure(
                    eval_dir, clip, theta_hash,
                    reason="invalid final regularization",
                    attempts=attempts,
                )
                raise LowerSolveFailure(
                    f"clip {clip} produced invalid regularization"
                )
            warm_dir = result.output_dir
            regularization = final_regularization
            previous_summary = summary
        continuation["status"] = "round_budget_exhausted"
        atomic_write_json(record_path, continuation)
        self._write_component_failure(
            eval_dir, clip, theta_hash,
            reason=(
                f"exhausted {self.policy.max_rounds} continuation rounds "
                "without a strict pass"
            ),
            attempts=attempts,
        )
        raise LowerSolveFailure(
            f"clip {clip} exhausted its {self.policy.max_rounds} "
            f"continuation rounds at theta {theta_hash[:12]}"
        )

    def _validate_component_result(
        self, clip: str, attempt: Path, expected_covariance_hash: str
    ) -> dict:
        summary = json.loads((attempt / "solve_summary.json").read_text())
        checks = {
            "solved": bool(summary.get("solved", False)),
            "zero_inner_rejects": int(
                summary.get("inner_stationarity_rejected", -1)
            ) == 0,
            "v3_mode": summary.get("contact_certification_mode") == (
                V3_GATE_SCHEMA["contact_certification_mode"]
            ),
            "defect": float(summary.get("defect_max", 1.0)) < (
                self.policy.defect_gate
            ),
            "horizon": int(summary.get("n_running_models", -1)) == (
                HORIZON_STATES
            ),
            "dimensions": (
                int(summary.get("nx", -1)) == 71
                and int(summary.get("ndx", -1)) == 70
            ),
            "covariance": summary.get("covariance_config_hash") == (
                expected_covariance_hash
            ),
            "contact_health": bool(
                summary.get("contact_health_passed", False)
            ),
        }
        if not all(checks.values()):
            raise LowerSolveFailure(
                f"clip {clip} failed strict validation {checks}: {attempt}"
            )
        return summary

    def _evaluate_component_locked(
        self,
        theta: np.ndarray,
        clip: str,
        eval_dir: Path,
        theta_hash: str,
        covariance_hash: str,
        *,
        label: str,
    ) -> ComponentEvaluation:
        """Evaluate one clip while the oracle lock is held."""
        component_started = time.perf_counter()
        failure_memo = eval_dir / clip / "component_failed.json"
        if failure_memo.is_file():
            memo = json.loads(failure_memo.read_text())
            self._ledger_line({
                "event": "failed_cache_hit",
                "label": label,
                "theta_hash": theta_hash,
                "clip": clip,
                "reason": memo["reason"],
            })
            raise LowerSolveFailure(
                f"exact theta {theta_hash[:12]} already failed for {clip} "
                f"under the frozen solver policy ({memo['reason']}); failed "
                "attempts are never reused"
            )
        cache_hit = False
        attempts: tuple[str, ...] = ()
        lower_attempts = 0
        existing = try_resolve_selected(eval_dir / clip)
        if existing is not None:
            selected = str(existing.relative_to(project_root()))
            cache_hit = True
        else:
            selected, attempt_list, lower_attempts = self._solve_component(
                theta, clip, eval_dir, theta_hash
            )
            attempts = tuple(attempt_list)
        attempt_dir = resolve_inside_root(selected)
        summary = self._validate_component_result(
            clip, attempt_dir, covariance_hash
        )
        xs = np.loadtxt(
            attempt_dir / "xs_results_fddp.csv", delimiter=",", ndmin=2
        )
        loss = trajectory_loss_arrays(xs[1:], self.truth_states(clip))
        component = ComponentEvaluation(
            theta_hash=theta_hash,
            clip=clip,
            selected_attempt=selected,
            loss=loss,
            attempts=attempts,
            lower_attempts=lower_attempts,
            cache_hit=cache_hit,
            wall_seconds=time.perf_counter() - component_started,
        )
        self._ledger_line({
            "event": "component_done",
            "label": label,
            "theta_hash": theta_hash,
            "clip": clip,
            "selected_attempt": selected,
            "loss": loss.value,
            "lower_attempts": lower_attempts,
            "solve_stop": summary.get("stop"),
            "defect_max": summary.get("defect_max"),
        })
        return component

    def evaluate_component(
        self, theta, clip: str, *, label: str = "component_trial"
    ) -> ComponentEvaluation:
        """Solve/evaluate exactly one released clip at ``theta``.

        This is the lower-level command API. Upper calibration continues to
        use :meth:`evaluate`, whose objective is always the two-clip mean.
        """
        if clip not in self.specs:
            raise ValueError(f"unknown clip {clip!r}; expected run1 or run2")
        theta = self.canonicalize_theta(theta)
        theta_hash = self.evaluation_hash(theta)
        eval_dir = self.evaluation_dir(theta)
        evaluation_file = eval_dir / "evaluation.json"
        if evaluation_file.is_file():
            aggregate = self._evaluation_from_payload(
                theta, json.loads(evaluation_file.read_text()), cache_hit=True
            )
            return next(item for item in aggregate.components if item.clip == clip)
        lock_path = self.root / ".oracle.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if evaluation_file.is_file():
                    aggregate = self._evaluation_from_payload(
                        theta,
                        json.loads(evaluation_file.read_text()),
                        cache_hit=True,
                    )
                    return next(
                        item for item in aggregate.components
                        if item.clip == clip
                    )
                covariance_hash = self._write_theta_inputs(theta, eval_dir)
                return self._evaluate_component_locked(
                    theta, clip, eval_dir, theta_hash, covariance_hash,
                    label=label,
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # -- public objective -------------------------------------------------

    def evaluate(self, theta, *, label: str = "trial") -> AggregateEvaluation:
        theta = self.canonicalize_theta(theta)
        theta_hash = self.evaluation_hash(theta)
        eval_dir = self.evaluation_dir(theta)
        evaluation_file = eval_dir / "evaluation.json"
        started = time.perf_counter()
        if evaluation_file.is_file():
            payload = json.loads(evaluation_file.read_text())
            if payload.get("problem_hash") != self.problem_hash:
                raise RuntimeError(
                    "cached evaluation belongs to a different problem "
                    f"definition: {evaluation_file}"
                )
            self.cache_hit_count += 1
            self._ledger_line(
                {
                    "event": "cache_hit",
                    "label": label,
                    "theta_hash": theta_hash,
                    "theta13": float(theta[RELEASED_INDEX]),
                    "loss": payload["loss"],
                }
            )
            return self._evaluation_from_payload(theta, payload, cache_hit=True)

        lock_path = self.root / ".oracle.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if evaluation_file.is_file():
                    payload = json.loads(evaluation_file.read_text())
                    self.cache_hit_count += 1
                    return self._evaluation_from_payload(
                        theta, payload, cache_hit=True
                    )
                self._ledger_line(
                    {
                        "event": "evaluation_requested",
                        "label": label,
                        "theta_hash": theta_hash,
                        "theta13": float(theta[RELEASED_INDEX]),
                    }
                )
                covariance_hash = self._write_theta_inputs(theta, eval_dir)
                components = [
                    self._evaluate_component_locked(
                        theta, clip, eval_dir, theta_hash, covariance_hash,
                        label=label,
                    )
                    for clip in sorted(self.specs)
                ]
                aggregate = float(
                    np.mean([component.loss.value for component in components])
                )
                payload = {
                    "schema": "g1cal_aggregate_evaluation_v1",
                    "evaluation_id": f"theta_{theta_hash[:16]}",
                    "problem_hash": self.problem_hash,
                    "loss_schema": SE3_LOG_SCHEMA,
                    "theta": theta.tolist(),
                    "theta_hash": theta_hash,
                    "theta13": float(theta[RELEASED_INDEX]),
                    "loss": aggregate,
                    "label": label,
                    "statement": CALIBRATION_STATEMENT,
                    "components": {
                        component.clip: {
                            "selected_attempt": component.selected_attempt,
                            "loss": asdict(component.loss),
                            "attempts": list(component.attempts),
                            "lower_attempts": component.lower_attempts,
                            "cache_hit": component.cache_hit,
                            "wall_seconds": component.wall_seconds,
                        }
                        for component in components
                    },
                    "wall_seconds": time.perf_counter() - started,
                }
                atomic_write_json(evaluation_file, payload)
                self.evaluation_count += 1
                self._ledger_line(
                    {
                        "event": "evaluation_done",
                        "label": label,
                        "theta_hash": theta_hash,
                        "loss": aggregate,
                        "wall_seconds": payload["wall_seconds"],
                    }
                )
                return AggregateEvaluation(
                    evaluation_id=payload["evaluation_id"],
                    theta=theta.copy(),
                    theta_hash=theta_hash,
                    loss=aggregate,
                    components=tuple(components),
                    cache_hit=False,
                    wall_seconds=payload["wall_seconds"],
                )
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _evaluation_from_payload(
        self, theta: np.ndarray, payload: dict, *, cache_hit: bool
    ) -> AggregateEvaluation:
        if payload.get("loss_schema") != SE3_LOG_SCHEMA:
            raise RuntimeError(
                "cached evaluation uses a different loss schema and cannot "
                "be an exact cache hit"
            )
        components = tuple(
            ComponentEvaluation(
                theta_hash=payload["theta_hash"],
                clip=clip,
                selected_attempt=entry["selected_attempt"],
                loss=LossResult(**entry["loss"]),
                attempts=tuple(entry["attempts"]),
                lower_attempts=entry["lower_attempts"],
                cache_hit=True,
                wall_seconds=entry["wall_seconds"],
            )
            for clip, entry in sorted(payload["components"].items())
        )
        return AggregateEvaluation(
            evaluation_id=payload["evaluation_id"],
            theta=theta.copy(),
            theta_hash=payload["theta_hash"],
            loss=float(payload["loss"]),
            components=components,
            cache_hit=cache_hit,
            wall_seconds=float(payload["wall_seconds"]),
        )

    def strict_evaluations(self) -> list[dict]:
        """All completed strict aggregate evaluations, for final selection."""
        results = []
        evaluations = self.root / "evaluations"
        if evaluations.is_dir():
            for theta_dir in sorted(evaluations.glob("theta_*")):
                payload_path = theta_dir / "evaluation.json"
                if payload_path.is_file():
                    payload = json.loads(payload_path.read_text())
                    if payload.get("problem_hash") == self.problem_hash:
                        results.append(payload)
        return results
