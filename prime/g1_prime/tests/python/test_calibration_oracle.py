"""Calibration oracle cache, component scope, gates, and data isolation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from g1cal.backend import MotionFieRequest, MotionFieResult
from g1cal.calibration import (
    CalibrationOracle,
    LowerSolveFailure,
    assert_no_truth_leakage,
)
from g1cal.paths import project_root, resolve_inside_root
from g1cal.profiles import load_model_profile
from g1cal.gradient import cached_gradient_fn


class _FakeBackend:
    """Writes a complete strict H=501 bundle from the released truth."""

    def __init__(self, *, solved: bool = True) -> None:
        self.solve_count = 0
        self.solved = solved

    def solve(self, request: MotionFieRequest) -> MotionFieResult:
        self.solve_count += 1
        output = resolve_inside_root(request.output_dir, must_exist=False)
        output.mkdir(parents=True, exist_ok=True)
        source_config = resolve_inside_root(request.config)
        shutil.copyfile(source_config, output / "request_config.xml")
        horizon = int(
            ET.parse(source_config).getroot().find("solver").get("horizon")
        )
        clip = "run1" if "run1" in request.request_id else "run2"
        truth = np.loadtxt(
            resolve_inside_root(f"data/clips/{clip}/upper_truth_h501.csv"),
            delimiter=",",
            ndmin=2,
        )
        xs = np.vstack((truth[0], truth))
        np.savetxt(output / "xs_results_fddp.csv", xs, delimiter=",",
                   fmt="%.17g")
        zero = lambda width: ",".join("0" for _ in range(width)) + "\n"
        (output / "us_results_fddp.csv").write_text(
            zero(70) + zero(35) * (horizon - 1)
        )
        (output / "f_rollout.csv").write_text(zero(24) * (horizon - 1))
        (output / "contact_diagnostics.csv").write_text(
            "knot,newton_converged,newton_termination,newton_iterations,"
            "newton_grad_norm,newton_relative_grad_norm,feasible_init_used,"
            "min_cone_margin,min_alpha,force_norm\n"
            + "".join(
                f"{k},1,gradient,1,0,0,1,1,1,0\n"
                for k in range(horizon - 1)
            )
        )
        (output / "contact_candidate_diagnostics.csv").write_text(
            "knot,candidate_index,seed_mode,termination,relative_grad_norm\n"
            + "".join(
                f"{k},{candidate},accepted,gradient,0\n"
                for k in range(horizon - 1) for candidate in range(3)
            )
        )
        (output / "contact_corner_diagnostics.csv").write_text(
            "knot,contact_index,contact_frame,force_norm,value\n"
            + "".join(
                f"{k},{corner},corner_{corner},0,0\n"
                for k in range(horizon - 1) for corner in range(8)
            )
        )
        precision = resolve_inside_root(request.covariance_precision_file)
        covariance_hash = precision.read_text().splitlines()[0].split("=", 1)[1]
        profile_key = load_model_profile(request.profile_id).cache_key
        summary = {
            "solved": self.solved,
            "contact_health_passed": self.solved,
            "contact_certification_mode": (
                "action_stationarity_plus_shooting_defect_v3"
            ),
            "inner_stationarity_rejected": 0,
            "defect_max": 1e-9 if self.solved else 1.0,
            "n_running_models": horizon,
            "nx": 71,
            "ndx": 70,
            "iterations": 1,
            "stop": 1e-10 if self.solved else 1.0,
            "final_cost": 1.0,
            "final_preg": 1.0,
            "covariance_config_hash": covariance_hash,
        }
        (output / "backend.stdout.log").write_text("fake backend\n")
        (output / "request.json").write_text(json.dumps({
            "request": request.__dict__,
            "request_hash": request.request_hash,
            "profile_key": profile_key,
        }))
        (output / "backend_result.json").write_text(json.dumps({
            "request_hash": request.request_hash,
            "profile_key": profile_key,
        }))
        (output / "solve_summary.json").write_text(json.dumps(summary))
        return MotionFieResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            profile_key=profile_key,
            solved=self.solved,
            return_code=0,
            wall_seconds=0.01,
            summary=summary,
            output_dir=request.output_dir,
            stdout_log=f"{request.output_dir}/backend.stdout.log",
            xs_path=f"{request.output_dir}/xs_results_fddp.csv",
            us_path=f"{request.output_dir}/us_results_fddp.csv",
            force_path=f"{request.output_dir}/f_rollout.csv",
        )


@pytest.fixture()
def oracle_output(fresh_scratch):
    return fresh_scratch("out/test_scratch/calibration_oracle")


def _oracle(root: Path, backend: _FakeBackend) -> CalibrationOracle:
    return CalibrationOracle(
        backend=backend,
        stream_output=False,
        output_root=str(root.relative_to(project_root())),
    )


def test_component_api_solves_only_requested_clip(oracle_output):
    backend = _FakeBackend()
    oracle = _oracle(oracle_output, backend)
    theta = oracle.theta0.copy()
    theta[13] = 0.5
    component = oracle.evaluate_component(theta, "run1", label="single")
    assert component.clip == "run1"
    assert backend.solve_count == 1
    assert not (oracle.evaluation_dir(theta) / "run2").exists()
    assert not (oracle.evaluation_dir(theta) / "evaluation.json").exists()


def test_aggregate_reuses_component_then_completes_atomically(oracle_output):
    backend = _FakeBackend()
    oracle = _oracle(oracle_output, backend)
    theta = oracle.theta0.copy()
    theta[13] = 0.6
    first = oracle.evaluate_component(theta, "run1", label="first")
    aggregate = oracle.evaluate(theta, label="aggregate")
    assert backend.solve_count == 2
    assert first.selected_attempt == aggregate.components[0].selected_attempt
    assert len(aggregate.components) == 2
    assert aggregate.loss == pytest.approx(0.0, abs=1e-14)
    assert (oracle.evaluation_dir(theta) / "evaluation.json").is_file()


def test_exact_aggregate_cache_survives_new_oracle_process(oracle_output):
    theta = CalibrationOracle(
        backend=_FakeBackend(), stream_output=False,
        output_root=str(oracle_output.relative_to(project_root())),
    ).theta0.copy()
    theta[13] = 0.7
    first_backend = _FakeBackend()
    first_oracle = _oracle(oracle_output, first_backend)
    first = first_oracle.evaluate(theta, label="first")
    second_backend = _FakeBackend()
    second = _oracle(oracle_output, second_backend).evaluate(
        theta, label="second"
    )
    assert first_backend.solve_count == 2
    assert second_backend.solve_count == 0
    assert second.cache_hit and second.loss == first.loss


def test_failed_component_never_creates_aggregate(oracle_output):
    backend = _FakeBackend(solved=False)
    oracle = _oracle(oracle_output, backend)
    theta = oracle.theta0.copy()
    theta[13] = 0.8
    with pytest.raises(LowerSolveFailure):
        oracle.evaluate(theta, label="failure")
    assert not (oracle.evaluation_dir(theta) / "evaluation.json").exists()
    assert (oracle.evaluation_dir(theta) / "run1/component_failed.json").is_file()


def test_truth_is_rejected_from_lower_request():
    assert_no_truth_leakage({"config": "configs/lower/h501_template.xml"})
    with pytest.raises(RuntimeError, match="privileged"):
        assert_no_truth_leakage({"input": "data/clips/run1/gt_clip.npz"})


def test_theta_contract_releases_only_index_13(oracle_output):
    oracle = _oracle(oracle_output, _FakeBackend())
    invalid = oracle.theta0.copy()
    invalid[12] += 1e-12
    with pytest.raises(ValueError, match=r"only theta\[13\]"):
        oracle.canonicalize_theta(invalid)
    outside = oracle.theta0.copy()
    outside[13] = 2.1
    with pytest.raises(ValueError, match="outside frozen"):
        oracle.canonicalize_theta(outside)


def test_problem_hash_binds_implementation_sources(
    oracle_output, monkeypatch
):
    import g1cal.calibration as calibration

    first = _oracle(oracle_output, _FakeBackend()).problem_hash
    monkeypatch.setattr(
        calibration, "_implementation_hashes", lambda: {"probe": "changed"}
    )
    second = _oracle(oracle_output, _FakeBackend()).problem_hash
    assert first != second


def test_gradient_cache_survives_new_oracle_process(oracle_output):
    theta = _oracle(oracle_output, _FakeBackend()).theta0.copy()
    theta[13] = 0.9
    first_backend = _FakeBackend()
    first_oracle = _oracle(oracle_output, first_backend)
    first_value, first_meta = cached_gradient_fn(first_oracle)(
        theta, label="first"
    )
    second_backend = _FakeBackend()
    second_oracle = _oracle(oracle_output, second_backend)
    second_value, second_meta = cached_gradient_fn(second_oracle)(
        theta, label="second"
    )
    assert first_backend.solve_count == 6
    assert second_backend.solve_count == 0
    assert second_value == first_value
    assert first_meta["gradient_cache_hit"] is False
    assert second_meta["gradient_cache_hit"] is True
