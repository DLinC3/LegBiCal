"""Upper-method tests over a fast exact quadratic oracle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g1cal.calibration import (
    LowerSolveFailure,
    RELEASED_INDEX,
    THETA13_LOWER,
    THETA13_UPPER,
)
from g1cal.optimizers import (
    OBJECTIVE_SCALE,
    SDP_DISCLOSURE,
    OptimizerRunner,
    VarianceCoordinate,
    VarianceOracle,
    solve_sdp_lmo,
)


def _theta0() -> np.ndarray:
    theta = np.zeros(17)
    theta[7:10] = np.log(4.0)
    theta[13] = np.log(4.0)
    return theta


class _FakeBaseOracle:
    def __init__(self, minimum_theta13: float = 0.6) -> None:
        self.theta0 = _theta0()
        self.minimum = minimum_theta13
        self.cache: dict[float, float] = {}
        self.evaluations = 0

    def loss_at(self, theta13: float) -> float:
        return 0.01 * (theta13 - self.minimum) ** 2 + 0.015

    def gradient_at(self, theta13: float) -> float:
        return 0.02 * (theta13 - self.minimum)

    def evaluate(self, theta, *, label: str):
        del label
        theta13 = float(np.asarray(theta)[RELEASED_INDEX])
        assert THETA13_LOWER <= theta13 <= THETA13_UPPER
        key = round(theta13, 14)
        cache_hit = key in self.cache
        if not cache_hit:
            self.cache[key] = self.loss_at(theta13)
            self.evaluations += 1
        loss = self.cache[key]
        component = SimpleNamespace(
            clip="run1", loss=SimpleNamespace(value=loss)
        )
        return SimpleNamespace(
            loss=loss,
            cache_hit=cache_hit,
            components=(component,),
            theta_hash=f"fake_{key}",
        )


class _FailingEndpointOracle(_FakeBaseOracle):
    def evaluate(self, theta, *, label: str):
        theta13 = float(np.asarray(theta)[RELEASED_INDEX])
        if abs(theta13 - THETA13_LOWER) < 1e-12:
            raise LowerSolveFailure("endpoint rejected by strict lower gate")
        return super().evaluate(theta, label=label)


def _gradient(base: _FakeBaseOracle):
    def gradient(theta, *, label: str):
        del label
        theta13 = float(np.asarray(theta)[RELEASED_INDEX])
        return base.gradient_at(theta13), {"method": "test_analytic"}
    return gradient


def _runner(base: _FakeBaseOracle, root) -> OptimizerRunner:
    oracle = VarianceOracle(
        base, _gradient(base), log_path=root / "oracle_calls.jsonl"
    )
    return OptimizerRunner(oracle, output_root=root)


def test_variance_coordinate_round_trip_and_chain_rule():
    coordinate = VarianceCoordinate()
    for theta13 in (-0.5, 0.0, 0.7, np.log(4.0), 2.0):
        variance = coordinate.variance(theta13)
        assert coordinate.theta13(variance) == pytest.approx(theta13)
        eta = coordinate.eta_from_s(variance)
        assert 0.0 <= eta <= 1.0
        assert coordinate.s_from_eta(eta) == pytest.approx(variance)
        step = 1e-7
        numeric = (
            coordinate.variance(theta13 + step)
            - coordinate.variance(theta13 - step)
        ) / (2.0 * step)
        assert coordinate.ds_dtheta(variance) == pytest.approx(
            numeric, rel=1e-6
        )


def test_variance_oracle_changes_only_released_coordinate():
    base = _FakeBaseOracle()
    oracle = VarianceOracle(base, _gradient(base))
    theta = oracle.expand_theta(oracle.coordinate.variance(0.9))
    frozen = [i for i in range(17) if i != RELEASED_INDEX]
    assert np.array_equal(theta[frozen], base.theta0[frozen])
    assert theta[RELEASED_INDEX] == pytest.approx(0.9)


@pytest.mark.parametrize("gradient,endpoint", [(2.5, 0), (-1.5, 1)])
def test_sdp_lmo_matches_analytic_endpoint(gradient, endpoint):
    coordinate = VarianceCoordinate()
    bounds = coordinate.s_bounds
    result = solve_sdp_lmo(gradient, coordinate=coordinate)
    assert result["analytic_vertex"] == pytest.approx(bounds[endpoint])
    assert result["sdp_vertex"] == pytest.approx(
        bounds[endpoint], abs=result["parity_tolerance"]
    )
    assert result["vertex_used"] == pytest.approx(bounds[endpoint])
    assert result["disclosure"] == SDP_DISCLOSURE


def test_value_gradient_chain_matches_numeric():
    base = _FakeBaseOracle()
    oracle = VarianceOracle(base, _gradient(base))
    coordinate = oracle.coordinate
    variance = coordinate.variance(1.0)
    _, derivative = oracle.value_and_gradient(variance, label="chain")
    step = 1e-9 * variance
    numeric = (
        base.loss_at(coordinate.theta13(variance + step))
        - base.loss_at(coordinate.theta13(variance - step))
    ) / (2.0 * step)
    assert derivative == pytest.approx(numeric, rel=1e-5)


def test_both_methods_share_baseline_and_improve(fresh_scratch):
    root = fresh_scratch("out/test_scratch/optimizer_shared")
    base = _FakeBaseOracle()
    runner = _runner(base, root)
    sqp = runner.run_sqp_bfgs(max_iterations=2)
    fw = runner.run_frank_wolfe_sdp(max_iterations=2)
    assert sqp["baseline"] == fw["baseline"]
    assert sqp["objective_scale"] == fw["objective_scale"] == OBJECTIVE_SCALE
    assert sqp["best_feasible"]["loss"] < sqp["baseline"]["loss"]
    assert fw["best_feasible"]["loss"] < fw["baseline"]["loss"]
    assert len(sqp["accepted_iterations"]) <= 2


def test_frank_wolfe_stationary_gap_stops_without_line_search(fresh_scratch):
    root = fresh_scratch("out/test_scratch/optimizer_stationary")
    base = _FakeBaseOracle(minimum_theta13=float(np.log(4.0)))
    record = _runner(base, root).run_frank_wolfe_sdp(max_iterations=2)
    assert record["stop_reason"] == "nonpositive_frank_wolfe_gap"
    assert "armijo_trials" not in record["iterations"][0]


def test_frank_wolfe_backtracks_after_strict_lower_failure(fresh_scratch):
    root = fresh_scratch("out/test_scratch/optimizer_backtrack")
    base = _FailingEndpointOracle(minimum_theta13=0.2)
    record = _runner(base, root).run_frank_wolfe_sdp(max_iterations=1)
    trials = record["iterations"][0]["armijo_trials"]
    assert trials[0]["lower_failed"] is True
    accepted = [item for item in trials if item.get("sufficient_decrease")]
    assert accepted and accepted[0]["gamma"] < 1.0


def test_cli_rejects_unpublished_optimizer(monkeypatch):
    import sys
    from g1cal.cli import main

    monkeypatch.setattr(sys, "argv", [
        "g1cal", "calibrate", "--optimizer", "l-bfgs-b"
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
