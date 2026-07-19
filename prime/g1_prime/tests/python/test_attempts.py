"""Immutable attempt lifecycle and strict promotion regression tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from g1cal.attempts import (
    ATTEMPT_RECORD_NAME,
    SELECTED_NAME,
    create_attempt,
    finalize_attempt,
    load_attempt_record,
    promote_attempt,
    resolve_selected,
    strict_gate_verdict,
    try_resolve_selected,
)


def _write_bundle(
    directory: Path,
    *,
    request_hash: str = "a" * 64,
    result_hash: str | None = None,
    solved: bool = True,
    contact_health: bool = True,
    horizon: int = 3,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    profile_key = "g1:test:test:test"
    (directory / "request.json").write_text(json.dumps({
        "request": {"request_id": "probe", "profile_id": "g1"},
        "request_hash": request_hash,
        "profile_key": profile_key,
    }))
    (directory / "request_config.xml").write_text(
        f'<config><solver horizon="{horizon}"/></config>\n'
    )
    (directory / "backend.stdout.log").write_text("fake backend\n")
    (directory / "backend_result.json").write_text(json.dumps({
        "request_hash": result_hash or request_hash,
        "profile_key": profile_key,
    }))
    (directory / "solve_summary.json").write_text(json.dumps({
        "solved": solved,
        "contact_health_passed": contact_health,
        "contact_certification_mode": (
            "action_stationarity_plus_shooting_defect_v3"
        ),
        "defect_max": 1e-9,
        "n_running_models": horizon,
    }))
    (directory / "execution.json").write_text(json.dumps({
        "horizon": horizon,
        "result": {"request_hash": request_hash},
    }))
    row = lambda width: ",".join("0" for _ in range(width)) + "\n"
    (directory / "xs_results_fddp.csv").write_text(row(71) * (horizon + 1))
    (directory / "us_results_fddp.csv").write_text(
        row(70) + row(35) * (horizon - 1)
    )
    (directory / "f_rollout.csv").write_text(row(24) * (horizon - 1))
    (directory / "contact_diagnostics.csv").write_text(
        "knot,newton_converged,newton_termination,newton_iterations,"
        "newton_grad_norm,newton_relative_grad_norm,feasible_init_used,"
        "min_cone_margin,min_alpha,force_norm\n"
        + "".join(
            f"{k},1,gradient,1,0,0,1,1,1,0\n"
            for k in range(horizon - 1)
        )
    )
    (directory / "contact_candidate_diagnostics.csv").write_text(
        "knot,candidate_index,seed_mode,termination,relative_grad_norm\n"
        + "".join(
            f"{k},{candidate},accepted,gradient,0\n"
            for k in range(horizon - 1) for candidate in range(3)
        )
    )
    (directory / "contact_corner_diagnostics.csv").write_text(
        "knot,contact_index,contact_frame,force_norm,value\n"
        + "".join(
            f"{k},{corner},corner_{corner},0,0\n"
            for k in range(horizon - 1) for corner in range(8)
        )
    )


def _completed(parent: Path, *, request_hash: str = "a" * 64) -> Path:
    attempt = create_attempt(parent, label="probe")
    _write_bundle(attempt, request_hash=request_hash)
    verdict = strict_gate_verdict(attempt)
    finalize_attempt(attempt, status="completed", extra={"gates": verdict})
    return attempt


def test_attempt_directories_are_unique_and_concurrent(tmp_path):
    parent = tmp_path / "case"
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(
            lambda _: create_attempt(parent, label="parallel"), range(16)
        ))
    assert len({attempt.name for attempt in attempts}) == 16
    assert all(load_attempt_record(path)["status"] == "running"
               for path in attempts)


def test_finalize_is_single_shot(tmp_path):
    attempt = create_attempt(tmp_path / "case", label="probe")
    finalize_attempt(attempt, status="completed")
    assert not (attempt / (ATTEMPT_RECORD_NAME + ".tmp")).exists()
    with pytest.raises(RuntimeError, match="immutable"):
        finalize_attempt(attempt, status="failed")


def test_strict_promotion_is_atomic_and_replaceable(tmp_path):
    parent = tmp_path / "case"
    first = _completed(parent, request_hash="b" * 64)
    promote_attempt(parent, first)
    assert resolve_selected(parent) == first
    second = _completed(parent, request_hash="c" * 64)
    promote_attempt(parent, second)
    assert resolve_selected(parent) == second
    assert first.is_dir()
    assert not (parent / (SELECTED_NAME + ".tmp")).exists()


def test_promotion_rejects_failed_gate_and_running_attempt(tmp_path):
    parent = tmp_path / "case"
    running = create_attempt(parent, label="running")
    _write_bundle(running)
    with pytest.raises(RuntimeError, match="expected 'completed'"):
        promote_attempt(parent, running, strict_gate_verdict(running))
    failed = create_attempt(parent, label="failed")
    _write_bundle(failed, solved=False)
    verdict = strict_gate_verdict(failed)
    finalize_attempt(failed, status="completed", extra={"gates": verdict})
    with pytest.raises(RuntimeError, match="recomputed gate failed"):
        promote_attempt(parent, failed, verdict)


def test_promotion_rejects_foreign_or_mutated_bundle(tmp_path):
    parent = tmp_path / "case"
    attempt = _completed(parent)
    foreign = _completed(tmp_path / "other")
    with pytest.raises(ValueError, match="not an attempt"):
        promote_attempt(parent, foreign)
    summary = json.loads((attempt / "solve_summary.json").read_text())
    summary["contact_health_passed"] = False
    (attempt / "solve_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="recomputed gate failed"):
        promote_attempt(parent, attempt)


def test_resolution_requires_an_explicit_selector(tmp_path):
    parent = tmp_path / "case"
    _write_bundle(parent)
    with pytest.raises(FileNotFoundError, match="no promoted attempt"):
        resolve_selected(parent)
    assert try_resolve_selected(parent) is None


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing_contact", "missing"),
        ("wrong_mode", "V3"),
        ("wrong_shape", "width"),
        ("nonfinite", "nonfinite"),
        ("hash_mismatch", "!="),
    ],
)
def test_complete_bundle_validation_rejects_invalid_data(
    tmp_path, mutation, reason
):
    attempt = create_attempt(tmp_path / "case", label="probe")
    _write_bundle(attempt)
    if mutation == "missing_contact":
        (attempt / "contact_diagnostics.csv").unlink()
    elif mutation == "wrong_mode":
        summary = json.loads((attempt / "solve_summary.json").read_text())
        summary["contact_certification_mode"] = "other"
        (attempt / "solve_summary.json").write_text(json.dumps(summary))
    elif mutation == "wrong_shape":
        (attempt / "xs_results_fddp.csv").write_text("0,0\n" * 4)
    elif mutation == "nonfinite":
        (attempt / "f_rollout.csv").write_text(
            ",".join(["nan"] + ["0"] * 23) + "\n"
            + ",".join(["0"] * 24) + "\n"
        )
    else:
        result = json.loads((attempt / "backend_result.json").read_text())
        result["request_hash"] = "f" * 64
        (attempt / "backend_result.json").write_text(json.dumps(result))
    verdict = strict_gate_verdict(attempt)
    assert not verdict["all_passed"]
    assert reason in verdict["consistency_reason"]
