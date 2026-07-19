"""Generic immutable-attempt long-horizon solve primitive.

Every long-horizon solve runs through the immutable attempt lifecycle:
the caller supplies an explicit generated config, immutable precision path,
canonical parent and audited warm source; each solve writes a fresh
``attempts/attempt_NNNN`` directory and only a passing recomputed gate moves
the atomic selector.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import time

from .attempts import (
    atomic_write_json,
    completion_gate_verdict,
    create_attempt,
    finalize_attempt,
    load_attempt_record,
    promote_attempt,
    strict_gate_verdict,
)
from .backend import MotionFieRequest, MotionFieResult, PrimeMotionFieBackend
from .profiles import load_model_profile
from .paths import project_root, resolve_inside_root


@contextmanager
def long_run_environment(
    checkpoint_interval: int,
    newton_max_iters: int,
    stream_output: bool,
    initial_regularization: float = 0.0,
):
    keys = (
        "G1CAL_CHECKPOINT_INTERVAL",
        "G1CAL_STREAM_OUTPUT",
        "G1CAL_NEWTON_MAX_ITERS",
        "G1CAL_INITIAL_REGULARIZATION",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["G1CAL_CHECKPOINT_INTERVAL"] = str(checkpoint_interval)
    os.environ["G1CAL_STREAM_OUTPUT"] = "1" if stream_output else "0"
    os.environ["G1CAL_NEWTON_MAX_ITERS"] = str(newton_max_iters)
    if initial_regularization > 0.0:
        os.environ["G1CAL_INITIAL_REGULARIZATION"] = str(
            initial_regularization
        )
    else:
        os.environ.pop("G1CAL_INITIAL_REGULARIZATION", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_warm_start_dir(warm: str, *, expected_profile_id: str) -> None:
    """Require warm files plus a matching frozen model-profile provenance."""
    warm_root = resolve_inside_root(warm, must_exist=False)
    required = ("xs_results_fddp.csv", "us_results_fddp.csv")
    missing = [name for name in required if not (warm_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"warm-start directory is missing {missing}: {warm_root}"
        )
    expected_profile = load_model_profile(expected_profile_id)
    warm_profile = ""
    reference_manifest = warm_root / "reference_manifest.json"
    request_path = warm_root / "request.json"
    if reference_manifest.is_file():
        manifest = json.loads(reference_manifest.read_text())
        if manifest.get("schema") != "g1cal_reference_solution_v1":
            raise RuntimeError("unknown reference-solution manifest schema")
        warm_profile = manifest.get("profile_key", "")
        for name, expected_hash in manifest.get("file_sha256", {}).items():
            source = warm_root / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"reference warm-start file is missing: {source}"
                )
            actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"reference warm-start hash mismatch for {name}"
                )
    elif request_path.is_file():
        warm_payload = json.loads(request_path.read_text())
        warm_profile = warm_payload.get("profile_key", "")
        if not warm_profile:
            warm_profile_id = warm_payload.get("request", {}).get(
                "profile_id", ""
            )
            if warm_profile_id:
                warm_profile = load_model_profile(warm_profile_id).cache_key
    if warm_profile != expected_profile.cache_key:
        raise RuntimeError(
            "warm-start model profile is missing or incompatible: "
            f"{warm_profile!r} != {expected_profile.cache_key!r}"
        )


def solve_horizon_attempt(
    *,
    parent: Path,
    request_id: str,
    config: str,
    profile_id: str,
    covariance_precision_file: str,
    prior_state_file: str = "",
    warm_start_dir: str = "",
    checkpoint_interval: int,
    newton_max_iters: int,
    initial_regularization: float,
    stream_output: bool,
    gate: str = "strict",
    attempt_label: str,
    attempt_metadata: dict,
    execution_record: dict,
    backend: PrimeMotionFieBackend | None = None,
) -> MotionFieResult:
    """One fresh immutable attempt: solve, record, gate, and promote.

    The caller supplies an explicit generated config, immutable precision
    path, canonical parent and audited warm source.  Every solve writes a new
    ``attempts/attempt_NNNN`` directory; only a passing recomputed gate moves
    the atomic selector.
    """
    if gate not in ("strict", "completion"):
        raise ValueError(f"unknown promotion gate: {gate!r}")
    if warm_start_dir:
        validate_warm_start_dir(
            warm_start_dir, expected_profile_id=profile_id
        )
    attempt_dir = create_attempt(
        parent, label=attempt_label, metadata=attempt_metadata
    )
    output = attempt_dir.relative_to(project_root())
    request = MotionFieRequest(
        request_id=request_id,
        config=config,
        output_dir=str(output),
        profile_id=profile_id,
        covariance_precision_file=covariance_precision_file,
        warm_start_dir=warm_start_dir,
        prior_state_file=prior_state_file,
    )
    runner = backend or PrimeMotionFieBackend()
    started = time.time()
    try:
        with long_run_environment(
            checkpoint_interval,
            newton_max_iters,
            stream_output,
            initial_regularization,
        ):
            result = runner.solve(request)
        record = dict(execution_record)
        record.update(
            {
                "started_unix": started,
                "attempt": attempt_dir.name,
                "canonical_parent": str(parent.relative_to(project_root())),
                "result": result.to_dict(),
            }
        )
        atomic_write_json(attempt_dir / "execution.json", record)
        verdict = (
            completion_gate_verdict(attempt_dir)
            if gate == "completion"
            else strict_gate_verdict(attempt_dir)
        )
        finalize_attempt(
            attempt_dir, status="completed", extra={"gates": verdict}
        )
        if verdict["all_passed"]:
            promote_attempt(parent, attempt_dir, verdict)
        return result
    except KeyboardInterrupt:
        if load_attempt_record(attempt_dir).get("status") == "running":
            finalize_attempt(attempt_dir, status="interrupted")
        raise
    except Exception as error:
        if load_attempt_record(attempt_dir).get("status") == "running":
            finalize_attempt(
                attempt_dir, status="failed", extra={"error": repr(error)}
            )
        raise
