"""Immutable solve-attempt directories with an atomic promotion selector.

Every solve writes into a fresh ``attempts/attempt_NNNN`` directory and
promotion is an atomic pointer update, so a canonical location never mixes
two attempts and never silently loses a preserved one.

Layout under one canonical parent:

.. code-block:: text

    <parent>/
      attempts/
        attempt_0000/
          attempt.json          status: running/completed/failed/interrupted
          request.json, solve_summary.json, xs/us/force/contact, ...
      selected_attempt.json     atomic promotion pointer (may be absent)

Rules enforced here and by ``PrimeMotionFieBackend``:

- an attempt directory receives exactly one solve; directories already
  containing solve products are rejected;
- ``selected_attempt.json`` is updated only through a temporary file plus
  ``os.replace`` and only after the caller's gate verdict passes.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

ATTEMPTS_DIRNAME = "attempts"
SELECTED_NAME = "selected_attempt.json"
ATTEMPT_RECORD_NAME = "attempt.json"

# Files only a solve writes.  Input staging (measurement CSVs, precision
# files) is deliberately absent so a dataset/covariance directory may be
# prepared before its one solve.
SOLVE_PRODUCT_NAMES = (
    "request.json",
    "request_config.xml",
    "backend.stdout.log",
    "backend_result.json",
    "solve_summary.json",
    "xs_results_fddp.csv",
    "us_results_fddp.csv",
    "f_rollout.csv",
    "contact_diagnostics.csv",
    "contact_candidate_diagnostics.csv",
    "contact_corner_diagnostics.csv",
    "progress.jsonl",
    "execution.json",
    "warm_xs.csv",
    "warm_arrival_u.csv",
    "warm_running_us.csv",
    "checkpoint",
)

# Minimum provenance bundle used by lightweight consistency checks.
MINIMUM_RESULT_BUNDLE_NAMES = (
    "request.json",
    "backend_result.json",
    "solve_summary.json",
    "xs_results_fddp.csv",
    "us_results_fddp.csv",
    "f_rollout.csv",
)

# New promotions require the complete scientific bundle.  Progress/checkpoint
# files are optional because a short solve may finish between checkpoints.
PROMOTION_RESULT_BUNDLE_NAMES = (
    "request.json",
    "request_config.xml",
    "backend.stdout.log",
    "backend_result.json",
    "solve_summary.json",
    "execution.json",
    "xs_results_fddp.csv",
    "us_results_fddp.csv",
    "f_rollout.csv",
    "contact_diagnostics.csv",
    "contact_candidate_diagnostics.csv",
    "contact_corner_diagnostics.csv",
)

STRICT_DEFECT_GATE = 1e-6


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, payload: dict) -> None:
    """Durably replace one JSON file with a unique temp and per-target lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def solve_products_present(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return [name for name in SOLVE_PRODUCT_NAMES if (directory / name).exists()]


def create_attempt(
    parent: Path, *, label: str = "", metadata: dict | None = None
) -> Path:
    """Create the next immutable attempt directory under ``parent``."""
    attempts = parent / ATTEMPTS_DIRNAME
    attempts.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        candidate = attempts / f"attempt_{index:04d}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        atomic_write_json(
            candidate / ATTEMPT_RECORD_NAME,
            {
                "schema": "g1cal_solve_attempt_v1",
                "attempt": candidate.name,
                "status": "running",
                "created_utc": _utc_now(),
                "pid": os.getpid(),
                "label": label,
                "metadata": metadata or {},
            },
        )
        return candidate
    raise RuntimeError(f"attempt namespace exhausted under {attempts}")


def load_attempt_record(attempt_dir: Path) -> dict:
    return json.loads((attempt_dir / ATTEMPT_RECORD_NAME).read_text())


def finalize_attempt(
    attempt_dir: Path, *, status: str, extra: dict | None = None
) -> dict:
    if status not in ("completed", "failed", "interrupted"):
        raise ValueError(f"invalid final attempt status: {status}")
    record = load_attempt_record(attempt_dir)
    if record.get("status") != "running":
        raise RuntimeError(
            f"attempt {attempt_dir} is already finalized "
            f"({record.get('status')}); attempts are immutable"
        )
    record["status"] = status
    record["finalized_utc"] = _utc_now()
    if extra:
        record.update(extra)
    atomic_write_json(attempt_dir / ATTEMPT_RECORD_NAME, record)
    return record


def _finite_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if not rows:
        raise ValueError(f"empty CSV: {path.name}")
    return rows


def _validate_numeric_rows(
    path: Path, *, expected_rows: int, expected_widths: tuple[int, ...]
) -> None:
    rows = _finite_csv_rows(path)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{path.name} rows {len(rows)} != expected {expected_rows}"
        )
    if len(expected_widths) == 1:
        widths = expected_widths * expected_rows
    elif len(expected_widths) == expected_rows:
        widths = expected_widths
    else:
        raise ValueError("internal expected-width specification error")
    for index, (row, width) in enumerate(zip(rows, widths, strict=True)):
        if len(row) != width:
            raise ValueError(
                f"{path.name} row {index} width {len(row)} != {width}"
            )
        try:
            values = [float(value) for value in row]
        except ValueError as error:
            raise ValueError(
                f"{path.name} row {index} contains nonnumeric data"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name} row {index} contains nonfinite data")


def _validate_contact_csv(path: Path, *, expected_rows: int) -> None:
    rows = _finite_csv_rows(path)
    header, data = rows[0], rows[1:]
    if len(data) != expected_rows:
        raise ValueError(
            f"{path.name} data rows {len(data)} != expected {expected_rows}"
        )
    if len(header) < 10 or len(set(header)) != len(header):
        raise ValueError(f"{path.name} has invalid/duplicate header")
    text_columns = {
        index for index, name in enumerate(header)
        if "termination" in name or name.endswith("_mode")
    }
    for row_index, row in enumerate(data):
        if len(row) != len(header):
            raise ValueError(
                f"{path.name} row {row_index} width {len(row)} "
                f"!= header {len(header)}"
            )
        if int(row[0]) != row_index:
            raise ValueError(
                f"{path.name} knot {row[0]} != expected {row_index}"
            )
        for index, value in enumerate(row):
            if index in text_columns:
                if not value:
                    raise ValueError(
                        f"{path.name} row {row_index} empty text field"
                    )
                continue
            try:
                numeric = float(value)
            except ValueError as error:
                raise ValueError(
                    f"{path.name} row {row_index} column {header[index]} "
                    "is nonnumeric"
                ) from error
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{path.name} row {row_index} contains nonfinite data"
                )


def _validate_grouped_diagnostics_csv(
    path: Path, *, expected_knots: int, rows_per_knot: int
) -> None:
    """Validate fixed-cardinality per-knot V3 candidate/corner evidence."""
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"{path.name} is empty")
    header, data = rows[0], rows[1:]
    expected_rows = expected_knots * rows_per_knot
    if len(data) != expected_rows:
        raise ValueError(
            f"{path.name} data rows {len(data)} != expected {expected_rows}"
        )
    if len(header) < 5 or len(set(header)) != len(header):
        raise ValueError(f"{path.name} has invalid/duplicate header")
    text_columns = {
        index for index, name in enumerate(header)
        if "termination" in name or name.endswith("_mode")
        or name.endswith("_frame")
    }
    for row_index, row in enumerate(data):
        if len(row) != len(header):
            raise ValueError(
                f"{path.name} row {row_index} width {len(row)} "
                f"!= header {len(header)}"
            )
        expected_knot = row_index // rows_per_knot
        if int(row[0]) != expected_knot:
            raise ValueError(
                f"{path.name} knot {row[0]} != expected {expected_knot}"
            )
        for index, value in enumerate(row):
            if index in text_columns:
                if not value:
                    raise ValueError(
                        f"{path.name} row {row_index} empty text field"
                    )
                continue
            try:
                numeric = float(value)
            except ValueError as error:
                raise ValueError(
                    f"{path.name} row {row_index} column {header[index]} "
                    "is nonnumeric"
                ) from error
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{path.name} row {row_index} contains nonfinite data"
                )


def validate_promotion_bundle(directory: Path) -> dict:
    """Validate provenance plus horizon-aware shapes for a new promotion."""
    missing = [
        name for name in PROMOTION_RESULT_BUNDLE_NAMES
        if not (directory / name).is_file()
    ]
    if missing:
        raise ValueError(f"incomplete promotion bundle, missing {missing}")
    request = json.loads((directory / "request.json").read_text())
    result = json.loads((directory / "backend_result.json").read_text())
    summary = json.loads((directory / "solve_summary.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    request_hash = request["request_hash"]
    if result["request_hash"] != request_hash:
        raise ValueError("request/backend result hash mismatch")
    execution_hash = execution.get("result", {}).get("request_hash")
    if execution_hash != request_hash:
        raise ValueError("request/execution result hash mismatch")
    profile_key = request.get("profile_key")
    if profile_key and result.get("profile_key") not in (None, profile_key):
        raise ValueError("request/backend profile key mismatch")

    root = ET.parse(directory / "request_config.xml").getroot()
    solver = root.find("solver")
    if solver is None or solver.get("horizon") is None:
        raise ValueError("request_config.xml has no solver horizon")
    horizon = int(solver.get("horizon"))
    if horizon < 2:
        raise ValueError(f"invalid horizon {horizon}")
    if int(summary.get("n_running_models", horizon)) != horizon:
        raise ValueError("summary/config horizon mismatch")
    if int(execution.get("horizon", horizon)) != horizon:
        raise ValueError("execution/config horizon mismatch")

    _validate_numeric_rows(
        directory / "xs_results_fddp.csv",
        expected_rows=horizon + 1,
        expected_widths=(71,),
    )
    _validate_numeric_rows(
        directory / "us_results_fddp.csv",
        expected_rows=horizon,
        expected_widths=(70,) + (35,) * (horizon - 1),
    )
    _validate_numeric_rows(
        directory / "f_rollout.csv",
        expected_rows=horizon - 1,
        expected_widths=(24,),
    )
    _validate_contact_csv(
        directory / "contact_diagnostics.csv", expected_rows=horizon - 1
    )
    _validate_grouped_diagnostics_csv(
        directory / "contact_candidate_diagnostics.csv",
        expected_knots=horizon - 1,
        rows_per_knot=3,
    )
    _validate_grouped_diagnostics_csv(
        directory / "contact_corner_diagnostics.csv",
        expected_knots=horizon - 1,
        rows_per_knot=8,
    )
    if summary.get("contact_certification_mode") != (
        "action_stationarity_plus_shooting_defect_v3"
    ):
        raise ValueError("strict promotion requires V3 contact certification")
    for key in ("defect_max",):
        value = float(summary.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"summary {key} is missing/nonfinite")
    return {
        "horizon": horizon,
        "request_hash": request_hash,
        "profile_key": profile_key or result.get("profile_key", ""),
        "bundle_schema": "complete_horizon_v1",
    }


def bundle_consistency(directory: Path, *, require_complete: bool = False) -> dict:
    """Check one-attempt provenance; optionally validate a full promotion bundle."""
    present = solve_products_present(directory)
    verdict = {
        "directory": str(directory),
        "products_present": present,
        "consistent": False,
        "reason": "",
        "request_hash": "",
    }
    if not present:
        verdict["reason"] = "no solve products"
        return verdict
    request_path = directory / "request.json"
    result_path = directory / "backend_result.json"
    if not request_path.is_file() or not result_path.is_file():
        verdict["reason"] = "request.json/backend_result.json missing"
        return verdict
    try:
        request_hash = json.loads(request_path.read_text())["request_hash"]
        result_hash = json.loads(result_path.read_text())["request_hash"]
    except (json.JSONDecodeError, KeyError) as error:
        verdict["reason"] = f"unreadable provenance record: {error!r}"
        return verdict
    if request_hash != result_hash:
        verdict["reason"] = (
            f"request.json hash {request_hash[:8]} != "
            f"backend_result.json hash {result_hash[:8]}"
        )
        return verdict
    missing = [
        name for name in MINIMUM_RESULT_BUNDLE_NAMES
        if not (directory / name).is_file()
    ]
    if missing:
        verdict["reason"] = f"incomplete result bundle, missing {missing}"
        return verdict
    if require_complete:
        try:
            verdict["bundle"] = validate_promotion_bundle(directory)
        except (ET.ParseError, json.JSONDecodeError, KeyError, TypeError,
                ValueError) as error:
            verdict["reason"] = f"invalid complete result bundle: {error}"
            return verdict
    verdict["consistent"] = True
    verdict["reason"] = (
        "single consistent complete solve bundle"
        if require_complete else "single consistent solve bundle"
    )
    verdict["request_hash"] = request_hash
    return verdict


def strict_gate_verdict(directory: Path) -> dict:
    """Ladder/monolithic promotion gate: solver, contact health, and defect."""
    consistency = bundle_consistency(directory, require_complete=True)
    checks = {
        "consistent_bundle": consistency["consistent"],
        "solved": False,
        "contact_health_passed": False,
        "defect_below_gate": False,
    }
    summary: dict = {}
    summary_path = directory / "solve_summary.json"
    if consistency["consistent"] and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        checks["solved"] = bool(summary.get("solved", False))
        checks["contact_health_passed"] = bool(
            summary.get("contact_health_passed", False)
        )
        checks["defect_below_gate"] = (
            float(summary.get("defect_max", float("inf"))) < STRICT_DEFECT_GATE
        )
    return {
        "gate": "strict",
        "defect_gate": STRICT_DEFECT_GATE,
        "checks": checks,
        "all_passed": all(checks.values()),
        "consistency_reason": consistency["reason"],
        "request_hash": consistency["request_hash"],
    }


def completion_gate_verdict(directory: Path) -> dict:
    """Initializer-tier gate: a complete, consistent bundle regardless of
    solver convergence (local windows are initializer evidence, not results)."""
    consistency = bundle_consistency(directory, require_complete=True)
    checks = {"complete_initializer_bundle": consistency["consistent"]}
    return {
        "gate": "completion",
        "checks": checks,
        "all_passed": all(checks.values()),
        "consistency_reason": consistency["reason"],
        "request_hash": consistency["request_hash"],
    }


def promote_attempt(
    parent: Path, attempt_dir: Path, verdict: dict | None = None
) -> Path:
    """Recompute gates and point the selector only at a completed attempt."""
    expected = parent / ATTEMPTS_DIRNAME / attempt_dir.name
    if expected.resolve() != attempt_dir.resolve():
        raise ValueError(
            f"{attempt_dir} is not an attempt of {parent}; refusing promotion"
        )
    record = load_attempt_record(attempt_dir)
    if record.get("status") != "completed":
        raise RuntimeError(
            f"promotion refused for {attempt_dir}: attempt status is "
            f"{record.get('status')!r}, expected 'completed'"
        )
    recorded = record.get("gates", {})
    gate_name = recorded.get("gate") or (verdict or {}).get("gate")
    if gate_name == "strict":
        recomputed = strict_gate_verdict(attempt_dir)
    elif gate_name == "completion":
        recomputed = completion_gate_verdict(attempt_dir)
    else:
        raise RuntimeError(
            f"promotion refused for {attempt_dir}: unknown/missing gate type"
        )
    supplied_hashes = {
        candidate.get("request_hash")
        for candidate in (recorded, verdict or {})
        if candidate
    }
    supplied_hashes.discard("")
    if any(value != recomputed["request_hash"] for value in supplied_hashes):
        raise RuntimeError(
            f"promotion refused for {attempt_dir}: stale gate/request hash"
        )
    if not recomputed.get("all_passed", False):
        raise RuntimeError(
            f"promotion refused for {attempt_dir}: recomputed gate failed "
            f"{recomputed.get('checks')}"
        )
    selector = parent / SELECTED_NAME
    atomic_write_json(
        selector,
        {
            "schema": "g1cal_selected_attempt_v1",
            "attempt": attempt_dir.name,
            "promoted_utc": _utc_now(),
            "request_hash": recomputed["request_hash"],
            "verdict": recomputed,
        },
    )
    return selector


def resolve_selected(parent: Path) -> Path:
    """Resolve the strictly promoted attempt under a canonical parent."""
    selector = parent / SELECTED_NAME
    if selector.is_file():
        payload = json.loads(selector.read_text())
        attempt = parent / ATTEMPTS_DIRNAME / payload["attempt"]
        if not attempt.is_dir():
            raise FileNotFoundError(
                f"selector points at missing attempt: {attempt}"
            )
        record = load_attempt_record(attempt)
        if record.get("status") != "completed":
            raise RuntimeError(
                f"selector points at non-completed attempt {attempt}"
            )
        consistency = bundle_consistency(attempt, require_complete=True)
        if not consistency["consistent"]:
            raise RuntimeError(
                f"promoted attempt {attempt} is inconsistent: "
                f"{consistency['reason']}"
            )
        selected_hash = payload.get("request_hash") or payload.get(
            "verdict", {}
        ).get("request_hash")
        if selected_hash and selected_hash != consistency["request_hash"]:
            raise RuntimeError(
                f"selector request hash does not match {attempt}"
            )
        return attempt
    raise FileNotFoundError(f"no promoted attempt under {parent}")


def try_resolve_selected(parent: Path) -> Path | None:
    """Return the promoted attempt, or ``None`` when none exists."""
    try:
        return resolve_selected(parent)
    except FileNotFoundError:
        return None
