"""Callable backends for the fixed-inertia motion-only FIE.

The subprocess boundary is retained as the reference execution path.  The
in-process binding executes the same C++ runner translation unit after explicit
output-parity checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from .attempts import solve_products_present
from .profiles import load_model_profile
from .paths import project_root, resolve_inside_root


@dataclass(frozen=True)
class MotionFieRequest:
    request_id: str = "unset"
    config: str = "configs/lower/h501_template.xml"
    output_dir: str = "out/unset"
    profile_id: str = "g1"
    covariance_precision_file: str = ""
    warm_start_dir: str = ""
    prior_state_file: str = ""

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def request_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MotionFieRequest":
        return cls(**raw)


@dataclass(frozen=True)
class MotionFieResult:
    request_id: str = "unset"
    request_hash: str = ""
    profile_key: str = ""
    solved: bool = False
    return_code: int = -1
    wall_seconds: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    stdout_log: str = ""
    xs_path: str = ""
    us_path: str = ""
    force_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrimeMotionFieBackend:
    """One request -> one C++ motion-only lower solve."""

    def __init__(self) -> None:
        self.solve_count = 0

    def _execute(self, command: list[str]) -> tuple[int, str]:
        if os.environ.get("G1CAL_STREAM_OUTPUT") == "1":
            process = subprocess.Popen(
                command,
                cwd=project_root(),
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            output: list[str] = []
            for line in process.stdout:
                output.append(line)
                print(line, end="", flush=True)
            return process.wait(), "".join(output)
        completed = subprocess.run(
            command,
            cwd=project_root(),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode, completed.stdout

    def solve(self, request: MotionFieRequest) -> MotionFieResult:
        profile = load_model_profile(request.profile_id)

        source_config = resolve_inside_root(request.config)
        output_dir = resolve_inside_root(request.output_dir, must_exist=False)
        existing_products = solve_products_present(output_dir)
        if existing_products:
            raise RuntimeError(
                f"output directory {request.output_dir} already contains solve "
                f"products {existing_products[:4]}; each attempt requires a "
                "fresh directory (no resume protocol is implemented). Retrying "
                "into a used directory would mix attempts and invalidate "
                "provenance."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        tree = ET.parse(source_config)
        xml_root = tree.getroot()
        if xml_root.find("identification") is not None:
            raise ValueError("motion-only backend rejects <identification>")
        paths = xml_root.find("paths")
        if paths is None:
            raise ValueError("config has no <paths>")
        paths.set("results", str(output_dir))
        robot = xml_root.find("robot")
        if robot is None:
            raise ValueError("config has no <robot>")
        robot.set("urdf", str(profile.urdf_path))
        robot.set("srdf", str(profile.srdf_path) if profile.srdf_path else "")
        if profile.srdf_path is None:
            robot.set("reference_configuration", "")
        generated_config = output_dir / "request_config.xml"
        ET.indent(tree, space="  ")
        tree.write(generated_config, encoding="utf-8", xml_declaration=True)

        request_record = {
            "request": asdict(request),
            "request_hash": request.request_hash,
            "profile_key": profile.cache_key,
        }
        (output_dir / "request.json").write_text(
            json.dumps(request_record, indent=2, sort_keys=True) + "\n"
        )

        binary = resolve_inside_root("build/overlay/g1_motion_fie")
        stdout_path = output_dir / "backend.stdout.log"
        self.solve_count += 1
        start = time.perf_counter()
        command = [str(binary), str(generated_config)]
        if request.covariance_precision_file:
            precision_path = resolve_inside_root(request.covariance_precision_file)
            command.append(str(precision_path))
        if request.warm_start_dir:
            if not request.covariance_precision_file:
                raise ValueError("warm start requires strict covariance mode")
            warm_dir = resolve_inside_root(request.warm_start_dir)
            warm_xs_source = warm_dir / "xs_results_fddp.csv"
            warm_us_source = warm_dir / "us_results_fddp.csv"
            if not warm_xs_source.is_file() or not warm_us_source.is_file():
                raise FileNotFoundError("warm-start result files missing")
            warm_xs = output_dir / "warm_xs.csv"
            warm_arrival = output_dir / "warm_arrival_u.csv"
            warm_running = output_dir / "warm_running_us.csv"
            warm_xs.write_bytes(warm_xs_source.read_bytes())
            lines = [line for line in warm_us_source.read_text().splitlines() if line]
            if not lines:
                raise ValueError("empty warm-start controls")
            warm_arrival.write_text(lines[0] + "\n")
            warm_running.write_text("\n".join(lines[1:]) + ("\n" if lines[1:] else ""))
            command.extend(
                [str(warm_xs), str(warm_arrival), str(warm_running)]
            )
        if request.prior_state_file:
            if not request.covariance_precision_file:
                raise ValueError("prior override requires strict covariance mode")
            command.append(str(resolve_inside_root(request.prior_state_file)))

        return_code, stdout = self._execute(command)
        wall = time.perf_counter() - start
        stdout_path.write_text(stdout)

        summary_path = output_dir / "solve_summary.json"
        if return_code != 0 or not summary_path.is_file():
            raise RuntimeError(
                f"lower solve failed rc={return_code}; see {stdout_path}"
            )
        summary = json.loads(summary_path.read_text())
        result = MotionFieResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            profile_key=profile.cache_key,
            solved=bool(summary.get("solved", False))
            and bool(summary.get("contact_health_passed", False)),
            return_code=return_code,
            wall_seconds=wall,
            summary=summary,
            output_dir=str(output_dir.relative_to(project_root())),
            stdout_log=str(stdout_path.relative_to(project_root())),
            xs_path=str((output_dir / "xs_results_fddp.csv").relative_to(project_root())),
            us_path=str((output_dir / "us_results_fddp.csv").relative_to(project_root())),
            force_path=str((output_dir / "f_rollout.csv").relative_to(project_root())),
        )
        (output_dir / "backend_result.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        return result


class InProcessPrimeMotionFieBackend(PrimeMotionFieBackend):
    """Persistent pybind backend used for final performance measurements."""

    def _execute(self, command: list[str]) -> tuple[int, str]:
        from . import _g1cal_cpp

        binary = resolve_inside_root("build/overlay/g1_motion_fie")
        if Path(command[0]).resolve() != binary.resolve():
            raise ValueError("in-process backend received an unknown executable")
        return_code, stdout = _g1cal_cpp.run_motion_fie(command[1:])
        return int(return_code), str(stdout)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request_json")
    parser.add_argument("result_json")
    args = parser.parse_args()
    request_path = resolve_inside_root(args.request_json)
    result_path = resolve_inside_root(args.result_json, must_exist=False)
    request = MotionFieRequest.from_dict(json.loads(request_path.read_text()))
    result = PrimeMotionFieBackend().solve(request)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if result.solved else 3


if __name__ == "__main__":
    raise SystemExit(_main())
