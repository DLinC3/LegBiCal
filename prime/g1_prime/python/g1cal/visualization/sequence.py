"""Build and persist validated viewer-independent motion/force sequences."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml

from ..profiles import ModelProfile, load_model_profile
from ..gt_clips import MujocoTruthClip
from ..paths import project_root, resolve_inside_root
from .force_geometry import (
    aggregate_foot_forces,
    align_mujoco_post_step_foot_forces,
    center_of_pressure_world,
    force_components_to_world,
    friction_utilization,
    support_polygon_xy,
)
from .types import (
    ContactDiagnostics,
    MotionForceSequence,
    VisualizationFrame,
    VisualizationStyle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_visualization_style(path: str) -> VisualizationStyle:
    source = resolve_inside_root(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != (
        "g1cal_visualization_style_v2"
    ):
        raise ValueError("unknown visualization style schema")
    values = raw.get("style")
    if not isinstance(values, dict):
        raise ValueError("visualization style must be a mapping")
    return VisualizationStyle(**values)


def _load_matrix(path: Path, columns: int) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",", ndmin=2)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{path.name} must have {columns} columns, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return np.asarray(matrix, dtype=float)


def _load_diagnostics(path: Path) -> ContactDiagnostics:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("contact diagnostics are empty")

    def floats(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    def ints(name: str) -> np.ndarray:
        return np.asarray([int(row[name]) for row in rows], dtype=int)

    def bools(name: str) -> np.ndarray:
        return np.asarray([bool(int(row[name])) for row in rows], dtype=bool)

    return ContactDiagnostics(
        knot=ints("knot"),
        newton_converged=bools("newton_converged"),
        newton_termination=tuple(row["newton_termination"] for row in rows),
        newton_iterations=ints("newton_iterations"),
        newton_grad_norm=floats("newton_grad_norm"),
        newton_relative_grad_norm=floats("newton_relative_grad_norm"),
        feasible_init_used=bools("feasible_init_used"),
        min_cone_margin=floats("min_cone_margin"),
        min_alpha=floats("min_alpha"),
        force_norm=floats("force_norm"),
    )


def _parse_config(
    path: Path,
) -> tuple[float, float, float, tuple[str, ...], int]:
    root = ET.parse(path).getroot()
    solver = root.find("solver")
    weights = root.find("weights")
    contacts = root.find("contacts")
    if solver is None or weights is None or contacts is None:
        raise ValueError("visualization config lacks solver/weights/contacts")
    down_sample = int(solver.get("down_sample", "0"))
    horizon = int(solver.get("horizon", "0"))
    dt = float(solver.get("interval", "0")) * down_sample
    mu = float(weights.get("mu", "0"))
    kappa = float(weights.get("kappa", "0"))
    names = tuple(frame.get("name", "") for frame in contacts.findall("frame"))
    if dt <= 0.0 or mu <= 0.0 or kappa <= 0.0 or horizon <= 0:
        raise ValueError("config dt/mu/kappa/horizon must be positive")
    if len(names) != 8 or any(not name for name in names):
        raise ValueError("config must name exactly eight contact frames")
    physical_knots = (horizon + down_sample - 1) // down_sample
    return dt, mu, kappa, names, physical_knots


def _add_profile_contact_frames(model: pin.Model, profile: ModelProfile) -> None:
    if profile.contact_frames_path is None:
        return
    raw = json.loads(profile.contact_frames_path.read_text())
    for name, spec in raw["frames"].items():
        if model.existFrame(name):
            continue
        parent_id = model.getFrameId(spec["parent"])
        if parent_id >= model.nframes:
            raise ValueError(f"missing contact parent frame {spec['parent']}")
        parent = model.frames[parent_id]
        offset = pin.SE3(np.eye(3), np.asarray(spec["xyz"], dtype=float))
        placement = parent.placement * offset
        model.addFrame(pin.Frame(
            name,
            parent.parentJoint,
            parent_id,
            placement,
            pin.FrameType.OP_FRAME,
        ), False)


def _build_pinocchio_model(
    profile: ModelProfile, contact_names: tuple[str, ...]
) -> pin.Model:
    model = pin.buildModelFromUrdf(
        str(profile.urdf_path), pin.JointModelFreeFlyer()
    )
    _add_profile_contact_frames(model, profile)
    missing = [name for name in contact_names if not model.existFrame(name)]
    if missing:
        raise ValueError(f"profile is missing contact frames: {missing}")
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"unexpected G1 dimensions nq={model.nq} nv={model.nv}")
    return model


class MotionForceSequenceBuilder:
    """Load preserved lower outputs without invoking a lower backend."""

    def __init__(self, style: VisualizationStyle | None = None) -> None:
        self.style = style or VisualizationStyle()

    def build(
        self,
        *,
        result_dir: str,
        config: str,
        profile_id: str,
        run_id: str | None = None,
        gt_clip: MujocoTruthClip | None = None,
    ) -> MotionForceSequence:
        result = resolve_inside_root(result_dir)
        config_path = resolve_inside_root(config)
        profile = load_model_profile(profile_id)
        dt, mu, kappa, contact_names, configured_knots = _parse_config(
            config_path
        )

        xs_path = result / "xs_results_fddp.csv"
        force_path = result / "f_rollout.csv"
        diagnostics_path = result / "contact_diagnostics.csv"
        summary_path = result / "solve_summary.json"
        for path in (xs_path, force_path, diagnostics_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        xs = _load_matrix(xs_path, 71)
        if xs.shape[0] < 3:
            raise ValueError("trajectory lacks arrival plus two physical states")
        physical_states = xs[1:].copy()
        if physical_states.shape[0] != configured_knots:
            raise ValueError(
                "arrival/force alignment failed: config expects "
                f"{configured_knots} physical states, result has "
                f"{physical_states.shape[0]}"
            )
        force_flat = _load_matrix(force_path, 24)
        expected_intervals = physical_states.shape[0] - 1
        if force_flat.shape[0] != expected_intervals:
            raise ValueError(
                "arrival/force alignment failed: "
                f"{physical_states.shape[0]} physical states require "
                f"{expected_intervals} force rows, got {force_flat.shape[0]}"
            )
        diagnostics = _load_diagnostics(diagnostics_path)
        if len(diagnostics) != expected_intervals:
            raise ValueError("contact diagnostics/force length mismatch")
        solve_summary = json.loads(summary_path.read_text())
        if int(solve_summary.get("nx", -1)) != 71 or int(
            solve_summary.get("ndx", -1)
        ) != 70:
            raise ValueError("solve summary is not motion-only G1")

        forces_contact = force_flat.reshape(expected_intervals, 8, 3)
        normal = np.array([0.0, 0.0, 1.0])
        forces_world = force_components_to_world(forces_contact, normal)
        utilization = friction_utilization(forces_contact, mu)
        active = np.linalg.norm(forces_world, axis=-1) >= (
            self.style.force_display_threshold_n
        )

        model = _build_pinocchio_model(profile, contact_names)
        data = model.createData()
        frame_ids = [model.getFrameId(name) for name in contact_names]
        positions = np.empty((expected_intervals, 8, 3))
        for index, state in enumerate(physical_states[:-1]):
            pin.forwardKinematics(model, data, state[:36])
            pin.updateFramePlacements(model, data)
            for contact, frame_id in enumerate(frame_ids):
                positions[index, contact] = data.oMf[frame_id].translation
        plane_height = 0.0
        gaps = np.einsum("...j,j->...", positions, normal) - plane_height

        gt_states = None
        gt_forces = None
        gt_foot_positions = None
        gt_rows = None
        gt_motion = None
        if gt_clip is not None:
            if gt_clip.states.shape != physical_states.shape:
                raise ValueError("GT clip states do not match physical trajectory")
            raw_gt_forces = np.asarray(
                gt_clip.contact_debug["contact_debug/per_foot_summed_grf"],
                dtype=float,
            )
            if raw_gt_forces.shape != (physical_states.shape[0], 2, 3):
                raise ValueError("GT foot-force rows do not match GT states")
            gt_states = gt_clip.states
            gt_forces = align_mujoco_post_step_foot_forces(raw_gt_forces)
            gt_corner_positions = np.empty((expected_intervals, 8, 3))
            # GT force row k+1 is reconstructed at the post-step GT pose k+1.
            for index, state in enumerate(gt_clip.states[1:]):
                pin.forwardKinematics(model, data, state[:36])
                pin.updateFramePlacements(model, data)
                for contact, frame_id in enumerate(frame_ids):
                    gt_corner_positions[index, contact] = (
                        data.oMf[frame_id].translation
                    )
            gt_foot_positions = np.stack((
                gt_corner_positions[:, :4].mean(axis=1),
                gt_corner_positions[:, 4:].mean(axis=1),
            ), axis=1)
            gt_rows = np.arange(gt_clip.source_start, gt_clip.source_stop)
            gt_motion = gt_clip.motion

        return MotionForceSequence(
            run_id=run_id or result.name,
            source_result_dir=str(result.relative_to(project_root())),
            source_config=str(config_path.relative_to(project_root())),
            profile_id=profile.profile_id,
            profile_key=profile.cache_key,
            urdf_sha256=profile.urdf_sha256,
            mjcf_sha256=profile.mjcf_sha256,
            total_mass_kg=float(sum(inertia.mass for inertia in model.inertias)),
            dt=dt,
            mu=mu,
            kappa=kappa,
            contact_names=contact_names,
            plane_normal_world=normal,
            plane_height_m=plane_height,
            physical_states=physical_states,
            interval_forces_contact=forces_contact,
            interval_forces_world=forces_world,
            contact_positions_world=positions,
            contact_gaps_m=gaps,
            friction_utilization=utilization,
            active_display_mask=active,
            diagnostics=diagnostics,
            solve_summary=solve_summary,
            source_hashes={
                "xs_results_fddp.csv": _sha256(xs_path),
                "f_rollout.csv": _sha256(force_path),
                "contact_diagnostics.csv": _sha256(diagnostics_path),
                "solve_summary.json": _sha256(summary_path),
                "config": _sha256(config_path),
            },
            style=self.style,
            gt_states=gt_states,
            gt_interval_foot_forces_world=gt_forces,
            gt_interval_foot_positions_world=gt_foot_positions,
            gt_source_rows=gt_rows,
            gt_motion=gt_motion,
        )


def build_visualization_frame(
    sequence: MotionForceSequence, transition_index: int
) -> VisualizationFrame:
    if transition_index < 0 or transition_index >= sequence.number_of_intervals:
        raise IndexError(transition_index)
    state = sequence.physical_states[transition_index]
    positions = sequence.contact_positions_world[transition_index]
    world_forces = sequence.interval_forces_world[transition_index]
    active = sequence.active_display_mask[transition_index]
    polygon = support_polygon_xy(positions, active)
    cop = center_of_pressure_world(
        positions,
        sequence.interval_forces_contact[transition_index],
        minimum_normal_force_n=sequence.style.cop_min_normal_force_n,
        plane_height_m=sequence.plane_height_m,
    )
    q_gt = v_gt = gt_forces = None
    if sequence.gt_states is not None:
        q_gt = sequence.gt_states[transition_index, :36]
        v_gt = sequence.gt_states[transition_index, 36:]
    if sequence.gt_interval_foot_forces_world is not None:
        gt_forces = sequence.gt_interval_foot_forces_world[transition_index]
    gt_positions = None
    if sequence.gt_interval_foot_positions_world is not None:
        gt_positions = sequence.gt_interval_foot_positions_world[
            transition_index
        ]
    return VisualizationFrame(
        time_seconds=transition_index * sequence.dt,
        state_index=transition_index,
        transition_index=transition_index,
        q_estimated=state[:36],
        v_estimated=state[36:],
        corner_positions_world=positions,
        corner_forces_world=world_forces,
        corner_friction_utilization=(
            sequence.friction_utilization[transition_index]
        ),
        active_display_mask=active,
        prime_foot_forces_world=aggregate_foot_forces(world_forces),
        support_polygon_xy=polygon,
        center_of_pressure_world=cop,
        diagnostics=sequence.diagnostics.row(transition_index),
        hidden_barrier_tail_count=int((~active).sum()),
        max_friction_utilization=float(
            np.max(sequence.friction_utilization[transition_index])
        ),
        q_gt=q_gt,
        v_gt=v_gt,
        gt_foot_forces_world=gt_forces,
        gt_foot_positions_world=gt_positions,
    )


def _manifest(sequence: MotionForceSequence) -> dict:
    style = asdict(sequence.style)
    return {
        "schema": "g1cal_motion_force_sequence_v1",
        "run_id": sequence.run_id,
        "source_result_dir": sequence.source_result_dir,
        "source_config": sequence.source_config,
        "source_hashes": sequence.source_hashes,
        "profile": {
            "id": sequence.profile_id,
            "key": sequence.profile_key,
            "urdf_sha256": sequence.urdf_sha256,
            "mjcf_sha256": sequence.mjcf_sha256,
            "total_mass_kg": sequence.total_mass_kg,
        },
        "timing": {
            "dt_seconds": sequence.dt,
            "physical_states": sequence.number_of_states,
            "force_intervals": sequence.number_of_intervals,
            "arrival_anchor_removed": True,
            "prime_force_semantics": "step-average over physical k->k+1",
            "mujoco_force_alignment": "post-step GT row k+1 when present",
        },
        "contact": {
            "frames": list(sequence.contact_names),
            "force_source_order": ["tangent_1", "tangent_2", "normal"],
            "force_frame": "world after deterministic basis conversion",
            "mu": sequence.mu,
            "kappa": sequence.kappa,
            "plane_normal_world": sequence.plane_normal_world.tolist(),
            "plane_height_m": sequence.plane_height_m,
        },
        "solve_summary": sequence.solve_summary,
        "style": style,
        "gt": {
            "present": sequence.gt_states is not None,
            "motion": sequence.gt_motion,
            "source_rows": (
                sequence.gt_source_rows.tolist()
                if sequence.gt_source_rows is not None else None
            ),
            "force_level": "foot aggregate only",
        },
        "claim_boundary": (
            "estimates from the PRIME-based lower-level solver at the "
            "released covariance; ground-truth comparison is foot-level "
            "aggregate GRF only"
        ),
    }


def write_sequence_artifacts(
    sequence: MotionForceSequence, output_dir: str
) -> tuple[Path, Path]:
    output = resolve_inside_root(output_dir, must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "visualization_manifest.json"
    sequence_path = output / "motion_force_sequence.npz"
    manifest_path.write_text(
        json.dumps(_manifest(sequence), indent=2, sort_keys=True) + "\n"
    )
    arrays = {
        "physical_states": sequence.physical_states,
        "interval_forces_contact": sequence.interval_forces_contact,
        "interval_forces_world": sequence.interval_forces_world,
        "contact_positions_world": sequence.contact_positions_world,
        "contact_gaps_m": sequence.contact_gaps_m,
        "friction_utilization": sequence.friction_utilization,
        "active_display_mask": sequence.active_display_mask,
    }
    if sequence.gt_states is not None:
        arrays["gt_states"] = sequence.gt_states
    if sequence.gt_interval_foot_forces_world is not None:
        arrays["gt_interval_foot_forces_world"] = (
            sequence.gt_interval_foot_forces_world
        )
    if sequence.gt_interval_foot_positions_world is not None:
        arrays["gt_interval_foot_positions_world"] = (
            sequence.gt_interval_foot_positions_world
        )
    np.savez_compressed(sequence_path, **arrays)
    return manifest_path, sequence_path
