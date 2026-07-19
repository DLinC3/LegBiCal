"""MuJoCo kinematic replay and user-scene contact-force rendering."""

from __future__ import annotations

import os

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..profiles import load_model_profile
from ..gt_clips import (
    pinocchio_qpos_to_mujoco,
    pinocchio_velocity_to_mujoco,
)
from .force_geometry import arrow_transforms
from .types import MotionForceSequence, VisualizationFrame


def _rgba(color: str, alpha: float = 1.0) -> np.ndarray:
    value = color.removeprefix("#")
    rgb = [int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4)]
    return np.asarray([*rgb, alpha], dtype=np.float32)


def apply_kinematic_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_pinocchio: np.ndarray,
    v_pinocchio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign one prescribed state and update kinematics without stepping."""
    qpos = pinocchio_qpos_to_mujoco(q_pinocchio)
    qvel = pinocchio_velocity_to_mujoco(qpos, v_pinocchio)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    return qpos, qvel


class MujocoKinematicRenderer:
    """Render prescribed states; never advances or re-solves dynamics."""

    def __init__(
        self,
        sequence: MotionForceSequence,
        *,
        width: int = 960,
        height: int = 720,
        annotate: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("renderer width/height must be positive")
        self.sequence = sequence
        self.profile = load_model_profile(sequence.profile_id)
        # Prefer the replay scene (robot plus the PRIME experiment MJCF's
        # checker groundplane/skybox/light, with thanks to the PRIME
        # authors); fall back to the bare robot MJCF.
        scene = self.profile.scene_mjcf_path or self.profile.mjcf_path
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        if self.model.nq != 36 or self.model.nv != 35:
            raise ValueError("MuJoCo profile does not have G1 dimensions")
        self.data = mujoco.MjData(self.model)
        # Renderer enforces the MJCF offscreen framebuffer declaration. Grow
        # only this loaded visualization model when a caller requests a larger
        # image; frozen source bytes and all dynamics values are untouched.
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), height
        )
        self.renderer = mujoco.Renderer(
            self.model, height=height, width=width, max_geom=10000
        )
        self.width = width
        self.height = height
        self.annotate = annotate
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.4
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0
        self.render_count = 0
        self.last_arrow_endpoints: dict[str, np.ndarray] = {}
        self.last_image: np.ndarray | None = None

    @property
    def mjcf_total_mass_kg(self) -> float:
        return float(self.model.body_mass.sum())

    def _append_connector(
        self,
        *,
        name: str,
        start: np.ndarray,
        end: np.ndarray,
        width: float,
        color: str,
        alpha: float = 1.0,
        geom_type=mujoco.mjtGeom.mjGEOM_ARROW,
    ) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo visualization scene geometry exhausted")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            _rgba(color, alpha),
        )
        mujoco.mjv_connector(
            geom, geom_type, width,
            np.asarray(start, dtype=float), np.asarray(end, dtype=float),
        )
        scene.ngeom += 1
        self.last_arrow_endpoints[name] = np.asarray(end, dtype=float).copy()

    def _append_sphere(
        self, position: np.ndarray, radius: float, color: str, alpha: float = 1.0
    ) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("MuJoCo visualization scene geometry exhausted")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0]),
            np.asarray(position, dtype=float),
            np.eye(3).reshape(-1),
            _rgba(color, alpha),
        )
        scene.ngeom += 1

    def _append_force_arrow(
        self,
        name: str,
        origin: np.ndarray,
        force: np.ndarray,
        color: str,
        *,
        width: float,
        alpha: float = 1.0,
    ) -> None:
        arrow = arrow_transforms(
            origin,
            force,
            scale_m_per_n=self.sequence.style.force_scale_m_per_n,
            display_threshold_n=self.sequence.style.force_display_threshold_n,
        )
        self.last_arrow_endpoints[name] = arrow.endpoint.copy()
        if arrow.visible:
            self._append_connector(
                name=name,
                start=arrow.origin,
                end=arrow.endpoint,
                width=width,
                color=color,
                alpha=alpha,
            )

    def _contact_color(self, foot_color: str, utilization: float) -> str:
        if utilization >= self.sequence.style.friction_critical:
            return self.sequence.style.colors["invalid"]
        if utilization >= self.sequence.style.friction_warning:
            return self.sequence.style.colors["warning"]
        return foot_color

    def render(self, frame: VisualizationFrame) -> np.ndarray:
        qpos, _ = apply_kinematic_state(
            self.model, self.data, frame.q_estimated, frame.v_estimated
        )
        self.camera.lookat[:] = qpos[:3]
        self.renderer.update_scene(self.data, camera=self.camera)
        self.last_arrow_endpoints = {}
        colors = self.sequence.style.colors
        if not np.any(
            self.model.geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            scene = self.renderer.scene
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_PLANE,
                np.array([2.0, 2.0, 0.01]),
                np.array([0.0, 0.0, self.sequence.plane_height_m]),
                np.eye(3).reshape(-1),
                _rgba("#707780", 0.55),
            )
            scene.ngeom += 1

        for index, (position, force, utilization, active) in enumerate(zip(
            frame.corner_positions_world,
            frame.corner_forces_world,
            frame.corner_friction_utilization,
            frame.active_display_mask,
            strict=True,
        )):
            foot_color = colors["prime_left" if index < 4 else "prime_right"]
            if active:
                self._append_sphere(
                    position, 0.012,
                    self._contact_color(foot_color, float(utilization)),
                )
            self._append_force_arrow(
                f"corner_{index}", position, force, foot_color,
                width=self.sequence.style.corner_shaft_radius_m,
            )

        for foot, contact_slice in enumerate((slice(0, 4), slice(4, 8))):
            side = "left" if foot == 0 else "right"
            origin = frame.corner_positions_world[contact_slice].mean(axis=0)
            self._append_force_arrow(
                f"prime_foot_{foot}", origin,
                frame.prime_foot_forces_world[foot], colors[f"prime_{side}"],
                width=(self.sequence.style.corner_shaft_radius_m
                       * self.sequence.style.foot_sum_radius_scale),
            )
            if (
                frame.gt_foot_forces_world is not None
                and frame.gt_foot_positions_world is not None
            ):
                self._append_force_arrow(
                    f"gt_foot_{foot}", frame.gt_foot_positions_world[foot],
                    frame.gt_foot_forces_world[foot], colors["mujoco_gt"],
                    width=(self.sequence.style.corner_shaft_radius_m
                           * self.sequence.style.foot_sum_radius_scale),
                    alpha=0.72,
                )

        polygon = frame.support_polygon_xy
        if polygon.shape[0] >= 2:
            closed = np.vstack((polygon, polygon[0]))
            for start_xy, end_xy in zip(closed[:-1], closed[1:], strict=True):
                self._append_connector(
                    name="support_polygon",
                    start=np.array([*start_xy, self.sequence.plane_height_m + 0.004]),
                    end=np.array([*end_xy, self.sequence.plane_height_m + 0.004]),
                    width=max(
                        1.0,
                        1000.0 * self.sequence.style.support_polygon_radius_m,
                    ),
                    color=self.sequence.style.support_polygon_color,
                    alpha=self.sequence.style.support_polygon_opacity,
                    geom_type=mujoco.mjtGeom.mjGEOM_LINE,
                )
        if frame.center_of_pressure_world is not None:
            self._append_sphere(
                frame.center_of_pressure_world + np.array([0.0, 0.0, 0.006]),
                self.sequence.style.cop_radius_m,
                self.sequence.style.cop_color,
            )
        image = self.renderer.render().copy()
        if self.annotate:
            image = self._annotate(image, frame)
        self.last_image = image
        self.render_count += 1
        return image

    def _annotate(
        self, image: np.ndarray, frame: VisualizationFrame
    ) -> np.ndarray:
        prime = frame.prime_foot_forces_world
        lines = [
            f"{self.sequence.run_id} | {self.sequence.profile_id}",
            f"t={frame.time_seconds:.3f}s transition={frame.transition_index}",
            (
                f"PRIME Fz L/R={prime[0,2]:.1f}/{prime[1,2]:.1f} N  "
                f"Fz/BW={prime[:,2].sum()/(self.sequence.total_mass_kg*9.81):.3f}  "
                f"max rho={frame.max_friction_utilization:.3f}"
            ),
            (
                f"URDF mass={self.sequence.total_mass_kg:.6f} kg  "
                f"MJCF mass={self.mjcf_total_mass_kg:.6f} kg"
            ),
            "PRIME force: latent step-average | diagnostic visualization",
            (
                f"force scale: 200 N = "
                f"{200.0 * self.sequence.style.force_scale_m_per_n:.2f} m | hidden < "
                f"{self.sequence.style.force_display_threshold_n:g} N: "
                f"{frame.hidden_barrier_tail_count}"
            ),
        ]
        if frame.gt_foot_forces_world is not None:
            gt = frame.gt_foot_forces_world
            lines.append(
                f"MuJoCo post-step GT Fz L/R={gt[0,2]:.1f}/{gt[1,2]:.1f} N"
            )
            lines.append("GT comparison: foot-level aggregate GRF only")
        relative = frame.diagnostics["newton_relative_grad_norm"]
        tolerance = float(
            self.sequence.solve_summary.get(
                "newton_relative_grad_tolerance", 1e-7
            )
        )
        invalid = (
            not np.isfinite(relative)
            or relative > tolerance
            or frame.diagnostics["min_cone_margin"] <= 0.0
            or frame.diagnostics["min_alpha"] <= 0.0
        )
        if invalid:
            health = "INVALID"
        elif not frame.diagnostics["newton_converged"]:
            health = "AMBER/raw Newton false; accepted relative gate"
        else:
            health = "PASS"
        lines.append(
            f"contact health={health} Newton rel={relative:.3e} "
            f"cone={frame.diagnostics['min_cone_margin']:.3e}"
        )
        if not bool(self.sequence.solve_summary.get("solved", False)):
            lines.append("NON-CONVERGED RECONSTRUCTION")
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        padding = 8
        line_height = 15
        draw.rectangle(
            [0, 0, min(self.width, 690), padding * 2 + line_height * len(lines)],
            fill=(0, 0, 0, 175),
        )
        for index, line in enumerate(lines):
            draw.text((padding, padding + index * line_height), line,
                      fill=(255, 255, 255))
        return np.asarray(pil)

    def close(self) -> None:
        self.renderer.close()


def replay_interactive(
    sequence: MotionForceSequence,
    *,
    loop: bool = True,
    realtime_factor: float = 1.0,
) -> None:
    """Interactive kinematic replay in the MuJoCo viewer.

    Assigns ``qpos/qvel`` and calls ``mj_forward`` per frame at the clip's
    physical rate; never ``mj_step``. The scene includes the PRIME
    experiment MJCF's checker floor, with thanks to the PRIME authors.
    """
    import time as _time

    import mujoco.viewer

    profile = load_model_profile(sequence.profile_id)
    scene = profile.scene_mjcf_path or profile.mjcf_path
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    states = sequence.physical_states
    dt = sequence.dt / max(realtime_factor, 1e-6)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for state in states:
                if not viewer.is_running():
                    break
                frame_start = _time.perf_counter()
                apply_kinematic_state(model, data, state[:36], state[36:])
                viewer.sync()
                remaining = dt - (_time.perf_counter() - frame_start)
                if remaining > 0.0:
                    _time.sleep(remaining)
            if not loop:
                break
