"""Meshcat renderer and Drake publish sink for shared visualization frames.

Style v2: force arrows render unlit (constant saturated color regardless of
scene lighting) with thicker shafts/heads; the CoP marker and support
polygon use light style-driven colors so they no longer out-contrast the
forces; the floor is a checker texture matching the PRIME experiment MJCF
groundplane, with thanks to the PRIME authors for their excellent work.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as transformations
from meshcat.animation import Animation
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from ..profiles import load_model_profile
from ..paths import resolve_inside_root
from .force_geometry import arrow_transforms
from .recording import sampled_playback_fps, sampled_transition_indices
from .sequence import build_visualization_frame
from .types import MotionForceSequence, VisualizationFrame


def _hex_color(value: str) -> int:
    return int(value.removeprefix("#"), 16)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    raw = value.removeprefix("#")
    return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))


def _checker_png(
    rgb1: str, rgb2: str, mark_rgb: str, *, cells: int = 8, pixels: int = 512
) -> bytes:
    """Programmatic checker tile matching MuJoCo's builtin=checker mark=edge."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (pixels, pixels))
    draw = ImageDraw.Draw(image)
    cell = pixels // cells
    color1, color2, mark = _hex_rgb(rgb1), _hex_rgb(rgb2), _hex_rgb(mark_rgb)
    edge = max(1, pixels // 300)
    for row in range(cells):
        for column in range(cells):
            x0, y0 = column * cell, row * cell
            x1, y1 = x0 + cell - 1, y0 + cell - 1
            fill = color1 if (row + column) % 2 == 0 else color2
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            draw.rectangle(
                [x0, y0, x1, y1], outline=mark, width=edge
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FrameRenderer(Protocol):
    def render(self, frame: VisualizationFrame) -> None: ...


class MeshcatMotionForceRenderer:
    """Persistent Meshcat geometry; all force math is supplied by the frame."""

    def __init__(
        self,
        sequence: MotionForceSequence,
        *,
        viewer: meshcat.Visualizer | None = None,
        open_browser: bool = False,
    ) -> None:
        self.sequence = sequence
        self.style = sequence.style
        self.profile = load_model_profile(sequence.profile_id)
        package_dirs = [
            str(self.profile.urdf_path.parent),
            str(self.profile.urdf_path.parent.parent),
        ]
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            str(self.profile.urdf_path),
            package_dirs,
            pin.JointModelFreeFlyer(),
        )
        self.model = model
        self.viewer = viewer or meshcat.Visualizer()
        self.robot = MeshcatVisualizer(model, collision_model, visual_model)
        self.robot.initViewer(viewer=self.viewer, open=open_browser,
                              loadModel=False)
        self.root = f"g1cal/{sequence.run_id}"
        estimated_rgb = np.array(
            [*(_v / 255.0 for _v in _hex_rgb(
                self.style.colors["robot_estimated"]
            )), 1.0]
        )
        self.robot.loadViewerModel(
            rootNodeName=f"{self.root}/estimated",
            visual_color=estimated_rgb,
        )
        self.gt_robot = None
        if sequence.gt_states is not None:
            self.gt_robot = MeshcatVisualizer(
                model, collision_model, visual_model, copy_models=True
            )
            self.gt_robot.initViewer(viewer=self.viewer, open=False,
                                     loadModel=False)
            gt_rgb = np.array(
                [*(_v / 255.0 for _v in _hex_rgb(
                    self.style.colors["robot_gt"]
                )), self.style.robot_gt_opacity]
            )
            self.gt_robot.loadViewerModel(
                rootNodeName=f"{self.root}/ground_truth",
                visual_color=gt_rgb,
            )
        self._initialize_scene()
        self.render_count = 0
        self.last_arrow_endpoints: dict[str, np.ndarray] = {}

    def _material(self, color: str, opacity: float = 1.0, *,
                  unlit: bool | None = None):
        unlit = self.style.arrow_unlit if unlit is None else unlit
        cls = (
            geometry.MeshBasicMaterial if unlit
            else geometry.MeshLambertMaterial
        )
        return cls(
            color=_hex_color(color), transparent=opacity < 1.0,
            opacity=opacity,
        )

    def _initialize_arrow(self, name: str, color: str, opacity: float = 1.0):
        material = self._material(color, opacity)
        self.viewer[f"{name}/shaft"].set_object(
            geometry.Cylinder(1.0, 1.0), material
        )
        self.viewer[f"{name}/head"].set_object(
            geometry.Cylinder(1.0, radiusTop=0.0, radiusBottom=1.0), material
        )

    def _floor_material(self, extent: np.ndarray):
        if not self.style.floor_checker:
            return self._material("#D8D8D8", 0.45, unlit=False)
        png = _checker_png(
            self.style.floor_rgb1,
            self.style.floor_rgb2,
            self.style.floor_mark_rgb,
        )
        tile_m = 8 * self.style.floor_cell_m
        repeat = [
            max(1, int(round(float(extent[0]) / tile_m))),
            max(1, int(round(float(extent[1]) / tile_m))),
        ]
        texture = geometry.ImageTexture(
            image=geometry.PngImage(png),
            wrap=[1000, 1000],  # three.js RepeatWrapping
            repeat=repeat,
        )
        return geometry.MeshLambertMaterial(map=texture)

    def _initialize_scene(self) -> None:
        style = self.style
        colors = style.colors
        # Meshcat's perspective camera has a built-in local offset of [3, 1, 0]
        # below the animated ``/Cameras/default`` node.  Neutralize that offset
        # so the parent poses recorded below are the actual world camera poses;
        # otherwise the rotated offset can put the camera many metres away from
        # a moving robot even though the requested eye-to-target distance is
        # close.
        self.viewer[
            "/Cameras/default/rotated/<object>"
        ].set_property("position", [0.0, 0.0, 0.0])
        # The textured plane is the single ground reference.  Meshcat's own
        # infinite grid/axes otherwise overlap it and dominate small README
        # previews.
        self.viewer["/Grid"].set_property("visible", False)
        self.viewer["/Axes"].set_property("visible", False)
        ground = f"{self.root}/ground"
        base_xy = self.sequence.physical_states[:, :2]
        if self.sequence.gt_states is not None:
            base_xy = np.vstack((base_xy, self.sequence.gt_states[:, :2]))
        lower = base_xy.min(axis=0) - 3.0
        upper = base_xy.max(axis=0) + 3.0
        extent = np.maximum(upper - lower, 6.0)
        center = 0.5 * (lower + upper)
        self.viewer[ground].set_object(
            geometry.Box([float(extent[0]), float(extent[1]), 0.002]),
            self._floor_material(extent),
        )
        self.viewer[ground].set_transform(
            transformations.translation_matrix([*center, -0.001])
        )
        for index in range(8):
            color = colors["prime_left" if index < 4 else "prime_right"]
            name = f"{self.root}/forces/corner_{index}"
            self._initialize_arrow(name, color)
            self.viewer[f"{self.root}/contacts/point_{index}"].set_object(
                geometry.Sphere(0.012), self._material(color)
            )
        for foot, side in enumerate(("left", "right")):
            color = colors[f"prime_{side}"]
            self._initialize_arrow(
                f"{self.root}/forces/prime_foot_{foot}", color
            )
            self._initialize_arrow(
                f"{self.root}/forces/gt_foot_{foot}",
                colors["mujoco_gt"], opacity=0.72,
            )
        self.viewer[f"{self.root}/cop"].set_object(
            geometry.Sphere(style.cop_radius_m),
            self._material(style.cop_color),
        )
        for index in range(8):
            node = self.viewer[f"{self.root}/support_segments/{index}"]
            node.set_object(
                geometry.Cylinder(1.0, 1.0),
                self._material(
                    style.support_polygon_color,
                    style.support_polygon_opacity,
                ),
            )
            node.set_property("visible", False)

    @staticmethod
    def _segment_transform(
        start: np.ndarray, end: np.ndarray, radius: float = 0.004
    ) -> np.ndarray:
        delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            return np.zeros((4, 4))
        direction = delta / length
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(direction @ reference)) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(direction, reference)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, direction)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((x_axis, direction, z_axis)) @ (
            np.diag([radius, length, radius])
        )
        transform[:3, 3] = 0.5 * (np.asarray(start) + np.asarray(end))
        return transform

    @staticmethod
    def _camera_transform(target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float) + np.array([0.0, 0.0, 0.55])
        eye = target + np.array([-1.0, -1.0, 0.8])
        z_axis = eye - target
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        transform[:3, 3] = eye
        return transform

    def _arrow_radii(self, radius_scale: float) -> dict:
        return {
            "shaft_radius_m": self.style.corner_shaft_radius_m * radius_scale,
            "head_radius_m": self.style.corner_head_radius_m * radius_scale,
        }

    def _set_arrow(
        self,
        name: str,
        origin: np.ndarray,
        force: np.ndarray,
        *,
        radius_scale: float = 1.0,
    ) -> None:
        arrow = arrow_transforms(
            origin,
            force,
            scale_m_per_n=self.style.force_scale_m_per_n,
            display_threshold_n=self.style.force_display_threshold_n,
            **self._arrow_radii(radius_scale),
        )
        self.last_arrow_endpoints[name] = arrow.endpoint.copy()
        for part, transform in (
            ("shaft", arrow.shaft_transform),
            ("head", arrow.head_transform),
        ):
            node = self.viewer[f"{name}/{part}"]
            node.set_transform(transform)
            node.set_property("visible", arrow.visible)

    def render(self, frame: VisualizationFrame) -> None:
        self.robot.display(frame.q_estimated)
        if self.gt_robot is not None and frame.q_gt is not None:
            self.gt_robot.display(frame.q_gt)
        for index, (position, force) in enumerate(zip(
            frame.corner_positions_world, frame.corner_forces_world, strict=True
        )):
            marker = self.viewer[f"{self.root}/contacts/point_{index}"]
            marker.set_transform(transformations.translation_matrix(position))
            marker.set_property("visible", bool(frame.active_display_mask[index]))
            self._set_arrow(
                f"{self.root}/forces/corner_{index}", position, force
            )
        for foot, contact_slice in enumerate((slice(0, 4), slice(4, 8))):
            origin = frame.corner_positions_world[contact_slice].mean(axis=0)
            self._set_arrow(
                f"{self.root}/forces/prime_foot_{foot}",
                origin,
                frame.prime_foot_forces_world[foot],
                radius_scale=self.style.foot_sum_radius_scale,
            )
            gt_visible = (
                frame.gt_foot_forces_world is not None
                and frame.gt_foot_positions_world is not None
            )
            if gt_visible:
                self._set_arrow(
                    f"{self.root}/forces/gt_foot_{foot}",
                    frame.gt_foot_positions_world[foot],
                    frame.gt_foot_forces_world[foot],
                    radius_scale=self.style.foot_sum_radius_scale,
                )
            else:
                for part in ("shaft", "head"):
                    self.viewer[
                        f"{self.root}/forces/gt_foot_{foot}/{part}"
                    ].set_property("visible", False)
        cop = frame.center_of_pressure_world
        self.viewer[f"{self.root}/cop"].set_property("visible", cop is not None)
        if cop is not None:
            self.viewer[f"{self.root}/cop"].set_transform(
                transformations.translation_matrix(cop)
            )
        self._render_support_polygon(frame.support_polygon_xy)
        self.render_count += 1

    def _render_support_polygon(self, polygon_xy: np.ndarray) -> None:
        """Update the eight persistent outline cylinders via transforms only.

        The recorded and live paths share this geometry, so recordings (which
        capture transform tracks, never per-frame ``set_object``) look
        identical to the live view, and no geometry is re-uploaded per frame.
        """
        closed = (
            np.vstack((polygon_xy, polygon_xy[0]))
            if polygon_xy.shape[0] >= 2 else np.empty((0, 2))
        )
        height = self.sequence.plane_height_m + 0.004
        for index in range(8):
            node = self.viewer[f"{self.root}/support_segments/{index}"]
            visible = index < max(0, closed.shape[0] - 1)
            if visible:
                node.set_transform(self._segment_transform(
                    np.array([*closed[index], height]),
                    np.array([*closed[index + 1], height]),
                    radius=self.style.support_polygon_radius_m,
                ))
            node.set_property("visible", visible)

    def export_html(self, output: str) -> Path:
        path = resolve_inside_root(output, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.viewer.static_html()
        follow_script = """
<script id="g1cal-follow-camera-v1">
(function installG1calFollowCamera() {
  if (!window.viewer || !viewer.controls || !viewer.scene) {
    window.requestAnimationFrame(installG1calFollowCamera);
    return;
  }
  let estimated = null;
  viewer.scene.traverse(function (node) {
    if (node.name === "estimated") estimated = node;
  });
  const pelvis = estimated && estimated.getObjectByName("pelvis_0");
  if (!pelvis) {
    window.requestAnimationFrame(installG1calFollowCamera);
    return;
  }
  const updateControls = viewer.controls.update.bind(viewer.controls);
  viewer.controls.update = function () {
    viewer.scene.updateMatrixWorld(true);
    pelvis.getWorldPosition(viewer.controls.target);
    viewer.controls.target.y += 0.55;
    return updateControls();
  };
  viewer.needs_render = true;
})();
</script>
<style id="g1cal-playback-rate-style-v1">
  #g1cal-playback-rate {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 10000;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 9px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 8px;
    background: rgba(13, 21, 32, 0.84);
    color: #edf4fb;
    font: 600 12px/1.2 system-ui, sans-serif;
  }
  #g1cal-playback-rate button {
    border: 1px solid rgba(255, 255, 255, 0.32);
    border-radius: 5px;
    padding: 4px 7px;
    background: #25384a;
    color: #edf4fb;
    cursor: pointer;
  }
  #g1cal-playback-rate button[aria-pressed="true"] {
    border-color: #00c690;
    background: #007d5b;
  }
</style>
<div id="g1cal-playback-rate" role="group" aria-label="Playback speed">
  <span>Playback</span>
  <button type="button" data-rate="0.5" aria-pressed="true">0.5&times;</button>
  <button type="button" data-rate="1" aria-pressed="false">1&times;</button>
</div>
<script id="g1cal-playback-rate-v1">
(function installG1calPlaybackRate() {
  if (!window.viewer || !viewer.animator || !viewer.animator.mixer ||
      !(viewer.animator.duration > 0)) {
    window.requestAnimationFrame(installG1calPlaybackRate);
    return;
  }
  const control = document.getElementById("g1cal-playback-rate");
  const buttons = Array.from(control.querySelectorAll("button[data-rate]"));
  function setRate(rate) {
    viewer.animator.mixer.timeScale = rate;
    control.dataset.rate = String(rate);
    buttons.forEach(function (button) {
      button.setAttribute(
        "aria-pressed", String(Number(button.dataset.rate) === rate)
      );
    });
    viewer.needs_render = true;
  }
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      setRate(Number(button.dataset.rate));
    });
  });
  const requested = Number(
    new URLSearchParams(window.location.search).get("speed")
  );
  setRate(requested === 1.0 ? 1.0 : 0.5);
})();
</script>
"""
        if "</body>" not in html:
            raise RuntimeError("Meshcat static HTML has no body terminator")
        path.write_text(html.replace("</body>", follow_script + "</body>", 1))
        return path

    @staticmethod
    def _record_scaled_transform(node, transform: np.ndarray, visible: bool) -> None:
        scale = np.linalg.norm(transform[:3, :3], axis=0)
        rigid = transform.copy()
        for index, value in enumerate(scale):
            if value > 0.0:
                rigid[:3, index] /= value
            else:
                rigid[:3, index] = np.eye(3)[:, index]
        node.set_transform(rigid)
        node.set_property("scale", "vector3", scale.tolist())
        node.set_property("visible", "boolean", bool(visible))

    def _record_arrow(
        self,
        frame_viewer,
        name: str,
        origin: np.ndarray,
        force: np.ndarray,
        *,
        radius_scale: float = 1.0,
    ) -> None:
        arrow = arrow_transforms(
            origin,
            force,
            scale_m_per_n=self.style.force_scale_m_per_n,
            display_threshold_n=self.style.force_display_threshold_n,
            **self._arrow_radii(radius_scale),
        )
        self._record_scaled_transform(
            frame_viewer[f"{name}/shaft"], arrow.shaft_transform, arrow.visible
        )
        self._record_scaled_transform(
            frame_viewer[f"{name}/head"], arrow.head_transform, arrow.visible
        )

    def record_animation(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        fps: float | None = None,
    ) -> Animation:
        """Attach a transform animation without rerunning any estimator."""
        transitions = sampled_transition_indices(
            self.sequence.number_of_intervals,
            start=start,
            stop=stop,
            stride=stride,
        )
        animation = Animation(
            default_framerate=(
                fps or sampled_playback_fps(transitions, self.sequence.dt)
            )
        )
        foot_scale = self.style.foot_sum_radius_scale
        for animation_index, transition in enumerate(transitions):
            frame = build_visualization_frame(self.sequence, transition)
            with animation.at_frame(self.viewer, animation_index) as frame_viewer:
                frame_viewer["/Cameras/default"].set_transform(
                    self._camera_transform(frame.q_estimated[:3])
                )
                robot_viewer = self.robot.viewer
                self.robot.viewer = frame_viewer
                try:
                    self.robot.display(frame.q_estimated)
                finally:
                    self.robot.viewer = robot_viewer
                if self.gt_robot is not None and frame.q_gt is not None:
                    gt_viewer = self.gt_robot.viewer
                    self.gt_robot.viewer = frame_viewer
                    try:
                        self.gt_robot.display(frame.q_gt)
                    finally:
                        self.gt_robot.viewer = gt_viewer
                for index, (position, force, active) in enumerate(zip(
                    frame.corner_positions_world,
                    frame.corner_forces_world,
                    frame.active_display_mask,
                    strict=True,
                )):
                    marker = frame_viewer[
                        f"{self.root}/contacts/point_{index}"
                    ]
                    marker.set_transform(
                        transformations.translation_matrix(position)
                    )
                    marker.set_property("visible", "boolean", bool(active))
                    self._record_arrow(
                        frame_viewer,
                        f"{self.root}/forces/corner_{index}",
                        position,
                        force,
                    )
                for foot, contact_slice in enumerate(
                    (slice(0, 4), slice(4, 8))
                ):
                    origin = frame.corner_positions_world[
                        contact_slice
                    ].mean(axis=0)
                    self._record_arrow(
                        frame_viewer,
                        f"{self.root}/forces/prime_foot_{foot}",
                        origin,
                        frame.prime_foot_forces_world[foot],
                        radius_scale=foot_scale,
                    )
                    has_gt = (
                        frame.gt_foot_forces_world is not None
                        and frame.gt_foot_positions_world is not None
                    )
                    if has_gt:
                        self._record_arrow(
                            frame_viewer,
                            f"{self.root}/forces/gt_foot_{foot}",
                            frame.gt_foot_positions_world[foot],
                            frame.gt_foot_forces_world[foot],
                            radius_scale=foot_scale,
                        )
                    else:
                        for part in ("shaft", "head"):
                            frame_viewer[
                                f"{self.root}/forces/gt_foot_{foot}/{part}"
                            ].set_property("visible", "boolean", False)
                cop = frame.center_of_pressure_world
                cop_node = frame_viewer[f"{self.root}/cop"]
                cop_node.set_property("visible", "boolean", cop is not None)
                if cop is not None:
                    cop_node.set_transform(
                        transformations.translation_matrix(cop)
                    )
                polygon = frame.support_polygon_xy
                closed = (
                    np.vstack((polygon, polygon[0]))
                    if polygon.shape[0] >= 2 else np.empty((0, 2))
                )
                for segment_index in range(8):
                    node = frame_viewer[
                        f"{self.root}/support_segments/{segment_index}"
                    ]
                    visible = segment_index < max(0, closed.shape[0] - 1)
                    if visible:
                        segment_start = np.array([
                            *closed[segment_index],
                            self.sequence.plane_height_m + 0.004,
                        ])
                        segment_end = np.array([
                            *closed[segment_index + 1],
                            self.sequence.plane_height_m + 0.004,
                        ])
                        self._record_scaled_transform(
                            node,
                            self._segment_transform(
                                segment_start, segment_end,
                                radius=self.style.support_polygon_radius_m,
                            ),
                            True,
                        )
                    else:
                        node.set_property("visible", "boolean", False)
        self.viewer.set_animation(animation, play=True, repetitions=1)
        return animation

    def close(self) -> None:
        self.viewer.delete()


def record_meshcat_html(
    sequence: MotionForceSequence,
    output: str,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    fps: float | None = None,
) -> Path:
    renderer = MeshcatMotionForceRenderer(sequence, open_browser=False)
    try:
        renderer.record_animation(
            start=start, stop=stop, stride=stride, fps=fps
        )
        return renderer.export_html(output)
    finally:
        renderer.close()


class MeshcatMotionForceSystem(LeafSystem):
    """Periodic Drake publish sink around a Meshcat renderer."""

    def __init__(
        self,
        sequence: MotionForceSequence,
        renderer: FrameRenderer,
        *,
        publish_period: float | None = None,
    ) -> None:
        super().__init__()
        self.set_name("MeshcatMotionForceSystem")
        self._renderer = renderer
        self._frame_port = self.DeclareAbstractInputPort(
            "frame", AbstractValue.Make(None)
        )
        self.DeclarePeriodicPublishEvent(
            publish_period or sequence.dt, 0.0, self._publish
        )

    def _publish(self, context):
        frame = self._frame_port.Eval(context)
        if not isinstance(frame, VisualizationFrame):
            raise TypeError("Meshcat frame input is not VisualizationFrame")
        self._renderer.render(frame)
        return None

    def get_frame_input_port(self):
        return self._frame_port
