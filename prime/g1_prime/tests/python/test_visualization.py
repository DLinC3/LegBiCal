"""Shared visualization contract, style v2, and replay scene tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import pytest

from g1cal.gt_clips import pinocchio_qpos_to_mujoco, pinocchio_velocity_to_mujoco
from g1cal.paths import project_root, resolve_inside_root
from g1cal.rendering import build_sequence
from g1cal.visualization.force_geometry import (
    aggregate_foot_forces,
    arrow_transforms,
    center_of_pressure_world,
    support_polygon_xy,
)
from g1cal.visualization.meshcat_system import (
    MeshcatMotionForceRenderer,
    _checker_png,
    record_meshcat_html,
)
from g1cal.visualization.mujoco_renderer import (
    MujocoKinematicRenderer,
    apply_kinematic_state,
)
from g1cal.visualization.recording import sampled_transition_indices
from g1cal.visualization.sequence import (
    build_visualization_frame,
    load_visualization_style,
)


@pytest.fixture(scope="module")
def run1_sequence():
    return build_sequence("run1")[0]


def test_style_v2_has_locked_prominence_and_colors():
    style = load_visualization_style("configs/visualization/default.yaml")
    assert style.arrow_unlit
    assert style.corner_shaft_radius_m == pytest.approx(0.007)
    assert style.corner_head_radius_m == pytest.approx(0.018)
    assert style.foot_sum_radius_scale == pytest.approx(2.0)
    assert style.force_scale_m_per_n == pytest.approx(0.0005)
    assert style.cop_color == "#009E73"
    assert style.support_polygon_color == "#8899AA"
    assert style.support_polygon_opacity == pytest.approx(0.6)


def test_checker_texture_matches_scene_palette():
    image = Image.open(BytesIO(_checker_png(
        "#334D66", "#1A334D", "#CCCCCC", cells=8, pixels=512
    )))
    colors = set(map(tuple, np.asarray(image).reshape((-1, 3))))
    assert {(51, 77, 102), (26, 51, 77), (204, 204, 204)} <= colors


def test_meshcat_follow_camera_is_close_and_targets_robot_center():
    base = np.array([4.0, -2.0, 0.7])
    transform = MeshcatMotionForceRenderer._camera_transform(base)
    target = base + np.array([0.0, 0.0, 0.55])
    eye = transform[:3, 3]
    assert np.linalg.norm(eye - target) == pytest.approx(
        np.linalg.norm([1.0, 1.0, 0.8])
    )
    assert np.allclose(
        transform[:3, 2], (eye - target) / np.linalg.norm(eye - target)
    )


def test_replay_scene_matches_prime_ground_contract():
    scene = resolve_inside_root("configs/replay/g1_scene.xml")
    root = ET.parse(scene).getroot()
    textures = {item.get("name", "skybox"): item
                for item in root.findall("asset/texture")}
    ground = textures["groundplane"]
    assert ground.get("builtin") == "checker"
    assert ground.get("rgb1") == "0.2 0.3 0.4"
    assert ground.get("rgb2") == "0.1 0.2 0.3"
    assert ground.get("markrgb") == "0.8 0.8 0.8"
    material = root.find("asset/material[@name='groundplane']")
    assert material is not None
    assert material.get("texrepeat") == "5 5"
    assert material.get("reflectance") == "0.2"
    sky = textures["skybox"]
    assert sky.get("rgb1") == "0.3 0.5 0.7" and sky.get("rgb2") == "0 0 0"
    light = root.find("worldbody/light")
    assert light is not None and light.get("directional") == "true"


def test_released_sequence_alignment_and_final_sampling(run1_sequence):
    sequence = run1_sequence
    assert sequence.number_of_states == 501
    assert sequence.number_of_intervals == 500
    assert sequence.physical_states.shape == (501, 71)
    assert sequence.interval_forces_world.shape == (500, 8, 3)
    assert sequence.gt_interval_foot_forces_world.shape == (500, 2, 3)
    indices = sampled_transition_indices(500, stride=5)
    assert indices[0] == 0 and indices[-1] == 499


def test_frame_force_geometry_is_consistent(run1_sequence):
    frame = build_visualization_frame(run1_sequence, 0)
    assert np.allclose(
        frame.prime_foot_forces_world,
        aggregate_foot_forces(frame.corner_forces_world),
    )
    assert np.array_equal(
        frame.support_polygon_xy,
        support_polygon_xy(
            frame.corner_positions_world, frame.active_display_mask
        ),
    )
    expected_cop = center_of_pressure_world(
        frame.corner_positions_world,
        run1_sequence.interval_forces_contact[0],
        minimum_normal_force_n=run1_sequence.style.cop_min_normal_force_n,
        plane_height_m=run1_sequence.plane_height_m,
    )
    assert np.allclose(frame.center_of_pressure_world, expected_cop)


def test_force_arrow_keeps_physical_length_and_v2_radii():
    arrow = arrow_transforms(
        np.zeros(3), np.array([0.0, 0.0, 200.0]),
        scale_m_per_n=0.0005,
        display_threshold_n=5.0,
        shaft_radius_m=0.007,
        head_radius_m=0.018,
    )
    assert arrow.visible and arrow.length_m == pytest.approx(0.1)
    assert np.linalg.norm(arrow.endpoint) == pytest.approx(0.1)
    assert np.linalg.norm(arrow.shaft_transform[:3, 0]) == pytest.approx(0.007)
    assert np.linalg.norm(arrow.head_transform[:3, 0]) == pytest.approx(0.018)


def test_kinematic_state_mapping_uses_velocity_frame_conversion(run1_sequence):
    model = mujoco.MjModel.from_xml_path(
        str(resolve_inside_root("configs/replay/g1_scene.xml"))
    )
    data = mujoco.MjData(model)
    state = run1_sequence.physical_states[10]
    qpos, qvel = apply_kinematic_state(
        model, data, state[:36], state[36:]
    )
    expected_qpos = pinocchio_qpos_to_mujoco(state[:36])
    expected_qvel = pinocchio_velocity_to_mujoco(expected_qpos, state[36:])
    assert np.allclose(qpos, expected_qpos)
    assert np.allclose(qvel, expected_qvel)
    assert np.allclose(data.qvel, expected_qvel)


def test_mujoco_offscreen_frame_uses_style_markers(run1_sequence):
    renderer = MujocoKinematicRenderer(
        run1_sequence, width=320, height=240, annotate=False
    )
    captured_connectors = []
    captured_spheres = []
    original_connector = renderer._append_connector
    original_sphere = renderer._append_sphere

    def connector(**kwargs):
        captured_connectors.append(kwargs.copy())
        return original_connector(**kwargs)

    def sphere(position, radius, color, alpha=1.0):
        captured_spheres.append((radius, color, alpha))
        return original_sphere(position, radius, color, alpha)

    renderer._append_connector = connector
    renderer._append_sphere = sphere
    try:
        image = renderer.render(build_visualization_frame(run1_sequence, 0))
    finally:
        renderer.close()
    assert image.shape == (240, 320, 3)
    polygon = [item for item in captured_connectors
               if item["name"] == "support_polygon"]
    assert polygon
    assert all(item["color"] == "#8899AA" for item in polygon)
    assert all(item["alpha"] == pytest.approx(0.6) for item in polygon)
    assert any(radius == pytest.approx(0.018) and color == "#009E73"
               for radius, color, _ in captured_spheres)


def test_meshcat_html_is_self_contained_and_unlit(
    run1_sequence, fresh_scratch
):
    root = fresh_scratch("out/test_scratch/meshcat_html")
    relative = str((root / "probe.html").relative_to(project_root()))
    path = record_meshcat_html(run1_sequence, relative, stride=500)
    html = path.read_text()
    assert "MeshBasicMaterial" in html
    assert 'id="g1cal-follow-camera-v1"' in html
    assert 'id="g1cal-playback-rate-v1"' in html
    assert 'data-rate="0.5"' in html
    assert "viewer.animator.mixer.timeScale = rate" in html
    assert "URLSearchParams(window.location.search)" in html
    assert "setRate(requested === 1.0 ? 1.0 : 0.5)" in html
    assert 'getObjectByName("pelvis_0")' in html
    assert "data:application/octet-binary;base64" in html
    assert path.stat().st_size > 1_000_000
    assert "http://127.0.0.1" not in html


def test_release_sources_define_no_video_container():
    sources = [
        path for path in project_root().rglob("*")
        if path.is_file()
        and not any(part in {"third_party", "build", "out", ".git"}
                    for part in path.parts)
        # Generated self-contained Meshcat pages embed the upstream JavaScript
        # bundle; scan authored release sources rather than third-party bundle
        # internals.
        and not ("docs" in path.parts and "media" in path.parts)
        and path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".html"}
    ]
    forbidden = "." + "mp4"
    assert all(forbidden not in path.read_text(errors="ignore").lower()
               for path in sources)
