"""Canonical Drake composition for synchronized visualization sinks."""

from __future__ import annotations

from dataclasses import dataclass

from pydrake.systems.framework import Diagram, DiagramBuilder

from .meshcat_system import FrameRenderer, MeshcatMotionForceSystem
from .mujoco_system import ImageFrameRenderer, MujocoMotionForceSystem
from .playback_system import MotionForcePlaybackSystem
from .types import MotionForceSequence


@dataclass(frozen=True)
class MotionForceVisualizationDiagram:
    diagram: Diagram
    playback: MotionForcePlaybackSystem
    meshcat: MeshcatMotionForceSystem | None
    mujoco: MujocoMotionForceSystem | None


def build_visualization_diagram(
    sequence: MotionForceSequence,
    *,
    meshcat_renderer: FrameRenderer | None = None,
    mujoco_renderer: ImageFrameRenderer | None = None,
    loop: bool = False,
    publish_period: float | None = None,
) -> MotionForceVisualizationDiagram:
    """Connect one time-indexed frame source to either or both viewers."""
    if meshcat_renderer is None and mujoco_renderer is None:
        raise ValueError("at least one visualization renderer is required")
    builder = DiagramBuilder()
    playback = builder.AddSystem(MotionForcePlaybackSystem(sequence, loop=loop))
    meshcat_system = None
    mujoco_system = None
    if meshcat_renderer is not None:
        meshcat_system = builder.AddSystem(MeshcatMotionForceSystem(
            sequence, meshcat_renderer, publish_period=publish_period
        ))
        builder.Connect(
            playback.get_frame_output_port(),
            meshcat_system.get_frame_input_port(),
        )
    if mujoco_renderer is not None:
        mujoco_system = builder.AddSystem(MujocoMotionForceSystem(
            sequence, mujoco_renderer, publish_period=publish_period
        ))
        builder.Connect(
            playback.get_frame_output_port(),
            mujoco_system.get_frame_input_port(),
        )
    return MotionForceVisualizationDiagram(
        diagram=builder.Build(), playback=playback,
        meshcat=meshcat_system, mujoco=mujoco_system,
    )
