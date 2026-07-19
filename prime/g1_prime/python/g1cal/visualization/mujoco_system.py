"""Drake publish sink for MuJoCo kinematic visualization."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from .types import MotionForceSequence, VisualizationFrame


class ImageFrameRenderer(Protocol):
    def render(self, frame: VisualizationFrame) -> np.ndarray: ...


class MujocoMotionForceSystem(LeafSystem):
    """Periodic publish System; owns no estimator or MuJoCo time stepping."""

    def __init__(
        self,
        sequence: MotionForceSequence,
        renderer: ImageFrameRenderer,
        *,
        publish_period: float | None = None,
    ) -> None:
        super().__init__()
        self.set_name("MujocoMotionForceSystem")
        self._renderer = renderer
        self._frame_port = self.DeclareAbstractInputPort(
            "frame", AbstractValue.Make(None)
        )
        self.last_image: np.ndarray | None = None
        self.DeclarePeriodicPublishEvent(
            publish_period or sequence.dt, 0.0, self._publish
        )

    def _publish(self, context):
        frame = self._frame_port.Eval(context)
        if not isinstance(frame, VisualizationFrame):
            raise TypeError("MuJoCo frame input is not VisualizationFrame")
        self.last_image = self._renderer.render(frame)
        return None

    def get_frame_input_port(self):
        return self._frame_port
