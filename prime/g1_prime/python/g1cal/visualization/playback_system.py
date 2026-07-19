"""Drake time ownership for immutable motion/force playback."""

from __future__ import annotations

import math

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import LeafSystem

from .sequence import build_visualization_frame
from .types import MotionForceSequence, VisualizationFrame


class MotionForcePlaybackSystem(LeafSystem):
    """Map Drake Context time to one scientific-knot visualization frame."""

    def __init__(self, sequence: MotionForceSequence, *, loop: bool = False):
        super().__init__()
        self.set_name("MotionForcePlaybackSystem")
        self._sequence = sequence
        self._loop = loop
        self._frame_port = self.DeclareAbstractOutputPort(
            "frame",
            lambda: AbstractValue.Make(build_visualization_frame(sequence, 0)),
            self._calc_frame,
            prerequisites_of_calc={self.time_ticket()},
        )

    @property
    def duration_seconds(self) -> float:
        return self._sequence.number_of_intervals * self._sequence.dt

    def transition_index(self, time_seconds: float) -> int:
        if not math.isfinite(time_seconds):
            raise ValueError("playback time must be finite")
        raw = max(0, int(math.floor(max(0.0, time_seconds) /
                                    self._sequence.dt + 1e-12)))
        if self._loop:
            return raw % self._sequence.number_of_intervals
        return min(raw, self._sequence.number_of_intervals - 1)

    def _calc_frame(self, context, output: AbstractValue) -> None:
        index = self.transition_index(context.get_time())
        output.set_value(build_visualization_frame(self._sequence, index))

    def get_frame_output_port(self):
        return self._frame_port
