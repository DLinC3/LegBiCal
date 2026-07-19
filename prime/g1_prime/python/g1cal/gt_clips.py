"""Ground-truth clip loading and MuJoCo<->Pinocchio coordinate conversions.

Each released clip ships a self-contained ``gt_clip.npz`` holding the MuJoCo
ground-truth slice (post-step states, actuator torques, foot-level contact
debug) for exactly the 501 released states. Ground truth is consumed only by
the calibration loss and the visualization; the lower-level estimator never
sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .paths import resolve_inside_root


def mujoco_qpos_to_pinocchio(qpos: np.ndarray) -> np.ndarray:
    """Convert MuJoCo free-joint [xyz,wxyz,joints] to Pinocchio xyzw."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape[-1] != 36:
        raise ValueError("qpos must end in dimension 36")
    out = qpos.copy()
    out[..., 3:7] = qpos[..., [4, 5, 6, 3]]
    return out


def pinocchio_qpos_to_mujoco(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=float)
    if qpos.shape[-1] != 36:
        raise ValueError("qpos must end in dimension 36")
    out = qpos.copy()
    out[..., 3:7] = qpos[..., [6, 3, 4, 5]]
    return out


def mujoco_velocity_to_pinocchio(qpos_mj: np.ndarray,
                                 qvel_mj: np.ndarray) -> np.ndarray:
    """Map MuJoCo world-linear/body-angular free velocity to Pin tangent."""
    qpos_mj = np.asarray(qpos_mj, dtype=float)
    qvel_mj = np.asarray(qvel_mj, dtype=float)
    if qpos_mj.shape[-1] != 36 or qvel_mj.shape[-1] != 35:
        raise ValueError("expected qpos 36 and qvel 35")
    rotation = Rotation.from_quat(qpos_mj[..., [4, 5, 6, 3]]).as_matrix()
    out = qvel_mj.copy()
    out[..., :3] = np.einsum("...ji,...j->...i", rotation,
                             qvel_mj[..., :3])
    return out


def pinocchio_velocity_to_mujoco(qpos_mj: np.ndarray,
                                 velocity_pin: np.ndarray) -> np.ndarray:
    qpos_mj = np.asarray(qpos_mj, dtype=float)
    velocity_pin = np.asarray(velocity_pin, dtype=float)
    rotation = Rotation.from_quat(qpos_mj[..., [4, 5, 6, 3]]).as_matrix()
    out = velocity_pin.copy()
    out[..., :3] = np.einsum("...ij,...j->...i", rotation,
                             velocity_pin[..., :3])
    return out


@dataclass(frozen=True)
class MujocoTruthClip:
    motion: str
    source_start: int
    source_stop: int
    qpos_pin: np.ndarray
    qvel_pin: np.ndarray
    actuator_force: np.ndarray
    contact_debug: dict[str, np.ndarray]

    @property
    def states(self) -> np.ndarray:
        return np.hstack((self.qpos_pin, self.qvel_pin))


def load_gt_clip(clip: str) -> MujocoTruthClip:
    """Load one released clip's self-contained ground-truth slice."""
    path = resolve_inside_root(f"data/clips/{clip}/gt_clip.npz")
    with np.load(path) as source:
        metadata = json.loads(str(source["metadata"]))
        qpos_mj = np.asarray(source["gt__qpos"], dtype=float)
        qvel_mj = np.asarray(source["gt__qvel"], dtype=float)
        torque = np.asarray(
            source["input__actuator_force_joint_order"], dtype=float
        )
        debug = {
            key.replace("__", "/"): np.asarray(source[key])
            for key in source.files
            if key.startswith("contact_debug__")
        }
    if qpos_mj.shape[0] != metadata["states"]:
        raise ValueError(f"gt clip state count mismatch: {path}")
    if torque.shape[0] != metadata["transitions"]:
        raise ValueError(f"gt clip torque count mismatch: {path}")
    return MujocoTruthClip(
        motion=metadata["motion"],
        source_start=metadata["source_start"],
        source_stop=metadata["source_stop"],
        qpos_pin=mujoco_qpos_to_pinocchio(qpos_mj),
        qvel_pin=mujoco_velocity_to_pinocchio(qpos_mj, qvel_mj),
        actuator_force=torque,
        contact_debug=debug,
    )
