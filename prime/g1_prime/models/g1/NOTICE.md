# Unitree G1 model attribution

The URDF/MJCF description and STL meshes in this directory describe the
Unitree G1 humanoid (29 DoF) and are derived from Unitree Robotics' published
robot descriptions under BSD 3-Clause.

The shipped mesh bytes were audited against these official repositories:

- `unitreerobotics/unitree_mujoco`, commit
  `ae6a8403e272733e9996ef59990880330496177f`;
- `unitreerobotics/unitree_ros`, commit
  `d96d8f63ae17a7108d4f7229c00ef875ba7129c9`.

Every shipped STL matches the same-named file in at least one of those pinned
trees. Their license texts are preserved as `LICENSE.unitree_mujoco` and
`LICENSE.unitree_ros`.

- `urdf/g1_custom_collision_29dof.urdf` — Pinocchio-compatible description
  (total mass 33.34014202 kg); mesh references point at `../mjcf/assets/`.
- `mjcf/g1.xml` — MuJoCo description (total mass 33.341142 kg).
- `../../configs/replay/g1_scene.xml` — replay scene adding the floor/sky/light
  setup used by
  PRIME's experiment MJCF (well-robotics/PRIME), with thanks to the PRIME
  authors for their excellent work.
- `contact_frames.json` — the eight foot corner frames used by the
  point-contact estimator, derived from the MJCF foot support geometry.

The URDF is a Pinocchio/contact-frame adaptation and the MJCF is the aligned
29-DoF experiment variant; neither is represented as an unmodified upstream
file. The BSD notices above cover the upstream model material.
