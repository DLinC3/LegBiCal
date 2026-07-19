# Third-party notices

This repository builds on the excellent work of several projects.

## PRIME (vendored subset)

`third_party/PRIME/` contains a vendored subset of
[PRIME](https://github.com/well-robotics/PRIME.git) (a Crocoddyl fork with
smoothed second-order-cone contact extensions), pinned at commit
`b848ceecd451f4786ce39dcefa59e96dbaa369ba`.

- License: BSD 3-Clause (see `third_party/PRIME/LICENSE`).
- Copyright: LAAS-CNRS (2018-2020), University of Edinburgh (2019-2020),
  Jiarong Kang, Legged AI Lab, University of Wisconsin-Madison (2026).
- The lower-level estimator, its contact solver, and the visualized results
  in this repository are built on top of PRIME's excellent estimator work.
- See `third_party/PRIME/VENDORED.md` for exactly which files were vendored
  and why.

## Crocoddyl

PRIME is a fork of [Crocoddyl](https://github.com/loco-3d/crocoddyl)
(BSD 3-Clause, LAAS-CNRS / University of Edinburgh); Crocoddyl's attribution
is preserved inside `third_party/PRIME/`.

## Unitree G1 robot model

`models/g1/` contains the Unitree G1 (29-DoF) URDF/MJCF description and STL
meshes. These files derive from Unitree Robotics' BSD-3-Clause model
repositories. Pinned commits, byte-level mesh provenance, and both upstream
license texts are recorded in `models/g1/NOTICE.md`.

## Motion data

`data/clips/` contains two 10-second sensor/ground-truth clips generated
from MuJoCo rollouts of retargeted human running motions. See
`data/clips/README.md` for provenance.

The motion lineage and release constraint are recorded in
`data/clips/README.md`. Public redistribution of the clip files requires
separate permission from the original motion rightsholder; this repository
must not be published with those files until that permission is documented.
