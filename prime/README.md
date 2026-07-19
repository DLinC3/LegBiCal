# PRIME implementations

Contact-aware full-information estimation and covariance calibration built on
[PRIME](https://github.com/well-robotics/PRIME) and Crocoddyl FDDP.

## Implementations

| Directory | Scope |
|---|---|
| [`stride_prime/`](stride_prime/README.md) | Planar STRIDE covariance and shin-geometry calibration |
| [`g1_prime/`](g1_prime/README.md) | Unitree G1 two-clip covariance calibration, Meshcat pages, and MuJoCo replay |

The implementations are deliberately independent: each carries the PRIME
source subset required by its established build. Both subsets are fixed to
upstream commit `b848ceecd451f4786ce39dcefa59e96dbaa369ba` and preserve the
BSD 3-Clause attribution. Build and run commands live in the corresponding
child README.

Return to the [repository overview](../README.md).
