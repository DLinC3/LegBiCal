# LegBiCal

Reference implementations for *Simultaneous Calibration of Noise Covariance
and Kinematics for State Estimation of Legged Robots via Bi-level
Optimization* ([arXiv:2510.11539](https://arxiv.org/abs/2510.11539)).

[Explore the calibrated G1 PRIME results](https://dlinc3.github.io/LegBiCal/)
in the interactive Meshcat viewer. Replays open at `0.5x`; `1x` is available
in the viewer and through direct links.

## Implementations

| Directory | Estimator and calibration path |
|---|---|
| [`matlab/`](matlab/README.md) | Stage-structured Fatrop FIE with covariance and kinematic calibration |
| [`python/`](python/README.md) | Hardware-oriented B1 FIE with sparse-adjoint bilevel calibration |
| [`cuda/`](cuda/README.md) | Batched Torch CPU/CUDA covariance calibration for a contact-aided InEKF |
| [`prime/`](prime/README.md) | Contact-aware PRIME FDDP implementations for STRIDE and Unitree G1 |

Each implementation owns its environment, commands, and tests. Start from its
README and follow the local links one level at a time.

## Project information

- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Repository license: [`LICENSE`](LICENSE)
- Third-party software and data: [`THIRD_PARTY.md`](THIRD_PARTY.md)
- Cross-implementation test entry point: [`tests/run.sh`](tests/run.sh)
