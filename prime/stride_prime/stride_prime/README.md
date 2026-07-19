# `stride_prime` package

Python orchestration for the STRIDE PRIME estimator and its upper-level
calibration.

## Modules

| Module | Responsibility |
|---|---|
| [`estimator.py`](estimator.py) | Converts the MAT data once, invokes `prime_fie`, and returns typed trajectories and dynamics |
| [`sensitivity.py`](sensitivity.py) | Computes the Gauss--Newton KKT adjoint and parameter gradient |
| [`calibration.py`](calibration.py) | Defines the upper loss and SQP, Frank--Wolfe, and projected-Adam updates |
| [`__init__.py`](__init__.py) | Package boundary |

The public executable is the parent
[`run_calibration.py`](../run_calibration.py); native estimator details remain
behind `PrimeEstimator`.

Return to the [STRIDE implementation](../README.md).
