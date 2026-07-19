# MATLAB implementation

This implementation combines a stage-structured Fatrop full-information
estimator with an adjoint KKT gradient. SQP--BFGS, Frank--Wolfe, and projected
Adam are interchangeable upper-level updates for covariance and kinematic
calibration.

## Contents

| Path | Responsibility |
|---|---|
| [`+legbical/`](+legbical/) | MATLAB package containing calibration, configuration, and estimation code |
| [`assets/`](assets/) | Paper figures generated from the experiments |
| [`data/`](data/) | STRIDE example signals and precomputed kinematic quantities |
| [`tests/`](tests/) | Fast-FIE regression test |
| [`run_calibration.m`](run_calibration.m) | Calibration entry point |
| [`setup.m`](setup.m) | Local MATLAB path setup |

## Run

MATLAB, Optimization Toolbox, and a CasADi build with Fatrop are required. Set
`CASADI_MATLAB_PATH` when CasADi is not already on the MATLAB path.

```matlab
cd matlab
result = run_calibration(Method="sqp", Horizon="demo");
```

`Method` also accepts `"frank-wolfe"` and `"adam"`; `Horizon="full"` uses the
complete stored trajectory.

## Test

```matlab
addpath('tests');
test_fast_fie
```

Return to the [repository overview](../README.md).
