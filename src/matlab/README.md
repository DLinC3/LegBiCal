# MATLAB Estimation

Planar full-information estimation and Frank-Wolfe calibration code. This
folder does not include the STRIDE simulator or controller.

## Files

- `FIE.m`: estimator
- `FIECalibrator.m`: calibration loop
- `calibrationOptions.m`: default theta, bounds, and solver settings
- `runCalibration.m`: main entry point
- `plotFIE.m`: plotting helper
- `main.m`, `estimation_FIE.m`: compatibility wrappers

## Usage

```matlab
options = calibrationOptions();
result = runCalibration(inputData, options);
```

`inputData` can be a MAT file or struct with either:

- `log.flow.q`, `log.flow.dq`, `log.flow.ddq`, `log.estimate.contact`, `log.estimate.t`, `log.groundtruth.x`
- `data.q`, `data.dq`, `data.ddq`, `data.contact`, `data.dt`, `xGroundTruth`

Contacts use `-1` for left foot, `0` for double support, and `1` for right foot.
State trajectories are `8-by-K`; `q`, `dq`, and `ddq` are `7-by-K`.

## Kinematics

By default, `FIE` expects `pLeftToe_d`, `pRightToe_d`, `J_leftToe_d`, and
`J_rightToe_d` on the MATLAB path. You can instead pass equivalent function
handles through the `model` argument of `FIE`.

## Dependencies

MATLAB, CasADi with IPOPT, YALMIP, and MOSEK or another YALMIP-compatible SDP
solver. Add external dependencies to the MATLAB path before running.
