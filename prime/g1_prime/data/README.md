# Released calibration data

Inputs and fixed outputs for the two 501-state G1 calibration components.

## Contents

| Path | Responsibility |
|---|---|
| [`calibrated/`](calibrated/) | Released covariance precision, parameter vector, and loss summary |
| [`clips/`](clips/) | run1/run2 measurements, truth slices, injection records, and strict reference solutions |

`calibrated/precision.csv` is the quickstart covariance. The covariance stored
in each clip's `injection.json` describes data generation and is not the
released estimator covariance.

Motion lineage, redistribution status, and requested citations are recorded in
[`clips/README.md`](clips/README.md). The local publication gate is represented
by [`clips/PUBLICATION_STATUS.json`](clips/PUBLICATION_STATUS.json).

Return to the [G1 implementation](../README.md).
