# Python implementation

This hardware-oriented B1 implementation uses a stage-structured Fatrop FIE,
a sparse adjoint, and a semidefinite Frank--Wolfe oracle to calibrate
covariance and kinematic parameters.

## Contents

| Path | Responsibility |
|---|---|
| [`bilevel/`](bilevel/) | Estimator, sensitivity, calibration, robot, data, and generated-code modules |
| [`tests/`](tests/) | Optimization and integration tests |
| [`tools/`](tools/) | Maintainer utility for regenerating portable kinematic C code |

The repository-root [`pyproject.toml`](../pyproject.toml) owns this package and
the `estimation-calibration` command. Data, URDF, and portable generated C
sources are packaged under `bilevel/resources/`; native kinematic functions
are compiled once into the user cache.

## Run

```bash
conda create -n legbical -c conda-forge python=3.12 pinocchio casadi
conda activate legbical
python -m pip install -e '.[dev]'
estimation-calibration --horizon 3000 --iterations 75
```

CasADi must provide the Fatrop plugin. CLARABEL is the default open-source
linear minimization oracle; select another installed CVXPY solver with
`--lmo-solver`.

## Test

```bash
pytest python/tests
```

Return to the [repository overview](../README.md).
