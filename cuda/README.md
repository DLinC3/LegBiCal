# CUDA implementation

A compact Torch package for covariance calibration of a contact-aided
invariant EKF. Fixed contact slots permit batched CPU execution, compiled CUDA
execution, and optional CUDA Graph replay without data-dependent Python
control flow.

## Contents

| Path | Responsibility |
|---|---|
| [`src/estimation_calibration_cuda/`](src/estimation_calibration_cuda/) | Public API, CLI, dataset validation, InEKF replay, and calibration |
| [`tests/`](tests/) | Numeric, gradient, data-contract, and release-surface tests |
| [`benchmarks/`](benchmarks/) | Replay profiling entry point |
| [`notebooks/`](notebooks/) | Calibration tutorial and CUDA benchmark |

## Install and run

Python 3.10--3.14 and Torch 2.11 or newer are required. Install the Torch build
for the target CPU or CUDA environment first.

```bash
cd cuda
python -m pip install .
estimation-calibration-cuda train example -o run \
  --device cpu --compile none --epochs 2 --chunk 32
estimation-calibration-cuda evaluate example --checkpoint run/checkpoint.pt \
  --device cpu
estimation-calibration-cuda inspect run
```

The packaged `example` dataset runs without external files. A run contains
`checkpoint.pt`, `covariances.npz`, `metrics.json`, and `manifest.json`.
Training reads the train split, validation selects the covariance state, and
the explicit `evaluate` command opens the test split once.

## Python API

```python
from estimation_calibration_cuda import CalibrationConfig, calibrate, load_dataset

result = calibrate(
    load_dataset("example"),
    CalibrationConfig(device="cpu", compile_mode="none", epochs=2, chunk=32),
    output_dir="run",
)
```

Custom datasets use `dataset_manifest.json` plus one NPZ per episode. The
manifest records each episode's name, split, source identity, file, and SHA-256
hash; array shapes and split lineage are validated before replay.

## Test

```bash
python -m pip install -e '.[dev]'
pytest
```

See the [tutorial notebook](notebooks/covariance_tuning_tutorial.ipynb) for a
small walkthrough or return to the [repository overview](../README.md).
