# Python package source

Packaging boundary for the installable `g1cal` library and command-line
interface. The parent [`pyproject.toml`](../pyproject.toml) maps this directory
as the setuptools package root.

## Contents

| Path | Responsibility |
|---|---|
| [`g1cal/`](g1cal/README.md) | Calibration, lower-solver orchestration, loss, rendering, and CLI modules |

The C++ build places `_g1cal_cpp` beside the package modules. Install from the
implementation root after building:

```bash
python -m pip install -e .
g1cal --help
```

Return to the [G1 implementation](../README.md).
