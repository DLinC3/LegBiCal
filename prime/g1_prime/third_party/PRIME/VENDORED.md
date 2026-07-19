# Vendored PRIME subset

This directory is a vendored subset of
[PRIME](https://github.com/well-robotics/PRIME.git), pinned at upstream
commit `b848ceecd451f4786ce39dcefa59e96dbaa369ba` (branch `main`).

PRIME is a Crocoddyl fork with smoothed second-order-cone contact
extensions; the lower-level estimator in this repository is built directly
on its excellent work. License: BSD 3-Clause (`LICENSE`), with Crocoddyl's
original attribution preserved (`NOTICE.md`, `THIRD_PARTY_NOTICES.md`).

## What is vendored

Only what is needed to build `libcrocoddyl.so` and compile against it:

- `include/` — full public headers (crocoddyl core + `contact_id/`);
- `src/` — full library sources;
- `experiments/common/` — three headers consumed by the overlay
  (`contact_id_model.hpp`, `contact_id_outputs.hpp`,
  `contact_id_preprocess.hpp`);
- `CMakeLists.txt` and `cmake/` — build system (jrl-cmakemodules), with
  `cmake/doxygen/MathJax/` removed (documentation-rendering payload only);
- `LICENSE`, `NOTICE.md`, `AUTHORS.md`, `THIRD_PARTY_NOTICES.md`,
  `CITATION.cff`, `README.md`, `package.xml` — verbatim.

## What was removed (and why)

- `experiments/` datasets and runners (~335 MB) — experiment data;
  the overlay only needs the three `experiments/common` headers;
- `media/` (~23 MB), `doc/` — documentation assets;
- `bindings/`, `unittest/`, `benchmark/`, `examples/` — not required for
  the library build (their `add_subdirectory` calls are behind
  `BUILD_*` options that this repository's superbuild disables);
- `cmake/doxygen/MathJax/` (~24 MB) — only used when building docs.

## Build options used by the superbuild

```
-DBUILD_PYTHON_INTERFACE=OFF -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF
-DBUILD_TESTING=OFF -DBUILD_CONTACT_ID_EXPERIMENTS=OFF -DBUILD_WITH_IPOPT=OFF
-DBUILD_WITH_MULTITHREADS=ON -DBUILD_WITH_NTHREADS=8
```

## Upstream release-metadata note

Upstream `AUTHORS.md` and `CITATION.cff` contain literal
"TODO ... before release" placeholders left by the PRIME authors; they are
preserved verbatim here rather than filled in on their behalf.
