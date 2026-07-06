# GPU Execution Notes

## Core Point

The baseline was already on CUDA. It was slow because it was launch-bound:
dynamic state dimensions, Python control flow per row, tiny matrix kernels,
and hidden device-to-host syncs.

The fast path is:

1. static shapes with masked contact slots;
2. batched rollouts;
3. whole-chunk CUDA graph capture for fwd+bwd.

## Key Files

- `invariant_ekf.py`: dynamic reference implementation; keep as oracle.
- `fixed_slot_inekf.py`: static-shape batched estimator.
- `batched_calibration.py`: batched trainer and `ChunkGraph`.
- `profile_replay.py`: benchmark and trace harness.
- `tests/`: required correctness gates.

## Design Rules

- Prefer static shape plus masks over dynamic tensor resize.
- Precompute event schedules on the host as tensors.
- Use `step(carry, inputs[:, t], schedule[:, t], params) -> (carry, out)`.
- Keep the hot path tensor-only and batch-first.
- Avoid `.item()`, `float(tensor)`, `.cpu()`, and checked linear algebra in the
  training loop.
- Use `cholesky_ex(check_errors=False)` plus `cholesky_solve` when the failure
  count can be accumulated on device.
- Keep filter linear algebra in float64 unless a specific replacement is
  proven stable.

## CUDA Graph Rules

- Warm up and capture on the same side stream.
- Do not keep parameter-referencing autograd graphs alive across capture.
- Use preallocated static buffers and `copy_` new chunk inputs into them.
- Keep non-capturable regularization outside the graph and add gradients
  explicitly.

## Quick Checklist

- `PYTHONPATH=src python -m pytest tests/` passes.
- Dynamic oracle parity and gradient parity stay within test gates.
- Padded rows remain exact no-ops.
- Chunked replay equals monolithic replay.
- Batched replay matches single-rollout replay within tolerance.
- Profiler trace does not regress launches/row, DtoH copies, or rows/s.
- Full calibration loss decreases and final covariance is finite,
  symmetric, and PSD.
