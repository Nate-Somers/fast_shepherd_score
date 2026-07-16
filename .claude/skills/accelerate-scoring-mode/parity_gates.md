# Parity gates

A mode is accelerated only when all four gates pass. Each gate isolates one link in the chain
reference → CPU kernel → GPU kernel → batched driver. Run them in order: a failure at gate N is
almost always a bug introduced at step N, not earlier.

## Gate 1 — numba kernel ≡ eager autograd reference
The numba kernel's value and gradient must match the reference `optimize_<mode>_overlay`
objective computed with autograd.
- **What to compare**: the scalar overlap and the full `dO/dq` vector, on the same inputs and the
  same pose, for several distinct molecule pairs (not just self-pairs).
- **Tolerance**: ~`1e-16`–`1e-17` in double precision. This is the tightest gate — the numba
  kernel and autograd are computing the *same math* two ways, so they should agree to machine
  epsilon. A larger gap means the kernel's analytic gradient is wrong, not "just fp noise".
- Run this before writing any Triton code.

## Gate 2 — Triton kernel ≡ numba kernel
The GPU twin must match the CPU kernel.
- **What to compare**: value and `dO/dq` from the Triton kernel vs the numba kernel on identical
  inputs.
- **Tolerance**: fp32 (`~1e-5`–`1e-6` relative). Triton runs fp32 on GPU; exact equality is not
  expected, but the two must agree to single precision.
- **Verify in-process.** Run both kernels in the same Python process and compare directly. Triton
  autotuning can pick different kernel configurations across processes, so a cached result from a
  separate run is not a reliable baseline — compare live, in one process.

## Gate 3 — batched driver ≡ per-pair reference
The batched coarse-to-fine driver must reproduce the per-pair result.
- **What to compare**: `MoleculePairBatch.align_with_<mode>()` scores vs looping
  `MoleculePair.align_with_<mode>()` over the same pairs, on distinct molecules.
- **Tolerance**: agree to ~4 decimals. Small differences from the multi-start schedule are
  acceptable; a systematic gap means the driver's padding, masking, or step schedule differs from
  the reference.
- This is where padding/masking bugs surface: pad slots must be masked by a real count, not left
  to contaminate the overlap.

## Gate 4 — self-copy = 1.000
A molecule aligned to a copy of itself scores exactly 1.000, on **both** backends.
- Run it under numba and under Triton. This is the cheap end-to-end smoke test that the whole
  path (kernel → driver → API) is wired correctly.

## Running them
The relevant suites already exist; add your mode's cases to them:
- `tests/test_numba_backend.py` — gates 1 and 4 (CPU).
- `tests/test_fast_batch_alignment.py` — gate 3, and gate 4 on GPU where available.
- Gate 2 needs a CUDA device; guard it with a `torch.cuda.is_available()` skip so the suite still
  runs on CPU-only machines (Triton parity is then checked wherever a GPU is present).

Do not weaken a tolerance to make a gate pass. If gate 1 fails at `1e-4`, the analytic gradient is
wrong — fix the kernel, not the threshold.
