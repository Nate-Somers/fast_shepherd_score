# Parity gates

A mode is accelerated only when all four gates pass — plus gate 5 (the screen round-trip) if the
mode is meant to screen a library. Each gate isolates one link in the chain
reference → CPU kernel → GPU kernel → batched driver → out-of-core store. Run them in order: a
failure at gate N is almost always a bug introduced at step N, not earlier.

## Gate 1 — numba kernel ≡ eager autograd reference
The numba kernel's value and gradient must match the reference `optimize_<mode>_overlay`
objective computed with autograd.
- **What to compare**: the scalar overlap and the `dO/dq` vector, on the same inputs and the same
  pose, for several distinct molecule pairs (not just self-pairs).
- **Compare the gradient in the tangent space of the unit quaternion, not raw.** The kernels emit
  a **raw** `dO/dq` — they do not project out the component parallel to `q` — because the optimizer
  (`fused_adam_qt_with_tangent_proj`) renormalizes `q` each step and discards that radial part.
  Autograd taken through the repo's *normalizing* `quaternion→rotation` map is already
  tangent-projected. So a naive raw-vs-autograd comparison **fails at ~0.4–0.8 even for a correct
  kernel** — the entire mismatch is the radial component. Project both gradients onto the tangent
  space at `q` (subtract the `q`-parallel part) before comparing, or differentiate a non-normalizing
  map. The tangent-space component is the physical part that drives the step.
- **Tolerance**: ~`1e-16`–`1e-17` in double precision *on the tangent-space component*. This is the
  tightest gate — kernel and autograd compute the same math two ways. A larger tangent-space gap
  means the analytic gradient is wrong, not "just fp noise".
- Run this before writing any Triton code. If you **reused** existing kernels instead of writing a
  new one (step 1), this gate validates that the blend + scaling in your driver is correct.

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

## Gate 5 — streamed screen ≡ per-pair `MoleculePairBatch` *(only if the mode screens)*
The out-of-core `screen` path must reproduce the in-memory result. This is the screen analog of
gate 3, one level further out: it exercises the store's serialization (step 10) and, once the mode
is in `_FAST_MODES`, the resident-tensor fast engine.
- **What to compare**: `screen(query, ProfileStore, mode=<mode>)` scores vs
  `MoleculePairBatch.align_with_<mode>()` on the *same centered molecules* (deep-copy + `center_to`
  each library molecule to its own heavy-atom COM to match a `pre_centered` store).
- **This test runs the FAST path** when the mode is in `_FAST_MODES` and the store is
  `pre_centered` — which is what you want: it proves the resident-tensor path is score-faithful, not
  just the object path. (A score gap *larger* than the tolerance below, where before it was
  bit-identical, is the signature of the fast path engaging — confirm that, do not "fix" it by
  reverting to the object path.)
- **Tolerance**: a single-channel or reduction mode is typically bit-identical (`abs=1e-4`) because
  the fast and object paths feed identical inputs to one driver. A basin-sensitive multi-channel
  mode can differ by ~`1e-3`: the fast and object query arrays differ by fp noise (~`2e-7`, from
  re-centering the query independently in the reference), and a rugged multi-start objective can let
  that flip which seed wins a near-tie. Use `abs=1e-2` there (the accel test's tolerance for the
  same mode). Do not loosen further — a real wiring bug moves scores by `>>1e-2`.
- **Also assert the data reached disk** for a Tier-B mode: load a `shard_*.npz` and check the new
  arrays (and the offset table for a variable-length set) are present. A mode that silently stores
  nothing still "passes" a score comparison when every molecule's data is empty — the disk check
  catches that, and it is the check that survives a loose score tolerance.

## Running them
The relevant suites already exist; add your mode's cases to them:
- `tests/test_numba_backend.py` — gates 1 and 4 (CPU).
- `tests/test_fast_batch_alignment.py` — gate 3, and gate 4 on GPU where available.
- `tests/test_screen.py` — gate 5 (models: `test_vol_tversky_stream_matches_object`,
  `test_esp_field_stream_matches_object`).
- Gate 2 needs a CUDA device; guard it with a `torch.cuda.is_available()` skip so the suite still
  runs on CPU-only machines (Triton parity is then checked wherever a GPU is present).

Do not weaken a tolerance to make a gate pass. If gate 1 fails at `1e-4`, the analytic gradient is
wrong — fix the kernel, not the threshold.
