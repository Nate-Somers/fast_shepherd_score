# Accel-layer seams

The files a mode's fast backend touches. All additive except the two paired registry edits in
step 7, which are optional.

| Layer | File | What you add |
|---|---|---|
| CPU kernel | `accel/kernels/cpu.py` (or `cpu_fused.py` / `cpu_soa.py`) | numba value+grad kernel emitting `dO/dq` in-register. Write and validate this first. |
| GPU kernel | `accel/kernels/<family>_triton.py` | Triton twin, **identical signature**, `tl.exp2` + `@triton.autotune`. |
| Dispatch | `accel/kernels/dispatch.py` | Thin wrapper routing a call to Triton (CUDA) or numba (CPU) by tensor device, per call; lazy Triton import. |
| Driver | `accel/drivers/<mode>.py` | Batched coarse-to-fine host loop; reuses `drivers/_graphed.py` (CUDA-graph fine loop) and `drivers/_common.py`. |
| Batched aligner | `accel/batch/aligners.py` | `_align_batch_<mode>(pairs, ...)`: pad → driver → write `transform_<mode>`/`sim_aligned_<mode>` back. Bind on `MoleculePair`. |
| Aligner export | `accel/batch/__init__.py` | Export `_align_batch_<mode>`. |
| Canonical promotion | `accel/_modes.py` **and** `tests/test_mode_registry.py` | Add rows to `MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`, and bump the hardcoded `len(CANONICAL_MODES)` count. Safe only after the aligner exists (step 6). |
| Multi-GPU/pool (optional) | `accel/batch/_dispatch.py` **and** `accel/_modes.py` | Paired: a `_MODE_SPEC` entry **and** adding the mode to `PROCESS_MODES`. Never one without the other. |
| Batched API | `container/_batch.py` | `MoleculePairBatch.align_with_<mode>(backend=None)`, device-aware resolve. |
| Tests | `tests/test_<mode>.py`, `tests/test_fast_batch_alignment.py`, `tests/test_numba_backend.py` | The four parity gates. |

## The registry invariant

`accel/_modes.py` is the single source of truth for canonical (screening) modes, and this skill is
where a mode *becomes* canonical. The `design-scoring-mode` skill registered the mode's result
slots in `_ALIGN_KEYS` (`container/_core.py`) but deliberately left `accel/_modes.py` untouched —
promoting it here is what makes `screen` / `multi_gpu` / `cpu_pool` pick it up.
`tests/test_mode_registry.py` pins the invariants; keep them green:

- `len(CANONICAL_MODES)` is a hardcoded count — bump it when you add your mode.
- `tuple(MODE_ATTRS) == CANONICAL_MODES`, and `set(MODE_SEEDS) == set(MODE_STEPS) ==
  set(CANONICAL_MODES)` — so `MODE_ATTRS`, `MODE_SEEDS`, and `MODE_STEPS` must all gain your mode
  together, never one alone.
- Every canonical mode must have an `accel.batch._align_batch_<mode>` (the `@_bind_batch_aligners`
  walk on `MoleculePair`, run at import). That is why promotion happens *after* you build the
  aligner, not in the reference skill.
- `PROCESS_MODES` (multi-GPU / CPU-pool) must equal `tuple(_MODE_SPEC)` in
  `accel/batch/_dispatch.py` — edit both or neither.

## Derive, do not hardcode

The whole point of the registry is that the front-end picks up a mode automatically:
`screen.py`, `accel/multi_gpu.py`, and `accel/cpu_pool.py` build their mode tables from `accel/_modes.py`
and the `_align_batch_*` function names. If your mode is registered and its aligner is bound and
exported, those front-ends already support it. If you feel the need to edit them, you have
diverged from the registry-driven design — find the derivation you missed instead.

## Which existing mode to read

Model your kernel + driver on the nearest existing family:

- shape-only → `accel/kernels/shape_triton.py` + `accel/drivers/shape.py`
- ESP → `accel/kernels/esp_triton.py` + `accel/drivers/esp.py` / `esp_combo.py`
- pharmacophore / combined → `accel/kernels/pharm_triton.py` (and its combined driver)

The numba twins for all of these live in `accel/kernels/cpu.py` (+ `cpu_fused.py`, `cpu_soa.py`).
Read the twin alongside the Triton kernel to see the identical-signature contract in practice.
