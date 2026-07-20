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
| **Screen store (step 10)** | `shepherd_score/screen.py` (+ `container/_core.py` if data is cached on `Molecule`) | **Tier A** (reuses stored data, e.g. `vol_tversky`): one `_store_supports` branch. **Tier B** (new per-mol data, e.g. `vol_lipo`): `MoleculeProfile` slot/`__init__`/`center_to`/`get_<data>()` accessor + `_schema_from_modes` flag + `_store_supports` + `_profile_from_schema` + `ProfileStore._concat`/`_reconstruct` (offset table for variable-length). Correct on the `fast=False` object path. |
| **Screen fast engine (step 10)** | `shepherd_score/screen.py` | Needed **before you report screening throughput** (the object path's per-shard `MoleculePair` rebuild understates aligns/s vs the resident-tensor fast modes; identical GPU kernel). Six edits: `_FAST_MODES` + `_FastPair.__slots__` (result + new tensor slots) + `_query_ref_arrays` + `_ref_tensors_from_arrays` + `_build_fit_fast_pairs` + `_fast_batch_kwargs`. Pre-set every tensor the driver's `_batch_upload` would fill so it skips the `.ref_molec`/`.fit_molec` lambdas. `vol_tversky` = drop-in copy of `vol`; `vol_lipo` adds four lipophilicity tensors. |
| Tests | `tests/test_<mode>.py`, `tests/test_fast_batch_alignment.py`, `tests/test_numba_backend.py`, `tests/test_screen.py` | The four parity gates **+ the screen round-trip** (`test_{vol_tversky,vol_lipo}_stream_matches_object`). |

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

## Derive routing; add data plumbing

Two different things flow through the front-end, and they are governed by opposite rules:

- **Routing is derived — never hardcode it.** `screen.py`, `accel/multi_gpu.py`, and
  `accel/cpu_pool.py` build their mode *tables* (which modes exist, which result attributes, which
  `_align_batch_*` to call) from `accel/_modes.py`. Register the mode + bind + export its aligner
  and that routing lights up automatically. If you catch yourself writing a mode-name list to
  decide dispatch, find the derivation you missed.
- **Per-molecule DATA is not derivable — you must add it (step 10).** The out-of-core store
  (`MoleculeProfile` + `ProfileStore` in `screen.py`) only carries the arrays it was taught to
  carry. The registry knows your mode's *name*, not that it reads, say, ESP field points. So a mode
  needing data beyond what's already stored *requires* a `screen.py` edit — that is by design, not a
  divergence. `multi_gpu` / `cpu_pool` inherit the store, so wiring `screen.py` covers them too
  (unless you also opt into the step-8 multi-GPU spec).

The tell for which you're doing: editing a **mode-name list** = wrong (derive it); adding a
**`_store_supports` branch or a `MoleculeProfile`/serialization field** = right (data plumbing).

## Which existing mode to read

For a *single-channel* mode, kernel and driver come from the same family:

- shape-only → `accel/kernels/shape_triton.py` + `accel/drivers/shape.py`
- ESP field alone → `accel/kernels/esp_triton.py` + `accel/drivers/esp.py`
- pharmacophore alone → `accel/kernels/pharm_triton.py` + its driver

For a *blended* mode, pick the **driver** and the **kernel(s)** from different modes (see SKILL.md
step 1) — they rarely coincide:

| Your objective | Driver to copy | Kernel(s) to reuse |
|---|---|---|
| shape + pharmacophore-color | `vol_color` driver (two-channel combined gradient) | shape kernel + pharm color kernel |
| shape + ESP field | `esp_combo` / `vol_and_surf_esp` driver | shape kernel + ESP kernel |
| **shape + any signed scalar field over atoms** (e.g. lipophilicity) | `vol_color` driver | shape kernel + **ESP kernel** (feed the scalar as its `charges`) |

The blended driver blends the per-channel `dQ`s that the reused kernels already emit — no new
kernel. The numba twins for all kernels live in `accel/kernels/cpu.py` (+ `cpu_fused.py`,
`cpu_soa.py`); read the twin alongside the Triton kernel to see the identical-signature contract.
