# Accel-layer seams

The files a mode's fast backend touches. Steps 3–5 are skipped by most modes; step 10 is optional.

| Layer | File | What you add |
|---|---|---|
| CPU kernel *(new math only)* | `accel/kernels/cpu.py` | `@njit(parallel=True, fastmath=True, cache=True)` inner kernel + torch-facing wrapper, emitting value and `dO/dq`. Write and validate first. |
| GPU kernel *(new math only)* | `accel/kernels/<family>_triton.py` | Triton twin, **identical signature**, `@triton.autotune` + `tl.exp2`. |
| Dispatch *(new math only)* | `accel/kernels/dispatch.py` | One `_make(name, triton_tag)` line. Routes per call by tensor device; lazy Triton import. |
| Driver | `accel/drivers/<mode>.py` | Batched coarse-to-fine host loop; reuses `drivers/_graphed.py` and `drivers/_common.py`. Often a parameterisation of an existing driver instead of a new file. |
| Batched aligner | `accel/batch/aligners.py` | `_align_batch_<mode>(pairs, ...)`: upload → bucket/pad → driver → write results back. |
| Aligner export | `accel/batch/__init__.py` | One explicit re-export line. |
| Canonical promotion | `accel/_modes.py` **and** `tests/test_mode_registry.py` | Rows in `MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`; bump the count 21 → 22. Safe only after the aligner exists. |
| Worker path *(optional)* | `accel/batch/_dispatch.py` **and** `accel/_modes.py` | Paired: a `_MODE_SPEC` entry **and** `PROCESS_MODES`. Never one without the other. |
| Batched API | `container/_batch.py` | `MoleculePairBatch.align_with_<mode>(backend=None, return_aligned=False)`. |
| Eager default switch | `container/_core.py` | Replace the reference method's literal seed/step defaults with `_default_seeds` / `_default_steps`. |
| **Screen store** | `shepherd_score/screen.py` | Tier A: one `_store_supports` line. Tier B: schema flag + `MoleculeProfile` slots/`__init__`/`center_to`/accessors + `_profile_from_schema` (extract **+ pre-center + canonical rotate**) + `_concat` + `_reconstruct`. |
| **Screen array path** | `screen.py` + `accel/batch/_arrays.py` | `_ARRAY_MODES`, `_build_fit_arrays_<mode>`, `_align_fast_arrays_<mode>`, `_ARRAY_BUILDERS`, `_ARRAY_ALIGNERS`, and `align_batch_<mode>_arrays`. Required before reporting screening throughput. |
| Tests | `tests/test_<mode>_accel.py` or `tests/test_new_modes_accel.py`, `tests/test_screen.py`, `tests/test_mode_registry.py` | The parity gates + the screen round-trip. |

## Directory map

```
accel/
  _modes.py        registry: MODE_ATTRS (21), CANONICAL_MODES, LEGACY_MODE_ALIASES,
                   canonical(), PROCESS_MODES (4), MODE_SEEDS, MODE_STEPS.  Pure data.
  _stats.py        opt-in fine-loop effort recorder: reset() / record() / summary() / disable().
                   Off by default; process-local, so an empty summary means UNMEASURED.
  cpu_pool.py      persistent single-threaded process pool for the numba backend.
  multi_gpu.py     process-per-GPU driver: align_multi_gpu, MultiGPUAligner.
  screen_parallel.py  fork-sharded CPU screening over an in-RAM Molecule list. Registry-driven.
  kernels/
    dispatch.py      the only kernel module drivers import. 11 wrappers.
    shape_triton.py  Gaussian volume value+grad, fused Adam, self-overlap.
    esp_triton.py    ESP-weighted overlap value+grad, ShaEP comparison, ESP self-overlap.
    pharm_triton.py  pharmacophore kernels: directional dQ, directionless colour, legacy R-grad.
    avoid_triton.py  linear hard-sphere excluded-volume value+grad.
    vol_color_triton.py  FUSED shape+colour single kernel, small pads only. NOT dispatched.
    cpu.py           numba twins of every dispatched kernel.
    cpu_fused.py     first-class fused CPU fine loop (no torch in the hot loop).
    cpu_soa.py       SoA + fp32 SVML-vectorized twins of two cpu.py inner kernels.
  drivers/
    _common.py   quaternion/SE(3) math, seed generation, coarse grid, canonical seeds.
    _graphed.py  shared CUDA-graph fine loop + LRU cache + graph_cap.
    shape.py esp.py esp_combo.py pharm.py vol_color.py vol_tversky.py vol_esp_tversky.py
    vol_lipo.py vol_lipo_tversky.py vol_color_tversky.py vol_atomtype.py vol_pharm.py
    vol_and_surf_esp_tversky.py avoid.py            (14 drivers for 21 modes)
    pharm_overlap.py   NOT a driver: eager torch pharmacophore overlap helpers.
  batch/
    aligners.py   the 21 _align_batch_<mode> functions + _batch_upload.
    _arrays.py    array-native screen aligners: align_batch_<mode>_arrays. ENABLED is a TEST SEAM.
    _bucket.py    PadSpec / Bucket / plan_buckets.
    _pad.py       _BAND, memory-aware sub-batching, _CPU_CHUNK_PAIRS, _FINE_CHUNK_POSES.
    _dispatch.py  multi-GPU gating + _MODE_SPEC (4 entries).
```

## Which existing mode to read

For a single-channel mode, kernel and driver come from the same family:

- shape only → `kernels/shape_triton.py` + `drivers/shape.py`
- ESP field alone → `kernels/esp_triton.py` + `drivers/esp.py`
- pharmacophore alone → `kernels/pharm_triton.py` + `drivers/pharm.py`

For a blended mode, driver and kernels come from different places:

| Your objective | Driver to copy | Kernels to reuse |
|---|---|---|
| shape + pharmacophore colour | `drivers/vol_color.py` | shape + `pharm_color_score_grad_se3_batch` |
| shape + directional pharmacophore | `drivers/vol_pharm.py` | shape + `pharm_grad_dq_se3_batch` |
| shape + surface ESP (ShaEP style) | `drivers/esp_combo.py` | shape + `esp_comparison_batch` |
| **shape + any signed scalar field over atoms** | `drivers/vol_lipo.py` — probably unchanged | shape + **ESP kernel**, field fed as its `charges` |
| any Tversky reduction of a parent | `drivers/vol_tversky.py` / `vol_esp_tversky.py` | the parent's kernel; the reduction is host-side |
| a genuinely new penalty term | `drivers/avoid.py` | a new kernel pair — the only recent case |

`drivers/vol_lipo.py` serves `vol_lipo`, `vol_mr` and `vol_fukui` with no per-mode code; the
aligners differ only in which array they pass. Check whether that is true of your mode before
adding a file.

## The registry invariant

`accel/_modes.py` is the single source of truth for canonical modes, and this skill is where a mode
*becomes* canonical. `tests/test_mode_registry.py` pins the invariants:

- `len(CANONICAL_MODES)` is a hardcoded `21` — bump it.
- `tuple(MODE_ATTRS) == CANONICAL_MODES`, and `set(MODE_SEEDS) == set(MODE_STEPS) ==
  set(CANONICAL_MODES)` — all three tables gain your mode together.
- Every canonical mode must have an `accel.batch._align_batch_<mode>`; the `@_bind_batch_aligners`
  walk runs at import, so a missing aligner is an `AttributeError` on
  `import shepherd_score.container`.
- `tuple(_MODE_SPEC) == PROCESS_MODES` — edit both or neither.
- `aligners._MODE_SEEDS is M.MODE_SEEDS` — object **identity**, so do not copy the table anywhere.
- `MoleculePair.__init__` must pre-init every `MODE_ATTRS` slot; that comes free from `_ALIGN_KEYS`.

## Hardcoded mode lists you must update

Routing derives from the registry, but these lists do not and will not:

| Location | Needed when |
|---|---|
| `tests/test_mode_registry.py` count | always |
| `screen.py` `_FAST_MODES` | the mode screens at all (it gates `fast`, which gates the array path) |
| `screen.py` `_ARRAY_MODES` + `_ARRAY_BUILDERS` + `_ARRAY_ALIGNERS` | the mode takes the array path |
| `screen.py` `_build_fit_fast_pairs` + `_fast_batch_kwargs` + `_query_ref_arrays` + `_ref_tensors_from_arrays` | the mode screens at all — each ends in `raise ValueError(mode)`, and gate 5's reference leg runs the first of them |
| `screen.py` `_SURF_ALPHA_MODES` | the mode reads surface points |
| `tests/test_new_modes_accel.py` `MODES` | you use the shared accel harness rather than a per-mode file |
| `tests/test_si_modes.py` dicts | you use the shared reference harness |
| `tests/test_screen.py` per-test `modes=(...)` tuples | you add a screen round-trip test |

Everything else — `_TRANSFORM_ATTR`, `_SCORE_ATTR`, `_VALID_MODES`, `_seeds_for` / `_steps_for`,
`_align_fast`'s `getattr`, `screen_parallel._ALIGN_ATTR`, the `MoleculePair` binds — derives
automatically.

## Two module-level seams that are not runtime switches

- `accel/batch/_arrays.py:ENABLED = True` — the array path is **on**. `ENABLED` survives only so
  the parity tests can flip it to force the object path for comparison. Nothing reads the
  environment for it.
- `kernels/cpu_fused._USE_SOA` — derived from whether numba has SVML, not a knob.

**The library reads no environment variable to change behaviour.** Every `os.environ` write in
`accel/` is a thread-count cap being *set* for a worker process, saved and restored. Tunables are
module-level constants you edit or monkey-patch: `_GRAPH_WORK_BUDGET`, `_GRAPH_CAP_CEIL`,
`_GRAPH_CAP_MIN`, `_GRAPH_ES_BLOCK`, `_GRAPH_ES_MARGIN`, `_GRAPH_CACHE_MAX` in `drivers/_graphed.py`;
`_FINE_CHUNK_POSES`, `_CPU_CHUNK_PAIRS`, `_BAND` in `batch/_pad.py`; `_ESP_STRIDE` in
`drivers/esp_combo.py`; `VOL_COLOR_FUSED_MAX_PAD` in `kernels/vol_color_triton.py`.
