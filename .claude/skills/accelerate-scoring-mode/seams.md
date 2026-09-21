# Accel-layer seams

What a mode's fast backend touches. Steps 3–5 are skipped by most modes; step 6 by nearly all.

| Layer | File | What you add |
|---|---|---|
| CPU kernel *(new math only)* | `accel/kernels/cpu.py` | `@njit(parallel=True, fastmath=True, cache=True)` inner kernel + torch-facing wrapper, emitting value and `dO/dq`. Write and validate first. |
| GPU kernel *(new math only)* | `accel/kernels/<family>_triton.py` | Triton twin, **identical signature**, `@triton.autotune` + `tl.exp2`. |
| Dispatch *(new math only)* | `accel/kernels/dispatch.py` | One `_make(name, triton_tag)` line. Routes per call by tensor device; lazy Triton import. |
| Term evaluation *(new math only)* | `accel/drivers/terms.py` | One branch in `evaluate`, one in `self_overlap` if the reduction needs one. |
| Fused CPU loop *(new math only)* | `accel/kernels/cpu_fused.py` | One closure in `_term_closure`, so the mode keeps the fused CPU path. |
| Per-molecule data *(new data only)* | `accel/channels.py` | A `Channel` row (+ a `BASES` entry and a `SCHEMA_FLAGS` entry if the basis is new). |
| Store plumbing *(new data only)* | `shepherd_score/screen.py` | One `_BASIS_TABLE` row + the `MoleculeProfile` slots and `get_<data>()` accessors. |
| **The mode** | `accel/_modes.py` | One `ModeSpec`. |
| Batched API | `container/_batch.py` | `MoleculePairBatch.align_with_<mode>(backend=None, return_aligned=False)`. |
| Eager default switch | `container/_core.py` | Replace the reference method's literal seed/step defaults with `_default_seeds` / `_default_steps`. |
| Registry count | `tests/test_mode_registry.py` | Bump the literal 21 → 22. The only hardcoded mode fact left. |

## Directory map

```
accel/
  _modes.py        THE REGISTRY. Term / ModeSpec / SPECS (21), plus the flat tables derived from
                   them: MODE_ATTRS, CANONICAL_MODES, MODE_SEEDS, MODE_STEPS, PROCESS_MODES,
                   CONST_SEED_MODES, LEGACY_MODE_ALIASES, canonical(), spec_of(). Pure data.
  channels.py      THE DATA TABLE. Channel rows: reader, pair attr, store key, basis/offset
                   table, schema flag, pad value, dtype, rotate/translate. Pure data + numpy.
  _stats.py        opt-in fine-loop effort recorder: reset() / record() / summary() / disable().
                   Off by default; process-local, so an empty summary means UNMEASURED.
  cpu_pool.py      persistent single-threaded process pool for the numba backend.
  multi_gpu.py     process-per-GPU driver: align_multi_gpu, MultiGPUAligner.
  screen_parallel.py  fork-sharded CPU screening over an in-RAM Molecule list. Registry-driven.
  kernels/
    dispatch.py      the only kernel module the engine imports. 11 wrappers.
    shape_triton.py  Gaussian volume value+grad, fused Adam, self-overlap.
    esp_triton.py    ESP-weighted overlap value+grad, ShaEP comparison, ESP self-overlap.
    pharm_triton.py  pharmacophore kernels: directional dQ, directionless colour, legacy R-grad.
    avoid_triton.py  linear hard-sphere excluded-volume value+grad.
    vol_color_triton.py  FUSED shape+colour single kernel, small pads only. NOT dispatched;
                     reached through ModeSpec.fused_pair.
    cpu.py           numba twins of every dispatched kernel.
    cpu_fused.py     the fused CPU fine loop (no torch in the hot loop), generic over terms.
    cpu_soa.py       SoA + fp32 SVML-vectorized twins of two cpu.py inner kernels.
  drivers/
    engine.py    THE FINE LOOP. assemble -> seeds -> per-pose expansion -> graph | fused | eager
                 -> per-pair best. One body for every mode.
    terms.py     per-Term value+grad evaluation, self-overlaps, lookup tables.
    _common.py   quaternion/SE(3) math, seed generation, coarse grid, canonical seeds.
    _graphed.py  shared CUDA-graph capture/replay + LRU cache + graph_cap.
    _shim.py     helpers for the driver entry points.
    shape.py esp.py esp_combo.py pharm.py vol_color.py vol_lipo.py vol_tversky.py
    vol_esp_tversky.py vol_color_tversky.py vol_lipo_tversky.py vol_atomtype.py vol_pharm.py
    vol_and_surf_esp_tversky.py avoid.py
                 THIN ENTRY POINTS ONLY -- each keeps its historical coarse_fine_* /
                 fast_optimize_*_batch signature for callers that hand it padded tensors, and
                 forwards to the engine. No fine loop lives in them any more.
    pharm_overlap.py   NOT a driver: eager torch pharmacophore overlap helpers, used by the
                 extended_points path.
  batch/
    aligners.py   ONE generic _align_batch body; the 21 named functions are generated from SPECS.
    aligners_legacy.py  the one objective with no kernel: pharm extended_points.
    _arrays.py    ONE generic array-native screen aligner (align_arrays) + generated per-mode
                  names. ENABLED is a TEST SEAM, not a runtime switch.
    _bucket.py    PadSpec / Bucket / plan_buckets.
    _pad.py       _BAND, memory-aware sub-batching, _CPU_CHUNK_PAIRS, _FINE_CHUNK_POSES.
    _dispatch.py  multi-GPU gating + _MODE_SPEC, DERIVED from each spec's channels (21 entries).
```

## Which existing mode to read

Read the **spec**, not a driver — the driver no longer holds the mode's arithmetic.

| Your objective | ModeSpec to copy | Kernels it names |
|---|---|---|
| shape only | `vol` (or `surf`, which adds `multipose`) | `shape` |
| a scalar field alone | `vol_esp` / `surf_esp` | `esp` |
| pharmacophore alone | `pharm` (note `pharm_style`, `center_clouds`) | `pharm` |
| shape + pharmacophore colour | `vol_color` (note `fused_pair`) | `shape` + `color` |
| shape + directional pharmacophore | `vol_pharm` | `shape` + `pharm` |
| shape + any signed scalar field | `vol_lipo` / `vol_mr` / `vol_fukui` — identical but for the channel | `shape` + `esp` |
| shape + element identity | `vol_atomtype` | `shape` + `color` with `tables="element"` |
| shape + surface ESP (ShaEP) | `vol_and_surf_esp` (note `work="combo"`, `channel_switch`) | `shape` + `esp_cmp` |
| any Tversky reduction of a parent | the parent with `reduction="tversky"` on its terms | the parent's |
| a subtracted penalty | `vol_avoid` (negative weight, `raw` reduction) | `shape` + `avoid` |

`vol_lipo`, `vol_mr` and `vol_fukui` differ **only** in which channel pair their field term names.
Check whether that is true of your mode before adding anything.

## The registry invariant

`accel/_modes.py` is the single source of truth, and this skill is where a mode *becomes* canonical.
`tests/test_mode_registry.py` pins:

- `len(CANONICAL_MODES)` is a hardcoded `21` — bump it.
- `tuple(MODE_ATTRS) == CANONICAL_MODES`, and `set(MODE_SEEDS) == set(MODE_STEPS) ==
  set(CANONICAL_MODES)` — all three are derived from `SPECS`, so registering a spec does all three.
- Every canonical mode must have an `accel.batch._align_batch_<mode>`; the `@_bind_batch_aligners`
  walk runs at import. Generation from `SPECS` satisfies this automatically.
- `tuple(_MODE_SPEC) == PROCESS_MODES` — both derived from `SPECS`.
- `aligners._MODE_SEEDS is M.MODE_SEEDS` — object **identity**, so do not copy the table anywhere.
- `MoleculePair.__init__` must pre-init every `MODE_ATTRS` slot; that comes free from `_ALIGN_KEYS`.

## Hardcoded mode lists you must update

One:

| Location | Needed when |
|---|---|
| `tests/test_mode_registry.py` count | always |

Everything else derives from `SPECS` / `CHANNELS`: `_TRANSFORM_ATTR`, `_SCORE_ATTR`, `_VALID_MODES`,
`_FAST_MODES`, `_ARRAY_MODES`, `_ARRAY_BUILDERS`, `_ARRAY_ALIGNERS`, `_SURF_ALPHA_MODES`,
`_schema_from_modes`, `_store_supports`, `_query_ref_arrays`, `_ref_tensors_from_arrays`,
`_build_fit_fast_pairs`, `_build_fit_arrays`, `_fast_batch_kwargs`, `_FastPair.__slots__`,
`_seeds_for` / `_steps_for`, `_MODE_SPEC`, `screen_parallel._ALIGN_ATTR`, the `MoleculePair` binds,
and the parity tests' parametrization.

## Module-level seams that are not runtime switches

- `accel/batch/_arrays.py:ENABLED = True` — the array path is **on**. `ENABLED` survives only so
  the parity tests can flip it to force the object path for comparison.
- `kernels/cpu_fused._USE_SOA` — derived from whether numba has SVML, not a knob.

**The library reads no environment variable to change behaviour.** Every `os.environ` write in
`accel/` is a thread-count cap being *set* for a worker process, saved and restored. Tunables are
module-level constants you edit or monkey-patch: `_GRAPH_WORK_BUDGET`, `_GRAPH_CAP_CEIL`,
`_GRAPH_CAP_MIN`, `_GRAPH_ES_BLOCK`, `_GRAPH_ES_MARGIN`, `_GRAPH_CACHE_MAX` in `drivers/_graphed.py`;
`_FINE_CHUNK_POSES`, `_CPU_CHUNK_PAIRS`, `_BAND` in `batch/_pad.py`; `_ESP_STRIDE` in
`drivers/engine.py`; `VOL_COLOR_FUSED_MAX_PAD` in `kernels/vol_color_triton.py`. Per-mode tunables
(graph budget, pose cap, fused-CPU opt-out, multipose) live on the mode's `ModeSpec`.
