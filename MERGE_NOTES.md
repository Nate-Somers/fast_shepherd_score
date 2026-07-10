# Merge notes — unifying `fast_shepherd_score` into `shepherd-score`

Status of preparing this fork to be merged into the canonical `shepherd-score` repo as a
single unified codebase. **The fork is canonical**: its renamed mode names win; the original's
old names are kept working as legacy aliases. This document is the verified record of *what
changed vs. upstream and why*, so the eventual merge (and review) is mechanical.

Verified against source on 2026-06-29 (not from prior docs, which lag the code).

## 1. The fork is almost entirely additive

Comparing `fast_shepherd_score/shepherd_score/` to `shepherd-score/shepherd_score/` at the
git-blob level (both repos use `.gitattributes` `* text=auto`, so blobs are already LF —
working-tree CRLF on Windows is just `core.autocrlf` and is not a real diff):

- **Zero files deleted** from upstream.
- **New, self-contained additions** (drop in as-is): the entire `accel/` package
  (`kernels/` Triton+numba, `drivers/` per-mode SE(3) optimizers, `batch/` orchestration,
  `cpu_pool.py`, `multi_gpu.py`), plus top-level `screen.py` and `surface_diagnostics.py`.
- **Only 15 shared files have real content changes.** Of those, 5 are pure-additive and 4
  more are additive behind default-off flags.

## 2. Deliberate behavior changes (the only things a reviewer must consciously accept)

1. **Mode rename + legacy aliases.** `esp → surf_esp`, `esp_combo → vol_and_surf_esp`.
   Canonical names are the new ones; old names still work everywhere via aliases:
   - `MoleculePair.align_with_esp` / `align_with_esp_combo` (method aliases, `container/_core.py`)
   - per-pair attribute aliases `sim_aligned_esp`/`transform_esp`/`..._esp_combo` (properties)
   - `accel/batch/aligners.py` `_align_batch_esp`/`_align_batch_esp_combo`
   - `accel/batch/__init__.py` re-exports the legacy names
   - `cpu_pool.align_pairs` and `multi_gpu` normalize via `_LEGACY_MODE_ALIASES`
     (`{"esp":"surf_esp","esp_combo":"vol_and_surf_esp"}`) before any `_MODE_SPEC` lookup.
   Note: internal error messages (e.g. `screen.py`'s alpha guard) report the **canonical**
   name even when a legacy name was passed — by design.

2. **`max_num_steps` default `200 → None`** in `container/_batch.py` `align_with_*`.
   `None` resolves via `_default_steps(mode)` (≈50). **Adopted intentionally** (the measured
   steps≈50 knee). This lowers the default fine-step count vs. upstream — the one default
   *numerics* change in the merge.

3. **`se3.apply_SE3_transform` R==1 collapse.** Reimplemented with a fused `torch.baddbmm`
   (mathematically `pts @ R.T + t`), and a `(1,N,3)+(1,4,4)` singleton batch now returns
   `(N,3)` instead of `(1,N,3)`. The accel/autograd/analytical optimizers rely on this so
   `num_repeats==1` batched calls agree with the unbatched path. Pinned by
   `tests/test_alignment_utils.py::TestSE3::test_apply_se3_transform_torch_R1_collapses_to_single`.

## 3. The 3 in-place "hotspots" — already isolated, now tested

These were the only existing function bodies rewritten in place (everything else is appended
or default-off). All three are already cleanly guarded so the original default path is intact:

- **`alignment/utils/se3.py:apply_SE3_transform`** — see §2.3. Docstring + contract test added.
- **`container/_core.py`** `align_with_surf_esp` / `align_with_vol_and_surf_esp` — the new
  `use_fast` path is fully behind `if use_fast and torch.cuda.is_available():` with a
  `try/except ImportError` fallthrough to the untouched original optimizer directly below.
  No restructure needed (it is already an additive insert, not an edit to the original path).
- **`generate_point_cloud.py`** — `import open3d` made lazy (`_LazyOpen3D`, avoids the slow
  fork-hostile import); `get_molecular_surface` body wrapped in a `method` dispatch with
  `method='mesh'` the default (original path preserved bit-for-bit; `smooth_sdf` is opt-in).
  Covered by `tests/test_smooth_surface.py` (`test_default_is_mesh`,
  `test_default_mesh_still_uses_open3d`).

The other modified files are additive-behind-default-off-flags (pharmacophore scoring
torch/np `directional`/`color_weight`; `pharm_utils/pharmacophore.py` `feature_set`/
`directionless`; `score/analytical_gradients/_torch.py` `directionless`), pure-additive
(`alignment/_torch.py` appends `objective_/optimize_vol_color_overlay`; `alignment/__init__.py`
+ `container/__init__.py` exports; `protonate.py` `from __future__`), or mechanical rename-only
(`evaluations/evaluate/evals.py`, `objective.py`, `_pipeline_eval_single.py`).

## 4. Verification status

- Full fork suite **green: 235 passed, 76 skipped, 0 failed** (CPU box: torch-cpu, numba,
  rdkit, scipy). The 76 skips are GPU/Triton, JAX, and Open3D-mesh paths not installed here.
- Fixed in this prep pass (were 5 pre-existing failures, all incomplete-rename test artifacts,
  **library code was already correct**): `tests/test_cpu_pool.py` reached into the private
  `_MODE_SPEC` dict with a legacy key (now normalizes like the library does);
  `tests/test_screen.py` had a stale assertion string (now name-agnostic).
- **Not runnable in this environment** (verify on a CUDA + JAX + Open3D box before merge):
  Triton kernels (need `triton>=3.6`), the jax backend, and the Open3D `mesh` surface path.

## 5. Remaining before/after merge (Phase 2 — systematization, in scope, not yet done)

- **Single mode registry.** A mode is currently declared across `_MODE_SEEDS`/`_MODE_STEPS`
  (`aligners.py`), `_MODE_SPEC` (`_dispatch.py`), `POOL_MODES` (`cpu_pool.py`),
  `_LEGACY_MODE_ALIASES` (×3 copies: `cpu_pool.py`, `multi_gpu.py`, and the container/aligner
  aliases), the container `align_with_*` methods, the `alignment/_torch.py` functions +
  `__init__` exports, and `screen.py` maps. Collapse to one declarative registry the rest
  reads from — this is what makes the two planned agent skills ("add a custom autodiff
  alignment", "port a mode to numba/triton") short and safe. The 3 duplicated
  `_LEGACY_MODE_ALIASES` dicts are the most obvious first consolidation.
- **Kernel-twin parity test** (introspect `dispatch._make` registrations; assert every Triton
  name has a numba twin in `cpu.py`).
- **Parametrized validation harness** for the 4 gates keyed on the registry.
- **Config dup:** `pytest.ini` and `pyproject.toml` both define pytest config; pytest warns it
  ignores the `pyproject.toml` one. Pick one during unification.

## 6. Gentle registry consolidation applied (post-Phase-2)

`container/_core.py` now derives two per-mode blocks from the registry instead of hand-listing
them (behavior-identical; suite green):
- `MoleculePair.__init__` result slots (`transform_<mode>`/`sim_aligned_<mode>` = eye/None) are a
  loop over `accel/_modes.MODE_ATTRS` + a 2-entry tuple for the legacy `no_H` variants
  (`transform_vol`, `transform_vol_esp`) that are not registry modes.
- The batched-aligner static-method binds are applied by an `@_bind_batch_aligners` class
  decorator (one per `CANONICAL_MODES` + `LEGACY_MODE_ALIASES`) instead of explicit lines.

Merge note: both regions were **already** fork-divergent — the static-method binds are fork-only
(the original `_core.py` has no `accel`/`_align_batch`/`_modes` references at all), and the
`__init__` slot block was already rewritten by the esp→surf_esp rename + `vol_color` addition +
property aliases. So this consolidation opens **no new** fork↔original conflict surface; it only
changes content within hunks that already diverge, and adds a `_core.py → accel/_modes` import
(inside the pre-existing `_core.py → accel.batch` coupling). Pinned by
`tests/test_mode_registry.py::test_moleculepair_batch_aligner_binds_are_registry_driven` and
`::test_moleculepair_init_result_slots_cover_registry`.
