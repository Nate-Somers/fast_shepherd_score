# What's New

This document describes the accelerated-alignment update to `shepherd-score`. It covers the
complete code diff, the organization of every new file, the full public API, every new feature,
and — explicitly — the four places where existing behavior changes.

The update is **additive in structure**: no upstream file is deleted, and no upstream public
name is removed. It is **not entirely additive in behavior**: four changes are visible to existing
callers and are documented in [Behavior changes](#behavior-changes-read-this) below. Read that
section before upgrading.

---

## Contents

1. [The diff](#1-the-diff)
2. [Installation and dependencies](#2-installation-and-dependencies)
3. [Organization: what every file does](#3-organization-what-every-file-does)
4. [New features](#4-new-features)
5. [API reference](#5-api-reference)
6. [Behavior changes (read this)](#behavior-changes-read-this)
7. [Testing and validation](#7-testing-and-validation)
8. [Known gaps](#8-known-gaps)

---

## 1. The diff

Measured against `coleygroup/shepherd-score` at `75f1b6b`:

```
62 files changed, 14,180 insertions(+), 253 deletions(-)
```

(Excluding this document.)

Broken down: **38 new files**, **24 modified** (15 library source, 1 test, 8 config/docs/example),
**0 deleted**. The fork's history already contains all of upstream, so the merge applies without
conflicts.

### New files (38)

| File | Lines | Role |
|---|---:|---|
| **`shepherd_score/accel/`** — the acceleration subpackage | **9,458** | |
| `accel/__init__.py` | 32 | Public surface: `has_triton`, `align_multi_gpu`, `MultiGPUAligner` |
| `accel/_modes.py` | 61 | Mode registry. Pure data, no heavy imports |
| `accel/kernels/__init__.py` | 8 | — |
| `accel/kernels/dispatch.py` | 112 | Per-call device dispatch: CUDA tensor → Triton, CPU tensor → numba |
| `accel/kernels/shape_triton.py` | 448 | Gaussian shape overlap + analytic SE(3) gradient (`vol`, `surf`) |
| `accel/kernels/esp_triton.py` | 425 | Shape overlap with ESP charge weighting (`vol_esp`, `surf_esp`) |
| `accel/kernels/pharm_triton.py` | 467 | Directional pharmacophore overlap + gradient |
| `accel/kernels/vol_color_triton.py` | 209 | Fused shape + directionless-color kernel |
| `accel/kernels/cpu.py` | 624 | numba mirrors of every kernel above (the CPU math) |
| `accel/kernels/cpu_fused.py` | 338 | Torch-free fused fine loop; imports its math from `cpu.py` |
| `accel/kernels/cpu_soa.py` | 111 | Structure-of-arrays fp32 variant, used when numba can emit SVML |
| `accel/drivers/__init__.py` | 8 | — |
| `accel/drivers/_common.py` | 529 | Shared seeding, coarse grid, Adam/Tanimoto tails |
| `accel/drivers/_graphed.py` | 193 | Mode-agnostic CUDA-graph fine loop |
| `accel/drivers/shape.py` | 313 | Driver for `vol` and `surf` |
| `accel/drivers/esp.py` | 558 | Driver for `vol_esp` and `surf_esp` |
| `accel/drivers/esp_combo.py` | 807 | Driver for `vol_and_surf_esp` |
| `accel/drivers/pharm.py` | 770 | Driver for `pharm` |
| `accel/drivers/pharm_overlap.py` | 475 | Pharmacophore overlap support for the pharm driver |
| `accel/drivers/vol_color.py` | 516 | Driver for `vol_color` |
| `accel/batch/__init__.py` | 21 | Re-exports the batch surface |
| `accel/batch/aligners.py` | 1,146 | The seven `_align_batch_<mode>` functions `MoleculePairBatch` calls |
| `accel/batch/_bucket.py` | 277 | Adaptive size bucketer |
| `accel/batch/_pad.py` | 126 | Padding, GPU-memory sub-batching, scatter |
| `accel/batch/_dispatch.py` | 142 | Multi-GPU dispatch + the per-mode tensor spec (`_MODE_SPEC`) |
| `accel/cpu_pool.py` | 220 | Persistent single-threaded CPU worker pool |
| `accel/multi_gpu.py` | 424 | Process-per-GPU data parallelism |
| `accel/screen_parallel.py` | 98 | Fork-based shard-parallel CPU screening |
| **Top-level modules** | | |
| `shepherd_score/screen.py` | 1,282 | Virtual-screening front-end: `ProfileStore`, `screen`, `screen_many` |
| `shepherd_score/surface_diagnostics.py` | 142 | Leak / crimp metrics for validating a surface generator |
| **Tests** (7 new files) | **1,658** | |
| `tests/test_fast_batch_alignment.py` | 396 | Triton/CUDA batch aligners |
| `tests/test_screen.py` | 393 | Screening front-end |
| `tests/test_vol_color.py` | 275 | The `vol_color` mode |
| `tests/test_cpu_pool.py` | 168 | CPU process pool |
| `tests/test_smooth_surface.py` | 165 | Smooth-SDF surfacer + diagnostics |
| `tests/test_numba_backend.py` | 148 | numba CPU kernels |
| `tests/test_mode_registry.py` | 113 | Registry invariants (guards against drift) |
| **Packaging** | | |
| `environment-cpu-svml.yml` | 26 | Conda env for the SVML-vectorized CPU kernels |

### Modified files — library source (15)

| File | Change | Additive? |
|---|---:|---|
| `shepherd_score/container/_core.py` | +306 / −71 | Mostly. See [Behavior changes](#behavior-changes-read-this) |
| `shepherd_score/container/_batch.py` | +453 / −26 | Mostly. Same |
| `shepherd_score/generate_point_cloud.py` | +249 / −13 | Yes — lazy Open3D + the opt-in `smooth_sdf` surfacer |
| `shepherd_score/alignment/_torch.py` | +255 / −0 | Yes — one appended hunk, zero upstream lines touched |
| `shepherd_score/score/pharmacophore_scoring.py` | +98 / −4 | Yes — `directional=True` default preserves behavior |
| `shepherd_score/pharm_utils/pharmacophore.py` | +68 / −20 | Yes |
| `shepherd_score/score/pharmacophore_scoring_np.py` | +55 / −16 | Yes |
| `shepherd_score/alignment/utils/se3.py` | +28 / −14 | **No** — see the `R==1` shape change |
| `shepherd_score/score/analytical_gradients/_torch.py` | +17 / −6 | Yes |
| `shepherd_score/container/__init__.py` | +7 / −1 | Yes — two new exports |
| `shepherd_score/evaluations/evaluate/evals.py` | +6 / −6 | Internal rename only |
| `shepherd_score/objective.py` | +5 / −5 | Internal rename only |
| `shepherd_score/alignment/__init__.py` | +4 / −0 | Yes — two new exports |
| `shepherd_score/evaluations/evaluate/_pipeline_eval_single.py` | +2 / −2 | Internal rename only |
| `shepherd_score/protonation/protonate.py` | +2 / −0 | `from __future__ import annotations` |

Also modified: `tests/test_alignment_utils.py` (+12 — one appended test, no existing assertion
changed), `pyproject.toml` (+19/−1), `pytest.ini` (+1), `.gitignore` (+3), `README.md` (+9),
`environment.yml` (newline), `docs/usage.rst` and `docs/api/container/molecule_pair_batch.rst`
(rename), `examples/02_scoring.ipynb` (+9/−62, rename + re-run).

---

## 2. Installation and dependencies

The core dependency set is unchanged except for one fix. `torch` and `open3d` were already
required; the accelerated backends add **no new hard dependency**.

```bash
pip install shepherd-score            # unchanged: rdkit, torch, open3d, py3Dmol, numpy, pandas, scipy, molscrub, tqdm
pip install "shepherd-score[gpu]"     # + triton   -> backend="triton"
pip install "shepherd-score[cpu]"     # + numba    -> backend="numba"
pip install "shepherd-score[jax]"     # + jax      -> backend="jax" (the default; unchanged)
```

- **`tqdm` moved into the core dependency list.** This is a bugfix: upstream already imported
  `tqdm` at module scope in `conformer_generation.py`, `objective.py`, and both `pipelines.py`
  modules, but never declared it.
- **`numba` and `triton` are genuinely optional.** They are imported lazily, on first dispatch.
  The package imports cleanly with `numba`, `triton`, `jax`, and `open3d` **all absent** —
  which upstream could not do, because it imported Open3D eagerly.
- **`triton` ships manylinux wheels only.** If you add `gpu` to an `all`-style install on
  Windows or macOS, resolution will fail. Install it explicitly on Linux instead.
- **SVML.** The numba CPU kernels reach full speed only when numba can emit Intel SVML vector
  math. `environment-cpu-svml.yml` builds an environment that can. Without it the kernels run
  unvectorized and a one-time `RuntimeWarning` is emitted, so the slow regime is never silent.

---

## 3. Organization: what every file does

The acceleration subpackage is layered strictly bottom-up. Nothing in a lower layer imports
from a higher one.

```
MoleculePairBatch.align_with_<mode>(backend="triton")
        │
        ├── accel/batch/aligners.py        one _align_batch_<mode> per mode
        │      ├── accel/batch/_bucket.py  group same-size pairs into padded workspaces
        │      ├── accel/batch/_pad.py     pad, sub-batch to fit GPU memory, scatter results
        │      └── accel/batch/_dispatch.py  multi-GPU sharding + the per-mode tensor spec
        │
        ├── accel/drivers/<mode>.py        batched coarse-to-fine SE(3) optimizer
        │      ├── accel/drivers/_common.py   seeding, coarse grid, Adam/Tanimoto tails
        │      └── accel/drivers/_graphed.py  CUDA-graph fine loop (shared by all modes)
        │
        └── accel/kernels/dispatch.py      pick a kernel by tensor device
               ├── accel/kernels/*_triton.py   CUDA
               └── accel/kernels/cpu*.py       CPU (numba)
```

**`accel/_modes.py` is the single source of truth for the seven modes.** It is pure data with
no heavy imports, so every layer can import it freely. It defines the canonical mode names,
the legacy aliases, the result-attribute map, which modes have a worker-process path, and the
per-mode seed/step defaults. `tests/test_mode_registry.py` asserts that the registry and the
torch-typed `_MODE_SPEC` in `accel/batch/_dispatch.py` cannot drift apart.

To add a mode, register it in `_modes.py` and add a driver; the batch layer and the screen
front-end pick it up from the registry.

---

## 4. New features

### 4.1 Two new alignment modes

The library now has **seven** modes. Two are new:

- **`vol_and_surf_esp`** — combined volumetric shape + surface-ESP scoring. Available on
  `MoleculePairBatch` for the first time (upstream had a combo mode on `MoleculePair` only).
- **`vol_color`** — atom-centred Gaussian **shape** overlap plus a **directionless
  pharmacophore ("color")** overlap, the ROCS/ROSHAMBO-style scoring function. The SE(3) step
  descends on the weighted objective, so *both* channels steer the pose:

  ```
  score = (1 - color_weight) * shape_Tanimoto + color_weight * color_Tanimoto
  ```

  New on both `MoleculePair` and `MoleculePairBatch`, and available to the screening front-end.

### 4.2 A `backend=` argument on every batch aligner

`MoleculePairBatch.align_with_*` takes `backend="jax" | "triton" | "numba"`.

- `"jax"` is the **default** and runs the original, unchanged JAX code path.
- `"triton"` runs hand-written Triton GPU kernels.
- `"numba"` runs op-for-op numba CPU mirrors of the same kernels.

Kernel selection is per call, by tensor device, so one process can run both — a CPU batch and a
GPU batch in the same program each get the right kernel.

### 4.3 CUDA-graph fine loop

All seven modes share one CUDA-graph implementation (`accel/drivers/_graphed.py`). One fine
step is captured and replayed, removing per-step host launch overhead. It engages automatically
on CUDA float32 batches inside a work budget; larger batches and capture failures fall back to
the eager loop.

### 4.4 Adaptive bucketing

`accel/batch/_bucket.py` replaces the hand-rolled fixed-width size bands with one planner used
by every mode. It is **result-identical** by construction: the kernels are one-CTA-per-pose and
mask padding to the real point counts, and seeds are keyed on the real counts rather than the
pad width, so padding two different-sized molecules into the same bucket cannot change a score.

### 4.5 A virtual-screening front-end (`shepherd_score.screen`)

New top-level module. It featurizes a molecule library once into an on-disk **`ProfileStore`**
(sharded `.npz` + a manifest), then streams shards through the accelerated aligners against one
or many queries, keeping a top-K heap.

```python
from shepherd_score.screen import ProfileStore, screen

store = ProfileStore.create("lib.store", num_surf_points=200, modes=["surf_esp", "vol"])
for mol in library:
    store.add(mol)
store.close()

hits = screen(query_mol, ProfileStore.open("lib.store"), mode="surf_esp", top_k=1000)
```

`screen_many` streams the library **once** for a list of queries. Both accept `ndev=` to shard
across GPUs.

### 4.6 Multi-GPU data parallelism

Alignment is host-bound rather than kernel-bound, so driving N GPUs from one process serialises
on the GIL. The path that scales is one OS process per GPU, exposed explicitly:

```python
from shepherd_score.container import MultiGPUAligner, align_multi_gpu

scores = align_multi_gpu(pairs, "surf", ndev=4)          # one-shot
aligner = MultiGPUAligner(pairs, ndev=4)                  # persistent; reuses each GPU's shard
scores = aligner.align("surf")
```

This is deliberately **opt-in**. A library must not spawn worker processes behind the user's
back, because `spawn` re-imports the caller's `__main__` and breaks scripts without an
`if __name__ == "__main__":` guard. A large batch on a multi-GPU host therefore runs on a
single GPU and emits a one-time warning pointing at `MultiGPUAligner`.

### 4.7 Two CPU parallelism paths

- **`num_workers=N` with `backend="numba"`** uses a persistent pool of single-threaded worker
  processes (`accel/cpu_pool.py`), sharding *pairs* across processes. Pairs are independent, so
  this does not change the optimization problem. Agreement with one large call is to convergence
  tolerance, not bitwise: the fine loop's early-stop tests a batch-global maximum, so a pair's
  step count depends on which pairs share its batch.
- **`accel.screen_parallel.screen_parallel`** forks workers for query-vs-library screening,
  sharing the featurized library copy-on-write.

### 4.8 A mesh-free surface generator (opt-in)

`Molecule(..., surface_method="smooth_sdf")` generates the surface point cloud with a smooth-min
signed-distance field plus stochastic sampling, instead of Open3D ball-pivoting + Poisson-disk.
It needs no Open3D and no mesh.

**The default is unchanged.** `surface_method="mesh"` reproduces the original code path exactly.
The smooth surfacer rounds the concave atom-border seams, so a model trained on mesh surfaces
sees a distribution shift — validate before using it in a generative pipeline.

`shepherd_score.surface_diagnostics` quantifies that: it measures how much a point cloud "leaks"
atom positions (points sitting exactly on one atom's sphere) and detects the crimps at
sphere-intersection seams.

### 4.9 Directionless ("color") pharmacophores

`Molecule(..., directionless=True)` and `get_pharmacophores(..., directionless=True)` build
isotropic, zero-vector pharmacophores for every family — the ROCS/ROSHAMBO "color" convention —
rather than oriented feature vectors. `feature_set="rdkit_base"` selects RDKit's base feature
definitions instead of ShEPhERD's SMARTS set.

Both default to the original behavior (`directionless=False`, `feature_set="shepherd"`).

### 4.10 Lazy Open3D

`generate_point_cloud.py` now imports Open3D on first use rather than at module load. Open3D is
a slow import and is fork-hostile — importing it poisons a later `fork` + CUDA, which would
break the fork-based worker pools. Alignment-only code paths never pay for it.

The observable consequence is an improvement: `import shepherd_score.container` now works
without Open3D installed. If Open3D is missing, the error moves from import time to the first
surface generation, and its type (`ModuleNotFoundError`) is unchanged.

### 4.11 `return_aligned=`

`MoleculePairBatch.align_with_*(return_aligned=False)` is the default on the accelerated
backends and skips materializing the aligned coordinate arrays, returning `[None] * N` in their
place. The transforms are still written to each pair. Pass `return_aligned=True` to get them.
The JAX path is unaffected.

---

## 5. API reference

### 5.1 `shepherd_score.container`

```python
__all__ = ["update_mol_coordinates", "Molecule", "MoleculePair", "MoleculePairBatch",
           "align_multi_gpu", "MultiGPUAligner"]
```

`align_multi_gpu` and `MultiGPUAligner` are new re-exports from `shepherd_score.accel`.

#### `Molecule.__init__`

Three new keyword arguments, all appended last, all default-preserving:

```python
Molecule(mol, num_surf_points=None, density=None, probe_radius=None, surface_points=None,
         partial_charges=None, electrostatics=None, pharm_multi_vector=None, pharm_types=None,
         pharm_ancs=None, pharm_vecs=None,
         feature_set='shepherd',      # NEW: 'shepherd' | 'rdkit_base'
         directionless=False,         # NEW: isotropic "color" pharmacophores
         surface_method='mesh')       # NEW: 'mesh' | 'smooth_sdf'
```

`Molecule.get_pharmacophore` gains the same `feature_set` and `directionless` arguments.

#### `MoleculePair` — alignment methods

| Method | Status |
|---|---|
| `align_with_vol(no_H=True, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None, use_jax=False, use_analytical=True, verbose=False)` | defaults changed |
| `align_with_vol_esp(lam=0.1, no_H=True, num_repeats=None, ..., max_num_steps=None, ...)` | defaults changed; `lam` now optional |
| `align_with_surf(alpha, num_repeats=None, ..., max_num_steps=None, ...)` | defaults changed |
| `align_with_surf_esp(alpha, lam=0.3, num_repeats=None, ..., max_num_steps=None, ...)` | **renamed** from `align_with_esp` |
| `align_with_vol_and_surf_esp(alpha, lam=0.001, probe_radius=1.0, esp_weight=0.5, ...)` | **renamed** from `align_with_esp_combo` |
| `align_with_pharm(similarity='tanimoto', ...)` | defaults changed |
| `align_with_vol_color(color_weight=0.5, alpha=0.81, similarity='tanimoto', directional=False, extended_points=False, only_extended=False, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None, verbose=False)` | **NEW** |

`align_with_esp` and `align_with_esp_combo` remain as aliases and are the same function objects.

#### `MoleculePair` — result attributes

| Mode | Transform | Score |
|---|---|---|
| `vol` | `transform_vol_noH` | `sim_aligned_vol_noH` |
| `vol_esp` | `transform_vol_esp_noH` | `sim_aligned_vol_esp_noH` |
| `surf` | `transform_surf` | `sim_aligned_surf` |
| `surf_esp` | `transform_surf_esp` | `sim_aligned_surf_esp` |
| `vol_and_surf_esp` | `transform_vol_and_surf_esp` | `sim_aligned_vol_and_surf_esp` |
| `pharm` | `transform_pharm` | `sim_aligned_pharm` |
| `vol_color` | `transform_vol_color` | `sim_aligned_vol_color` |

`transform_esp`, `sim_aligned_esp`, `transform_esp_combo` and `sim_aligned_esp_combo` remain as
read/write properties forwarding to the new names.

#### `MoleculePairBatch`

Every method gains `backend='jax'` and `return_aligned=False`. `align_with_vol` and
`align_with_vol_esp` additionally gain `alpha=0.81`. Two methods are new
(`align_with_vol_and_surf_esp`, `align_with_vol_color`) and two are renamed with aliases, exactly
as on `MoleculePair`.

```python
align_with_vol(no_H=True, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None,
               num_workers=1, use_shmap=True, num_buckets=1, verbose=False,
               backend='jax', alpha=0.81, return_aligned=False)

align_with_surf(alpha, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None,
                use_jax=True, use_analytical=True, num_workers=1, use_shmap=False,
                verbose=False, backend='jax', return_aligned=False)

align_with_vol_color(color_weight=0.5, alpha=0.81, num_repeats=None, trans_init=False, lr=0.1,
                     max_num_steps=None, verbose=False, backend='jax', return_aligned=False)
```

> **Note.** `alpha=` on `align_with_vol` / `align_with_vol_esp` reaches the Triton and numba
> kernels only. The JAX path still hardcodes `alpha=0.81`, exactly as upstream did. The default
> matches, so nothing changes — but passing a different `alpha` with `backend="jax"` is silently
> ignored.

### 5.2 `shepherd_score.screen` (new)

```python
__all__ = ["MoleculeProfile", "ProfileStore", "screen", "screen_many", "Hit"]

ProfileStore.create(path, *, num_surf_points, modes, dtype='float16', shard_size=100_000,
                    pre_centered=True, overwrite=False) -> ProfileStore
ProfileStore.open(path) -> ProfileStore
    .add(molecule, id=None)          .add_profile(profile, id=None)
    .supports(mode) -> bool          .iter_shards() -> Iterator[List[MoleculeProfile]]
    .read_shard(idx) -> tuple        .read_profiles(idx) -> List[MoleculeProfile]
    .close()

screen(query, store, mode='surf_esp', *, backend=None, do_center=None, top_k=1000, ndev=None,
       scores_out=None, alpha=None, progress=False, **align_kwargs) -> List[Hit]

screen_many(queries, store, mode='surf_esp', *, backend=None, do_center=None, top_k=1000,
            ndev=None, scores_out=None, alpha=None, progress=False, **align_kwargs)
        -> List[List[Hit]]
```

`MoleculeProfile` fields: `atom_pos`, `atom_pos_noH`, `surf_pos`, `surf_esp`, `partial_charges`,
`radii`, `pharm_types`, `pharm_ancs`, `pharm_vecs`, `num_surf_points`, `mol`, `id`.

The on-disk format carries `VERSION = 1`. It should be treated as provisional.

### 5.3 `shepherd_score.accel`

```python
__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner"]

has_triton() -> bool

align_multi_gpu(pairs, mode, *, ndev=None, threads=None, backend='triton', do_center=False,
                write_back=True, return_timing=False, **align_kwargs)

class MultiGPUAligner:
    __init__(pairs, *, ndev=None, threads=None, do_center=False, start_method=None)
    align(mode, *, backend='triton', return_timing=False, **align_kwargs)
    close()

# shepherd_score.accel.screen_parallel
screen_parallel(query, library, mode, n_workers=None, **align_kwargs)
```

#### The mode registry — `shepherd_score.accel._modes`

```python
CANONICAL_MODES     = ('vol', 'vol_esp', 'surf', 'surf_esp', 'vol_and_surf_esp', 'pharm', 'vol_color')
LEGACY_MODE_ALIASES = {'esp': 'surf_esp', 'esp_combo': 'vol_and_surf_esp'}
PROCESS_MODES       = ('vol', 'surf', 'surf_esp', 'pharm')   # modes with a worker-process path
MODE_ATTRS          = {mode: (transform_attr, score_attr)}
MODE_SEEDS          = {'vol': 10, 'surf': 8, 'surf_esp': 8, 'vol_esp': 16,
                       'vol_and_surf_esp': 8, 'pharm': 32, 'vol_color': 16}
MODE_STEPS          = {'vol': 50, 'surf': 40, 'surf_esp': 40, 'vol_esp': 50,
                       'vol_and_surf_esp': 60, 'pharm': 50, 'vol_color': 40}
canonical(mode) -> str      # resolve a legacy name; unknown names pass through
```

`MODE_SEEDS` and `MODE_STEPS` are what `num_repeats=None` and `max_num_steps=None` resolve to.

### 5.4 Surface generation

```python
get_molecular_surface(centers, radii, num_points=None, num_samples_per_atom=None,
                      probe_radius=1.2, ball_radii=[1.2],
                      method='mesh',        # NEW: 'mesh' | 'smooth_sdf'
                      sdf_s=10.0, sdf_iters=6, sdf_knn=8, sdf_jitter=0.0,
                      even='fps', seed=None) -> np.ndarray

get_molecular_surface_smooth_sdf(centers, radii, num_points=None, num_samples_per_atom=None,
                                 probe_radius=1.2, s=10.0, iters=6, knn=8, jitter=0.0,
                                 even='fps', seed=None) -> np.ndarray   # NEW
```

All new parameters are appended after `ball_radii`, so positional callers are unaffected.
`num_samples_per_atom` changed from `25` to `None`; on the `'mesh'` path `None` maps back to
`25`, so the default surface is byte-identical.

### 5.5 Surface diagnostics (new)

```python
leak_metrics(points, centers, radii, probe_radius=1.2) -> Dict[str, float]
crimp_points(points, centers, radii, probe_radius=1.2, on_shell_tol=0.1, seam_tol=0.35)
center_recovery_attack(points, centers, radii, probe_radius=1.2, min_pts=8) -> Dict[str, float]
local_curvature(points, k=12) -> np.ndarray
summarize(points, centers, radii, probe_radius=1.2) -> Dict[str, float]
```

numpy + scipy only; no Open3D.

### 5.6 Scoring and alignment

```python
# shepherd_score.alignment — two new exports
objective_vol_color_overlay(...)
optimize_vol_color_overlay(ref_centers, fit_centers, ref_pharms, fit_pharms, ref_anchors,
                           fit_anchors, ref_vectors, fit_vectors, alpha=0.81, color_weight=0.5,
                           similarity='tanimoto', directional=False, extended_points=False,
                           only_extended=False, num_repeats=50, trans_centers=None, lr=0.1,
                           max_num_steps=200, verbose=False)

# shepherd_score.score.pharmacophore_scoring{,_np} — new kwargs, defaults preserve behavior
get_overlap_pharm(..., directional=True)
get_pharm_combo_score(..., color_weight=0.5, directional=True)

# shepherd_score.pharm_utils.pharmacophore
get_pharmacophores(mol, ..., feature_set='shepherd', directionless=False)
get_pharmacophores_dict(...)  # same two new kwargs

# shepherd_score.score.analytical_gradients._torch
build_lookup_tables(..., directionless=False)
build_lookup_tables_cached(..., directionless=False)
```

`get_pharm_combo_score`'s final combination changed from `(pharm + shape) / 2` to
`(1 - color_weight) * shape + color_weight * pharm`. At the default `color_weight=0.5` these are
**exactly** equal in IEEE-754 (multiplication by 0.5 is exact), so the result is bit-identical.

---

## Behavior changes (read this)

Four changes are visible to code written against the previous release.

### B1. `num_repeats` and `max_num_steps` defaults — the important one

On **all six** pre-existing alignment methods, on **both** `MoleculePair` and
`MoleculePairBatch`:

```python
num_repeats:   int = 50   ->  Optional[int] = None
max_num_steps: int = 200  ->  Optional[int] = None
```

`None` resolves per-mode via `accel/_modes.py`:

| mode | seeds (was → now) | steps (was → now) |
|---|---|---|
| `vol` | 50 → **10** | 200 → **50** |
| `vol_esp` | 50 → **16** | 200 → **50** |
| `surf` | 50 → **8** | 200 → **40** |
| `surf_esp` | 50 → **8** | 200 → **40** |
| `vol_and_surf_esp` | 50 → **8** | 200 → **60** |
| `pharm` | 50 → **32** | 200 → **50** |

**This changes results.** It is not merely "fewer restarts". `alignment/_torch.py` and
`alignment/_jax.py` contain a special case:

```python
if num_repeats == 50:
    se3_params[5:, :4] = _get_45_fibo()          # a precomputed 45-point Fibonacci set
else:
    se3_params[5:, :4] = _quats_from_fibo(num_repeats - 5)
```

The old default of exactly `50` hit the precomputed branch. None of the new defaults are 50, so
every default call now takes the `else` branch and gets a **different set of orientations**, not
a subset of the old one. That seeding code is unchanged between releases — only the default that
feeds it moved.

Measured over 56 ordered cross-pairs of drug-like molecules: **every pair's score changed.** The
aggregate cost is small — `vol` mean Δ = −0.00013, ≥99.98% of the achievable overlap — but
individual pairs move by up to several percent (worst observed: `pharm`, −4.55% relative).

These defaults sit at the accuracy/throughput knee: seed count is cheap in retrospective-screening
ROC-AUC but expensive in throughput, and ROC-AUC plateaus at low seed counts for every mode.

**To restore the previous behavior exactly**, pass the old values:

```python
pair.align_with_vol(num_repeats=50, max_num_steps=200)
```

If you have a pinned regression baseline, a golden-file test, or a published score table, do this.

### B2. Mode rename

`esp` → `surf_esp` and `esp_combo` → `vol_and_surf_esp`, across method names, result attributes,
and every mode string (the screen front-end, the CPU pool, the multi-GPU driver).

**Input-side compatibility is complete.** The legacy method names are aliases to the same
function objects, the legacy result attributes are read/write properties, and every mode string
is canonicalized at the entry point. Existing code that *names* these things keeps working.

**Output-side compatibility is not complete.** The canonical attributes are the real instance
storage; the legacy names exist only as class-level properties. So:

```python
pair.transform_esp                  # works
'transform_esp' in pair.__dict__    # now False
vars(pair)                          # now shows transform_surf_esp
```

Code that introspects `MoleculePair.__dict__` — rather than naming the attribute — will see the
new names.

### B3. Unpickling old `MoleculePair` objects

A `MoleculePair` pickled by the previous release carries `transform_esp` / `sim_aligned_esp` in
its `__dict__` and *not* the canonical keys. Because the legacy names are now data descriptors,
they take precedence over the instance dict and forward to a canonical attribute that the old
pickle does not have:

```
AttributeError: 'MoleculePair' object has no attribute 'transform_surf_esp'
```

**Old pickles do not round-trip.** Re-run the alignment, or add a `__setstate__` that remaps the
legacy keys.

### B4. `apply_SE3_transform` collapses a singleton batch

`shepherd_score.alignment.utils.se3.apply_SE3_transform` was reimplemented with `baddbmm`. For a
batch of exactly one:

```python
apply_SE3_transform(points[None], transform[None])   # was (1, N, 3), now (N, 3)
```

Values are unchanged; only the shape collapses. No call site inside the library is affected, and
the batched and gradient paths remain bitwise identical to before. But two caveats matter:

- **`apply_SO3_transform` was not given the same collapse.** With an `R == 1` batch the two
  functions now return different ranks, and upstream pairs them on adjacent lines in five places.
  This mis-broadcasts silently rather than failing loudly.
- The single-instance (non-batched) path differs by about **one float32 ULP** (~1.2e-7) from the
  old `(R @ P.T).T + t`. Harmless numerically, but it breaks bit-for-bit reproduction of golden
  values on the `num_repeats == 1` path.

### Minor

- `align_with_vol_esp(lam=...)` — `lam` was a required positional; it now defaults to `0.1`.
  Strictly widening.
- `backend="triton"` on `vol` and `surf` ignores `num_repeats`, `trans_init`, and `lr` (the
  kernels re-derive seeds internally). Every other mode passes them through. `backend="jax"` is
  unaffected.

---

## 7. Testing and validation

- **244 passed, 76 skipped, 0 failed.** All of upstream's tests still pass. The only upstream
  test file touched is `tests/test_alignment_utils.py`, which gains one appended test pinning the
  `R == 1` contract from B4 — no existing assertion was changed.
- The 76 skips are the Triton, CUDA, and JAX tests, which need hardware or optional
  dependencies this environment lacks. Verify them on a CUDA + JAX box.
- `ruff check shepherd_score/ tests/` passes clean.
- The package imports with `numba`, `triton`, `jax`, and `open3d` **all absent**.
- Behavior on the default path was checked against a 180-array golden baseline (`vol`,
  `vol_color`, `pharm` over 30 cross-pairs) captured before the final cleanup pass: **every array
  is bit-identical**.

## 8. Known gaps

- **`accel/` and `screen.py` have no Sphinx API pages.** They are not in `docs/api/`.
- **The parity gates for three optimizations are not in the test suite.** The adaptive bucketer,
  the CUDA-graph fine loop, and the fused `vol_color` kernel were validated by standalone scripts
  that lived in the (now removed) `benchmarks/` tree. They should be rewritten as CUDA-gated
  tests. Nothing in `tests/` currently covers those three paths.
- **No JAX ↔ Triton/numba cross-backend parity test exists.** Backends are validated against
  self-copy recovery and against the torch reference, not against each other.
- **The `ProfileStore` on-disk format is provisional** (`VERSION = 1`).
