# What's New

This document describes the accelerated-alignment update to `shepherd-score`. It covers the
code, the organization of every new file, the full public API, and every new feature.

> ## ⚠️ Read first — three changes affect existing results and installs
>
> 1. **The default batch backend changed, and it changes scores.** `MoleculePairBatch.align_with_*`
>    used to default to JAX; it now defaults to the accelerated backends — **Triton on a CUDA host,
>    numba on CPU**. Those backends use a *different SE(3) seed set* than JAX, so a default
>    `align_with_*` call now returns **different (not worse) scores** than before. If you need the
>    old numbers, pass `backend="jax"` explicitly. See [§4.2](#42-a-backend-argument-on-every-batch-aligner)
>    and [Behavior changes B5](#7-behavior-changes-read-this).
> 2. **`numba` is now a core dependency**, not an optional extra. The package no longer imports
>    without it. See [§2](#2-installation-and-dependencies).
> 3. **The default charge model for the ESP modes is now gfn2-xTB, not MMFF94, and it changes ESP
>    scores.** `Molecule(...)` with no `partial_charges` now defaults to `charge_model="xtb"`. Charges
>    are generated **lazily** — only when something actually reads them (an ESP mode, or the surface
>    ESP built when a surface is generated) — so pure volumetric-shape / colour / pharmacophore work
>    never pays the xTB subprocess. If the `xtb` binary is missing or fails on a molecule, it falls
>    back to MMFF94 with a warning. Pass `charge_model="mmff"` (or explicit `partial_charges=`) for the
>    old behavior. xTB is **not uniformly better**: on the DUDE-Z enrichment benchmark it helps the
>    surface-ESP modes (`vol_and_surf_esp` +0.036 ROC-AUC) but *hurts* atom-centred `vol_esp`
>    (−0.041); pick the charge model to match the representation. See [Behavior changes B6](#7-behavior-changes-read-this).

Structurally the fork is still additive on top of upstream (no upstream file deleted, no upstream
public name removed), and this release also **merges upstream's `Molecule` refactor** (the
`Surface`/`Pharmacophore`/`AlignmentResult` dataclasses); fss's flat attributes still work because
upstream exposes them as backward-compatible properties. The behavior changes visible to existing
callers are enumerated in [Behavior changes](#7-behavior-changes-read-this) — read that before upgrading.

---

## Contents

1. [The diff](#1-the-diff)
2. [Installation and dependencies](#2-installation-and-dependencies)
3. [Organization: what every file does](#3-organization-what-every-file-does)
4. [New features](#4-new-features)
5. [API reference](#5-api-reference)
6. [Extensibility: the two agent skills](#6-extensibility-the-two-agent-skills) — **not implemented yet**
7. [Behavior changes (read this)](#7-behavior-changes-read-this)
8. [Testing and validation](#8-testing-and-validation)
9. [Known gaps](#9-known-gaps)
10. [Session update: nine new alignment modes via the agent skills](#10-session-update-nine-new-alignment-modes-via-the-agent-skills)

---

## 1. The diff

Measured against `coleygroup/shepherd-score` at **`446f358`** (the current upstream, after this
release merged it in — so these are the fork's net additions on top of the new upstream):

```
61 files changed, 14,278 insertions(+), 253 deletions(-)
```

(Excluding this document.)

Broken down: **37 new files**, **24 modified** (15 library source, 2 test, 7 config/docs/example),
**0 deleted**. The upstream `Molecule` refactor was merged with a 3-way merge; only three files
conflicted (`_core.py`, `pharmacophore.py`, `container/__init__.py`), all resolved.

### New files (37, excluding this document)

| File | Lines | Role |
|---|---:|---|
| **`shepherd_score/accel/`** — the acceleration subpackage | **9,498** | |
| `accel/__init__.py` | 52 | Public surface: `has_triton`, `clear_caches`, `align_multi_gpu`, `MultiGPUAligner` |
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
| `accel/drivers/_common.py` | 538 | Shared seeding, coarse grid, Adam/Tanimoto tails |
| `accel/drivers/_graphed.py` | 193 | Mode-agnostic CUDA-graph fine loop |
| `accel/drivers/shape.py` | 313 | Driver for `vol` and `surf` |
| `accel/drivers/esp.py` | 563 | Driver for `vol_esp` and `surf_esp` |
| `accel/drivers/esp_combo.py` | 807 | Driver for `vol_and_surf_esp` |
| `accel/drivers/pharm.py` | 775 | Driver for `pharm` |
| `accel/drivers/pharm_overlap.py` | 475 | Pharmacophore overlap support for the pharm driver |
| `accel/drivers/vol_color.py` | 518 | Driver for `vol_color` |
| `accel/batch/__init__.py` | 21 | Re-exports the batch surface |
| `accel/batch/aligners.py` | 1,146 | The seven `_align_batch_<mode>` functions `MoleculePairBatch` calls |
| `accel/batch/_bucket.py` | 277 | Adaptive size bucketer |
| `accel/batch/_pad.py` | 125 | Padding, GPU-memory sub-batching, scatter |
| `accel/batch/_dispatch.py` | 142 | Multi-GPU dispatch + the per-mode tensor spec (`_MODE_SPEC`) |
| `accel/cpu_pool.py` | 220 | Persistent single-threaded CPU worker pool |
| `accel/multi_gpu.py` | 424 | Process-per-GPU data parallelism |
| `accel/screen_parallel.py` | 98 | Fork-based shard-parallel CPU screening |
| **Top-level modules** | | |
| `shepherd_score/screen.py` | 1,297 | Virtual-screening front-end: `ProfileStore`, `screen`, `screen_many` |
| `shepherd_score/surface_diagnostics.py` | 142 | Leak / crimp metrics for validating a surface generator |
| **Tests** (7 new files) | **1,658** | |
| `tests/test_fast_batch_alignment.py` | 396 | Triton/CUDA batch aligners |
| `tests/test_screen.py` | 393 | Screening front-end |
| `tests/test_vol_color.py` | 275 | The `vol_color` mode |
| `tests/test_cpu_pool.py` | 168 | CPU process pool |
| `tests/test_smooth_surface.py` | 165 | Smooth-SDF surfacer + diagnostics |
| `tests/test_numba_backend.py` | 148 | numba CPU kernels |
| `tests/test_mode_registry.py` | 113 | Registry invariants (guards against drift) |

(The consolidated conda `environment.yml` is a *modified* file — see below.)

### Modified files — library source (15)

Numbers are the fork's net change on top of the new upstream `446f358`.

| File | Change | Additive? |
|---|---:|---|
| `shepherd_score/container/_batch.py` | +472 / −27 | Mostly. The default-backend change is a behavior change — see [B5](#7-behavior-changes-read-this) |
| `shepherd_score/container/_core.py` | +298 / −52 | Mostly. Merge-reconciled onto upstream's Surface/Pharmacophore Molecule; see [Behavior changes](#7-behavior-changes-read-this) |
| `shepherd_score/alignment/_torch.py` | +255 / −0 | Yes — one appended hunk (the `vol_color` objective/optimizer) |
| `shepherd_score/generate_point_cloud.py` | +249 / −13 | Yes — lazy Open3D + the opt-in `smooth_sdf` surfacer |
| `shepherd_score/score/pharmacophore_scoring.py` | +101 / −4 | Yes — the `directionless` scoring kwarg (see [B7](#7-behavior-changes-read-this)) |
| `shepherd_score/pharm_utils/pharmacophore.py` | +78 / −26 | Yes — `feature_set`/`directionless` re-applied onto upstream's rewritten extractor |
| `shepherd_score/score/pharmacophore_scoring_np.py` | +55 / −16 | Yes — `directionless` (numpy oracle) |
| `shepherd_score/alignment/utils/se3.py` | +28 / −14 | **No** — see the `R==1` shape change ([B4](#7-behavior-changes-read-this)) |
| `shepherd_score/score/analytical_gradients/_torch.py` | +17 / −6 | Yes |
| `shepherd_score/container/__init__.py` | +5 / −0 | Yes — union of upstream's + fork's exports |
| `shepherd_score/evaluations/evaluate/evals.py` | +6 / −6 | Internal rename only |
| `shepherd_score/objective.py` | +5 / −5 | Internal rename only |
| `shepherd_score/alignment/__init__.py` | +4 / −0 | Yes — two new exports |
| `shepherd_score/evaluations/evaluate/_pipeline_eval_single.py` | +2 / −2 | Internal rename only |
| `shepherd_score/protonation/protonate.py` | +2 / −0 | `from __future__ import annotations` |

Also modified: `tests/test_alignment_utils.py` (+12 — one appended test, no existing assertion
changed), `pyproject.toml` (+33/−1 — numba core, `triton>=3.6`, ruff scoping), `environment.yml`
(+34/−14 — rewritten to the single SVML env; `environment-cpu-svml.yml` removed), `README.md`
(+9), `.gitignore` (+3), `pytest.ini` (+1), `docs/usage.rst` (+4/−4) and
`docs/api/container/molecule_pair_batch.rst` (rename), `examples/02_scoring.ipynb` (+9/−62,
rename + re-run).

---

## 2. Installation and dependencies

```bash
pip install shepherd-score            # core: rdkit, torch, open3d, py3Dmol, numpy, pandas, scipy, molscrub, tqdm, numba
pip install "shepherd-score[gpu]"     # + triton>=3.6  -> backend="triton"  (needs a CUDA torch build)
pip install "shepherd-score[jax]"     # + jax          -> backend="jax"    (the OLD default; now opt-in)
```

- **`numba` is now a CORE dependency**, no longer optional. It is the default CPU backend for the
  batched aligners (see [§4.2](#42-a-backend-argument-on-every-batch-aligner)), so the package no
  longer imports cleanly with numba absent. The `[cpu]` extra is retained as a redundant alias.
  numba is unpinned in pip because the fast SVML build cannot be delivered by pip — use the conda
  `environment.yml` for that (below).
- **`triton` is optional** (the `[gpu]` extra) and lazily imported, so the package still imports
  without it. `jax` is optional too — but note it is **no longer the default backend**.
- **`tqdm` moved into the core dependency list.** This is a bugfix: upstream already imported
  `tqdm` at module scope but never declared it.
- **The `gpu` extra has two install traps** (both flagged in `pyproject.toml`):
  - **A CUDA build of torch is required.** The default CPU torch wheel has no GPU and bundles no
    Triton, so `pip install torch` alone will not run the GPU path — install a `+cuXXX` build.
  - **Triton must be ≥ 3.6.** The overlap kernels use `@triton.autotune(cache_results=…)`, an API
    that only exists in Triton 3.6+. On older Triton (e.g. the 3.1.0 that ships with a CPU-cloned
    env) every GPU kernel dies at compile with a missing-attribute error and there is **no CPU
    fallback**. The `gpu` extra pins `triton>=3.6`; a known-good combination is `torch==2.11.0+cu128`
    with `triton==3.6.0` from the cu128 index.
- **`triton` ships manylinux wheels only** — install it explicitly on Linux, not via an
  `all`-style extra on Windows/macOS.
- **One conda `environment.yml` now serves both CPU and GPU.** It pins the SVML-vectorized numba
  stack (`numba 0.59.1 + llvmlite 0.42 + icc_rt + numpy 1.26`) plus the full runtime deps (open3d,
  xtb, rdkit, …). The separate `environment-cpu-svml.yml` was folded into it and removed. SVML
  makes the CPU kernels ~2–4× faster **and changes precision** — it switches the shape/ESP inner
  loops to fp32 structure-of-arrays kernels (values ~1e-6 rel, gradients ~1e-4). Without SVML the
  kernels are correct but slower and emit a one-time `RuntimeWarning`, so the slow regime is never
  silent. For the GPU path, add a CUDA torch + `triton>=3.6` on top with pip.

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

### 4.1 The seven modes — and what each one actually scores

Every mode uses a Gaussian Tanimoto `V_AB / (V_AA + V_BB − V_AB)`, with self-overlaps computed
through the same kernel at the identity pose.

| Mode | Scores | Pose is steered by |
|---|---|---|
| `vol` | Gaussian shape overlap of heavy-atom centers | shape gradient |
| `surf` | the same, over surface points | shape gradient |
| `vol_esp` | shape overlap, each pair term weighted by `exp(−(C_i − C_j)²/λ)` on partial charges | shape+ESP gradient |
| `surf_esp` | the same, over surface points with surface ESP | shape+ESP gradient |
| `vol_and_surf_esp` | `esp_weight · ESP_sim + (1 − esp_weight) · shape_sim`, where `ESP_sim` is a ShaEP-style masked Gaussian of the ESP difference at each surface point | **shape gradient only** — see below |
| `pharm` | typed pharmacophore Gaussians (per-type α), same-type only, with a direction weight | pharmacophore gradient |
| `vol_color` | `(1 − color_weight) · shape_Tanimoto + color_weight · color_Tanimoto`; color is the pharmacophore Gaussian with all direction weighting removed (the ROCS/ROSHAMBO convention) | **joint** — both channels |

Two modes are new:

**`vol_and_surf_esp`** — combined volumetric shape + surface-ESP scoring. New on
`MoleculePairBatch` (upstream had a combo mode on `MoleculePair` only).

> **The SE(3) descent direction for this mode is the shape gradient alone.** The electrostatic
> term enters only the *tracked score*, never the derivative — the mode optimizes shape and
> *reports* a shape+ESP score. Because the trajectory is shape-driven, the expensive ESP score is
> evaluated only every 5th step plus the final step (`_ESP_STRIDE` in
> `accel/drivers/esp_combo.py`); set it to 1 to score densely.

**`vol_color`** — atom-centred Gaussian shape overlap plus a **directionless pharmacophore
("color")** overlap. Unlike `vol_and_surf_esp`, the SE(3) step here descends on the **joint**
weighted objective, so *both* channels steer the pose. New on both `MoleculePair` and
`MoleculePairBatch`.

The two `vol_color` signatures differ: `MoleculePair.align_with_vol_color` accepts `similarity`,
`directionless`, `extended_points` and `only_extended`; the batch version accepts none of them.

`pharm` also supports `similarity='tversky' | 'tversky_ref' | 'tversky_fit'` alongside the default
`'tanimoto'`. Tversky forfeits both the CUDA-graph and the fused-CPU fast paths.

### 4.2 A `backend=` argument on every batch aligner

`MoleculePairBatch.align_with_*` takes `backend="jax" | "triton" | "numba"`, defaulting to `None`.
Aliases are accepted: `"cuda"` / `"gpu"` → triton, `"cpu"` → numba.

- **`None` (the default) is device-aware: Triton on a CUDA host, numba on CPU.** JAX is *not* the
  default any more — pass `backend="jax"` to select it.
- `"triton"` runs hand-written Triton GPU kernels.
- `"numba"` runs numba CPU kernels.
- `"jax"` runs the original JAX path — **except on the two new modes** (`align_with_vol_and_surf_esp`,
  `align_with_vol_color`), which have no JAX kernel and fall through to the per-pair PyTorch path.

Kernel selection is per call, by tensor device, so one process can run both — a CPU batch and a
GPU batch in the same program each get the right kernel.

> #### ⚠️ The default now changes scores vs the old JAX default
>
> Before this release, `align_with_*` defaulted to JAX. It now defaults to the accelerated backends,
> and **they use a different SE(3) seed set than JAX**, so *a default call returns different scores
> than it did before*. Both begin with identity + 4 PCA-alignment quaternions, but the accelerated
> seeder then adds up to **six structured ±90° rotations about each reference principal axis** —
> covering the axis *swaps* PCA alignment alone misses — before falling back to a Fibonacci fill. At
> the shipped per-mode seed counts those structured seeds absorb most of the budget: at `vol`'s
> default of 10 seeds the accelerated path emits **zero** Fibonacci rotations, where JAX emits five.
>
> The backends explore different orientations and return different (not worse) scores. `backend=` is
> **not** a pure-performance switch, and Triton/numba will not reproduce a JAX baseline. If you have
> a pinned baseline or a published score table from a previous release, **pass `backend="jax"`
> explicitly** to reproduce it. (Changing the default this way also fixes a latent bug: the old JAX
> default was not installable by default — `jax` is an optional extra — so a fresh install hitting
> the default batch path raised `ImportError`.)

Three backend-specific limits:

- **`no_H=False` is unsupported** on `triton`/`numba` for `vol` and `vol_esp` — they align heavy
  atoms only and raise `NotImplementedError`. It still works on `jax`.
- `vol` and `surf` **ignore `num_repeats`, `trans_init` and `lr`** on `triton`/`numba`; the kernels
  re-derive seeds internally. Every other mode passes them through. `jax` is unaffected.
- **`backend="numba"` permanently moves the batch to CPU.** It sets `pair.device = cpu` on every
  pair and never restores it, so a subsequent `backend="triton"` call on the same
  `MoleculePairBatch` will not run on the GPU. Rebuild the batch to switch back.

### 4.3 The optimizer: seeds, analytic gradients, CUDA graphs

**There is no coarse grid and no top-k prune on the default path.** Despite the "coarse-to-fine"
naming, every seed goes straight into the fine loop and the per-pair maximum is taken. Ranking
seeds on raw un-optimized overlap repeatedly discarded the true basin for pseudo-symmetric
molecules, so it was removed. The coarse-grid + top-k path runs **only** when `trans_init=True`.
The practical consequence: cost scales linearly with `num_repeats`, and there is no cheap-prune
knob.

**Gradients are analytic, not autograd.** Every kernel emits closed-form `dV/dq` (4) and `dV/dt`
(3), so no autograd graph is built in the fine loop at all. The pharmacophore and color kernels go
further and emit `dO/dq` in-register, dropping the rotation-matrix→quaternion projection tail.

**The Adam is not `torch.optim.Adam`.** β₁=0.9, β₂=0.999, ε=1e-8 *inside* the sqrt, **no bias
correction**, a quaternion tangent-space projection fused into the kernel, and unit-quaternion
renormalization every step. `lr` does not mean what it means in `torch.optim.Adam`. The drivers'
internal default is `lr=0.075`, not the `lr=0.1` the public API advertises.

**Early stopping is on, per-mode, and batch-global.** Patience is 2 for `vol`/`surf`/`vol_color`
and 5 for the ESP and pharmacophore modes; tolerance 1e-5; checked every 5 steps to avoid a
per-step GPU→CPU sync. The criterion is a **batch-global maximum**, so a pair's step count depends
on which pairs share its batch — in every path, not just the worker-pool one.

**CUDA graphs.** All seven modes share one implementation (`accel/drivers/_graphed.py`): one fine
step is captured and replayed, removing per-step host launch overhead. Engagement is *not* uniform
— `pharm` is graphed only for `tanimoto` + `extended_points=False`, `vol_and_surf_esp` is graphed
with early-stop disabled (it runs the full step count), and the work budget differs per mode.
Captured graphs live in a bounded LRU cache of 24 and **pin GPU buffers for the process lifetime**;
call `accel.drivers._graphed.reset_graph_cache()` to free them if you hit fragmentation between
large runs.

**Triton autotune is cached to disk.** Every kernel is autotuned on `(N_pad, M_pad)` with
`cache_results=True`, so the roughly-4-seconds-per-shape sweep is paid once per machine rather
than once per process. A first run on a new shape looks slow; later runs (and fresh processes)
do not. The legacy `BLOCK` / `num_warps` / `num_stages` kwargs on the kernel wrappers are accepted
and **ignored**.

### 4.4 Adaptive bucketing and automatic GPU-memory sub-batching

`accel/batch/_bucket.py` replaces the hand-rolled fixed-width size bands with one planner used by
every mode. It is **result-identical** by construction: the kernels are one-CTA-per-pose and mask
padding to the real point counts, and seeds key on the real counts rather than the pad width, so
padding two different-sized molecules into one bucket cannot change a score.

Under-occupied buckets are merged toward a full CTA wave (`SM_count × 16` poses), capped at two
waves. On CPU the wave floor is 1, so occupancy merging is disabled entirely. A bucket's hoisted
memory is capped at 25% of *current free* device memory — read at call time — and oversized buckets
are split, so buckets shrink automatically on a busy or smaller GPU.

Separately, `accel/batch/_pad.py` sub-batches each bucket to keep peak memory under 70% of free
memory, learns the per-pair byte footprint per `(device, mode, pad shape, seed count)`, and
**halves the chunk and retries on OOM**. Both are result-identical, because pairs are independent.

### 4.5 A virtual-screening front-end (`shepherd_score.screen`)

New top-level module. It featurizes a molecule library once into an on-disk **`ProfileStore`**
(sharded `.npz` + a `manifest.json`), then streams shards through the accelerated aligners against
one or many queries, keeping a top-K heap.

```python
from shepherd_score.container import Molecule
from shepherd_score.screen import ProfileStore, screen

# The store's num_surf_points must match how the Molecules were built.
with ProfileStore.create("lib.store", num_surf_points=200,
                         modes=["surf_esp", "vol"]) as store:
    for rdmol in library:
        store.add(Molecule(rdmol, num_surf_points=200))   # Molecule, not a raw RDKit Mol

query = Molecule(query_rdmol, num_surf_points=200)
hits = screen(query, ProfileStore.open("lib.store"), mode="surf_esp",
              alpha=0.81, top_k=1000)
# each hit is a Hit(score, id, transform); transform is a 4x4 numpy array
```

`screen_many` streams the library **once** for a list of queries. Both accept `ndev=` to shard
across GPUs, and `scores_out=` to write full score vectors (memmap-friendly; single-process only —
passing it with `ndev>1` raises).

Things that will bite you if you don't know them:

- **`ProfileStore.add` takes a `Molecule`, not an RDKit `Mol`** — it reads `atom_pos`, `surf_pos`,
  and friends. Every molecule in a store must share the same `num_surf_points`, and a mismatch
  fails at flush time, potentially thousands of `add()` calls later.
- **The default `dtype="float16"` is lossy** (~0.01 Å, the surface-resampling noise floor). Screen
  scores will differ slightly from a pairwise `MoleculePair` run for this reason alone. Pass
  `dtype="float32"` if you need comparable numbers.
- **A store only serves modes it has the arrays for.** `vol` works on any store; `vol_color` needs
  a pharmacophore store; `vol_and_surf_esp` needs surfaces, ESP, charges, radii and with-H centers,
  so build for it explicitly. An unsupported mode raises `ValueError`.
- **Some modes require explicit parameters.** `vol_and_surf_esp` requires `alpha=`, `vol_esp`
  requires `lam=`, and `no_H=False` is rejected outright — screening is heavy-atom-only. `alpha`
  auto-fills for `surf`/`surf_esp` only.
- **`Hit.transform` is in the pre-centered frame.** On a `pre_centered=True` store (the default)
  library molecules are stored shifted by their own centre of mass and the query is centered too,
  so the 4×4 maps *centered* onto *centered*. To apply it to original coordinates, subtract that
  molecule's COM first — or build the store with `pre_centered=False`.
- **`shard_size` is a GPU-memory knob**, not just an I/O knob: on the fast path a whole shard is
  uploaded as device tensors at once.
- **`trans_init=True`, `backend="jax"`, or a non-pre-centered store** silently drop you off the
  fast path onto a much slower object path.
- The manifest is rewritten after every shard flush, so **a killed build leaves a readable store**
  containing every completed shard. A store directory is **single-writer**.

### 4.6 Multi-GPU data parallelism

Alignment is host-bound rather than kernel-bound, so driving N GPUs from one process serialises on
the GIL. The path that scales is one OS process per GPU, exposed explicitly:

```python
from shepherd_score.container import MultiGPUAligner, align_multi_gpu

if __name__ == "__main__":                       # required: align_multi_gpu spawns
    scores, transforms = align_multi_gpu(pairs, "surf", ndev=4, alpha=0.81)

    with MultiGPUAligner(pairs, ndev=4) as pool:  # persistent; reuses each GPU's shard
        scores, transforms = pool.align("surf", alpha=0.81)
```

Both return `(scores, transforms)` — and both forward `**align_kwargs` straight into
`align_with_<mode>`, so a mode's **required** arguments are required here too (`alpha` for `surf`,
`alpha`+`lam` for `surf_esp`, and so on).

This is deliberately **opt-in**: a library must not spawn worker processes behind the user's back,
because `spawn` re-imports the caller's `__main__` and breaks unguarded scripts. A large batch on a
multi-GPU host therefore runs on a **single GPU** and emits a one-time warning pointing at
`MultiGPUAligner`. (The warning only fires above 4,096 pairs per GPU.)

- **`align_multi_gpu` always spawns**, so it always needs the `__main__` guard.
- **`MultiGPUAligner` prefers `fork`** for fast startup, and falls back to `spawn` — with a warning
  — only if CUDA is already initialized or Open3D is already imported, since either poisons
  `fork`+CUDA. So: build the pool *before* doing CUDA work, and the `__main__` guard is only needed
  on the spawn fallback.
- **Only `vol`, `surf`, `surf_esp` and `pharm` have a worker-process path.** `vol_esp`,
  `vol_and_surf_esp` and `vol_color` raise `ValueError` here. (`screen(ndev>1)` is a *separate*
  multi-GPU implementation and does serve all seven.)
- Worker threads are capped to `cores // ndev`; the cap is mandatory, since uncapped workers
  oversubscribe and scaling collapses below 1×.
- `Molecule` objects must be picklable; they are what crosses the process boundary.

### 4.7 Two CPU parallelism paths, and the SVML kernels

**`num_workers=N` with `backend="numba"`** uses a persistent pool of single-threaded worker
processes (`accel/cpu_pool.py`), sharding *pairs* across processes. Pairs are independent, so this
does not change the optimization problem — but agreement with one large call is to **convergence
tolerance, not bitwise**, because the fine loop's early-stop tests a batch-global maximum and a
pair's step count depends on which pairs share its shard.

- It uses `spawn`, so it needs an **`if __name__ == "__main__":` guard**.
- It applies to **`vol`, `surf`, `surf_esp` and `pharm` only**. On `vol_esp`, `vol_and_surf_esp`
  and `vol_color` it is a silent no-op (the latter two do not even accept `num_workers`). It is
  also ignored on CUDA tensors.

**`accel.screen_parallel.screen_parallel(query, library, mode, n_workers=...)`** forks workers for
query-vs-library screening, sharing the featurized library copy-on-write. It is a *different* API
from `screen()`: it takes a RAM-resident list of `Molecule`s rather than a `ProfileStore`, is
CPU/numba-only, and returns a plain list of scores in library order — no `Hit`s, no transforms, no
top-K. It accepts canonical mode names only.

> It **always forks**, so it is POSIX-only. And it must run **before** any in-process numba
> alignment: a live libgomp thread pool at `fork` aborts the child. Featurize, then screen.

**The SVML CPU kernels change precision, not just speed.** When numba can emit Intel SVML vector
math, the fused CPU loop automatically switches the shape and ESP inner loops to **fp32
structure-of-arrays** kernels. That is where the CPU speed comes from, but it means *enabling* SVML
changes results: values agree to ~1e-6 relative, gradients to ~1e-4. There is no way to turn it off.
The fused CPU loop is also not applied uniformly — `surf_esp` is deliberately excluded from it (as
the most shape-degenerate mode, the fused trajectory settles in different, equally valid basins),
and `pharm` is fused only for `tanimoto` without extended points.

The numba kernels run with `fastmath=True` and `parallel=True`, using `NUMBA_NUM_THREADS`
(defaulting to *all* cores). That oversubscribes badly if you also use the process pool — which is
exactly why `screen_parallel` pins it to 1.

### 4.8 A mesh-free surface generator (opt-in)

`Molecule(..., surface_method="smooth_sdf")` generates the surface point cloud with a smooth-min
signed-distance field plus stochastic sampling, instead of Open3D ball-pivoting + Poisson-disk.
It needs no Open3D and no mesh.

**The default is unchanged.** `surface_method="mesh"` reproduces the original code path exactly.
The smooth surfacer rounds the concave atom-border seams, so a model trained on mesh surfaces sees
a **distribution shift** — validate before using it in a generative pipeline.

`surface_method="smooth_sdf"` requires `num_surf_points` and **raises `ValueError` if you pass
`density=`** instead. Aliases `'sdf'`, `'smooth'` and `'fast'` are accepted; an unknown method
raises. Its tunables (`sdf_s`, `sdf_iters`, `sdf_knn`, `sdf_jitter`, `even`, `seed`) are exposed on
`get_molecular_surface`, with defaults in the module-level `SMOOTH_SDF_*` constants. It samples 15
candidates per atom, against the mesh path's 25.

`shepherd_score.surface_diagnostics` is how you validate a surfacer. Its core quantity is the
**shell residual** — the distance from a point to an atom's *sphere*, not to its center:

- `leak_metrics` → how much the cloud "leaks" atom positions. A point sitting exactly on one
  sphere makes that atom's center directly recoverable. **Lower residual = more leak.** For scale,
  the `mesh` surfacer gives a ~0.010 Å median.
- `crimp_points` → points on one sphere *and* near a second: the concave atom-border seams.
- `center_recovery_attack` → a strongest-case attack that Kasa-fits a sphere to recover each atom.
  **Higher error = safer.**
- `local_curvature`, `summarize` → per-point non-flatness, and the whole gate in one call.

### 4.9 Directionless ("color") pharmacophores

`Molecule(..., directionless=True)` and `get_pharmacophores(..., directionless=True)` build
isotropic, zero-vector pharmacophores for every family — the ROCS/ROSHAMBO "color" convention —
rather than oriented feature vectors. Both default to the original behavior.

> **`directionless=True` changes the pharmacophore *count*, not just the vectors.** The
> donor/acceptor multi-vector branches are skipped entirely, so a donor that upstream expanded into
> one anchor *per hydrogen* now emits a single anchor at the feature position.

`feature_set="rdkit_base"` selects RDKit's base feature definitions instead of ShEPhERD's SMARTS
set. It keeps six families and renames three (`PosIonizable`→`Cation`, `NegIonizable`→`Anion`,
`LumpedHydrophobe`→`Hydrophobe`) — **`Halogen` and `ZnBinder` disappear.**

### 4.10 Lazy Open3D

`generate_point_cloud.py` now imports Open3D on first use rather than at module load. Open3D is
a slow import and is fork-hostile — importing it poisons a later `fork` + CUDA, which would
break the fork-based worker pools. Alignment-only code paths never pay for it.

The observable consequence is an improvement: `import shepherd_score.container` now works without
Open3D installed. If Open3D is missing, the error moves from import time to the first surface
generation, and its type (`ModuleNotFoundError`) is unchanged.

One caveat: the module global `o3d` is now a lazy **proxy object**, not the module. Attribute
access works, but `from shepherd_score.generate_point_cloud import o3d` no longer yields something
`isinstance`/`inspect` will treat as a module.

Note also that `import shepherd_score.container` now eagerly imports `shepherd_score.accel.batch`
and `shepherd_score.accel.multi_gpu`, so the accel stack loads with the container. This adds no new
dependency (torch was already required), but it is not free.

### 4.11 `return_aligned=`

`MoleculePairBatch.align_with_*(return_aligned=False)` is the default on the accelerated backends
and skips materializing the aligned coordinate arrays, returning `[None] * N` in their place. The
transforms are still written to each pair. Pass `return_aligned=True` to get them. The JAX path is
unaffected.

`align_with_pharm` returns a **3-tuple** `(scores, [None]*N, [None]*N)` where the other modes
return a 2-tuple, so code that indexes `[1]` expecting arrays gets `None`s.

### 4.12 Upstream `Molecule` refactor: `Surface` / `Pharmacophore` / `AlignmentResult`

This release merges upstream `446f358`, which restructured `Molecule` around dataclasses. fss adopts
it wholesale; the following are **new public API** a user of the merged package gets. Existing flat
attributes are unaffected — upstream exposes them as backward-compatible properties.

- **`Surface`** (`shepherd_score.container.profiles`) — holds `positions` / `esp` / `probe_radius`.
  Backs `Molecule.surf_pos` / `surf_esp` / `probe_radius` and is reachable via `Molecule.surface`.
- **`Pharmacophore`** — holds `types` / `positions` / `vectors`, plus `mol` / `atom_ids` / `labels`.
  It **unpacks as the old `(types, positions, vectors)` 3-tuple**, so `X, P, V = get_pharmacophores(mol)`
  still works. Reachable via `Molecule.pharmacophore`; backs `pharm_types` / `pharm_ancs` / `pharm_vecs`.
- **`AlignmentResult`** — one `(score, transform)` per mode. `MoleculePair` now stores results in a
  `_alignments` dict of these; `transform_<mode>` / `sim_aligned_<mode>` are properties over it, and
  the legacy `transform_esp` / `transform_esp_combo` names delegate to `surf_esp` / `vol_and_surf_esp`.

**Priority pharmacophores.** `get_pharmacophores` (and `Molecule.get_pharmacophore`) gained:
- `return_atom_ids=False` → retain per-pharmacophore atom-id sets on `Pharmacophore.atom_ids`.
- `priority_atoms=None` → compute 0/1 priority labels (ring-aware) into `Pharmacophore.labels`.
- `min_ring_priority_atoms=3` → the ring threshold for the aromatic/hydrophobe priority rule.
- `Pharmacophore.priority_labels(priority_atoms, ...)` computes labels lazily for any atom set.

These are opt-in and do not affect the arrays the accel/kernels consume (they read only
`types`/`positions`/`vectors`). `feature_set` and `directionless` (fss's extraction controls) sit
alongside them on the same signature.

---

## 5. API reference

### 5.1 `shepherd_score.container`

```python
__all__ = ["update_mol_coordinates", "Molecule", "MoleculePair", "MoleculePairBatch",
           "Surface", "Pharmacophore", "AlignmentResult",   # from upstream's Molecule refactor
           "align_multi_gpu", "MultiGPUAligner"]             # from shepherd_score.accel
```

`Surface`, `Pharmacophore` and `AlignmentResult` are the dataclasses upstream's refactor introduced
(see [§4.12](#412-upstream-molecule-refactor-surface--pharmacophore--alignmentresult)).
`align_multi_gpu` / `MultiGPUAligner` are re-exports from `shepherd_score.accel`.

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

Surface and pharmacophore are now stored on `Molecule._surface` (a `Surface`) and
`Molecule._pharmacophore` (a `Pharmacophore`), exposed through the `.surface` / `.pharmacophore`
properties and the unchanged flat `surf_pos` / `surf_esp` / `probe_radius` / `pharm_types` /
`pharm_ancs` / `pharm_vecs` accessors (now property+setter over those dataclasses).

New read helpers: `Molecule.get_positions(no_H=True)` and `Molecule.get_charges(no_H=True)` return
the heavy-atom (or all-atom) coordinate / partial-charge arrays.

`Molecule.get_pharmacophore` gains `feature_set` and `directionless` (fss) **and** upstream's
`return_atom_ids` / `priority_atoms` / `min_ring_priority_atoms` (see §4.12).

#### `MoleculePair` — alignment methods

| Method | Status |
|---|---|
| `align_with_vol(no_H=True, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None, use_jax=False, use_analytical=True, verbose=False)` | defaults changed |
| `align_with_vol_esp(lam=0.1, no_H=True, num_repeats=None, ..., max_num_steps=None, ...)` | defaults changed; `lam` now optional |
| `align_with_surf(alpha, num_repeats=None, ..., max_num_steps=None, ...)` | defaults changed |
| `align_with_surf_esp(alpha, lam=0.3, num_repeats=None, ..., max_num_steps=None, ...)` | **renamed** from `align_with_esp` |
| `align_with_vol_and_surf_esp(alpha, lam=0.001, probe_radius=1.0, esp_weight=0.5, ...)` | **renamed** from `align_with_esp_combo` |
| `align_with_pharm(similarity='tanimoto', ...)` | defaults changed |
| `align_with_vol_color(color_weight=0.5, alpha=0.81, similarity='tanimoto', directionless=True, extended_points=False, only_extended=False, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None, verbose=False)` | **NEW** |

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
(`align_with_vol_and_surf_esp`, `align_with_vol_color`) and two are renamed with aliases.

`MoleculePairBatch.align_with_vol_color` does **not** take `similarity`, `directionless`,
`extended_points` or `only_extended` — those exist only on the `MoleculePair` version.

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
__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner", "clear_caches"]

has_triton() -> bool
clear_caches() -> None      # free the process-global accel caches (workspaces, footprint table, graph cache)

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
MODE_STEPS          = {'vol': 30, 'surf': 40, 'surf_esp': 40, 'vol_esp': 50,
                       'vol_and_surf_esp': 60, 'pharm': 50, 'vol_color': 40}
canonical(mode) -> str      # resolve a legacy name; unknown names pass through
```

`MODE_SEEDS` and `MODE_STEPS` are what `num_repeats=None` and `max_num_steps=None` resolve to.

#### Lower-level entry points

These are importable and callable, though most users will not need them. They are **not** covered
by a stability promise.

```python
# shepherd_score.accel.kernels.dispatch — device-dispatched kernels (CUDA -> Triton, CPU -> numba)
overlap_score_grad_se3_batch(...)        # shape overlap + dV/dq, dV/dt
overlap_score_grad_esp_se3_batch(...)    # shape+ESP overlap + gradient
esp_comparison_batch(...)                # ShaEP-style masked ESP comparison
pharm_score_grad_se3_batch(...)          # directional pharmacophore + gradient
pharm_grad_dq_se3_batch(...)             # ... with in-register dO/dq
pharm_color_score_grad_se3_batch(...)    # directionless (color) variant
fused_adam_qt(...)                       # the fused Adam step
fused_adam_qt_with_tangent_proj(...)     # ... with quaternion tangent projection
has_triton() -> bool

# shepherd_score.accel.drivers.<mode> — the batched optimizers
coarse_fine_align_many(...)              # shape.py   (vol, surf)
coarse_fine_esp_align_many(...)          # esp.py     (vol_esp, surf_esp)
coarse_fine_esp_combo_align_many(...)    # esp_combo.py
coarse_fine_pharm_align_many(...)        # pharm.py
coarse_fine_vol_color_align_many(...)    # vol_color.py

# shepherd_score.accel.drivers._graphed
graph_cap(work) -> int                   # the pose cap at which graphs engage
reset_graph_cache()                      # free all captured graphs + their pinned GPU buffers

# shepherd_score.accel.cpu_pool
get_pool(num_workers) -> CpuAlignPool    # process-global singleton
align_pairs(...)
```

#### Tunable module-level constants

None of these are env vars any more; they are constants you can edit or monkey-patch.

| Constant | Module | Default | Effect |
|---|---|---|---|
| `_ESP_STRIDE` | `accel/drivers/esp_combo.py` | 5 | ESP re-scoring interval; 1 = dense |
| `VOL_COLOR_FUSED_MAX_PAD` | `accel/kernels/vol_color_triton.py` | 32 | Above this pad, the fused vol_color kernel is not used |
| `_GRAPH_WORK_BUDGET`, `_GRAPH_CAP_CEIL`, `_GRAPH_CAP_MIN` | `accel/drivers/_graphed.py` | 300M, 262144, 2000 | When a CUDA graph engages, and its memory ceiling |
| `_GRAPH_CACHE_MAX` | `accel/drivers/_graphed.py` | 24 | Live captured graphs (each pins GPU buffers) |
| `_BAND` | `accel/batch/_pad.py` | 16 | Legacy fixed-band pad granularity |
| `SMOOTH_SDF_*` | `generate_point_cloud.py` | — | Smooth-SDF surfacer defaults |

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
                           similarity='tanimoto', directionless=True, extended_points=False,
                           only_extended=False, num_repeats=50, trans_centers=None, lr=0.1,
                           max_num_steps=200, verbose=False)

# shepherd_score.score.pharmacophore_scoring{,_np} — new kwargs, defaults preserve behavior
get_overlap_pharm(..., directionless=False)
get_pharm_combo_score(..., color_weight=0.5, directionless=False)

# shepherd_score.pharm_utils.pharmacophore  -> returns a Pharmacophore (unpacks as (X, P, V))
get_pharmacophores(mol, multi_vector=True, exclude=[], check_access=False, scale=1.0,
                   feature_set='shepherd', directionless=False,        # fss extraction controls
                   return_atom_ids=False, priority_atoms=None, min_ring_priority_atoms=3)  # upstream
get_pharmacophores_dict(...)  # same feature_set/directionless/return_atom_ids kwargs
# Pharmacophore.priority_labels(priority_atoms, min_ring_priority_atoms=3) -> np.ndarray

# shepherd_score.score.analytical_gradients._torch
build_lookup_tables(..., directionless=False)
build_lookup_tables_cached(..., directionless=False)
```

`get_pharm_combo_score`'s final combination changed from `(pharm + shape) / 2` to
`(1 - color_weight) * shape + color_weight * pharm`. At the default `color_weight=0.5` these are
**exactly** equal in IEEE-754 (multiplication by 0.5 is exact), so the result is bit-identical.

---

## 6. Extensibility: the two agent skills

> **Status update: both skills now exist and have been exercised, repeatedly.** The two playbooks
> this section originally reserved as a roadmap are now real, packaged skills —
> `.claude/skills/design-scoring-mode/` and `.claude/skills/accelerate-scoring-mode/` — and have
> been used end to end to add `vol_tversky`, `vol_lipo`, `vol_esp_tversky`, and, in one session, a
> further **nine** new modes at once. See [§10](#10-session-update-nine-new-alignment-modes-via-the-agent-skills)
> for that session's results. The four-step path and worked example below are unchanged and still
> accurate — read on for how the skills map to it.

Adding a new alignment type to this library is a four-step path. Every step already exists as a
real, shipped extension point — `vol_color` was built by walking exactly this path:

| # | Step | What you write | Extension point |
|---|---|---|---|
| 1 | **Define representation + overlap objective** | A feature channel and a normalized (Tanimoto) overlap, in torch | `score/` |
| 2 | **Reference + eager optimizer** | A per-pair Adam optimizer over `objective_<mode>_overlay`. This is your autograd ground truth | `alignment/_torch.py` |
| 3 | **Kernel twins** | A numba (CPU) and a Triton (GPU) kernel with *identical signatures* | `accel/kernels/`, `accel/drivers/` |
| 4 | **Validate** | Parity against the step-2 oracle, plus a throughput gate | `tests/` |

The output plugs into the same optimizer, the same public API, and the same backends as the seven
built-in modes, and produces the same kind of outputs.

The two planned skills each cover half of that path:

### Skill 1 — *author a new alignment type* (steps 1–2)

Takes a scoring idea to a working, correct, **eager** implementation: define the representation
and the overlap objective, then wrap it in the per-pair Adam optimizer so it becomes the gradient
oracle everything downstream is checked against. A user who stops here has a correct new mode on
the existing torch path — just not an accelerated one.

Anchor: register the mode in `shepherd_score/accel/_modes.py` (name, result attributes, seed and
step defaults). The batch layer and the screening front-end pick it up from the registry;
`tests/test_mode_registry.py` will fail if the registry and the tensor spec drift apart.

### Skill 2 — *add an optimized backend* (steps 3–4)

Takes that working eager objective onto the fast path: port it to a fused Triton GPU kernel and
its op-for-op numba CPU twin, wire them into the per-call device dispatcher, and gate the result
on parity against the step-2 oracle plus a throughput measurement.

The intended shortcut, and the reason step 2 matters, is that a correct autograd objective can
first be *batched* over pairs for a no-kernel GPU speedup. Only if the per-step autograd is still
the bottleneck do you need to write kernels at all.

### Status

- Both skills exist and ship (`.claude/skills/design-scoring-mode/`,
  `.claude/skills/accelerate-scoring-mode/`) and have been used repeatedly, most recently to add
  nine modes in one session — see [§10](#10-session-update-nine-new-alignment-modes-via-the-agent-skills).
- The parity and throughput gates the skills depend on for step 4 are still **not fully in the
  committed test suite** for every optimization path — see [Known gaps](#9-known-gaps).

---

## 7. Behavior changes (read this)

Seven changes are visible to code written against the previous release. **B5 (the default batch
backend) and B6 (numba required) are the ones most likely to affect you.**

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
| `vol` | 50 → **10** | 200 → **30** |
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
  functions now return different ranks, and upstream pairs them on adjacent lines in **three**
  places (`alignment/_torch.py:1100`, `:1265`, and `_torch_analytical.py:315`). This
  mis-broadcasts silently rather than failing loudly.
- The single-instance (non-batched) path differs by about **one float32 ULP** (~1.2e-7) from the
  old `(R @ P.T).T + t`. Harmless numerically, but it breaks bit-for-bit reproduction of golden
  values on the `num_repeats == 1` path.

`se3.py` also gains a new public function, `quaternions_to_SE3_batch(q, t) -> (K, 4, 4)`, used by
all the batched aligners.

### B5. The default batch backend changed (scores change) — the important one

`MoleculePairBatch.align_with_*` no longer defaults to JAX. The default is now device-aware —
**Triton on a CUDA host, numba on CPU** — and those backends use a **different SE(3) seed set**
than JAX, so **a default `align_with_*` call returns different scores than it did before**
(different, not worse — see [§4.2](#42-a-backend-argument-on-every-batch-aligner) for the seed-set
detail).

```python
batch.align_with_surf(alpha)                    # was JAX; now Triton/numba -> different scores
batch.align_with_surf(alpha, backend="jax")     # explicit -> the old numbers
```

This only affects `MoleculePairBatch`. The per-pair `MoleculePair.align_with_*` methods are
unaffected (they still default to the torch path). **If you have a pinned baseline or a published
score table from a previous release, pass `backend="jax"`.** Changing the default also fixes a
latent bug — the old JAX default was not installable by default (jax is an optional extra), so a
fresh install hitting the default batch path raised `ImportError`.

### B6. `numba` is now a required dependency

It is the default CPU backend, so it is a core dependency and the package no longer imports without
it. If you relied on `import shepherd_score` working in a numba-free environment, install numba (it
is a pure-pip wheel) or pin the previous release.

### B7. Pharmacophore scoring: `directional` → `directionless`

The fork-only scoring kwarg `directional` (on `get_overlap_pharm` / `get_pharm_combo_score` and
forwarded through `align_with_vol_color`) is **renamed to `directionless` with inverted meaning**,
so extraction and scoring now share one polarity (`directionless=True` = orientation-blind). There
is no alias — `directional=` was never in upstream and is fork-only, so update any call:
`directional=False` → `directionless=True`, `directional=True` → `directionless=False`.

### B8. Default charge model for ESP is now gfn2-xTB (ESP scores change), computed lazily

`Molecule(...)` with no `partial_charges` used to fall back to MMFF94. It now takes a
`charge_model` argument that **defaults to `"xtb"`** (gfn2-xTB), so a default-constructed molecule's
ESP modes score on xTB charges instead of MMFF94 — **different ESP scores**. Shape, colour, and
pharmacophore modes are unaffected (they never read charges).

```python
Molecule(rd, num_surf_points=200)                      # ESP now uses gfn2-xTB charges (new default)
Molecule(rd, num_surf_points=200, charge_model="mmff") # the old MMFF94 behaviour
Molecule(rd, num_surf_points=200, partial_charges=q)   # explicit charges (unchanged; skips generation)
```

Two things make this safe rather than a performance cliff:

- **Lazy.** Charges are generated the first time they are actually read (an ESP mode, or the surface
  ESP that is built when a surface is generated), then cached. A molecule used only for volumetric
  shape / colour / pharmacophore never invokes the xTB subprocess — so screening throughput for the
  non-ESP modes is unchanged. (Constructing *with a surface* does build its surface ESP eagerly, as
  before, and so does pay the charge cost; pass `charge_model="mmff"` for a fast pure-surface-shape
  prep.)
- **Graceful fallback.** If the `xtb` binary is missing or fails to converge on a molecule, it falls
  back to MMFF94 and emits a `RuntimeWarning` rather than crashing. `xtb` is therefore *recommended*
  but not a hard import dependency.

**Why xTB, and a caveat.** gfn2-xTB gives quantum-derived charges that are more transferable than
MMFF94's atom-typed empirical charges, and it is what the paper's ESP benchmark uses. But on the
DUDE-Z retrospective-enrichment benchmark the effect is **mode-dependent, not uniformly positive**:
xTB helps the surface-ESP modes (`vol_and_surf_esp` +0.036 ROC-AUC, `surf_esp` ~flat) but *hurts*
the atom-centred `vol_esp` (−0.041), which does better on MMFF94. The representation and the charge
model interact — surface ESP rewards a faithful field, atom-centred ESP suits atom-centred charges —
so choose `charge_model` per mode rather than assuming xTB is always best.

### Minor

- `align_with_vol_esp(lam=...)` — `lam` was a required positional; it now defaults to `0.1`.
  Strictly widening, so no existing call breaks.
- `MoleculePair.__init__` now eagerly allocates two torch tensors on the target device, so
  constructing a pair touches GPU memory before any `align_*` call.
- `get_overlap_pharm(directionless=True)` raises `ValueError` if combined with
  `precomputed_self_overlaps`, and silently forces `extended_points` / `only_extended` off. The
  numpy variant forces the flags off without raising. Neither affects the `directionless=False`
  default.
- `get_pharm_combo_score`'s combination changed from `(pharm + shape) / 2` to
  `(1 − color_weight) · shape + color_weight · pharm`. At the default `color_weight=0.5` these are
  **exactly** equal in IEEE-754, so results are bit-identical. (Verified over 2 × 10⁶ random pairs
  in both float32 and float64: zero mismatches.)

The `backend=`-specific limits — the different seed set, `no_H=False`, the ignored
`num_repeats`/`trans_init`/`lr` on `vol`/`surf`, and the permanent CPU move under `numba` — are in
[§4.2](#42-a-backend-argument-on-every-batch-aligner), since they affect only the new backends and
not any existing caller.

---

## 8. Testing and validation

- **244 passed, 76 skipped, 0 failed.** All of upstream's tests still pass. The only upstream
  test file touched is `tests/test_alignment_utils.py`, which gains one appended test pinning the
  `R == 1` contract from B4 — no existing assertion was changed.
- The 76 skips are the Triton, CUDA, and JAX tests, which need hardware or optional
  dependencies this environment lacks. Verify them on a CUDA + JAX box.
- `ruff check shepherd_score/ tests/` passes clean.
- The package imports with `triton`, `jax`, and `open3d` absent (all optional/lazy). It now
  **requires `numba`** (the default CPU backend), so numba is no longer in that set.
- Behavior on the default path was checked against a 180-array golden baseline (`vol`,
  `vol_color`, `pharm` over 30 cross-pairs) captured before the final cleanup pass: **every array
  is bit-identical**.

## 9. Known gaps

**Validation**

- **The parity gates for three optimizations are not in the test suite.** The adaptive bucketer,
  the CUDA-graph fine loop, and the fused `vol_color` kernel were validated by standalone scripts
  in the (now removed) `benchmarks/` tree. Nothing in `tests/` currently covers those three paths.
  They should be rewritten as CUDA-gated tests — and the two agent skills in
  [§6](#6-extensibility-the-two-agent-skills) depend on exactly this gate existing.
- **No JAX ↔ Triton/numba cross-backend parity test exists** — and, given the different seed sets
  ([§4.2](#42-a-backend-argument-on-every-batch-aligner)), a naive one would fail. What is needed
  is a *quality* comparison (does each backend recover the same optimum?), not an equality test.
- Also untested: the sparse-ESP stride, the SoA/SVML fp32 kernels, the fused CPU loop's
  basin agreement with eager, and `vol_and_surf_esp`'s shape-only gradient.
- Of the 61 new tests, **28 are skipped in the CI core matrix** without numba/triton/GPU: three
  whole files — `test_screen.py` (15), `test_cpu_pool.py` (4), `test_numba_backend.py` (4) — skip at
  collection on `importorskip("numba")`, `test_fast_batch_alignment.py` skips 4 (three CUDA, one
  numba) and `test_vol_color.py` skips 1. **Fixed:** CI now installs `.[dev,cpu]`, so numba is
  present and 24 of the 28 run on the existing CPU runners; only the ~4 CUDA tests still need a GPU.

**Documentation**

- **`accel/` and `screen.py` have no Sphinx API pages.** They are not wired into `docs/api/`, so
  none of the API in [§5](#5-api-reference) renders on the docs site. (When adding them, set
  `autodoc_mock_imports = ["triton", "numba"]` so the docs build without those installed.)
- The two agent skills ([§6](#6-extensibility-the-two-agent-skills)) are **not implemented**.

**Interface** — the four items below were **fixed in this release**:

- **`ProfileStore` on-disk format version is now validated on open.** `ProfileStore.open` reads
  `manifest["version"]` and raises `ValueError` if it does not match the reader's `VERSION`, so a
  future format bump fails loudly instead of misreading. (The format is still provisional at
  `VERSION = 1`.)
- **`screen(ndev>1)` now raises on `scores_out=`** rather than silently leaving the caller's array
  unwritten (multi-GPU workers return top-K hits only; there is no full-vector path back). `progress`
  is still ignored under `ndev>1`. Both work single-process.
- **`ProfileStore.create(overwrite=True)` now deletes only this store's own files** — its manifest
  and `shard_NNNNN.npz` sequence — instead of every `.npz` in the directory.
- **`shepherd_score.accel.clear_caches()`** frees the process-global caches (`_ALIGN_WORKSPACES`,
  `_INT_BUFFER_CACHE`, `_PAIR_FOOTPRINT_BYTES`, and the captured-graph LRU) so a long-lived process
  that has seen many distinct molecule sizes can reclaim device memory between workloads. They still
  do not shrink on their own.

## 10. Session update: nine new alignment modes via the agent skills

> **Status: uncommitted, on this branch.** Everything below was built in one working session by
> running the two packaged skills ([§6](#6-extensibility-the-two-agent-skills)) end to end for nine
> new modes at once — the largest single exercise of that path so far. The canonical registry now
> has **19 modes** (10 from before this session + these 9). Nothing here has landed on `main`; this
> section documents what exists on the working branch right now, including what is still running.

### What was added

Nine modes, in two families:

**A Tversky scoring option for six modes that lacked one** (`vol`/`vol_esp` already shipped
Tversky variants): `surf_tversky`, `surf_esp_tversky`, `vol_and_surf_esp_tversky` (Tversky on the
shape channel only — the ShaEP-style surface-ESP *agreement* channel is a masked point-to-point
average, not an overlap ratio, so Tversky does not apply to it), `vol_color_tversky`,
`vol_lipo_tversky`, `pharm_tversky`.

**Three genuinely new objectives:**

| Mode | Scores | New per-atom `Molecule` data |
|---|---|---|
| `vol_pharm` | shape + **directional** pharmacophore overlap (a ROCS *ColorTanimoto* analogue — `vol_color` with `directionless=False`) | none (reuses `pharm_vecs`) |
| `vol_atomtype` | shape + a **categorical** Gaussian overlap keyed by element (atomic number) — only same-element atoms contribute | `get_atomic_numbers()`, `get_atomtype_positions()` |
| `vol_mr` | shape + per-atom Crippen **molar refractivity** (polarizability), overlaid like an ESP field (structurally identical to `vol_lipo`, swapping the Crippen `logP` element for the `MR` element of the same `_CalcCrippenContribs` call) | `get_molar_refractivity_contribs()`, `get_molar_refractivity()`, `get_mr_positions()` |

`vol_atomtype` is the only one of the nine that needed a new *kernel* concept: it reuses the
existing directionless-colour Triton/numba kernel (`pharm_color_score_grad_se3_batch`) with a
purpose-built **element-indexed lookup table** (`accel/drivers/vol_atomtype.py::build_element_tables`)
in place of the pharmacophore-type table — no new kernel code, same in-register `dO/dq` kernel the
`vol_color`/`pharm` modes already validate.

### Both skill layers, both run

**Reference layer** (`design-scoring-mode`): a new `score/atomtype_scoring.py` module (the
element-identity overlap primitive), four new eager optimizers in `alignment/_torch.py`
(`vol_atomtype`, `vol_color_tversky`, `vol_lipo_tversky`, `vol_and_surf_esp_tversky`), nine new
`MoleculePair.align_with_<mode>` methods, and the `_ALIGN_KEYS` / export wiring. Validated with the
standard gate set — self-overlap = 1.000, autograd ≡ finite-difference gradient (non-identity pose),
determinism, and the retained-H per-atom basis — in `tests/test_si_modes.py` (26 cases; the 3
surface-mode gates need Open3D and are skipped on a CPU-only box, but their math is exercised via a
synthetic-surface test that doesn't need it).

**Accelerated layer** (`accelerate-scoring-mode`): nine driver files in `accel/drivers/`, nine
`_align_batch_<mode>` functions in `accel/batch/aligners.py`, canonical registration in
`accel/_modes.py` (`MODE_ATTRS`/`MODE_SEEDS`/`MODE_STEPS`, bumping `CANONICAL_MODES` 10 → 19), and
nine `MoleculePairBatch.align_with_<mode>` API methods. Six of the nine reuse an existing driver
outright (`vol_mr` → the `vol_lipo` driver, `surf_tversky`/`surf_esp_tversky` → the
`vol_tversky`/`vol_esp_tversky` drivers over surface data, `pharm_tversky` → the `pharm` driver,
which already accepted `similarity='tversky'`); three needed a new driver (`vol_lipo_tversky`,
`vol_color_tversky` — the Tversky reduction `V_AB / (k·V_AB + C)` proven in `vol_tversky`, applied
to a second channel; `vol_pharm` — the `vol_color` two-channel joint-gradient driver with the
directional pharmacophore kernel swapped in for the directionless one; `vol_and_surf_esp_tversky` —
the `esp_combo` driver with the shape channel's Tanimoto reduction swapped for Tversky).

Parity, measured on an NVIDIA L40S (`node3615`, `pi_melkin`):

- **numba ≡ per-pair reference**: within 8e-5 for every non-surface mode (added to
  `tests/test_new_modes_accel.py`).
- **Triton ≡ numba**: agreement to **~1e-7** for all nine modes (single precision, effectively
  bit-identical) — including the three surface modes, verified with a shared-surface self-pair
  (self-overlap = 1.00000 on both backends; a *separately re-built* surface pair reads ~0.97-0.98,
  which is surface-cloud reproducibility across two independent builds, not a backend or mode bug).
- **Screen (`ProfileStore`) wiring was explicitly *not* done for these nine modes** — that is a
  separate, large per-mode `screen.py` change (§10 of the accel skill's own docs) and was out of
  scope for this session. All nine work through `MoleculePairBatch` (in-memory, GPU/CPU) only; none
  are screen-able from an on-disk `ProfileStore` yet.

Zero regressions: the full suite went from 305 → 317 passing (the new parity cases) with the same
11 pre-existing environment-only failures (`open3d` absent locally) both before and after.

### Benchmarks

Both run on `pi_melkin` (priority partition), against the exact DUDE-Z / speed protocols the
paper's Figures 2–3 use, so the numbers are directly comparable to the shipped baselines:

- **Throughput** (`paper/SI/speed/run_speed.py`, batched `MoleculePairBatch`, Triton): all 19
  modes measured at both ~3k and 100k alignments/mode. The nine new modes screen at throughput
  comparable to their parent mode (e.g. `surf_tversky` ≈ `surf`, `vol_mr` ≈ `vol_lipo`).
- **Accuracy** (`paper/SI/accuracy/run_accuracy.py`, mirrors `fig3_enrichment`'s protocol exactly):
  a 41-target DUDE-Z retrospective screen, leave-one-out over 8 query actives + 3000 decoys per
  target, at each mode's shipped default SE(3) budget. **28 of 41 targets complete as of this
  writing**; the campaign is still running (SLURM array `18646996`, 4-way concurrent).

Full write-up, per-mode math, and the aggregated results (regenerated automatically as the campaign
finishes) live in `paper/SI/` in the sibling `Shepherd-Score-Paper` repo — see
`paper/SI/README.md` and `paper/SI/modes/MODES.md`.
