# What's New

What this package adds to `coleygroup/shepherd-score`, and what an existing caller has to know.
Scoped to the current behaviour of both: upstream at `20ebed7`, this fork at `main`.

> ## ⚠️ Five changes affect existing results or installs
>
> 1. **`MoleculePairBatch.align_with_*` no longer defaults to JAX.** The default is now device-aware
>    — Triton on a CUDA host, numba on CPU — and those backends use a **different SE(3) seed set**
>    than JAX, so a default call returns **different (not worse) scores**. Pass `backend="jax"` for
>    the old numbers. See [§3](#3-backends) and [B5](#b5-the-default-batch-backend-changed).
> 2. **`numba` is a core dependency.** The package does not import without it. See
>    [§1](#1-installation).
> 3. **ESP modes default to gfn2-xTB charges, not MMFF94.** `Molecule(...)` with no
>    `partial_charges` now uses `charge_model="xtb"`, so ESP scores differ. Charges are generated
>    lazily, so shape/colour/pharmacophore work never invokes xTB. Pass `charge_model="mmff"` for
>    the old behaviour. xTB is not uniformly better — see [B8](#b8-esp-charges-default-to-gfn2-xtb).
> 4. **Alignment scores can go UP.** Early stopping is per-pair rather than batch-global, so a pair
>    that used to be cut short by a converged neighbour now finishes its budget. Search effort is
>    unchanged. See [B9](#b9-early-stopping-is-per-pair).
> 5. **`pharm` CPU scores changed.** Its fused CPU loop landed in different optima than the eager
>    one, so it is disabled; `pharm` on CPU now matches the eager reference its GPU and pairwise
>    paths already produced. See [B12](#b12-pharm-cpu-scores-changed).

The fork is additive: no upstream file is deleted and no upstream public name is removed. It also
merges upstream's own `Molecule` refactor and interaction-subselection work, so a fork→upstream
merge is a fast-forward.

---

## Contents

1. [Installation](#1-installation)
2. [The 21 alignment modes](#2-the-21-alignment-modes)
3. [Backends](#3-backends)
4. [The optimizer](#4-the-optimizer)
5. [Virtual screening](#5-virtual-screening)
6. [Parallelism](#6-parallelism)
7. [Surfaces and pharmacophores](#7-surfaces-and-pharmacophores)
8. [API reference](#8-api-reference)
9. [Behavior changes](#9-behavior-changes)
10. [Limitations](#10-limitations)

---

## 1. Installation

```bash
pip install shepherd-score            # core: rdkit, torch, open3d, py3Dmol, numpy, pandas,
                                      #       scipy, molscrub, tqdm, numba
pip install "shepherd-score[gpu]"     # + triton>=3.6   -> backend="triton"
pip install "shepherd-score[jax]"     # + jax           -> backend="jax"  (the OLD default)
```

- **`numba` is core, not an extra.** It is the default CPU backend, so the package no longer
  imports without it. `[cpu]` is kept as a redundant alias.
- **`triton` is optional and lazily imported**, so the package imports without it. `jax` is
  optional too, and is no longer the default backend.
- **`tqdm` moved into core** — upstream imported it at module scope without declaring it.
- **The `gpu` extra has two traps.** It needs a **CUDA build of torch** (the default CPU wheel
  bundles no Triton and will not run the GPU path), and **Triton ≥ 3.6** — the kernels use
  `@triton.autotune(cache_results=…)`, which older Triton lacks, and there is no CPU fallback. A
  known-good pairing is `torch==2.11.0+cu128` with `triton==3.6.0`. Triton ships manylinux wheels
  only; install it explicitly on Linux rather than through an `all`-style extra.
- **One conda `environment.yml` serves CPU and GPU.** It pins the SVML-vectorized numba stack
  (`numba 0.59.1 + llvmlite 0.42 + icc_rt + numpy 1.26`) plus open3d and xtb. SVML makes the CPU
  kernels substantially faster **and changes precision** — it switches the shape/ESP inner loops to
  fp32 structure-of-arrays kernels (values ~1e-6 relative, gradients ~1e-4). Without it the kernels
  are correct but slower and emit a one-time `RuntimeWarning`, so the slow regime is never silent.

---

## 2. The 21 alignment modes

Upstream has six. All 21 here are reachable as `MoleculePair.align_with_<mode>` and
`MoleculePairBatch.align_with_<mode>`, all have Triton and numba backends, and all are registered in
`accel/_modes.py` — which is the single source of truth for their names, result attributes and
per-mode defaults.

Every mode reduces a Gaussian overlap with a Tanimoto `V_AB / (V_AA + V_BB − V_AB)` unless it is a
Tversky variant, with self-overlaps computed through the same kernel at the identity pose.

### The seven core modes

| Mode | Scores | Pose steered by | seeds × steps |
|---|---|---|---:|
| `vol` | Gaussian shape overlap of heavy-atom centers | shape gradient | 10 × 30 |
| `surf` | the same, over surface points | shape gradient | 8 × 40 |
| `vol_esp` | shape overlap, each pair term weighted by `exp(−(C_i − C_j)²/λ)` on partial charges | shape+ESP gradient | 16 × 50 |
| `surf_esp` | the same, over surface points with surface ESP | shape+ESP gradient | 8 × 40 |
| `vol_and_surf_esp` | `esp_weight · ESP_sim + (1 − esp_weight) · shape_sim`, `ESP_sim` a ShaEP-style masked Gaussian of the ESP difference at each surface point | **shape gradient only** | 8 × 60 |
| `pharm` | typed pharmacophore Gaussians (per-type α), same-type only, direction-weighted | pharmacophore gradient | 32 × 50 |
| `vol_color` | `(1 − color_weight) · shape_T + color_weight · color_T`; colour is the pharmacophore Gaussian with direction weighting removed (the ROCS/ROSHAMBO convention) | **joint** | 16 × 40 |

> **`vol_and_surf_esp` descends on shape alone.** The electrostatic term enters the *tracked score*,
> never the derivative — the mode optimizes shape and *reports* shape+ESP. Because the trajectory is
> shape-driven, the ESP score is evaluated every 5th step plus the final step (`_ESP_STRIDE`); set it
> to 1 to score densely.

> **`vol_color`'s two signatures differ.** `MoleculePair.align_with_vol_color` accepts `similarity`,
> `directionless`, `extended_points` and `only_extended`; the batch version accepts none of them.

### Six further per-atom channels

Each pairs atom-centred shape with a second per-atom field, scored
`(1 − w) · shape_T + w · channel_T` on a joint gradient.

| Mode | Second channel | seeds × steps |
|---|---|---:|
| `vol_lipo` | per-atom Crippen logP | 16 × 50 |
| `vol_mr` | per-atom Crippen molar refractivity | 16 × 50 |
| `vol_fukui` | condensed Fukui dual descriptor `f⁺ − f⁻` (three xTB single-points) | 16 × 50 |
| `vol_atomtype` | categorical element identity — a term only for same-element pairs | 16 × 40 |
| `vol_pharm` | **directional** pharmacophores (the directional twin of `vol_color`) | 32 × 50 |
| `vol_avoid` | shape Tanimoto **minus** a hard-sphere excluded-volume penalty against a fixed avoid cloud | 16 × 50 |

> **`vol_avoid` is pairwise-only.** It takes a third, non-molecule input (`avoid_points`), so it is
> not wired into screening and has no worker-process path.

> **`vol_fukui` needs xTB.** `Molecule.fukui` is generated by three gfn2-xTB single points and has
> **no MMFF fallback**, unlike `partial_charges`. Without `xtb` on PATH, pass a precomputed `fukui=`
> array to the constructor or building the mode's inputs fails.

### Eight Tversky variants

`<parent>_tversky` is the parent's overlap reduced asymmetrically:

```
T = AB / (AB + ta·(AA − AB) + tb·(BB − AB))
```

`AA` and `BB` are SE(3)-invariant, so they are precomputed once per pair and only `AB` moves. The
pose is the parent's; what changes is the ranking, which is what asymmetric similarity is for —
substructure-style queries where a small reference should match a large fit.

`vol_tversky` (10 × 40), `vol_esp_tversky` (16 × 50), `surf_tversky` (8 × 40),
`surf_esp_tversky` (8 × 40), `vol_color_tversky` (16 × 40), `vol_lipo_tversky` (16 × 50),
`pharm_tversky` (32 × 50), `vol_and_surf_esp_tversky` (8 × 60).

`pharm` also takes `similarity='tversky' | 'tversky_ref' | 'tversky_fit'` directly. Tversky forfeits
both the CUDA-graph and the fused-CPU fast paths.

### What is wired where

- **All 21** have a pairwise and a batched path.
- **11 are screen-capable and all 11 use the array-native screen path**: the seven core modes plus
  `vol_tversky`, `vol_esp_tversky`, `vol_lipo`, `vol_fukui`.
- **4 have a worker-process path** (`vol`, `surf`, `surf_esp`, `pharm`).

### Renamed modes

`esp` → `surf_esp` and `esp_combo` → `vol_and_surf_esp`. Both legacy names still work everywhere —
method names, result attributes and mode strings — see [B2](#b2-mode-rename).

---

## 3. Backends

`MoleculePairBatch.align_with_*` takes `backend="jax" | "triton" | "numba"`, defaulting to `None`.
Aliases: `"cuda"`/`"gpu"` → triton, `"cpu"` → numba.

- **`None` is device-aware: Triton on a CUDA host, numba on CPU.** JAX is not the default.
- `"triton"` runs hand-written Triton GPU kernels; `"numba"` runs numba CPU kernels.
- `"jax"` runs the original JAX path, **except on the modes that have no JAX kernel**, which fall
  through to the per-pair PyTorch path.

Kernel selection is per call, by tensor device, so one process can run a CPU batch and a GPU batch
and each gets the right kernel.

> #### ⚠️ The default changes scores relative to the old JAX default
>
> Both seeders begin with identity plus 4 PCA-alignment quaternions, but the accelerated one then
> adds up to **six structured ±90° rotations about each reference principal axis** — covering the
> axis *swaps* PCA alignment alone misses — before falling back to a Fibonacci fill. At the shipped
> seed counts those structured seeds absorb most of the budget: at `vol`'s 10 seeds the accelerated
> path emits **zero** Fibonacci rotations where JAX emits five.
>
> The backends explore different orientations and return different scores. `backend=` is **not** a
> pure-performance switch. If you have a pinned baseline, pass `backend="jax"`.

Three backend-specific limits:

- **`no_H=False` is unsupported** on triton/numba for `vol` and `vol_esp` — they align heavy atoms
  only and raise `NotImplementedError`. It still works on jax.
- `vol` and `surf` **ignore `num_repeats`, `trans_init` and `lr`** on triton/numba; the kernels
  re-derive seeds internally. jax is unaffected.
- **`backend="numba"` permanently moves the batch to CPU.** It sets `pair.device = cpu` on every
  pair and never restores it, so a later `backend="triton"` call on the same batch will not use the
  GPU. Rebuild the batch to switch back.

---

## 4. The optimizer

**`num_repeats` and `max_num_steps` default to `None`**, resolving per mode through
`accel/_modes.py` (the tables in [§2](#2-the-21-alignment-modes)). Upstream defaulted to 50 and 200
for every mode. This changes results — see [B1](#b1-num_repeats-and-max_num_steps-defaults).

**There is no coarse grid and no top-k prune on the default path.** Despite the "coarse-to-fine"
naming, every seed goes straight into the fine loop and the per-pair maximum is taken. Ranking seeds
on raw un-optimized overlap repeatedly discarded the true basin for pseudo-symmetric molecules. The
coarse-grid path runs **only** when `trans_init=True`. So cost scales linearly with seed count and
there is no cheap-prune knob.

**Gradients are analytic, not autograd.** Every kernel emits closed-form `dV/dq` and `dV/dt`, so no
autograd graph is built in the fine loop. The pharmacophore and colour kernels emit `dO/dq`
in-register, dropping the rotation-matrix→quaternion projection tail.

**The Adam is not `torch.optim.Adam`.** β₁=0.9, β₂=0.999, ε=1e-8 *inside* the sqrt, **no bias
correction**, a quaternion tangent-space projection fused into the kernel, and unit-quaternion
renormalization every step. `lr` does not mean what it means in `torch.optim.Adam`; the drivers'
internal default is `lr=0.075`, not the `0.1` the public API advertises.

**Early stopping is per-mode and per-pair.** Patience 2 for `vol`/`surf`/`vol_color`, 5 for the ESP
and pharmacophore modes; tolerance 1e-5; checked every 5 steps to avoid a per-step GPU→CPU sync. A
pair has converged when its own best stops improving, and the loop stops only once every pair in the
batch has. A pair's step count therefore still depends on which pairs share its batch, but no pair
is cut short by another converging first.

**CUDA graphs.** Every mode shares one implementation (`accel/drivers/_graphed.py`): one fine step is
captured and replayed, removing per-step host launch overhead. Engagement is not uniform — `pharm`
is graphed only for `tanimoto` with `extended_points=False`, `vol_and_surf_esp` is graphed with
early-stop disabled, and the work budget differs per mode. Whether a given bucket graphs is a
function of its pad shape and the mode's budget, not of allocator state. Captured graphs live in a
bounded LRU of 24 and **pin GPU buffers for the process lifetime**; call
`accel.drivers._graphed.reset_graph_cache()` to free them if you hit fragmentation between runs.

**Triton autotune is cached to disk.** Every kernel autotunes on `(N_pad, M_pad)` with
`cache_results=True`, so the sweep is paid once per machine rather than once per process. A first run
on a new shape looks slow; later runs do not.

### Bucketing and memory

`accel/batch/_bucket.py` groups same-size pairs into padded workspaces. It is **result-identical** by
construction: the kernels are one-CTA-per-pose and mask padding to the real point counts, and seeds
key on the real counts rather than the pad width, so padding two different-sized molecules into one
bucket cannot change a score. Under-occupied buckets merge toward a full CTA wave; on CPU the wave
floor is 1, so occupancy merging is disabled.

`accel/batch/_pad.py` sub-batches each bucket. On CUDA it keeps peak memory under a fraction of free
memory, learns the per-pair footprint per `(device, mode, pad shape, seed count)`, and halves the
chunk and retries on OOM. On CPU there is no device allocator to exhaust, so it applies a fixed cap
(`_CPU_CHUNK_PAIRS = 10,000`) purely to bound host RSS — without it an out-of-core CPU screen holds
a whole 100k-pair batch's intermediates live at once.

---

## 5. Virtual screening

`shepherd_score.screen` is new. It featurizes a library once into an on-disk **`ProfileStore`**
(sharded arrays plus a `manifest.json`), then streams shards through the accelerated aligners against
one or many queries, keeping a top-K heap.

```python
from shepherd_score.container import Molecule
from shepherd_score.screen import ProfileStore, screen

# The store's num_surf_points must match how the Molecules were built.
with ProfileStore.create("lib.store", num_surf_points=200,
                         modes=["surf_esp", "vol"]) as store:
    for rdmol in library:
        store.add(Molecule(rdmol, num_surf_points=200))   # a Molecule, not a raw RDKit Mol

query = Molecule(query_rdmol, num_surf_points=200)
hits = screen(query, ProfileStore.open("lib.store"), mode="surf_esp",
              alpha=0.81, top_k=1000)
# each hit is a Hit(score, id, transform); transform is a 4x4 numpy array
```

`screen_many` streams the library **once** for a list of queries. Both accept `ndev=` to shard across
GPUs and `scores_out=` to write full score vectors (memmap-friendly; single-process only — passing it
with `ndev>1` raises).

**The default shard format is `shard_format="npy"`** — one `.npy` per array, memory-mapped on read.
The single-`.npz`-per-shard format is still written and read (`shard_format="npz"`), but `.npy` skips
the zip decompress and the full-shard materialization.

Things that will bite you:

- **`ProfileStore.add` takes a `Molecule`, not an RDKit `Mol`.** Every molecule in a store must share
  the same `num_surf_points`, and a mismatch fails at flush time — potentially thousands of `add()`
  calls later.
- **The default `dtype="float16"` is lossy** (~0.01 Å, the surface-resampling noise floor). Screen
  scores will differ slightly from a pairwise run for this reason alone. Pass `dtype="float32"` for
  comparable numbers.
- **A store only serves modes it has the arrays for.** `vol` works on any store; `vol_color` needs a
  pharmacophore store; `vol_and_surf_esp` needs surfaces, ESP, charges, radii and with-H centers.
  An unsupported mode raises `ValueError`.
- **Some modes require explicit parameters.** `vol_and_surf_esp` requires `alpha=`, `vol_esp` requires
  `lam=`, and `no_H=False` is rejected — screening is heavy-atom-only. `alpha` auto-fills for
  `surf`/`surf_esp` only.
- **`Hit.transform` is in the pre-centered frame.** On a `pre_centered=True` store (the default),
  library molecules are stored shifted by their own centre of mass and the query is centered too, so
  the 4×4 maps *centered* onto *centered*. Subtract that molecule's COM first, or build with
  `pre_centered=False`.
- **`shard_size` is a GPU-memory knob**, not just an I/O knob: a whole shard is uploaded as device
  tensors at once.
- **A screen holds two shards in host RAM**, the one aligning plus one read ahead on a background
  thread. Size `shard_size` for two resident shards.
- **`trans_init=True`, `backend="jax"`, or a non-pre-centered store** drop you off the fast path onto
  a much slower object path.
- The manifest is rewritten after every shard flush, so **a killed build leaves a readable store**
  containing every completed shard. A store directory is **single-writer**.

`canonical=True` pre-rotates each molecule to its principal axes at build time, which makes the
per-query seed generation a constant set instead of a per-molecule eigensolve. It **changes scores**
(the seed set differs), so it is opt-in and currently applied only where it has been validated.

---

## 6. Parallelism

### Multi-GPU

Alignment is host-bound rather than kernel-bound, so driving N GPUs from one process serialises on the
GIL. The path that scales is one OS process per GPU, exposed explicitly:

```python
from shepherd_score.container import MultiGPUAligner, align_multi_gpu

if __name__ == "__main__":                        # required: align_multi_gpu spawns
    scores, transforms = align_multi_gpu(pairs, "surf", ndev=4, alpha=0.81)

    with MultiGPUAligner(pairs, ndev=4) as pool:  # persistent; reuses each GPU's shard
        scores, transforms = pool.align("surf", alpha=0.81)
```

Both forward `**align_kwargs` straight into `align_with_<mode>`, so a mode's required arguments are
required here too.

This is deliberately **opt-in**: a library must not spawn worker processes behind the user's back,
because `spawn` re-imports the caller's `__main__` and breaks unguarded scripts. A large batch on a
multi-GPU host therefore runs on a **single GPU** and emits a one-time warning (above 4,096 pairs per
GPU) pointing at `MultiGPUAligner`.

- **`align_multi_gpu` always spawns**, so it always needs the `__main__` guard.
- **`MultiGPUAligner` prefers `fork`** and falls back to `spawn` — with a warning — only if CUDA is
  already initialized or Open3D is already imported, since either poisons `fork`+CUDA. Build the pool
  *before* doing CUDA work and the guard is only needed on the spawn fallback.
- **Only `vol`, `surf`, `surf_esp` and `pharm` have a worker-process path.** Every other mode raises
  `ValueError` here. `screen(ndev>1)` is a separate implementation and is not restricted to those
  four.
- Worker threads are capped to `cores // ndev`; the cap is mandatory, since uncapped workers
  oversubscribe and scaling collapses below 1×.
- `Molecule` objects must be picklable; they are what crosses the process boundary.

### CPU

**`num_workers=N` with `backend="numba"`** shards pairs across a persistent pool of single-threaded
worker processes (`accel/cpu_pool.py`). Pairs are independent, so this does not change the
optimization problem — but agreement with one large call is to **convergence tolerance, not
bitwise**, because a pair's step count depends on which pairs share its shard.

- It uses `spawn`, so it needs an `if __name__ == "__main__":` guard.
- It applies to **`vol`, `surf`, `surf_esp` and `pharm` only**. On every other mode it is a silent
  no-op, and it is ignored on CUDA tensors.

**`accel.screen_parallel.screen_parallel(query, library, mode, n_workers=...)`** forks workers for
query-vs-library screening, sharing the featurized library copy-on-write. It is a *different* API
from `screen()`: it takes a RAM-resident list of `Molecule`s rather than a `ProfileStore`, is
CPU/numba-only, and returns a plain list of scores in library order — no `Hit`s, no transforms, no
top-K. Canonical mode names only.

> It **always forks**, so it is POSIX-only. And it must run **before** any in-process numba
> alignment: a live libgomp thread pool at `fork` aborts the child. Featurize, then screen.

**Torch is pinned to one intra-op thread for the duration of a CPU batch alignment**, then restored.
The numba kernels own the cores on that path; left unpinned, torch's pool spin-waits against them.
The setting is scoped rather than global, so it does not reconfigure the caller's torch. If you
already export `OMP_NUM_THREADS=1`, torch was already pinned and nothing changes.

**The SVML CPU kernels change precision, not just speed.** When numba can emit Intel SVML, the fused
CPU loop switches the shape and ESP inner loops to **fp32 structure-of-arrays** kernels. Values agree
to ~1e-6 relative, gradients to ~1e-4. There is no way to turn it off. The fused loop is also not
applied uniformly — `surf_esp` is excluded above 100 padded points (as the most shape-degenerate
mode, the fused trajectory settles in different, equally valid basins), `pharm` is excluded
entirely ([B12](#b12-pharm-cpu-scores-changed)), and the remaining 16 modes have no fused call site
at all.

The numba kernels run with `fastmath=True` and `parallel=True`, using `NUMBA_NUM_THREADS` (defaulting
to *all* cores). That oversubscribes badly if you also use the process pool — which is why
`screen_parallel` pins it to 1.

---

## 7. Surfaces and pharmacophores

### A mesh-free surface generator (opt-in)

`Molecule(..., surface_method="smooth_sdf")` generates the surface point cloud with a smooth-min
signed-distance field plus stochastic sampling, instead of Open3D ball-pivoting + Poisson-disk. It
needs no Open3D and no mesh.

**The default is unchanged.** `surface_method="mesh"` reproduces the original path exactly. The
smooth surfacer rounds the concave atom-border seams, so a model trained on mesh surfaces sees a
**distribution shift** — validate before using it in a generative pipeline.

`"smooth_sdf"` requires `num_surf_points` and **raises `ValueError` if you pass `density=`**. Aliases
`'sdf'`, `'smooth'`, `'fast'` are accepted; an unknown method raises. Tunables (`sdf_s`, `sdf_iters`,
`sdf_knn`, `sdf_jitter`, `even`, `seed`) are exposed on `get_molecular_surface`, with defaults in the
module-level `SMOOTH_SDF_*` constants. It samples 15 candidates per atom against the mesh path's 25.

`shepherd_score.surface_diagnostics` validates a surfacer. Its core quantity is the **shell
residual** — the distance from a point to an atom's *sphere*, not its center:

- `leak_metrics` → how much the cloud leaks atom positions. A point exactly on one sphere makes that
  atom's center directly recoverable. **Lower residual = more leak.** The `mesh` surfacer gives a
  ~0.010 Å median.
- `crimp_points` → points on one sphere *and* near a second: the concave seams.
- `center_recovery_attack` → a strongest-case attack that Kasa-fits a sphere per atom. **Higher error
  = safer.**
- `local_curvature`, `summarize` → per-point non-flatness, and the whole gate in one call.

### Directionless ("color") pharmacophores

`Molecule(..., directionless=True)` and `get_pharmacophores(..., directionless=True)` build
isotropic, zero-vector pharmacophores for every family — the ROCS/ROSHAMBO "color" convention —
rather than oriented feature vectors. Both default to the original behavior.

> **`directionless=True` changes the pharmacophore *count*, not just the vectors.** The
> donor/acceptor multi-vector branches are skipped, so a donor that upstream expanded into one anchor
> *per hydrogen* now emits a single anchor at the feature position.

`feature_set="rdkit_base"` selects RDKit's base feature definitions instead of ShEPhERD's SMARTS set.
It keeps six families and renames three (`PosIonizable`→`Cation`, `NegIonizable`→`Anion`,
`LumpedHydrophobe`→`Hydrophobe`) — **`Halogen` and `ZnBinder` disappear.**

### Lazy Open3D

`generate_point_cloud.py` imports Open3D on first use rather than at module load. Open3D is a slow
import and is fork-hostile — importing it poisons a later `fork`+CUDA, which would break the
fork-based worker pools.

The observable consequence is an improvement: `import shepherd_score.container` now works without
Open3D installed. If it is missing, the error moves from import time to the first surface generation;
its type (`ModuleNotFoundError`) is unchanged.

One caveat: the module global `o3d` is a lazy **proxy object**, not the module. Attribute access
works, but `from shepherd_score.generate_point_cloud import o3d` no longer yields something
`isinstance`/`inspect` treats as a module.

Note also that `import shepherd_score.container` eagerly imports `shepherd_score.accel.batch` and
`shepherd_score.accel.multi_gpu`, so the accel stack loads with the container. No new dependency —
torch was already required — but it is not free.

---

## 8. API reference

### `shepherd_score.container`

```python
__all__ = ["update_mol_coordinates", "Molecule", "MoleculePair", "MoleculePairBatch",
           "Surface", "Pharmacophore", "AlignmentResult",   # upstream's Molecule refactor
           "align_multi_gpu", "MultiGPUAligner"]             # from shepherd_score.accel
```

#### `Molecule.__init__` — five new keyword arguments

```python
Molecule(mol, num_surf_points=None, density=None, probe_radius=None, surface_points=None,
         partial_charges=None,
         charge_model='xtb',          # NEW: 'xtb' | 'mmff' -- CHANGES ESP SCORES, see B8
         electrostatics=None, pharm_multi_vector=None, pharm_types=None,
         pharm_ancs=None, pharm_vecs=None,
         feature_set='shepherd',      # NEW: 'shepherd' | 'rdkit_base'
         directionless=False,         # NEW: isotropic "color" pharmacophores
         surface_method='mesh',       # NEW: 'mesh' | 'smooth_sdf'
         fukui=None)                  # NEW: precomputed Fukui field for vol_fukui
```

Four are default-preserving; `charge_model` is not. Note it sits after `partial_charges` rather than
appended last, so a caller passing more than six positional arguments is affected.

Surface and pharmacophore are stored on `Molecule._surface` (a `Surface`) and
`Molecule._pharmacophore` (a `Pharmacophore`), exposed through `.surface` / `.pharmacophore` and the
unchanged flat `surf_pos` / `surf_esp` / `probe_radius` / `pharm_types` / `pharm_ancs` / `pharm_vecs`
accessors, now property+setter over those dataclasses.

New read helpers: `get_positions(no_H=True)`, `get_charges(no_H=True)`. Per-channel accessors for the
new modes: `get_lipophilicity`, `get_molar_refractivity`, `get_fukui`, `get_atomic_numbers`, and
matching `get_*_positions` that return the strict-heavy centres each channel is 1:1 with.

`get_pharmacophore` gains `feature_set` and `directionless` (fork) plus upstream's `return_atom_ids`
/ `priority_atoms` / `min_ring_priority_atoms`.

#### `MoleculePair` — result attributes

One `(transform, score)` pair per mode, from `MODE_ATTRS`. Two of the 21 keep the historical `_noH`
spelling; the other 19 follow `transform_<mode>` / `sim_aligned_<mode>`.

| Mode | Transform | Score |
|---|---|---|
| `vol` | `transform_vol_noH` | `sim_aligned_vol_noH` |
| `vol_esp` | `transform_vol_esp_noH` | `sim_aligned_vol_esp_noH` |
| every other mode | `transform_<mode>` | `sim_aligned_<mode>` |

`transform_esp`, `sim_aligned_esp`, `transform_esp_combo` and `sim_aligned_esp_combo` remain as
read/write properties forwarding to the canonical names.

#### `MoleculePairBatch`

Every method gains `backend=None` and `return_aligned=False`. `align_with_vol` and
`align_with_vol_esp` additionally gain `alpha=0.81`.

```python
align_with_vol(no_H=True, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None,
               num_workers=1, use_shmap=True, num_buckets=1, verbose=False,
               backend=None, alpha=0.81, return_aligned=False)

align_with_surf(alpha, num_repeats=None, trans_init=False, lr=0.1, max_num_steps=None,
                use_jax=True, use_analytical=True, num_workers=1, use_shmap=False,
                verbose=False, backend=None, return_aligned=False)
```

> **`alpha=` on `align_with_vol` / `align_with_vol_esp` reaches Triton and numba only.** The JAX path
> hardcodes `alpha=0.81`, as upstream did. The default matches, so nothing changes — but passing a
> different `alpha` with `backend="jax"` is silently ignored.

**`return_aligned=False` is the default on the accelerated backends** and skips materializing the
aligned coordinate arrays, returning `[None] * N` in their place. Transforms are still written to
each pair. The JAX path is unaffected. `align_with_pharm` returns a **3-tuple**
`(scores, [None]*N, [None]*N)` where other modes return a 2-tuple, so code indexing `[1]` for arrays
gets `None`s.

**`num_repeats` is accepted and ignored** by `align_with_vol_lipo` and `align_with_vol_color`; their
seed count comes from `MODE_SEEDS`.

### `shepherd_score.screen`

```python
__all__ = ["MoleculeProfile", "ProfileStore", "screen", "screen_many", "Hit"]

ProfileStore.create(path, *, num_surf_points, modes, dtype='float16', shard_size=100_000,
                    pre_centered=True, overwrite=False,
                    canonical=False,        # pre-rotate to principal axes at build time
                    shard_format='npy')     # 'npy' (mmap, default) | 'npz'
        -> ProfileStore
ProfileStore.open(path) -> ProfileStore
    .add(molecule, id=None)          .add_profile(profile, id=None)
    .supports(mode) -> bool          .iter_shards() -> Iterator[List[MoleculeProfile]]
    .read_shard(idx) -> tuple        .read_profiles(idx) -> List[MoleculeProfile]
    .close()

screen(query, store, mode='surf_esp', *, backend=None, do_center=None, top_k=1000, ndev=None,
       scores_out=None, alpha=None, progress=False, **align_kwargs) -> List[Hit]

screen_many(queries, store, mode='surf_esp', *, ...) -> List[List[Hit]]
```

`MoleculeProfile` fields: `atom_pos`, `atom_pos_noH`, `surf_pos`, `surf_esp`, `partial_charges`,
`radii`, `_nonH_atoms_idx`, `pharm_types`, `pharm_ancs`, `pharm_vecs`, `lipo_pos`, `lipophilicity`,
`fukui_pos`, `fukui`, `num_surf_points`, `mol`, `id`, `rot`. A store only materializes the arrays its
`modes=` need. The on-disk format carries `VERSION = 1`, validated on open, and should be treated as
provisional.

### `shepherd_score.accel`

```python
__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner", "clear_caches"]

has_triton() -> bool
clear_caches() -> None      # free the process-global accel caches

align_multi_gpu(pairs, mode, *, ndev=None, threads=None, backend='triton', do_center=False,
                write_back=True, return_timing=False, **align_kwargs)

class MultiGPUAligner:
    __init__(pairs, *, ndev=None, threads=None, do_center=False, start_method=None)
    align(mode, *, backend='triton', return_timing=False, **align_kwargs)
    close()

screen_parallel(query, library, mode, n_workers=None, **align_kwargs)   # accel.screen_parallel
```

#### The mode registry — `accel._modes`

```python
CANONICAL_MODES     = (...)   # the 21, in public order
LEGACY_MODE_ALIASES = {'esp': 'surf_esp', 'esp_combo': 'vol_and_surf_esp'}
PROCESS_MODES       = ('vol', 'surf', 'surf_esp', 'pharm')
MODE_ATTRS          = {mode: (transform_attr, score_attr)}
MODE_SEEDS          = {mode: int}     # what num_repeats=None resolves to
MODE_STEPS          = {mode: int}     # what max_num_steps=None resolves to
canonical(mode) -> str                # resolve a legacy name; unknown names pass through
```

#### Effort recorder — `accel._stats`

A no-op until armed, so the library pays nothing by default:

```python
from shepherd_score.accel import _stats
_stats.reset()                      # clear + enable
batch.align_with_vol()
_stats.summary()                    # {'calls', 'graphed', 'steps_min', 'steps_max', 'steps_mean',
                                    #  'steps_configured', 'early_stop_frac', ...}
_stats.disable()
```

`graphed` counts how many fine-loop calls ran the CUDA-graph path rather than the eager one. It is
process-local and not thread-safe, so a forked or spawned worker (`cpu_pool`, `screen_parallel`,
`screen(ndev>1)`) records only in its own process — **an empty summary means unmeasured, not full
effort.** The reference per-pair optimizers in `alignment/_torch.py` and the JAX path do not record,
so measurements must go through `MoleculePairBatch.align_with_*` or `screen()`.

#### Tunable module-level constants

Not env vars — constants you can edit or monkey-patch.

| Constant | Module | Default | Effect |
|---|---|---|---|
| `_ESP_STRIDE` | `accel/drivers/esp_combo.py` | 5 | ESP re-scoring interval; 1 = dense |
| `VOL_COLOR_FUSED_MAX_PAD` | `accel/kernels/vol_color_triton.py` | 32 | Above this pad, the fused vol_color kernel is not used |
| `_GRAPH_WORK_BUDGET` | `accel/drivers/_graphed.py` | 300,000,000 | Default pose budget for graph engagement |
| `_GRAPH_CAP_CEIL` / `_GRAPH_CAP_MIN` | `accel/drivers/_graphed.py` | 262,144 / 2,000 | Hard bounds on the graph pose cap |
| `_GRAPH_CACHE_MAX` | `accel/drivers/_graphed.py` | 24 | Live captured graphs (each pins GPU buffers) |
| `_FINE_CHUNK_POSES` | `accel/batch/_pad.py` | 81,920 | Pose cap per fine-loop sub-batch on the vol screen path |
| `_CPU_CHUNK_PAIRS` | `accel/batch/_pad.py` | 10,000 | Pairs per call on the CPU path; bounds host RSS |
| `_BAND` | `accel/batch/_pad.py` | 16 | Legacy fixed-band pad granularity |
| `SMOOTH_SDF_*` | `generate_point_cloud.py` | — | Smooth-SDF surfacer defaults |

### Surface generation and scoring

```python
get_molecular_surface(centers, radii, num_points=None, num_samples_per_atom=None,
                      probe_radius=1.2, ball_radii=[1.2],
                      method='mesh',        # NEW: 'mesh' | 'smooth_sdf'
                      sdf_s=10.0, sdf_iters=6, sdf_knn=8, sdf_jitter=0.0,
                      even='fps', seed=None) -> np.ndarray
get_molecular_surface_smooth_sdf(...) -> np.ndarray                      # NEW

# surface_diagnostics (new): numpy + scipy only, no Open3D
leak_metrics(points, centers, radii, probe_radius=1.2) -> Dict[str, float]
crimp_points(points, centers, radii, probe_radius=1.2, on_shell_tol=0.1, seam_tol=0.35)
center_recovery_attack(points, centers, radii, probe_radius=1.2, min_pts=8) -> Dict[str, float]
local_curvature(points, k=12) -> np.ndarray
summarize(points, centers, radii, probe_radius=1.2) -> Dict[str, float]

# shepherd_score.alignment: 16 new exports -- an objective_<mode>_overlay and an
# optimize_<mode>_overlay for each of vol_color, vol_tversky, vol_esp_tversky, vol_lipo,
# vol_atomtype, vol_color_tversky, vol_lipo_tversky, vol_and_surf_esp_tversky.
# These eager torch references are what the accelerated backends are checked against.

# new kwargs; defaults preserve behavior
get_overlap_pharm(..., directionless=False)
get_pharm_combo_score(..., color_weight=0.5, directionless=False)
get_pharmacophores(mol, multi_vector=True, exclude=[], check_access=False, scale=1.0,
                   feature_set='shepherd', directionless=False,
                   return_atom_ids=False, priority_atoms=None, min_ring_priority_atoms=3)
build_lookup_tables(..., directionless=False)          # analytical_gradients._torch
```

`get_pharm_combo_score`'s combination changed from `(pharm + shape) / 2` to
`(1 - color_weight) * shape + color_weight * pharm`. At the default `color_weight=0.5` these are
**exactly** equal in IEEE-754, so the result is bit-identical.

`get_pharmacophores` now returns a `Pharmacophore` that unpacks as the old `(types, positions,
vectors)` 3-tuple, so `X, P, V = get_pharmacophores(mol)` still works.

### Upstream's refactor and subselection, integrated

This release merges upstream through `20ebed7`, and the following are new public API a user of the
merged package gets:

- **`Surface`** — `positions` / `esp` / `probe_radius`. Backs `Molecule.surf_pos` / `surf_esp` /
  `probe_radius`, reachable via `Molecule.surface`.
- **`Pharmacophore`** — `types` / `positions` / `vectors`, plus `mol` / `atom_ids` / `labels`.
  Unpacks as the old 3-tuple. Reachable via `Molecule.pharmacophore`.
- **`AlignmentResult`** — one `(score, transform)` per mode. `MoleculePair` stores results in an
  `_alignments` dict of these.
- **`Molecule.select_atoms(atom_indices, ...)`** returns a new `Molecule` whose surface, ESP and
  pharmacophore are restricted to an atom subset — the binding-site workflow. It expands the seed set
  to the pharmacophore-consistent superset by default, can decouple which atoms define retained
  pharmacophores from which the surface is regenerated over (`surface_atom_indices=`), and can pull
  in bonded hydrogens (`include_h=True`). On this fork it also carries `charge_model`,
  `surface_method` and `fukui` through to the returned `Molecule`.
- **`Molecule.get_pc(atom_indices=None, radial_buffer=0.0)`** and
  **`get_electrostatic_potential(atom_indices=None, surf_pos=None)`** gained the parameters that make
  that possible. Both defaults are the previous behavior.
- **`Pharmacophore.expand_atom_selection` / `.subset_to_atoms`**, and priority-pharmacophore support:
  `return_atom_ids`, `priority_atoms`, `min_ring_priority_atoms`, and
  `Pharmacophore.priority_labels(...)` for lazy label computation over any atom set.

### Extensibility

Two agent skills ship in `.claude/skills/` — `design-scoring-mode` (define a representation and an
overlap objective, then wrap it in the per-pair Adam optimizer that becomes the gradient oracle) and
`accelerate-scoring-mode` (port that objective to a fused Triton kernel and its numba twin, wire them
into the per-call device dispatcher, and gate on parity against the oracle). The 14 modes beyond the
original seven were built through them. Each carries a small eval suite covering the cases that broke
in practice: reusing an existing kernel, adding a new channel kernel, and the screening-wiring tier.

To add a mode: register it in `accel/_modes.py` (name, result attributes, seed and step defaults) and
add a driver. The batch layer and the screening front-end pick it up from the registry;
`tests/test_mode_registry.py` fails if the registry and the torch-typed `_MODE_SPEC` drift apart.

---

## 9. Behavior changes

Twelve changes are visible to code written against upstream. **B5 (default backend), B6 (numba
required), B8 (xTB charges), B9 (per-pair early stopping) and B12 (pharm CPU scores) are the ones
most likely to affect you.**

### B1. `num_repeats` and `max_num_steps` defaults

On all six pre-existing alignment methods, on both `MoleculePair` and `MoleculePairBatch`:

```python
num_repeats:   int = 50   ->  Optional[int] = None
max_num_steps: int = 200  ->  Optional[int] = None
```

`None` resolves per mode (see [§2](#2-the-21-alignment-modes)). **This changes results**, and not
merely as "fewer restarts": `alignment/_torch.py` and `_jax.py` contain a special case where
`num_repeats == 50` loads a precomputed 45-point Fibonacci quaternion set, and any other value takes
a branch that produces a **different set of orientations**, not a subset. The old default of exactly
50 hit the precomputed branch; none of the new defaults do.

Individual pairs move by up to a few percent. The aggregate cost is small — these defaults sit at the
accuracy/throughput knee, where retrospective-screening ROC-AUC plateaus at low seed counts — but it
is not a no-op on any call.

**To restore the previous behavior exactly**, pass the old values:

```python
pair.align_with_vol(num_repeats=50, max_num_steps=200)
```

Do this if you have a pinned regression baseline, a golden-file test, or a published score table.

### B2. Mode rename

`esp` → `surf_esp`, `esp_combo` → `vol_and_surf_esp`, across method names, result attributes and
every mode string (the screen front-end, the CPU pool, the multi-GPU driver).

**Input-side compatibility is complete.** The legacy method names are aliases to the same function
objects, the legacy result attributes are read/write properties, and every mode string is
canonicalized at the entry point. Code that *names* these things keeps working.

**Output-side compatibility is not.** The canonical attributes are the real instance storage; the
legacy names exist only as class-level properties:

```python
pair.transform_esp                  # works
'transform_esp' in pair.__dict__    # now False
vars(pair)                          # now shows transform_surf_esp
```

Code that introspects `__dict__` rather than naming the attribute will see the new names.

### B3. Old pickles round-trip

The `Surface`/`Pharmacophore`/`AlignmentResult` refactor turned the flat attributes into data
descriptors, which take precedence over the instance dict. A pickle from an earlier release restores
its flat keys and then raises `AttributeError` on the first read of any of them — six attributes on
`Molecule`, every `transform_<mode>`/`sim_aligned_<mode>` on `MoleculePair`.

**Both classes define `__setstate__`, so old pickles work.** The hook remaps the flat keys into
`Surface` / `Pharmacophore` / `AlignmentResult`, honours the renamed-mode aliases, and defaults the
state introduced *after* the flat layout, which no old pickle can carry: `surface_method` (read by
`get_pc()`), `_charge_model` (read by `partial_charges` when the pickle carried no charges) and
`_fukui`. New-format pickles are unaffected; stale flat duplicates are stripped either way so a
shadowed copy cannot drift from the canonical entry.

### B4. `apply_SE3_transform` collapses a singleton batch

Reimplemented with `baddbmm`. For a batch of exactly one:

```python
apply_SE3_transform(points[None], transform[None])   # was (1, N, 3), now (N, 3)
```

Values are unchanged; only the shape collapses. No call site inside the library is affected, and the
batched and gradient paths remain bitwise identical. Two caveats:

- **`apply_SO3_transform` was not given the same collapse.** With an `R == 1` batch the two return
  different ranks, and upstream pairs them on adjacent lines in three places. This mis-broadcasts
  silently rather than failing loudly.
- The single-instance path differs by about **one float32 ULP** from the old `(R @ P.T).T + t`.
  Harmless numerically, but it breaks bit-for-bit reproduction of golden values on the
  `num_repeats == 1` path.

`se3.py` also gains `quaternions_to_SE3_batch(q, t) -> (K, 4, 4)`, used by all the batched aligners.

### B5. The default batch backend changed

`MoleculePairBatch.align_with_*` no longer defaults to JAX. The default is device-aware — Triton on
CUDA, numba on CPU — and those backends use a different SE(3) seed set, so a default call returns
**different scores** ([§3](#3-backends)).

```python
batch.align_with_surf(alpha)                    # was JAX; now Triton/numba -> different scores
batch.align_with_surf(alpha, backend="jax")     # explicit -> the old numbers
```

This affects `MoleculePairBatch` only; per-pair `MoleculePair.align_with_*` still defaults to the
torch path. The change also fixes a latent bug: the old JAX default was not installable by default,
since `jax` is an optional extra, so a fresh install hitting the default batch path raised
`ImportError`.

### B6. `numba` is a required dependency

It is the default CPU backend, so the package no longer imports without it. If you relied on
`import shepherd_score` working in a numba-free environment, install numba or pin the previous
release.

### B7. Pharmacophore scoring: `directional` → `directionless`

The scoring kwarg `directional` (on `get_overlap_pharm` / `get_pharm_combo_score`, forwarded through
`align_with_vol_color`) is **renamed to `directionless` with inverted meaning**, so extraction and
scoring share one polarity (`directionless=True` = orientation-blind). There is no alias —
`directional=` was fork-only — so update any call: `directional=False` → `directionless=True`.

### B8. ESP charges default to gfn2-xTB

`Molecule(...)` with no `partial_charges` used to fall back to MMFF94. It now takes `charge_model`,
defaulting to `"xtb"`, so a default-constructed molecule's ESP modes score on xTB charges —
**different ESP scores**. Shape, colour and pharmacophore modes are unaffected; they never read
charges.

```python
Molecule(rd, num_surf_points=200)                      # gfn2-xTB charges (new default)
Molecule(rd, num_surf_points=200, charge_model="mmff") # the old MMFF94 behaviour
Molecule(rd, num_surf_points=200, partial_charges=q)   # explicit charges (unchanged)
```

Two things make this safe rather than a performance cliff:

- **Lazy.** Charges are generated the first time they are read, then cached. A molecule used only for
  shape / colour / pharmacophore never invokes the xTB subprocess, so screening throughput for those
  modes is unchanged. Constructing *with a surface* does build its surface ESP eagerly and so does
  pay the charge cost; pass `charge_model="mmff"` for a fast pure-surface-shape prep.
- **Graceful fallback.** If `xtb` is missing or fails to converge, it falls back to MMFF94 with a
  `RuntimeWarning` rather than crashing. `xtb` is recommended, not a hard dependency.

**xTB is not uniformly better.** On retrospective enrichment it helps the surface-ESP modes and
*hurts* atom-centred `vol_esp`, which does better on MMFF94. Surface ESP rewards a faithful field;
atom-centred ESP suits atom-centred charges. Choose `charge_model` per mode.

### B9. Early stopping is per-pair

Every accelerated fine loop used to decide when to stop from a **maximum over the whole batch**, so a
pair that converged early — most cheaply a molecule aligned against itself, but any easy pair does it
— pinned that maximum and halted optimization for every other pair sharing its bucket. With patience
2 and a check every 5 steps the floor was 11 executed steps, whatever the configured budget.

The criterion is now **per pair**: a pair improves when its own best gains more than
`early_stop_tol`, and the loop breaks only after `early_stop_patience` consecutive checks in which
**no** pair improved.

**This changes results, and only upward.** A pair that was cut short now finishes its budget and
lands on a better or identical optimum.

- **Search effort is unchanged.** `MODE_SEEDS`, `MODE_STEPS`, every patience and every tolerance are
  as before. The extra work is work that was already configured and being skipped.
- **Early stopping is not disabled.** A bucket in which every pair genuinely converges still stops
  early.
- **Throughput drops on any workload that was truncating**, and that is the honest direction. **Any
  score table, throughput number or enrichment figure produced before this was computed on truncated
  optimizations and must be re-measured, not re-labelled** — including, because truncation depended
  on which pairs shared a bucket, results that were never reproducible in the first place.
- **On a large screening bucket early stopping now effectively never fires**, since thousands of
  pairs must stall on the same check. Those buckets run the full budget; the check itself is cheap
  but can no longer terminate anything.
- The per-pair rule is *not* provably never-earlier. A trajectory whose per-check gains straddle
  `early_stop_tol` can spend a baseline reset the global rule would still be holding, and stop one
  5-step block sooner. It has not been observed on real workloads, only in a generator built to
  provoke it.

### B10. `screen(ndev>1)` returned the wrong mode's answers

The multi-GPU screen worker dispatched the `vol` array builder and aligner for **every** mode, so any
mode whose reference tensors happen to satisfy `vol`'s contract silently ran `vol` and returned its
answer under another mode's name; `pharm`, whose reference dict has no shared key, raised `KeyError`
instead. The worker and the in-process driver now share one dispatch selector, so a mode added to the
tables lights up on both paths.

**Any `screen(ndev>1)` result for a non-`vol` mode produced before this is invalid.**

### B11. The screening host path is faster, and results are reproducible

The screen reduce no longer touches Python once per library molecule, all 11 screen-capable modes
align straight from the store's contiguous arrays rather than through per-molecule objects, and the
next shard is read on a background thread while the current one aligns. **Hit lists, hit order
(including exact score ties), ids, transforms and `scores_out` are unchanged** — this is a front-end
refactor with no arithmetic in it.

Two reproducibility problems are also gone. Sub-batch sizing used to be derived from *free* device
memory, so whether a bucket's pose count landed under the CUDA-graph budget depended on what the
allocator happened to be holding — and the graph and eager paths do not always agree, so the same
screen could return different scores on different runs. A fixed pose cap makes that decision a
function of the bucket's shape. Separately, the per-pair byte-footprint calibration charged a chunk
for memory *other* objects held, which under memory pressure collapsed the chunk size to one pair;
it now charges only its own allocation growth.

**Screen scores from a release before this may not reproduce even against themselves.** Scores from
this release are stable.

### B12. `pharm` CPU scores changed

`pharm`'s fused numba CPU loop and its eager loop land in **different optima**. The fused loop
applies its Adam tail before the early-stop check, so an N-iteration run is N evaluations *and* N
updates where the eager loop interleaves them the other way; with 32 seeds — the most of any mode —
and a strongly multi-basin objective, that relocates the optimum rather than perturbing it.

The fused path is therefore **disabled for `pharm`**, which is the same standard `surf_esp` is already
excluded under. `pharm` on CPU now produces the eager reference its GPU and pairwise paths already
produced, at a few percent less throughput.

**Any `pharm` CPU score produced before this came from the fused trajectory.**

### Minor

- `align_with_vol_esp(lam=...)` — `lam` was a required positional; it now defaults to `0.1`. Strictly
  widening, so no existing call breaks.
- `MoleculePair.__init__` eagerly allocates two torch tensors on the target device, so constructing a
  pair touches GPU memory before any `align_*` call.
- `get_overlap_pharm(directionless=True)` raises `ValueError` if combined with
  `precomputed_self_overlaps`, and silently forces `extended_points` / `only_extended` off. The numpy
  variant forces the flags off without raising. Neither affects the `directionless=False` default.
- `get_pharm_combo_score`'s combination changed as described in [§8](#8-api-reference); bit-identical
  at the default `color_weight=0.5`.

The `backend=`-specific limits — the different seed set, `no_H=False`, the ignored
`num_repeats`/`trans_init`/`lr` on `vol`/`surf`, and the permanent CPU move under `numba` — are in
[§3](#3-backends), since they affect only the new backends.

---

## 10. Limitations

- **The JAX path is the least exercised surface.** Almost every skipped test in a routine run is a
  JAX test, so a normal run validates none of it. Exercise it before relying on `backend="jax"` as
  the reproducibility escape hatch [B5](#b5-the-default-batch-backend-changed) advertises.
- **No JAX ↔ Triton/numba cross-backend parity test exists**, and given the different seed sets a
  naive equality test would fail. What is needed is a *quality* comparison — does each backend
  recover the same optimum — not an equality one.
- **Ten of the 21 modes have never been benchmarked**: `vol_mr`, `surf_tversky`, `surf_esp_tversky`,
  `vol_lipo_tversky`, `vol_color_tversky`, `vol_atomtype`, `vol_pharm`, `pharm_tversky`,
  `vol_and_surf_esp_tversky`, `vol_avoid`. They are correctness-tested, not performance-characterised,
  and several carry per-mode tuning constants inherited from a sibling rather than measured for
  themselves.
- **Nine modes are not screen-capable.** `screen.ProfileStore.supports()` has no branch for them even
  against a full-schema store, so they are pairwise/batch only. `vol_avoid` is pairwise-only by
  design.
- **`accel/` and `screen.py` have no Sphinx API pages.** They are not wired into `docs/api/`, so none
  of [§8](#8-api-reference) renders on the docs site. When adding them, set
  `autodoc_mock_imports = ["triton", "numba"]` so the docs build without those installed.
- **Bit-identity results come from non-early-stopping workloads.** The parity checks behind the
  screen path and the backend comparisons were taken where the fine loop ran its full budget. A very
  small chunk or a low-diversity library could fire the early stop and re-open a divergence between
  the graph and eager paths.
- **`vol_and_surf_esp`'s graph and eager paths compute different algorithms.** The eager loop applies
  `_ESP_STRIDE`; the captured graph step does not, so it scores and best-tracks every step and returns
  uniformly higher scores. The streaming screen runs the eager path. Which is intended is unsettled,
  so treat pairwise-batch scores for this mode as not interchangeable with screen scores.
