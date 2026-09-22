# What's New

What this package adds to `coleygroup/shepherd-score`, and what an existing caller must change.
Upstream at `20ebed7`, this fork at `main`.

> ## ⚠️ Six changes affect existing results or installs
>
> 1. **`align_with_*` on a batch no longer defaults to JAX** — it is device-aware (Triton on CUDA,
>    numba on CPU), and those backends use a different SE(3) seed set, so a default call returns
>    different scores. Pass `backend="jax"` for the old numbers. [§3](#3-backends) · [B5](#b5-the-default-batch-backend-changed)
> 2. **`numba` is a core dependency** — the package does not import without it. [§1](#1-installation)
> 3. **ESP modes default to gfn2-xTB charges, not MMFF94**, so ESP scores differ. `charge_model="mmff"`
>    restores the old behaviour. [B8](#b8-esp-charges-default-to-gfn2-xtb)
> 4. **Alignment scores can go UP** — early stopping is per-pair, so a pair is no longer cut short by
>    a converged neighbour. Search effort is unchanged. [B9](#b9-early-stopping-is-per-pair)
> 5. **`pharm` CPU scores changed** — its fused CPU loop found different optima, so it is disabled.
>    [B12](#b12-pharm-cpu-scores-changed)
> 6. **Profile stores that serve any atom-cloud-seeded mode are canonical by default** — `vol`,
>    `vol_color`, `vol_esp`, `vol_lipo`, `vol_fukui`, the volumetric Tversky modes and
>    `vol_and_surf_esp`. Such a store holds each molecule in its principal-axis frame, and a screen in
>    those modes runs one constant seed set instead of a per-molecule eigensolve, so their screening
>    scores differ slightly (1e-3 on average) from a store built before; existing stores are
>    unchanged. `ProfileStore.create(..., canonical=False)` restores the old layout. [B13](#b13-profile-stores-are-canonical-by-default)

The fork is additive — no upstream file deleted, no upstream public name removed — and it merges
upstream's own `Molecule` refactor and interaction-subselection work, so a fork→upstream merge is a
fast-forward.

**Contents** — [1 Installation](#1-installation) · [2 Modes](#2-the-21-alignment-modes) ·
[3 Backends](#3-backends) · [4 Optimizer](#4-the-optimizer) · [5 Screening](#5-virtual-screening) ·
[6 Parallelism](#6-parallelism) · [7 Surfaces](#7-surfaces-and-pharmacophores) · [8 API](#8-api-reference) ·
[9 Behavior changes](#9-behavior-changes) · [10 Limitations](#10-limitations)

## 1. Installation

```bash
pip install shepherd-score            # core: rdkit, torch, open3d, py3Dmol, numpy, pandas,
                                      #       scipy, molscrub, tqdm, numba, threadpoolctl
pip install "shepherd-score[gpu]"     # + triton>=3.6, torch>=2.6 -> backend="triton"
pip install "shepherd-score[jax]"     # + jax           -> backend="jax"  (the OLD default)
```

- **`numba` is core, not an extra** — it is the default CPU backend. `[cpu]` is a redundant alias.
- **`triton` and `jax` are optional and lazily imported.** Neither blocks import.
- **`tqdm` moved into core** — upstream imported it without declaring it.
- **The `gpu` extra needs a CUDA torch build, Triton ≥ 3.6 and torch ≥ 2.6.** The default CPU wheel
  bundles no Triton; older Triton lacks `@triton.autotune(cache_results=…)` and older torch lacks
  `torch.cuda.graph(capture_error_mode=…)`, which the graphed fine loop passes. Neither degrades — a
  non-OOM capture failure re-raises. The base `torch>=1.12` floor stands because nothing on the CPU
  path reaches capture. Known-good: `torch==2.11.0+cu128` with `triton==3.6.0`; Triton is
  manylinux-only, so install it explicitly.
- **One conda `environment.yml` serves CPU and GPU.** It pins the SVML numba stack
  (`numba 0.59.1 + llvmlite 0.42 + icc_rt + numpy 1.26`) plus open3d and xtb. SVML is much faster
  **and changes precision** — it switches the shape/ESP inner loops to fp32 structure-of-arrays
  kernels (values ~1e-6 relative, gradients ~1e-4). Without it they are correct but slower, and emit
  a one-time `RuntimeWarning`, so the slow regime is never silent.

## 2. The 21 alignment modes

Upstream has six. All 21 are reachable as `MoleculePair.align_with_<mode>` and
`MoleculePairBatch.align_with_<mode>`, all have Triton and numba backends, and all are registered in
`accel/_modes.py` — the single source of truth for names, result attributes and per-mode defaults.

Every mode reduces a Gaussian overlap with a Tanimoto `V_AB / (V_AA + V_BB − V_AB)` unless it is a
Tversky variant, with self-overlaps computed through the same kernel at the identity pose.

### Seven core modes

| Mode | Scores | Pose steered by | seeds × steps |
|---|---|---|---:|
| `vol` | Gaussian shape overlap of heavy-atom centers | shape | 10 × 30 |
| `surf` | the same, over surface points | shape | 8 × 40 |
| `vol_esp` | shape overlap, each pair term weighted by `exp(−(C_i − C_j)²/λ)` on partial charges | shape+ESP | 16 × 50 |
| `surf_esp` | the same, over surface points with surface ESP | shape+ESP | 8 × 40 |
| `vol_and_surf_esp` | `esp_weight · ESP_sim + (1 − esp_weight) · shape_sim`; `ESP_sim` is a ShaEP-style masked Gaussian of the ESP difference per surface point | **shape only** | 8 × 60 |
| `pharm` | typed pharmacophore Gaussians (per-type α), same-type only, direction-weighted | pharmacophore | 32 × 50 |
| `vol_color` | `(1 − w) · shape_T + w · color_T`; colour is the pharmacophore Gaussian with direction weighting removed (the ROCS/ROSHAMBO convention) | **joint** | 16 × 40 |

> **`vol_and_surf_esp` descends on shape alone** — the electrostatic term enters the tracked score,
> never the derivative. Because the trajectory is shape-driven, ESP is evaluated every 5th step plus
> the last (`_ESP_STRIDE`); set it to 1 to score densely.

> **`vol_color`'s two signatures differ.** The `MoleculePair` version accepts `similarity`,
> `directionless`, `extended_points` and `only_extended`; the batch version accepts none of them.

### Six further per-atom channels

Atom-centred shape plus a second per-atom field, scored `(1 − w) · shape_T + w · channel_T` on a
joint gradient.

| Mode | Second channel | seeds × steps |
|---|---|---:|
| `vol_lipo` | per-atom Crippen logP | 16 × 50 |
| `vol_mr` | per-atom Crippen molar refractivity | 16 × 50 |
| `vol_fukui` | condensed Fukui dual descriptor `f⁺ − f⁻` | 16 × 50 |
| `vol_atomtype` | element identity — a term only for same-element pairs | 16 × 40 |
| `vol_pharm` | **directional** pharmacophores (the directional twin of `vol_color`) | 32 × 50 |
| `vol_avoid` | shape Tanimoto **minus** a hard-sphere excluded-volume penalty vs a fixed avoid cloud | 16 × 50 |

> **`vol_avoid` takes a third, non-molecule input** (`avoid_points`). It belongs to the QUERY, not
> to a library molecule, so a screen passes it as `screen(..., avoid_points=...)` rather than storing
> it per molecule. **`vol_fukui` needs xTB**: `Molecule.fukui` comes
> from three gfn2-xTB single points and has **no MMFF fallback**, unlike `partial_charges`, so without
> `xtb` on PATH you must pass a precomputed `fukui=` array.

### Eight Tversky variants

`<parent>_tversky` is the parent's overlap reduced asymmetrically,
`T = AB / (AB + ta·(AA − AB) + tb·(BB − AB))`. `AA` and `BB` are SE(3)-invariant, so only `AB` moves.
The pose is the parent's; the ranking changes — which is the point, for substructure-style queries
where a small reference should match a large fit. `vol_tversky` (10 × 40), `vol_esp_tversky` (16 × 50),
`surf_tversky` and `surf_esp_tversky` (8 × 40), `vol_color_tversky` (16 × 40), `vol_lipo_tversky`
(16 × 50), `pharm_tversky` (32 × 50), `vol_and_surf_esp_tversky` (8 × 60). `pharm` also takes
`similarity='tversky'|'tversky_ref'|'tversky_fit'` directly. Tversky keeps BOTH fast paths: the
fused CPU loop takes its reduction from the mode registry now, so the Tanimoto-only restriction that
used to exclude these modes is gone.

### Wiring, and the rename

**All 21 have every path.** Each mode is a `ModeSpec` in `accel/_modes.py` — the channels it reads,
the terms it optimises, how they reduce and blend, and its schedule — and one generic engine runs all
of them, so the batched aligner, the CUDA-graph and fused-CPU fine loops, the array-native screen,
the multi-GPU screen and the worker-process path are all derived from that spec rather than written
per mode. Where the counts stood before: worker-process path 4 → 21, fused CPU loop 4 → 19 (the
pharmacophore family opts out, [B12](#b12-pharm-cpu-scores-changed)), screen-capable 11 → 21,
array-native screen 11 → 21, canonical-store constant seeds 8 → 15.

`esp` → `surf_esp` and `esp_combo` → `vol_and_surf_esp`; both legacy names still work everywhere
([B2](#b2-mode-rename)).

## 3. Backends

`MoleculePairBatch.align_with_*` takes `backend="jax" | "triton" | "numba"`, defaulting to `None`.
Aliases: `"cuda"`/`"gpu"` → triton, `"cpu"` → numba.

**`None` is device-aware: Triton on CUDA, numba on CPU**, selected per call by tensor device, so one
process can run a CPU batch and a GPU batch and each gets the right kernel. `"jax"` runs the original
path except on modes with no JAX kernel, which fall through to the per-pair PyTorch path.

> #### ⚠️ The default changes scores relative to the old JAX default
>
> Both seeders start with identity plus 4 PCA-alignment quaternions, but the accelerated one then adds
> up to **six structured ±90° rotations about each reference principal axis** — covering the axis
> *swaps* PCA alignment alone misses — before falling back to a Fibonacci fill. At the shipped seed
> counts those absorb most of the budget: at `vol`'s 10 seeds the accelerated path emits **zero**
> Fibonacci rotations where JAX emits five. `backend=` is **not** a pure-performance switch. If you
> have a pinned baseline, pass `backend="jax"`.

Three backend-specific limits. **`no_H=False` is unsupported** on triton/numba for `vol` and
`vol_esp` — heavy atoms only, raising `NotImplementedError`; it still works on jax. **`vol` and `surf`
ignore `num_repeats`, `trans_init` and `lr`** on triton/numba, since the kernels re-derive seeds
internally. And **`backend="numba"` permanently moves the batch to CPU** — it sets `pair.device = cpu`
on every pair and never restores it, so a later `backend="triton"` call on the same batch will not use
the GPU. Rebuild the batch to switch back.

## 4. The optimizer

**`num_repeats` and `max_num_steps` default to `None`**, resolving per mode via `accel/_modes.py`
(tables in [§2](#2-the-21-alignment-modes)). Upstream defaulted to 50 and 200 for every mode. This
changes results — see [B1](#b1-num_repeats-and-max_num_steps-defaults).

**No coarse grid, no top-k prune on the default path.** Despite the "coarse-to-fine" naming, every
seed goes straight into the fine loop and the per-pair maximum is taken — ranking seeds on raw
un-optimized overlap repeatedly discarded the true basin for pseudo-symmetric molecules. The
coarse-grid path runs **only** when `trans_init=True`, so cost scales linearly with seed count.

**The coarse grid is not always built from the cloud the mode seeds from.** `vol_and_surf_esp`
builds it from the **surface** clouds while seeding from the volume centres; every other mode uses
its seed cloud. The pre-registry drivers differed here by accident rather than by design, but the
behaviour ships, so it is carried as `ModeSpec.coarse_channel` — routing the combo mode through its
seed channel instead moves its `trans_init` scores by up to 1.24% relative. A new mode inherits the
default (its seed cloud) and needs no entry. `tests/test_trans_init_accel.py` pins the choice per
mode; that path had no test before. The same file records the one mode whose `trans_init` self-copy
does not reach 1.0 (`vol_and_surf_esp`, ~0.54), which is long-standing and reproduces exactly on
591f695.

**Gradients are analytic, not autograd** — every kernel emits closed-form `dV/dq` and `dV/dt`, so no
autograd graph is built in the fine loop, and the pharmacophore and colour kernels emit `dO/dq`
in-register, dropping the rotation-matrix→quaternion projection tail.

**The Adam is not `torch.optim.Adam`** — β₁=0.9, β₂=0.999, ε=1e-8 *inside* the sqrt, **no bias
correction**, a quaternion tangent-space projection fused into the kernel, and unit-quaternion
renormalization every step. `lr` does not mean what it means in `torch.optim.Adam`, and the drivers'
internal default is `0.075`, not the `0.1` the public API advertises.

**Early stopping is per-mode and per-pair** — patience 2 for `vol`/`surf`/`vol_color`, 5 for the ESP
and pharmacophore modes, tolerance 1e-5, checked every 5 steps ([B9](#b9-early-stopping-is-per-pair)).

**CUDA graphs.** All modes share one implementation: one fine step is captured and replayed.
Engagement is not uniform — `pharm` only for `tanimoto` with `extended_points=False`,
`vol_and_surf_esp` with early stop disabled, and the budget differs per mode — but whether a bucket
graphs is a function of its pad shape and that budget, not of allocator state. Graphs live in a
bounded LRU of 24 and **pin GPU buffers for the process lifetime**; call
`accel.drivers._graphed.reset_graph_cache()` if you hit fragmentation. Triton autotune is cached to
disk, so the per-shape sweep is paid once per machine, not once per process.

**Bucketing is result-identical by construction**: the kernels are one-CTA-per-pose and mask padding
to the real point counts, and seeds key on real counts rather than pad width, so padding two
different-sized molecules into one bucket cannot change a score. Sub-batching bounds peak memory — on
CUDA from free memory with an OOM halve-and-retry, on CPU by a fixed `_CPU_CHUNK_PAIRS = 10,000` cap
that bounds host RSS.

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

hits = screen(Molecule(query_rdmol, num_surf_points=200),
              ProfileStore.open("lib.store"), mode="surf_esp", alpha=0.81, top_k=1000)
# each hit is a Hit(score, id, transform); transform is a 4x4 numpy array
```

`screen_many` streams the library **once** for a list of queries. Both accept `ndev=` to shard across
GPUs and `scores_out=` to write full score vectors (memmap-friendly; single-process only — passing it
with `ndev>1` raises). The `ndev>1` workers — one process per device, each streaming its share of the
shards with the same read-ahead as the single-process screen — are spawned on the first call and kept
for later screens; `screen.close_multigpu_pool()` releases them (also run at interpreter exit). On a
canonical store the workers run the same constant `vol` seed set as the single-process screen (they
ran per-molecule seeds before, at 2.1× the cost per shard, so two devices screened no faster than one).

Things that will bite you:

- **`add` takes a `Molecule`, not an RDKit `Mol`**, and every molecule must share the store's
  `num_surf_points` — a mismatch fails at flush time, thousands of `add()` calls later.
- **The default `dtype="float16"` is lossy** (~0.01 Å, the surface-resampling noise floor), so screen
  scores differ slightly from a pairwise run for that reason alone. Use `dtype="float32"` to compare.
- **A store only serves modes it has arrays for** — `vol` works on any store, `vol_color` needs a
  pharmacophore store, `vol_and_surf_esp` needs surfaces, ESP, charges, radii and with-H centers. An
  unsupported mode raises `ValueError`.
- **Some modes need explicit parameters**: `vol_and_surf_esp` requires `alpha=`, `vol_esp` requires
  `lam=`, and `no_H=False` is rejected — screening is heavy-atom-only.
- **`Hit.transform` is in the pre-centered frame.** On a `pre_centered=True` store the 4×4 maps
  *centered* onto *centered*, so subtract that molecule's COM first — or build with
  `pre_centered=False`.
- **`shard_size` is a GPU-memory knob**, not just I/O: a whole shard uploads as device tensors at
  once, and a screen holds **two** shards in host RAM (the one aligning plus one read ahead).
- **Stores that serve `vol` are canonical by default** (`canonical=None`): each molecule is stored
  rotated into its principal-axis frame, which lets a `vol` screen skip the per-molecule eigensolve
  for one constant seed set (roughly 1.5–2× on GPU). Transforms are composed back to the centered
  frame, so `Hit.transform` reads the same as on a non-canonical store. Stores without `vol` keep
  raw centered coordinates. See [B13](#b13-profile-stores-are-canonical-by-default).
- **`trans_init=True`, `backend="jax"`, or a non-pre-centered store** drop you onto a much slower
  object path.
- A killed build leaves a **readable store** of every completed shard. A store is **single-writer**.

Two build options matter. `shard_format="npy"` (the default) writes one `.npy` per array and
memory-maps on read; `"npz"` is still supported. `canonical=True` pre-rotates each molecule to its
principal axes so per-query seed generation becomes a constant set — this **changes scores**, so it is
opt-in.

## 6. Parallelism

### Multi-GPU

Alignment is host-bound rather than kernel-bound, so driving N GPUs from one process serialises on the
GIL. The path that scales is one process per GPU: `align_multi_gpu(pairs, mode, ndev=4, **kwargs)` for
a one-shot call, or `MultiGPUAligner(pairs, ndev=4)` as a context manager that keeps each GPU's shard
resident across `.align(mode, **kwargs)` calls. Both forward `**align_kwargs` into
`align_with_<mode>`, so a mode's required arguments are required here too, and **both need an
`if __name__ == "__main__":` guard** (see below).

This is deliberately **opt-in**: a library must not spawn workers behind the user's back, because
`spawn` re-imports `__main__` and breaks unguarded scripts. A large batch on a multi-GPU host
therefore runs on **one GPU** and warns once, above 4,096 pairs per GPU.

**`align_multi_gpu` always spawns**, so it always needs the guard, while **`MultiGPUAligner` prefers
`fork`** and falls back to `spawn` with a warning if CUDA is already initialized or Open3D already
imported — either poisons `fork`+CUDA, so build the pool *before* doing CUDA work. **Only `vol`,
`surf`, `surf_esp` and `pharm` have a worker-process path**; every other mode raises `ValueError`
(`screen(ndev>1)` is separate and not restricted to those four). Worker threads are capped to
`cores // ndev` — mandatory, since uncapped workers oversubscribe and scaling collapses below 1× — and
`Molecule` objects must be picklable, since they cross the boundary.

### CPU

**`num_workers=N` with `backend="numba"`** shards pairs across a persistent pool of single-threaded
processes. Pairs are independent, so the optimization problem is unchanged — but agreement with one
large call is to **convergence tolerance, not bitwise**, since a pair's step count depends on which
pairs share its shard. It uses `spawn` (so it needs a `__main__` guard) and applies to
`vol`/`surf`/`surf_esp`/`pharm` only — elsewhere a silent no-op, and ignored on CUDA tensors.

**`accel.screen_parallel.screen_parallel(query, library, mode, n_workers=...)`** forks workers for
query-vs-library screening, sharing the featurized library copy-on-write. A different API from
`screen()`: a RAM-resident list of `Molecule`s rather than a `ProfileStore`, CPU/numba-only, canonical
mode names only, returning a plain list of scores in library order — no `Hit`s, transforms or top-K.

> It **always forks**, so POSIX-only, and must run **before** any in-process numba alignment — a live
> libgomp thread pool at `fork` aborts the child. Featurize, then screen.

The forked pool is **kept** for later calls against the same library (keyed by the list object's
identity and length, so a new list or a resized one forks afresh, while molecules mutated in place
after the first call are not seen by the workers); `screen_parallel_close()` releases it, and it is
also run at interpreter exit. Forking was the per-call cost that grew with the worker count. Each
worker pins itself to its own physical core (one CPU per hardware-thread sibling group of the process's
affinity mask; a no-op without `/sys` topology) and the library is dealt out strided — worker *w* gets
*w*, *w+k*, *w+2k*, … — because unpinned workers landed on sibling threads of busy cores (8 of 64 on a
96-core node, 1.8× the ideal time) and contiguous ranges of compound ensembles handed one worker the
largest compounds.

**Torch is pinned to one intra-op thread for a CPU batch alignment, then restored.** The numba kernels
own the cores there; unpinned, torch's pool spin-waits against them. Scoped rather than global, so it
does not reconfigure the caller's torch; if you already export `OMP_NUM_THREADS=1`, nothing changes.

**The fused CPU loop now serves 19 of the 21 modes** (it served 4). It is generic over a mode's
terms and reductions, so a mode joins it by existing. One exclusion remains, on the mode's own
`ModeSpec`: the pharmacophore family, entirely ([B12](#b12-pharm-cpu-scores-changed)). `vol_esp`
keeps a `cpu_fused_max_pad=100` bound, which its cloud never approaches.

**`surf_esp` lost its fused-loop exclusion, worth 13.4x on CPU.** It used to carry the same
`cpu_fused_max_pad=100`, not as a size bound but as a way of excluding the mode: the number was
chosen because atom clouds fall below it and 200-point surface clouds above it. The stated reason
was that the library's most shape-degenerate mode might settle in a different, equally valid basin
under the fused loop, while callers rely on pose-exact agreement. That does not happen. Measured on
real surfaces with MMFF charges, lifting it moves scores 0.0005% and poses 0.045 deg at 200 surface
points, and 0.0011% / 0.045 deg at 400. The exclusion always bound, refusing the fused loop above
about 96 surface points, so it never once fired at the 200-point default and `surf_esp` was the only
mode declaring `cpu_fused=True` while always running eager. **Any CPU `surf_esp` throughput number
measured before this is stale**; GPU numbers are unaffected, since the fused loop is a CPU path. The numba kernels run `fastmath=True, parallel=True` over `NUMBA_NUM_THREADS` (default: *all*
cores), which oversubscribes alongside the process pool — hence `screen_parallel` pinning it to 1.

**The fused loop and the eager loop agree, and that is now tested.** `engine.align` falls back to
the eager torch loop on any exception from the fused one, silently, so a mode whose two loops
disagreed would return different scores depending on an unrelated failure. Measured across all 21
modes: worst disagreement 1.699e-06 absolute, 0.0004% relative (`surf_esp`), with the pharmacophore
pair identical because both runs take the eager loop.
`tests/test_cpu_fine_loops_agree.py` holds that bound and picks up new modes from the registry.

## 7. Surfaces and pharmacophores

**A mesh-free surface generator (opt-in).** `Molecule(..., surface_method="smooth_sdf")` builds the
surface from a smooth-min signed-distance field plus stochastic sampling instead of Open3D
ball-pivoting + Poisson-disk — no Open3D, no mesh. **The default is unchanged**; `"mesh"` reproduces
the original path exactly. The smooth surfacer rounds the concave atom-border seams, so a model
trained on mesh surfaces sees a **distribution shift** — validate before using it generatively. It
requires `num_surf_points` and **raises `ValueError` if given `density=`**; aliases `'sdf'`,
`'smooth'`, `'fast'` work, unknown methods raise, tunables live on `get_molecular_surface` with
defaults in `SMOOTH_SDF_*`, and it samples 15 candidates per atom against the mesh path's 25.

**`shepherd_score.surface_diagnostics`** validates a surfacer (numpy + scipy only). Its core quantity
is the **shell residual** — distance from a point to an atom's *sphere*, not its center.
`leak_metrics` measures how much the cloud leaks atom positions, where **lower residual = more leak**
(`mesh` gives a ~0.010 Å median); `crimp_points` finds points on one sphere and near a second;
`center_recovery_attack` Kasa-fits a sphere per atom, where **higher error = safer**; `local_curvature`
and `summarize` give per-point non-flatness and the whole gate in one call.

**Directionless ("color") pharmacophores.** `Molecule(..., directionless=True)` and
`get_pharmacophores(..., directionless=True)` build isotropic zero-vector pharmacophores for every
family — the ROCS/ROSHAMBO "color" convention. Both default to the original behaviour.

> **This changes the pharmacophore *count*, not just the vectors.** The donor/acceptor multi-vector
> branches are skipped, so a donor upstream expanded into one anchor *per hydrogen* now emits a single
> anchor at the feature position.

`feature_set="rdkit_base"` selects RDKit's base definitions instead of ShEPhERD's SMARTS set. It keeps
six families and renames three (`PosIonizable`→`Cation`, `NegIonizable`→`Anion`,
`LumpedHydrophobe`→`Hydrophobe`) — **`Halogen` and `ZnBinder` disappear.**

**Lazy Open3D.** `generate_point_cloud.py` imports Open3D on first use, not at module load — it is
slow to import and fork-hostile, and importing it poisons a later `fork`+CUDA, which would break the
fork-based pools. So **`import shepherd_score.container` now works without Open3D**; if it is missing,
the error moves from import time to first surface generation, with its type (`ModuleNotFoundError`)
unchanged. One caveat: the module global `o3d` is a lazy **proxy**, so
`from shepherd_score.generate_point_cloud import o3d` no longer yields something `isinstance` or
`inspect` treats as a module. Note also that importing `container` eagerly imports `accel.batch` and
`accel.multi_gpu` — no new dependency, torch was already required, but not free.

## 8. API reference

Only what is new or changed. Full signatures are in the code; this is the delta.

### `container`

`__all__` gains `Surface`, `Pharmacophore`, `AlignmentResult` (upstream's refactor) plus
`align_multi_gpu` and `MultiGPUAligner`, re-exported from `accel`.

**`Molecule.__init__` gains five keyword arguments.** Four are default-preserving; `charge_model` is
not, and it sits after `partial_charges` rather than appended last, so a caller passing more than six
positional arguments is affected.

```python
charge_model='xtb'        # 'xtb' | 'mmff'  -- CHANGES ESP SCORES, see B8
feature_set='shepherd'    # 'shepherd' | 'rdkit_base'
directionless=False       # isotropic "color" pharmacophores
surface_method='mesh'     # 'mesh' | 'smooth_sdf'
fukui=None                # precomputed Fukui field for vol_fukui
```

Also new: `get_positions(no_H=True)` / `get_charges(no_H=True)`, and per-channel accessors
`get_lipophilicity` / `get_molar_refractivity` / `get_fukui` / `get_atomic_numbers`, each with a
`get_*_positions` returning the strict-heavy centres the channel is 1:1 with. `get_pharmacophore`
gains `feature_set` and `directionless` plus upstream's `return_atom_ids` / `priority_atoms` /
`min_ring_priority_atoms`.

**`MoleculePair`** carries one `(transform, score)` pair per mode. `vol` and `vol_esp` keep the
historical `_noH` spelling; the other 19 follow `transform_<mode>` / `sim_aligned_<mode>`. The legacy
`transform_esp` / `sim_aligned_esp` / `*_esp_combo` names remain as read/write properties.

**`MoleculePairBatch`**: every method gains `backend=None` and `return_aligned=False`;
`align_with_vol` and `align_with_vol_esp` also gain `alpha=0.81`. Four gotchas:

- **`return_aligned=False` is the default** on the accelerated backends, returning `[None] * N` in
  place of the aligned coordinates. Transforms are still written to each pair; JAX is unaffected.
- **`align_with_pharm` returns a 3-tuple** where other modes return a 2-tuple, so code indexing `[1]`
  for arrays gets `None`s.
- **`alpha=` reaches Triton and numba only** — JAX hardcodes `0.81`, as upstream did, so a different
  `alpha` with `backend="jax"` is silently ignored.
- **`num_repeats` is accepted and ignored** by `align_with_vol_lipo` and `align_with_vol_color`, whose
  seed count comes from `MODE_SEEDS`.

### `screen` (new)

```python
ProfileStore.create(path, *, num_surf_points, modes, dtype='float16', shard_size=100_000,
                    pre_centered=True, overwrite=False, canonical=False, shard_format='npy')
screen(query, store, mode='surf_esp', *, backend=None, do_center=None, top_k=1000, ndev=None,
       scores_out=None, alpha=None, progress=False, **align_kwargs) -> List[Hit]
screen_many(queries, store, ...) -> List[List[Hit]]     # same kwargs; streams the library once
```

Also exported: `MoleculeProfile`, `Hit`. A store materializes only the arrays its `modes=` need, and
its on-disk format carries `VERSION = 1` — validated on open, and provisional.

### `accel` (new)

`has_triton()`, `align_multi_gpu(...)`, `MultiGPUAligner`, `clear_caches()`, plus
`accel.screen_parallel.screen_parallel(...)`.

**`accel._modes`** is the mode registry, and the modes are DATA. `SPECS` holds one `ModeSpec` per
mode — its channels, its objective terms (kernel, reduction, blend weight), its optimiser schedule
and its per-mode tuning (graph budget, pose cap, fused-CPU opt-out, multipose) — and the flat tables
every consumer reads are derived from it: `CANONICAL_MODES` (the 21, in public order),
`LEGACY_MODE_ALIASES`, `PROCESS_MODES`, `MODE_ATTRS`, `MODE_SEEDS`/`MODE_STEPS` (what
`num_repeats=None` / `max_num_steps=None` resolve to), `CONST_SEED_MODES`, `canonical(mode)` and
`spec_of(mode)`.

**`accel.channels`** is the matching per-molecule data table: one `Channel` per named array (the
heavy-atom cloud, the surface points, the partial charges, the pharmacophore anchors, ...) carrying
how to read it off a `Molecule`, which pair tensor holds it, how a `ProfileStore` persists it, and
whether it rotates and translates under the canonical frame. Adding per-molecule data is a row here,
not an edit in five files.

**`accel._stats`** records fine-loop effort and is a no-op until armed. `reset()` enables it,
`summary()` returns `calls`, `graphed` (calls that took the CUDA-graph path), `steps_min/max/mean`,
`steps_configured` and `early_stop_frac`, and `disable()` turns it off. It is process-local, so a
forked or spawned worker records only its own process — **an empty summary means unmeasured, not full
effort** — and the reference per-pair optimizers and the JAX path do not record at all.

**Tunable module-level constants** — not env vars; edit or monkey-patch them.

| Constant | Module | Default | Effect |
|---|---|---|---|
| `_ESP_STRIDE` | `drivers/engine.py` | 5 | ESP re-scoring interval; 1 = dense |
| `VOL_COLOR_FUSED_MAX_PAD` | `kernels/vol_color_triton.py` | 32 | Above this pad, the fused vol_color kernel is skipped |
| `_GRAPH_WORK_BUDGET` | `drivers/_graphed.py` | 300,000,000 | Default pose budget for graph engagement |
| `_GRAPH_CAP_CEIL` / `_GRAPH_CAP_MIN` | `drivers/_graphed.py` | 262,144 / 2,000 | Hard bounds on the graph pose cap |
| `_GRAPH_CACHE_MAX` | `drivers/_graphed.py` | 24 | Live captured graphs (each pins GPU buffers) |
| `_FINE_CHUNK_POSES` | `batch/_pad.py` | 81,920 | Pose cap per fine-loop sub-batch, vol screen path |
| `_CPU_CHUNK_PAIRS` | `batch/_pad.py` | 10,000 | Pairs per call on the CPU path; bounds host RSS |
| `_BAND` | `batch/_pad.py` | 16 | Legacy fixed-band pad granularity |
| `SMOOTH_SDF_*` | `generate_point_cloud.py` | — | Smooth-SDF surfacer defaults |

### Scoring and surfaces

`get_molecular_surface` gains `method='mesh'|'smooth_sdf'` and the `sdf_*` / `even` / `seed` tunables;
`get_molecular_surface_smooth_sdf` is new, as is the whole `surface_diagnostics` module
(`leak_metrics`, `crimp_points`, `center_recovery_attack`, `local_curvature`, `summarize`).

`get_overlap_pharm`, `get_pharm_combo_score`, `get_pharmacophores` and `build_lookup_tables` gain
`directionless=`, and `get_pharmacophores` also `feature_set` / `return_atom_ids` / `priority_atoms` /
`min_ring_priority_atoms`. All defaults preserve behaviour. `get_pharm_combo_score`'s combination
changed from `(pharm + shape) / 2` to `(1 - w) * shape + w * pharm` — **exactly** equal at the default
`w=0.5` in IEEE-754. `get_pharmacophores` returns a `Pharmacophore` that unpacks as the old 3-tuple,
so `X, P, V = get_pharmacophores(mol)` still works.

`shepherd_score.alignment` gains **16 exports** — an `objective_<mode>_overlay` and an
`optimize_<mode>_overlay` for each of `vol_color`, `vol_tversky`, `vol_esp_tversky`, `vol_lipo`,
`vol_atomtype`, `vol_color_tversky`, `vol_lipo_tversky`, `vol_and_surf_esp_tversky`. These eager torch
references are what the accelerated backends are checked against.

### From upstream, integrated

`Surface`, `Pharmacophore` and `AlignmentResult`, via `Molecule.surface` / `.pharmacophore` /
`MoleculePair._alignments`. Plus interaction subselection: **`Molecule.select_atoms(atom_indices, ...)`**
returns a new `Molecule` whose surface, ESP and pharmacophore are restricted to an atom subset, built on
`get_pc(atom_indices=, radial_buffer=)`, `get_electrostatic_potential(atom_indices=, surf_pos=)` and
`Pharmacophore.expand_atom_selection` / `.subset_to_atoms` / `.priority_labels(...)`. Defaults preserve
behaviour; on this fork `select_atoms` also carries `charge_model`, `surface_method` and `fukui` through.

### Extensibility

Two agent skills ship in `.claude/skills/` — `design-scoring-mode` (representation plus overlap
objective, wrapped in the per-pair Adam optimizer that becomes the gradient oracle) and
`accelerate-scoring-mode` (port it to a fused Triton kernel and its numba twin, wire them into the
device dispatcher, gate on parity against that oracle); the 14 modes beyond the original seven were
built through them. To add a mode: register it in `accel/_modes.py` and add a driver — the batch layer
and screen front-end pick it up from the registry, and `tests/test_mode_registry.py` fails if the
registry and the torch-typed `_MODE_SPEC` drift apart.

## 9. Behavior changes

Twelve are visible to code written against upstream. **B5, B6, B8, B9 and B12 matter most.**

### B1. `num_repeats` and `max_num_steps` defaults

On all six pre-existing alignment methods, on both `MoleculePair` and `MoleculePairBatch`, `50`/`200`
became `None`, resolving per mode ([§2](#2-the-21-alignment-modes)).

**This changes results**, and not merely as "fewer restarts": `alignment/_torch.py` and `_jax.py`
special-case `num_repeats == 50` to load a precomputed 45-point Fibonacci quaternion set, and any
other value takes a branch producing a **different set of orientations**, not a subset. Individual
pairs move by up to a few percent; the aggregate cost is small, since these defaults sit at the
accuracy/throughput knee. **To restore the old behaviour**, pass `num_repeats=50, max_num_steps=200`.

### B2. Mode rename

`esp` → `surf_esp`, `esp_combo` → `vol_and_surf_esp`, across method names, result attributes and mode
strings. **Input-side compatibility is complete** — legacy method names are aliases to the same
function objects, legacy result attributes are read/write properties, and every mode string is
canonicalized at the entry point. **Output-side is not**: `pair.transform_esp` works, but
`'transform_esp' in pair.__dict__` is now `False` and `vars(pair)` shows `transform_surf_esp`, so code
that introspects `__dict__` sees the new names.

### B3. Old pickles round-trip

The refactor turned the flat attributes into data descriptors, which win over the instance dict, so an
older pickle restored its flat keys and then raised `AttributeError` on first read. **Both classes now
define `__setstate__`**, remapping flat keys into `Surface`/`Pharmacophore`/`AlignmentResult`,
honouring the renamed-mode aliases, and defaulting state added after the flat layout
(`surface_method`, `_charge_model`, `_fukui`). New-format pickles are unaffected; stale flat
duplicates are stripped so a shadowed copy cannot drift.

### B4. `apply_SE3_transform` collapses a singleton batch

`apply_SE3_transform(points[None], transform[None])` was `(1, N, 3)` and is now `(N, 3)`. Values are
unchanged and the batched and gradient paths stay bitwise identical, but **`apply_SO3_transform` was
not given the same collapse** — with an `R == 1` batch the two return different ranks, and upstream
pairs them on adjacent lines in three places, which mis-broadcasts silently. The single-instance path
also differs by about **one float32 ULP**, breaking bit-for-bit golden values on the
`num_repeats == 1` path. `se3.py` also gains `quaternions_to_SE3_batch(q, t)`.

### B5. The default batch backend changed

`MoleculePairBatch.align_with_*` no longer defaults to JAX; the default is device-aware and uses a
different seed set, so a default call returns **different scores** ([§3](#3-backends)). Pass
`backend="jax"` for the old numbers. Per-pair `MoleculePair.align_with_*` is unaffected. This also
fixes a latent bug — the old JAX default was not installable by default, since `jax` is an optional
extra, so a fresh install hitting the default batch path raised `ImportError`.

### B6. `numba` is a required dependency

It is the default CPU backend, so the package no longer imports without it.

### B7. Pharmacophore scoring: `directional` → `directionless`

Renamed **with inverted meaning**, so extraction and scoring share one polarity. No alias —
`directional=` was fork-only — so update any call: `directional=False` → `directionless=True`.

### B8. ESP charges default to gfn2-xTB

`Molecule(...)` with no `partial_charges` used to fall back to MMFF94; it now defaults to
`charge_model="xtb"`, so ESP modes score on xTB charges — **different ESP scores**. Shape, colour and
pharmacophore modes never read charges. Pass `charge_model="mmff"` for the old behaviour.

Two things make this safe. **Charges are lazy** — generated on first read, then cached — so a molecule
used only for shape/colour/pharmacophore never invokes the xTB subprocess. (Constructing *with a
surface* builds its surface ESP eagerly and pays the cost.) And a missing or non-converging `xtb`
**falls back to MMFF94 with a `RuntimeWarning`**, so it is recommended, not required.

**xTB is not uniformly better.** On retrospective enrichment it helps the surface-ESP modes and
*hurts* atom-centred `vol_esp` — surface ESP rewards a faithful field, atom-centred ESP suits
atom-centred charges. Choose per mode.

### B9. Early stopping is per-pair

Every accelerated fine loop used to stop on a **maximum over the whole batch**, so one pair that
converged early — most cheaply a molecule aligned against itself — halted optimization for every other
pair in its bucket, flooring it at 11 executed steps whatever the budget said. The criterion is now
each pair's own best, and the loop breaks only after `early_stop_patience` checks in which **no** pair
improved.

**This changes results, and only upward**, at unchanged search effort — the extra work is work that
was already configured and being skipped. Throughput drops on any workload that was truncating.
**Any score table, throughput number or enrichment figure produced before this must be re-measured,
not re-labelled** — including results that were never reproducible, since truncation depended on
which pairs shared a bucket.

### B10. `screen(ndev>1)` returned the wrong mode's answers

The multi-GPU screen worker dispatched the `vol` builder and aligner for **every** mode, so any mode
whose reference tensors satisfied `vol`'s contract silently ran `vol` under another mode's name;
`pharm` raised `KeyError` instead. Both drivers now share one dispatch selector. **Any
`screen(ndev>1)` result for a non-`vol` mode produced before this is invalid.**

### B11. The screening host path is faster, and reproducible

The reduce no longer touches Python per library molecule, all 11 screen-capable modes align straight
from the store's contiguous arrays, and the next shard is read on a background thread. **Hit lists,
hit order (including exact ties), ids, transforms and `scores_out` are unchanged** — a front-end
refactor with no arithmetic in it.

Two reproducibility problems are also gone. Sub-batch sizing derived from *free* device memory, so
whether a bucket landed under the CUDA-graph budget depended on what the allocator held — and the two
paths do not always agree, so the same screen could return different scores run to run; a fixed pose
cap makes that a function of the bucket's shape instead. And the per-pair footprint calibration
charged a chunk for memory *other* objects held, collapsing it to one pair under memory pressure.
**Screen scores from a release before this may not reproduce even against themselves.**

### B12. `pharm` CPU scores changed

`pharm`'s fused numba CPU loop and its eager loop land in **different optima**: the fused loop applies
its Adam tail before the early-stop check, so an N-iteration run is N evaluations *and* N updates
where the eager loop interleaves them the other way. With 32 seeds — the most of any mode — and a
strongly multi-basin objective, that relocates the optimum rather than perturbing it. The fused path
is therefore **disabled for `pharm`** — the standard `surf_esp` is excluded under — at a few percent
less throughput. **Any `pharm` CPU score produced before this came from the fused trajectory.**

### B13. Profile stores are canonical by default

`ProfileStore.create` now defaults `canonical` to `True` when the store's modes include any mode that
seeds from the heavy-atom cloud — `accel._modes.CONST_SEED_MODES`: `vol`, `vol_color`, `vol_esp`,
`vol_lipo`, `vol_fukui`, `vol_tversky`, `vol_esp_tversky` and `vol_and_surf_esp` (at `alpha=0.81`) —
and `pre_centered` is on, so such a store holds every molecule rotated into its own principal-axis
frame with the rotation kept. Screening any of those modes against it runs one constant seed set
instead of a per-molecule eigensolve — for `vol` this is what the paper's screening throughput was
measured on; the others gained it later, with the same set (`canonical_seed_quats`) and the same
zero translations, because their drivers seed from the same cloud — and the transforms it returns are
composed back to the centered frame. Because that seed set differs from
the per-molecule one, **`vol` scores from a canonical store differ from a non-canonical store's, and
from `MoleculePair`, by about 1e-3 on average** (Spearman 0.997, top-1000 overlap 99% at N=99,000),
with a few molecules landing in a different optimizer basin. The same holds for the other
constant-seed modes. `surf`, `surf_esp` and `pharm` seed from the surface or anchor clouds, which the
store does not canonicalise, so they keep their per-molecule generator and gain no speed; a store
serving only them stays non-canonical and matches the pairwise path to ~1e-4. Stores already on disk keep their manifest's
`canonical=False` and are unaffected. Before this became the default, screening scores
and DUDE-Z enrichment on canonical stores were validated against non-canonical stores and against the
pairwise path (Shepherd-Score-Paper, `paper/fig2_speed/validate_canonical.py` and its
`results/CANONICAL_validation.json`). Pass `canonical=False` for the previous layout.

### Minor

- `align_with_vol_esp(lam=...)` — `lam` was a required positional, now defaults to `0.1`.
- `MoleculePair.__init__` eagerly allocates two torch tensors on the target device, so constructing a
  pair touches GPU memory before any `align_*` call.
- `get_overlap_pharm(directionless=True)` raises `ValueError` with `precomputed_self_overlaps` and
  silently forces `extended_points`/`only_extended` off; the numpy variant forces them off without
  raising. Neither affects the `directionless=False` default.

## 10. Limitations

- **The JAX path is the least exercised surface** — almost every skipped test in a routine run is a
  JAX test, so exercise it before relying on it as [B5](#b5-the-default-batch-backend-changed)'s
  reproducibility escape hatch. There is also **no JAX ↔ Triton/numba parity test** — the seed sets
  differ, so what is needed is a *quality* comparison, not an equality test.
- **Ten of the 21 modes have never been benchmarked** — `vol_mr`, `vol_atomtype`, `vol_pharm`,
  `vol_avoid` and the six Tversky variants other than `vol_tversky`. They are correctness-tested, not
  performance-characterised, and several carry tuning constants inherited from a sibling rather than
  measured for themselves. They all screen now, on the array-native path, but **no throughput number
  has been measured for any of them** — do not quote one by analogy with a sibling.
- **Parity was measured on SMALL batches, and two bugs lived outside that envelope.** Every
  pairwise, screen, graph-vs-eager and array-vs-object comparison behind the sub-0.1% claims above
  ran on 6- to 12-molecule fixtures, and every test store held all modes at once. Two real
  regressions survived that sweep untouched: the kernel-launch slice past **65,535 poses**, which
  put Tanimotos as high as 1.1e5 on a 30,000-pair GPU batch, and a store built for **one mode
  alone**, which silently lacked the arrays that mode reads. Both are fixed and both now have
  gates (`tests/test_grid_chunked_launch.py`, `tests/test_store_minimal_schema.py`), but the
  lesson stands: before trusting a parity number here, run a GPU batch past 65,535 poses, build a
  store per mode, and use the real harness cell size — the figure's pairwise cell is N=100,000,
  not a toy fixture.
- **The 65,535 launch slice is gone; launches are sliced at the real bound.** Every kernel here
  launches a **1-D** grid (`grid = (K,)`, `[(P,)]`), so the limit that applies is `grid.x` at
  2^31-1 — `grid.z <= 65535`, which the old comment cited as a hardware limit, never entered into
  it. The bound that does exist is the int32 pointer offset the kernels form as `mol * N_pad * 3`,
  and `terms._launch_step` now derives that ceiling from the actual pads at call time (22.4M poses
  at a 32-atom pad, 3.4M at a 208-point surface, int32-safe at a 1,024-point cloud). At `vol`'s
  81,920-pose chunk the old slice split every captured fine step into two launches plus three
  output copies; it is now one direct launch, as the pre-registry driver made it. **Measured, one
  process per variant** (the captured-graph cache is keyed by shape, so an in-process A/B replays
  the first variant's graph and reads 0.4% — a trap worth knowing): `vol` screen **+3.4% at 1e5,
  +3.5% at 1e6**, `vol_esp` +4.9%, `surf` +4.4%, every score bit-identical. **The bug fixed in
  8127da9 was never about the limit** — it was that `_chunked`, once it slices, must slice the
  per-molecule arguments along with the molecules; that contract is unchanged.
- **Where the engine rewrite's throughput went, bisected.** Against the pre-branch baseline
  (b3599dc) at N=1e6 on one node, `vol` screen sat at −6.5% and the step is entirely inside the
  engine commit — b3599dc, d3bf4bf and 591f695 are flat. Two changes recover most of it: the
  launch ceiling above (~3.5 points on the screen path) and buffering the fine-loop reduction so
  the captured step allocates nothing (~0.8 points on the screen path, roughly 8 on the pairwise
  cell, where per-step allocation across many steps of one large batch mattered most). A profile
  of the tree after both changes put GPU time EQUAL to the base (75,585 vs 75,439 us per screen)
  and the whole remaining gap in host work between launches, of which the largest named piece
  was `assemble` recomputing every term's self-overlaps for every chunk the sub-batcher handed
  it -- two eager kernel launches per chunk, each building an identity quaternion on the host,
  where the old driver hoisted them once per bucket. `engine.term_self_overlaps` now does that
  once per bucket and the chunk loop slices it (`self_overlaps=` on `align`/`assemble`); a caller
  that passes nothing keeps the per-call behaviour. Six of the seven modes measured were already
  faster than the baseline before any of this, `pharm` by 10.8%.
- **The typed CPU kernel wrappers convert their invariant inputs once per bucket.** The pharm and
  colour wrappers in `kernels/cpu.py` converted the type arrays, the alpha/K/category tables and
  the real-atom counts to numpy int64/float64 on every call, and the eager fine loop calls them
  every step with the same tensors; on a threaded pharm screen the per-step copies of the
  `(K, N_pad)` type arrays alone were 13% of the wall time. `_np_cached` now stashes the converted
  copy on the tensor, keyed by storage pointer, shape and in-place version counter, so it is
  redone only when the tensor is rebuilt or written. Screen scores, pairwise scores and poses are
  bit-identical (2,000 Platinum molecules, 3 queries, 200 pairs, compared within one compile
  state). Laptop pharm screen: +7% at one numba thread, +19% to +75% at eight depending on SVML;
  the cluster number is pending. Only the CPU eager path changes: `pharm` and `pharm_tversky` by
  default; `vol_color`, `vol_color_tversky` and `vol_atomtype` (the colour wrapper) and
  `vol_pharm` (the pharm wrapper) only in their once-per-bucket self-overlap call, or when
  their fused loop is bypassed. The GPU and fused-loop paths are untouched.
- **`accel/` and `screen.py` have no Sphinx API pages**, so none of [§8](#8-api-reference) renders on
  the docs site. When adding them, set `autodoc_mock_imports = ["triton", "numba"]`.
- **Bit-identity results come from non-early-stopping workloads.** A very small chunk or a
  low-diversity library could fire the early stop and re-open a graph-vs-eager divergence.
- **`vol_and_surf_esp`'s graph and eager paths compute different algorithms** — the eager loop applies
  `_ESP_STRIDE` and the captured graph step does not, so the graph scores every step and returns
  uniformly higher values. The screen runs eager: treat its scores as not interchangeable.
