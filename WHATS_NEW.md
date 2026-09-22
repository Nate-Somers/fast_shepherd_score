# What's New

What this release adds to `shepherd-score`, and what an existing caller has to change.

> ## Six changes affect existing results or installs
>
> 1. **`align_with_*` on a batch no longer defaults to JAX.** The default is device-aware, Triton
>    on CUDA and numba on CPU, and those backends use a different SE(3) seed set, so a default
>    call returns different scores. Pass `backend="jax"` for the old numbers.
>    [§3](#3-backends) · [B5](#b5-the-default-batch-backend-changed)
> 2. **`numba` is a core dependency.** The package does not import without it. [§1](#1-installation)
> 3. **ESP modes default to gfn2-xTB charges, not MMFF94**, so ESP scores differ.
>    `charge_model="mmff"` restores the old behaviour. [B8](#b8-esp-charges-default-to-gfn2-xtb)
> 4. **Alignment scores can go up.** Early stopping is per pair, so a pair is no longer cut short
>    by a converged neighbour. Search effort is unchanged. [B9](#b9-early-stopping-is-per-pair)
> 5. **`pharm` CPU scores changed.** Its fused CPU loop found different optima, so that mode no
>    longer uses it. [B12](#b12-pharm-cpu-scores-changed)
> 6. **Profile stores that serve an atom-cloud-seeded mode are canonical by default**, which
>    changes their screening scores slightly. Existing stores on disk are unaffected.
>    [B13](#b13-profile-stores-are-canonical-by-default)

The release is additive: no upstream file is deleted and no upstream public name is removed.

**Contents** — [1 Installation](#1-installation) · [2 Modes](#2-the-21-alignment-modes) ·
[3 Backends](#3-backends) · [4 Optimizer](#4-the-optimizer) · [5 Screening](#5-virtual-screening) ·
[6 Parallelism](#6-parallelism) · [7 Surfaces](#7-surfaces-and-pharmacophores) ·
[8 API](#8-api-reference) · [9 Behavior changes](#9-behavior-changes) ·
[10 Limitations](#10-limitations)

## 1. Installation

```bash
pip install shepherd-score            # core: rdkit, torch, open3d, py3Dmol, numpy, pandas,
                                      #       scipy, molscrub, tqdm, numba, threadpoolctl
pip install "shepherd-score[gpu]"     # + triton>=3.6, torch>=2.6 -> backend="triton"
pip install "shepherd-score[jax]"     # + jax                     -> backend="jax"
```

- **`numba` is core, not an extra**, because it is the default CPU backend. The `[cpu]` extra is
  kept as a redundant alias.
- **`triton` and `jax` are optional and imported lazily.** Neither blocks import.
- **`tqdm` moved into core.** Upstream imported it without declaring it.
- **The `gpu` extra needs a CUDA build of torch.** The default CPU wheel bundles no Triton.
  Triton must be 3.6 or newer for `@triton.autotune(cache_results=...)`, and torch 2.6 or newer
  for `torch.cuda.graph(capture_error_mode=...)`, which the graphed fine loop passes. Neither
  degrades quietly: a capture failure that is not an out-of-memory error is re-raised. Triton
  publishes Linux wheels only, so install it explicitly.
- **One conda `environment.yml` covers CPU and GPU.** It pins the SVML numba stack, which the
  CPU kernels need for vectorised `exp`. SVML also changes precision: it selects fp32
  structure-of-arrays kernels. Without it the kernels are correct but slower, and warn once.

## 2. The 21 alignment modes

Upstream has six. All 21 are reachable as `MoleculePair.align_with_<mode>` and
`MoleculePairBatch.align_with_<mode>`, all have Triton and numba backends, and all are declared
in `accel/_modes.py`, the single source of truth for names, result attributes and per-mode
defaults.

Every mode reduces a Gaussian overlap with a Tanimoto `V_AB / (V_AA + V_BB - V_AB)` unless it is
a Tversky variant, with self-overlaps computed through the same kernel at the identity pose.

### Seven core modes

| Mode | Scores | Pose steered by | seeds x steps |
|---|---|---|---:|
| `vol` | Gaussian shape overlap of heavy-atom centers | shape | 10 x 30 |
| `surf` | the same, over surface points | shape | 8 x 40 |
| `vol_esp` | shape overlap, each pair term weighted by `exp(-(C_i - C_j)^2/lam)` on partial charges | shape+ESP | 16 x 50 |
| `surf_esp` | the same, over surface points with surface ESP | shape+ESP | 8 x 40 |
| `vol_and_surf_esp` | `esp_weight * ESP_sim + (1 - esp_weight) * shape_sim`, where `ESP_sim` is a ShaEP-style masked Gaussian of the per-point ESP difference | shape only | 8 x 60 |
| `pharm` | typed pharmacophore Gaussians, per-type alpha, same-type only, direction-weighted | pharmacophore | 32 x 50 |
| `vol_color` | `(1 - w) * shape_T + w * color_T`, colour being the pharmacophore Gaussian without direction weighting | joint | 16 x 40 |

`vol_and_surf_esp` descends on shape alone: the electrostatic term enters the tracked score but
never the derivative. Because the trajectory is shape-driven, the eager loop evaluates ESP every
`_ESP_STRIDE` steps plus the last; set that constant to 1 to score densely.

`vol_color`'s two signatures differ. The `MoleculePair` version accepts `similarity`,
`directionless`, `extended_points` and `only_extended`; the batch version accepts none of them.

### Six further per-atom channels

Atom-centred shape plus a second per-atom field, scored `(1 - w) * shape_T + w * channel_T` on a
joint gradient.

| Mode | Second channel | seeds x steps |
|---|---|---:|
| `vol_lipo` | per-atom Crippen logP | 16 x 50 |
| `vol_mr` | per-atom Crippen molar refractivity | 16 x 50 |
| `vol_fukui` | condensed Fukui dual descriptor `f+ - f-` | 16 x 50 |
| `vol_atomtype` | element identity, contributing only for same-element pairs | 16 x 40 |
| `vol_pharm` | directional pharmacophores, the directional twin of `vol_color` | 32 x 50 |
| `vol_avoid` | shape Tanimoto minus a hard-sphere penalty against a fixed avoid cloud | 16 x 50 |

`vol_avoid` takes a third, non-molecule input (`avoid_points`). It belongs to the query rather
than to a library molecule, so a screen passes it as `screen(..., avoid_points=...)` instead of
storing it per molecule.

`vol_fukui` needs xTB. `Molecule.fukui` comes from three gfn2-xTB single points and has no MMFF
fallback, unlike `partial_charges`, so without `xtb` on `PATH` you must pass a precomputed
`fukui=` array.

### Eight Tversky variants

`<parent>_tversky` is the parent's overlap reduced asymmetrically,
`T = AB / (AB + ta*(AA - AB) + tb*(BB - AB))`. `AA` and `BB` are SE(3)-invariant, so only `AB`
moves and the pose is the parent's. The ranking changes, which is the point for
substructure-style queries where a small reference should match a large fit.

`vol_tversky` (10 x 40), `vol_esp_tversky` (16 x 50), `surf_tversky` and `surf_esp_tversky`
(8 x 40), `vol_color_tversky` (16 x 40), `vol_lipo_tversky` (16 x 50), `pharm_tversky` (32 x 50),
`vol_and_surf_esp_tversky` (8 x 60). `pharm` also takes
`similarity='tversky'|'tversky_ref'|'tversky_fit'` directly.

### Wiring, and the rename

Every mode has every path. Each is a `ModeSpec` in `accel/_modes.py` giving the channels it
reads, the terms it optimises, how they reduce and blend, and its schedule; one generic engine
runs all of them. The batched aligner, the CUDA-graph and fused-CPU fine loops, the
array-native screen, the multi-GPU screen and the worker-process path are all derived from that
spec rather than written per mode, so all 21 modes are screen-capable and all 21 have a worker
path.

`esp` became `surf_esp` and `esp_combo` became `vol_and_surf_esp`. Both legacy names still work
everywhere ([B2](#b2-mode-rename)).

## 3. Backends

`MoleculePairBatch.align_with_*` takes `backend="jax" | "triton" | "numba"` and defaults to
`None`. Aliases: `"cuda"` and `"gpu"` select triton, `"cpu"` selects numba.

**`None` is device-aware**, selected per call from the tensor device, so one process can run a
CPU batch and a GPU batch and each gets the right kernel. `"jax"` runs the original path except
on modes with no JAX kernel, which fall through to the per-pair PyTorch path.

> #### The default changes scores relative to the old JAX default
>
> Both seeders start from the identity plus four PCA-alignment quaternions, but the accelerated
> one then adds up to six structured 90-degree rotations about each reference principal axis,
> covering the axis swaps PCA alignment alone misses, before falling back to a Fibonacci fill.
> At the shipped seed counts those absorb most of the budget, so `backend=` is not a
> pure-performance switch. If you have a pinned baseline, pass `backend="jax"`.

Three backend-specific limits:

- **`no_H=False` is unsupported** on triton and numba for `vol` and `vol_esp`, which are
  heavy-atom only and raise `NotImplementedError`. It still works on jax.
- **`vol` and `surf` ignore `num_repeats`, `trans_init` and `lr`** on triton and numba, because
  those kernels re-derive seeds internally.
- **`backend="numba"` moves the batch to CPU permanently.** It sets `pair.device = cpu` on every
  pair and never restores it, so a later `backend="triton"` call on the same batch will not use
  the GPU. Rebuild the batch to switch back.

## 4. The optimizer

**`num_repeats` and `max_num_steps` default to `None`**, resolving per mode through
`accel/_modes.py` (tables in [§2](#2-the-21-alignment-modes)). Upstream defaulted to 50 and 200
for every mode. This changes results; see [B1](#b1-num_repeats-and-max_num_steps-defaults).

**No coarse grid and no top-k prune on the default path.** Despite the coarse-to-fine naming,
every seed goes straight into the fine loop and the per-pair maximum is taken, because ranking
seeds on un-optimized overlap discarded the true basin for pseudo-symmetric molecules. The
coarse grid runs only when `trans_init=True`, so cost scales linearly with seed count.

**The coarse grid is not always built from the cloud the mode seeds from.**
`vol_and_surf_esp` builds it from the surface clouds while seeding from the volume centres;
every other mode uses its seed cloud. That difference is carried explicitly as
`ModeSpec.coarse_channel`, and a new mode inherits the default.
`tests/test_trans_init_accel.py` pins the choice per mode.

**Gradients are analytic, not autograd.** Every kernel emits closed-form `dV/dq` and `dV/dt`, so
no autograd graph is built in the fine loop, and the pharmacophore and colour kernels emit
`dO/dq` in-register, dropping the rotation-matrix-to-quaternion projection tail.

**The Adam is not `torch.optim.Adam`.** Beta1 0.9, beta2 0.999, epsilon 1e-8 inside the square
root, no bias correction, a quaternion tangent-space projection fused into the kernel, and
unit-quaternion renormalization every step. `lr` therefore does not mean what it means in
`torch.optim.Adam`, and the drivers' internal default is 0.075 rather than the 0.1 the public
signatures advertise.

**Early stopping is per mode and per pair**, patience 2 for `vol`, `surf` and `vol_color` and 5
for the ESP and pharmacophore modes, tolerance 1e-5, checked every 5 steps
([B9](#b9-early-stopping-is-per-pair)).

**CUDA graphs.** One fine step is captured and replayed, with one implementation for all modes.
Engagement is not uniform: `pharm` graphs only for `tanimoto` with `extended_points=False`,
`vol_and_surf_esp` only with early stop disabled, and the work budget differs per mode. Whether
a bucket graphs is a function of its pad shape and that budget. Captured graphs live in a
bounded cache and pin GPU buffers for the process lifetime; call
`accel.drivers._graphed.reset_graph_cache()` if you hit fragmentation. Triton autotune results
are cached to disk, so the per-shape sweep is paid once per machine.

**Bucketing is result-identical by construction.** The kernels run one CTA per pose and mask
padding to the real point counts, and seeds key on real counts rather than pad width, so padding
two different-sized molecules into one bucket cannot change a score. Sub-batching bounds peak
memory: on CUDA from free memory with a halve-and-retry on out-of-memory, on CPU by a fixed
per-call pair cap that bounds host resident memory.

## 5. Virtual screening

`shepherd_score.screen` is new. It featurizes a library once into an on-disk `ProfileStore`
(sharded arrays plus a `manifest.json`), then streams shards through the accelerated aligners
against one or many queries, keeping a top-K heap.

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

`screen_many` streams the library once for a list of queries. Both accept `ndev=` to shard
across GPUs and `scores_out=` to write full score vectors, which is memmap-friendly and
single-process only: passing it with `ndev>1` raises. The `ndev>1` workers, one process per
device each streaming its share of the shards, are spawned on the first call and kept for later
screens; `screen.close_multigpu_pool()` releases them, and it also runs at interpreter exit.

Things that will bite you:

- **`add` takes a `Molecule`, not an RDKit `Mol`**, and every molecule must share the store's
  `num_surf_points`. A mismatch fails at flush time, many `add()` calls later.
- **The default `dtype="float16"` is lossy**, at roughly the surface-resampling noise floor, so
  screen scores differ slightly from a pairwise run for that reason alone. Use `"float32"` to
  compare them.
- **A store only serves modes it has arrays for.** `vol` works on any store, `vol_color` needs a
  pharmacophore store, `vol_and_surf_esp` needs surfaces, ESP, charges, radii and with-H
  centers. An unsupported mode raises `ValueError`.
- **Some modes need explicit parameters.** `vol_and_surf_esp` requires `alpha=`, `vol_esp`
  requires `lam=`, and `no_H=False` is rejected: screening is heavy-atom only.
- **`Hit.transform` is in the pre-centered frame.** On a `pre_centered=True` store the 4x4 maps
  centered onto centered, so subtract that molecule's centre of mass first, or build the store
  with `pre_centered=False`.
- **`shard_size` is a GPU-memory knob**, not only an I/O one: a whole shard uploads as device
  tensors at once, and a screen holds two shards in host memory, the one aligning plus one read
  ahead.
- **Stores that serve an atom-cloud-seeded mode are canonical by default.** Each molecule is
  stored rotated into its principal-axis frame, which lets those modes run one constant seed set
  instead of a per-molecule eigensolve. Transforms are composed back to the centered frame, so
  `Hit.transform` reads the same either way. See
  [B13](#b13-profile-stores-are-canonical-by-default).
- **`trans_init=True`, `backend="jax"`, or a store that is not pre-centered** drop you onto a
  much slower object path.
- A killed build leaves a readable store of every completed shard. A store is single-writer.

`shard_format="npy"`, the default, writes one `.npy` per array and memory-maps on read; `"npz"`
is still supported.

## 6. Parallelism

### Multi-GPU

Alignment is host-bound rather than kernel-bound, so driving several GPUs from one process
serialises on the interpreter lock. The path that scales is one process per GPU:
`align_multi_gpu(pairs, mode, ndev=4, **kwargs)` for a one-shot call, or
`MultiGPUAligner(pairs, ndev=4)` as a context manager that keeps each GPU's shard resident
across `.align(mode, **kwargs)` calls. Both forward `**align_kwargs` into `align_with_<mode>`,
so a mode's required arguments are required here too, and **both need an
`if __name__ == "__main__":` guard**.

This is deliberately opt-in: a library must not spawn workers behind the caller's back, because
`spawn` re-imports `__main__` and breaks unguarded scripts. A large batch on a multi-GPU host
therefore runs on one GPU and warns once.

`align_multi_gpu` always spawns, so it always needs the guard. `MultiGPUAligner` prefers `fork`
and falls back to `spawn` with a warning if CUDA is already initialized or Open3D already
imported, either of which makes `fork` unsafe, so build the pool before doing CUDA work. Worker
threads are capped to `cores // ndev`, which is mandatory because uncapped workers oversubscribe
the machine, and `Molecule` objects must be picklable since they cross the process boundary.

### CPU

**`num_workers=N` with `backend="numba"`** shards pairs across a persistent pool of
single-threaded processes. Pairs are independent, so the optimization problem is unchanged, but
agreement with one large call is to convergence tolerance rather than bitwise, since a pair's
step count depends on which pairs share its shard. It uses `spawn`, so it needs a `__main__`
guard, and it is ignored on CUDA tensors.

**`accel.screen_parallel.screen_parallel(query, library, mode, n_workers=...)`** forks workers
for query-against-library screening, sharing the featurized library copy-on-write. It is a
different API from `screen()`: a memory-resident list of `Molecule` objects rather than a
`ProfileStore`, CPU and numba only, canonical mode names only, returning a plain list of scores
in library order with no hits, transforms or top-K.

> It always forks, so it is POSIX-only, and it must run before any in-process numba alignment: a
> live libgomp thread pool at `fork` aborts the child. Featurize, then screen.

The forked pool is kept for later calls against the same library, keyed by the list object's
identity and length, so a new or resized list forks afresh while molecules mutated in place
after the first call are not seen by the workers. `screen_parallel_close()` releases it, and it
also runs at interpreter exit. Each worker pins itself to its own physical core and the library
is dealt out strided, because unpinned workers land on sibling threads of busy cores and
contiguous ranges hand one worker the largest compounds.

**Torch is pinned to one intra-op thread for a CPU batch alignment, then restored.** The numba
kernels own the cores there, and an unpinned torch pool spin-waits against them. The change is
scoped rather than global, so it does not reconfigure the caller's torch.

**The fused CPU loop serves 19 of the 21 modes.** It is generic over a mode's terms and
reductions, so a mode joins it by existing. The exclusion is declared on the mode's own
`ModeSpec`, and only the pharmacophore family carries it
([B12](#b12-pharm-cpu-scores-changed)). The numba kernels run with `fastmath=True,
parallel=True` over `NUMBA_NUM_THREADS`, which defaults to every core and would oversubscribe
alongside a process pool, so `screen_parallel` pins it to one.

**The fused and eager CPU loops agree, and that is tested.** `engine.align` falls back to the
eager torch loop on any exception from the fused one, so a mode whose two loops disagreed would
return different scores depending on an unrelated failure.
`tests/test_cpu_fine_loops_agree.py` holds a bound on their disagreement across every mode and
picks up new modes from the registry.

## 7. Surfaces and pharmacophores

**A mesh-free surface generator, opt-in.** `Molecule(..., surface_method="smooth_sdf")` builds
the surface from a smooth-minimum signed-distance field with stochastic sampling instead of
Open3D ball-pivoting and Poisson-disk sampling: no Open3D and no mesh. The default is unchanged,
and `"mesh"` reproduces the original path exactly. The smooth surfacer rounds the concave
atom-border seams, so a model trained on mesh surfaces sees a distribution shift; validate
before using it generatively. It requires `num_surf_points` and raises `ValueError` if given
`density=`. Tunables live on `get_molecular_surface` with defaults in `SMOOTH_SDF_*`.

**`shepherd_score.surface_diagnostics`** validates a surfacer using only numpy and scipy. Its
core quantity is the shell residual, the distance from a point to an atom's sphere rather than
to its center. `leak_metrics` measures how much a cloud leaks atom positions, where a lower
residual means more leak; `crimp_points` finds points on one sphere and near a second;
`center_recovery_attack` fits a sphere per atom, where a higher error is safer; `local_curvature`
and `summarize` give per-point non-flatness and the whole gate in one call.

**Directionless colour pharmacophores.** `Molecule(..., directionless=True)` and
`get_pharmacophores(..., directionless=True)` build isotropic zero-vector pharmacophores for
every family, the ROCS and ROSHAMBO colour convention. Both default to the original behaviour.

> This changes the pharmacophore count, not only the vectors. The donor and acceptor
> multi-vector branches are skipped, so a donor that upstream expanded into one anchor per
> hydrogen now emits a single anchor at the feature position.

`feature_set="rdkit_base"` selects RDKit's base definitions instead of the ShEPhERD SMARTS set.
It keeps six families and renames three (`PosIonizable` to `Cation`, `NegIonizable` to `Anion`,
`LumpedHydrophobe` to `Hydrophobe`); `Halogen` and `ZnBinder` disappear.

**Open3D is imported lazily.** `generate_point_cloud.py` imports it on first use rather than at
module load, because it is slow to import and importing it makes a later `fork` with CUDA
unsafe, which would break the fork-based pools. So `import shepherd_score.container` now works
without Open3D; if it is missing, the error moves from import time to first surface generation,
with its type unchanged. One caveat: the module global `o3d` is a lazy proxy, so
`from shepherd_score.generate_point_cloud import o3d` no longer yields something `isinstance` or
`inspect` treats as a module. Note also that importing `container` eagerly imports `accel.batch`
and `accel.multi_gpu`, which adds no new dependency but is not free.

## 8. API reference

Only what is new or changed. Full signatures are in the code and the API documentation.

### `container`

`__all__` gains `Surface`, `Pharmacophore` and `AlignmentResult` from upstream's refactor, plus
`align_multi_gpu` and `MultiGPUAligner` re-exported from `accel`.

**`Molecule.__init__` gains five keyword arguments.** Four are default-preserving;
`charge_model` is not, and it sits after `partial_charges` rather than last, so a caller passing
more than six positional arguments is affected.

```python
charge_model='xtb'        # 'xtb' | 'mmff'  -- changes ESP scores, see B8
feature_set='shepherd'    # 'shepherd' | 'rdkit_base'
directionless=False       # isotropic "color" pharmacophores
surface_method='mesh'     # 'mesh' | 'smooth_sdf'
fukui=None                # precomputed Fukui field for vol_fukui
```

Also new: `get_positions(no_H=True)` and `get_charges(no_H=True)`, and the per-channel accessors
`get_lipophilicity`, `get_molar_refractivity`, `get_fukui` and `get_atomic_numbers`, each with a
`get_*_positions` returning the heavy-atom centres the channel is aligned with.
`get_pharmacophore` gains `feature_set` and `directionless` alongside upstream's
`return_atom_ids`, `priority_atoms` and `min_ring_priority_atoms`.

**`MoleculePair`** carries one transform and score per mode. `vol` and `vol_esp` keep the
historical `_noH` spelling; the other 19 follow `transform_<mode>` and `sim_aligned_<mode>`. The
legacy `transform_esp`, `sim_aligned_esp` and `*_esp_combo` names remain as read/write
properties.

**`MoleculePairBatch`**: every method gains `backend=None` and `return_aligned=False`, and
`align_with_vol` and `align_with_vol_esp` also gain `alpha=0.81`. Four gotchas:

- **`return_aligned=False` is the default** on the accelerated backends, returning `[None] * N`
  in place of the aligned coordinates. Transforms are still written to each pair; JAX is
  unaffected.
- **`align_with_pharm` returns a 3-tuple** where other modes return a 2-tuple, so code indexing
  `[1]` for arrays gets `None`s.
- **`alpha=` reaches Triton and numba only.** JAX hardcodes 0.81, as upstream did, so a
  different `alpha` with `backend="jax"` is silently ignored.
- **`num_repeats` is accepted and ignored** by `align_with_vol_lipo` and `align_with_vol_color`,
  whose seed count comes from the registry.

### `screen` (new)

```python
ProfileStore.create(path, *, num_surf_points, modes, dtype='float16', shard_size=100_000,
                    pre_centered=True, overwrite=False, canonical=None, shard_format='npy')
screen(query, store, mode='surf_esp', *, backend=None, do_center=None, top_k=1000, ndev=None,
       scores_out=None, alpha=None, progress=False, **align_kwargs) -> List[Hit]
screen_many(queries, store, ...) -> List[List[Hit]]     # same kwargs; streams the library once
```

Also exported: `MoleculeProfile` and `Hit`, plus `close_multigpu_pool`. A store materializes only
the arrays its `modes=` need, and its on-disk format carries a version that is validated on open.

### `accel` (new)

`has_triton()`, `align_multi_gpu(...)`, `MultiGPUAligner` and `clear_caches()`.
`screen_parallel` is reached through its own module,
`from shepherd_score.accel.screen_parallel import screen_parallel`.

**`accel._modes`** is the mode registry, and the modes are data. `SPECS` holds one `ModeSpec`
per mode giving its channels, its objective terms (kernel, reduction, blend weight), its
optimiser schedule and its per-mode tuning. The flat tables every consumer reads are derived
from it: `CANONICAL_MODES`, `LEGACY_MODE_ALIASES`, `PROCESS_MODES`, `MODE_ATTRS`, `MODE_SEEDS`
and `MODE_STEPS` (what `num_repeats=None` and `max_num_steps=None` resolve to),
`CONST_SEED_MODES`, `canonical(mode)` and `spec_of(mode)`.

**`accel.channels`** is the matching per-molecule data table: one `Channel` per named array,
carrying how to read it off a `Molecule`, which pair tensor holds it, how a `ProfileStore`
persists it, and whether it rotates and translates under the canonical frame. Adding
per-molecule data is a row here rather than an edit in five files.

**`accel._stats`** records fine-loop effort and is a no-op until armed. `reset()` enables it,
`summary()` returns the call count, how many took the CUDA-graph path, the step statistics
against the configured budget, and the early-stop fraction, and `disable()` turns it off. It is
process-local, so a forked or spawned worker records only its own process, and an empty summary
means unmeasured rather than full effort.

**Tunable module-level constants**, not environment variables. Edit or monkey-patch them.

| Constant | Module | Default | Effect |
|---|---|---|---|
| `_ESP_STRIDE` | `drivers/engine.py` | 5 | ESP re-scoring interval; 1 is dense |
| `VOL_COLOR_FUSED_MAX_PAD` | `kernels/vol_color_triton.py` | 32 | Above this pad, the fused vol_color kernel is skipped |
| `_GRAPH_WORK_BUDGET` | `drivers/_graphed.py` | 300,000,000 | Default pose budget for graph engagement |
| `_GRAPH_CAP_CEIL` / `_GRAPH_CAP_MIN` | `drivers/_graphed.py` | 262,144 / 2,000 | Hard bounds on the graph pose cap |
| `_GRAPH_CACHE_MAX` | `drivers/_graphed.py` | 24 | Live captured graphs, each pinning GPU buffers |
| `_FINE_CHUNK_POSES` | `batch/_pad.py` | 81,920 | Pose cap per fine-loop sub-batch on the screen path |
| `_CPU_CHUNK_PAIRS` | `batch/_pad.py` | 10,000 | Pairs per call on the CPU path; bounds host memory |
| `_BAND` | `batch/_pad.py` | 16 | Fixed-band pad granularity |
| `SMOOTH_SDF_*` | `generate_point_cloud.py` | | Smooth-SDF surfacer defaults |

### Scoring and surfaces

`get_molecular_surface` gains `method='mesh'|'smooth_sdf'` and the `sdf_*`, `even` and `seed`
tunables; `get_molecular_surface_smooth_sdf` is new, as is the whole `surface_diagnostics`
module.

`get_overlap_pharm`, `get_pharm_combo_score`, `get_pharmacophores` and `build_lookup_tables`
gain `directionless=`, and `get_pharmacophores` also `feature_set`, `return_atom_ids`,
`priority_atoms` and `min_ring_priority_atoms`. All defaults preserve behaviour.
`get_pharm_combo_score`'s combination changed from `(pharm + shape) / 2` to
`(1 - w) * shape + w * pharm`, which is exactly equal at the default `w=0.5` in IEEE-754.
`get_pharmacophores` returns a `Pharmacophore` that unpacks as the old 3-tuple, so
`X, P, V = get_pharmacophores(mol)` still works.

`shepherd_score.alignment` gains an `objective_<mode>_overlay` and an `optimize_<mode>_overlay`
for each of `vol_color`, `vol_tversky`, `vol_esp_tversky`, `vol_lipo`, `vol_atomtype`,
`vol_color_tversky`, `vol_lipo_tversky` and `vol_and_surf_esp_tversky`. These eager torch
references are what the accelerated backends are checked against.

### From upstream, integrated

`Surface`, `Pharmacophore` and `AlignmentResult`, through `Molecule.surface`,
`Molecule.pharmacophore` and `MoleculePair._alignments`. Plus interaction subselection:
`Molecule.select_atoms(atom_indices, ...)` returns a new `Molecule` whose surface, ESP and
pharmacophore are restricted to an atom subset, built on `get_pc(atom_indices=, radial_buffer=)`,
`get_electrostatic_potential(atom_indices=, surf_pos=)` and `Pharmacophore.expand_atom_selection`,
`.subset_to_atoms` and `.priority_labels(...)`. Defaults preserve behaviour; here `select_atoms`
also carries `charge_model`, `surface_method` and `fukui` through to the new molecule.

### Extensibility

Two agent skills ship in `.claude/skills/`: `design-scoring-mode`, which turns a scoring
objective into an autograd reference implementation, and `accelerate-scoring-mode`, which ports
that reference to a Triton kernel and its numba twin, wires them into the device dispatcher and
gates them on parity against the reference. The 14 modes beyond the original seven were built
through them.

To add a mode by hand: register it in `accel/_modes.py` and add a driver. The batch layer and
the screen front-end pick it up from the registry, and `tests/test_mode_registry.py` fails if
the registry and the torch-typed mode spec drift apart.

## 9. Behavior changes

Thirteen are visible to code written against upstream. B5, B6, B8, B9 and B12 matter most.

### B1. `num_repeats` and `max_num_steps` defaults

On all six pre-existing alignment methods, on both `MoleculePair` and `MoleculePairBatch`, 50
and 200 became `None`, resolving per mode ([§2](#2-the-21-alignment-modes)).

This changes results, and not merely as fewer restarts: `alignment/_torch.py` and `_jax.py`
special-case `num_repeats == 50` to load a precomputed 45-point Fibonacci quaternion set, and any
other value takes a branch producing a different set of orientations rather than a subset.
Individual pairs move by up to a few percent. To restore the old behaviour, pass
`num_repeats=50, max_num_steps=200`.

### B2. Mode rename

`esp` became `surf_esp` and `esp_combo` became `vol_and_surf_esp`, across method names, result
attributes and mode strings. Input-side compatibility is complete: legacy method names are
aliases to the same function objects, legacy result attributes are read/write properties, and
every mode string is canonicalized at the entry point. Output-side is not:
`pair.transform_esp` works, but `'transform_esp' in pair.__dict__` is now `False` and
`vars(pair)` shows `transform_surf_esp`, so code that introspects `__dict__` sees the new names.

### B3. Old pickles round-trip

The refactor turned the flat attributes into data descriptors, which take precedence over the
instance dictionary, so an older pickle restored its flat keys and then raised `AttributeError`
on first read. Both classes now define `__setstate__`, remapping flat keys into `Surface`,
`Pharmacophore` and `AlignmentResult`, honouring the renamed-mode aliases, and defaulting state
added after the flat layout (`surface_method`, `_charge_model`, `_fukui`). New-format pickles
are unaffected, and stale flat duplicates are stripped so a shadowed copy cannot drift.

### B4. `apply_SE3_transform` collapses a singleton batch

`apply_SE3_transform(points[None], transform[None])` was `(1, N, 3)` and is now `(N, 3)`. Values
are unchanged and the batched and gradient paths stay bitwise identical, but
`apply_SO3_transform` was not given the same collapse, so with a single-element batch the two
return different ranks, and upstream pairs them on adjacent lines in three places, which
mis-broadcasts silently. The single-instance path also differs by about one float32 unit in the
last place, which breaks bit-for-bit golden values on the `num_repeats == 1` path. `se3.py` also
gains `quaternions_to_SE3_batch(q, t)`.

### B5. The default batch backend changed

`MoleculePairBatch.align_with_*` no longer defaults to JAX. The default is device-aware and uses
a different seed set, so a default call returns different scores ([§3](#3-backends)). Pass
`backend="jax"` for the old numbers; per-pair `MoleculePair.align_with_*` is unaffected. This
also fixes a latent bug: the old JAX default was not installable by default, since `jax` is an
optional extra, so a fresh install hitting the default batch path raised `ImportError`.

### B6. `numba` is a required dependency

It is the default CPU backend, so the package no longer imports without it.

### B7. Pharmacophore scoring: `directional` became `directionless`

Renamed with inverted meaning, so extraction and scoring share one polarity. There is no alias,
because `directional=` did not exist upstream, so update any call: `directional=False` becomes
`directionless=True`.

### B8. ESP charges default to gfn2-xTB

`Molecule(...)` with no `partial_charges` used to fall back to MMFF94. It now defaults to
`charge_model="xtb"`, so ESP modes score on xTB charges and ESP scores differ. Shape, colour and
pharmacophore modes never read charges. Pass `charge_model="mmff"` for the old behaviour.

Two things make this safe. Charges are lazy, generated on first read and then cached, so a
molecule used only for shape, colour or pharmacophore scoring never invokes the xTB subprocess;
constructing a molecule with a surface does build its surface ESP eagerly and pays the cost. And
a missing or non-converging `xtb` falls back to MMFF94 with a `RuntimeWarning`, so xTB is
recommended rather than required.

xTB is not uniformly better. On retrospective enrichment it helps the surface-ESP modes and
hurts atom-centred `vol_esp`: surface ESP rewards a faithful field, while atom-centred ESP suits
atom-centred charges. Choose per mode.

### B9. Early stopping is per pair

Every accelerated fine loop used to stop on a maximum over the whole batch, so one pair that
converged early halted optimization for every other pair in its bucket. The criterion is now
each pair's own best, and the loop breaks only after `early_stop_patience` checks in which no
pair improved.

This changes results, and only upward, at unchanged search effort: the extra work is work that
was already configured and being skipped. Throughput drops on any workload that was truncating.
Scores, throughput numbers and enrichment figures produced before this change should be
re-measured rather than re-labelled, since truncation depended on which pairs shared a bucket.

### B10. `screen(ndev>1)` returned the wrong mode's answers

The multi-GPU screen worker dispatched the `vol` builder and aligner for every mode, so any mode
whose reference tensors satisfied `vol`'s contract silently ran `vol` under another mode's name,
and `pharm` raised `KeyError` instead. Both drivers now share one dispatch selector. Any
`screen(ndev>1)` result for a non-`vol` mode produced before this fix is invalid.

### B11. The screening host path is faster, and reproducible

The reduce no longer touches Python per library molecule, every screen-capable mode aligns
straight from the store's contiguous arrays, and the next shard is read on a background thread.
Hit lists, hit order including exact ties, ids, transforms and `scores_out` are unchanged: this
is a front-end change with no arithmetic in it.

Two reproducibility problems are also gone. Sub-batch sizing derived from free device memory, so
whether a bucket landed under the CUDA-graph budget depended on what the allocator held, and the
two paths do not always agree, so the same screen could return different scores from run to run;
a fixed pose cap makes that a function of the bucket's shape instead. And the per-pair footprint
calibration charged a chunk for memory other objects held, collapsing it to one pair under
memory pressure. Screen scores from a release before this may not reproduce even against
themselves.

### B12. `pharm` CPU scores changed

`pharm`'s fused numba CPU loop and its eager loop land in different optima, because the fused
loop applies its Adam tail before the early-stop check where the eager loop interleaves them the
other way. With 32 seeds, the most of any mode, and a strongly multi-basin objective, that
relocates the optimum rather than perturbing it. The fused path is therefore disabled for the
pharmacophore family, at some cost in throughput. Any `pharm` CPU score produced before this
came from the fused trajectory.

### B13. Profile stores are canonical by default

`ProfileStore.create` now defaults `canonical` to `True` when the store's modes include any mode
that seeds from the heavy-atom cloud (`accel._modes.CONST_SEED_MODES`) and `pre_centered` is on.
Such a store holds every molecule rotated into its own principal-axis frame, with the rotation
kept, so screening those modes runs one constant seed set instead of a per-molecule eigensolve,
and the transforms it returns are composed back to the centered frame.

Because that seed set differs from the per-molecule one, scores from a canonical store differ
slightly from a non-canonical store's and from `MoleculePair`, and a few molecules land in a
different optimizer basin. `surf`, `surf_esp` and `pharm` seed from the surface or anchor
clouds, which the store does not canonicalise, so they keep their per-molecule generator and
gain no speed; a store serving only those stays non-canonical. Stores already on disk keep their
manifest's setting and are unaffected. Pass `canonical=False` for the previous layout.

### Minor

- `align_with_vol_esp(lam=...)` was a required positional and now defaults to `0.1`.
- `MoleculePair.__init__` eagerly allocates two torch tensors on the target device, so
  constructing a pair touches GPU memory before any `align_*` call.
- `get_overlap_pharm(directionless=True)` raises `ValueError` with `precomputed_self_overlaps`
  and silently forces `extended_points` and `only_extended` off; the numpy variant forces them
  off without raising. Neither affects the `directionless=False` default.

## 10. Limitations

- **The JAX path is the least exercised surface.** Almost every skipped test in a routine run is
  a JAX test, so exercise it before relying on it as B5's reproducibility escape hatch. There is
  also no JAX-against-Triton parity test: the seed sets differ, so what is needed is a quality
  comparison rather than an equality test.
- **Ten of the 21 modes have no published throughput number**: `vol_mr`, `vol_atomtype`,
  `vol_pharm`, `vol_avoid` and six of the Tversky variants. They are correctness-tested, not
  performance-characterised, and several carry tuning constants inherited from a sibling mode
  rather than measured for themselves. Do not quote a rate for them by analogy.
- **Parity claims are strongest at small batch sizes.** The pairwise, screen, graph-against-eager
  and array-against-object comparisons behind them were built on small fixtures, and two real
  regressions once survived that sweep: a kernel-launch slice that corrupted scores past a pose
  count no fixture reached, and a store built for a single mode that lacked the arrays that mode
  reads. Both are fixed and both now have gates
  (`tests/test_grid_chunked_launch.py`, `tests/test_store_minimal_schema.py`). Before trusting a
  parity number, run a GPU batch large enough to be sliced, build a store per mode, and use a
  realistic batch size.
- **`vol_and_surf_esp`'s graph and eager paths compute different algorithms.** The eager loop
  applies `_ESP_STRIDE` and the captured graph step does not, so the graph scores every step and
  returns uniformly higher values. The screen runs eager; treat the two as not interchangeable.
- **Bit-identity results come from workloads that do not early-stop.** A very small chunk or a
  low-diversity library can fire the early stop and reopen a graph-against-eager divergence.
