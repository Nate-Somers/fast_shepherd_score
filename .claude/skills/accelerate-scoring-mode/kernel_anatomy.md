# Kernel anatomy

How the accel kernels, dispatch, drivers and graph loop fit together. Read this alongside an
existing family's kernel and driver before writing your own.

## The value+gradient kernel

Each channel has one fused kernel computing, in a single launch, both the overlap value and its
gradient with respect to the SE(3) pose. It does not return an intermediate the host has to
differentiate — the analytic gradient is computed in-register.

- **Inputs**: padded batch tensors (reference and fit positions / charges / pharm types, anchors
  and vectors, as the mode needs), the current unit quaternion `q` and translation `t`, and the
  real per-pair counts `N_real` / `M_real` used to mask padding.
- **Outputs**: the scalar overlap per pair, `dO/dq` (unit-quaternion space) and the translation
  gradient.
- **`NEED_GRAD`** is a flag on every kernel, so value-only passes share one implementation.

### Emit `dO/dq` in-register

Older code computed a gradient in rotation-matrix space and projected it back to the quaternion on
the host. Do not do that. Compute `dO/dq` inside the kernel, reusing the shape channel's validated
`dR/dq` tail (`_quat_grad_tail` in `shape_triton.py`):

- **Positional term**: force ⊗ fit-anchor, fed through the tail.
- **Directional term** (if the mode weights orientation vectors): Σ over vector pairs of the
  per-feature coefficient times (ref-vec ⊗ fit-vec), through the same tail.

`q` is kept unit each step (the Adam renormalizes), so pass a unit quaternion and the kernel need
not carry the normalization Jacobian in the hot loop. The kernels therefore emit a **raw** `dO/dq`
including the radial component, which the optimizer discards — this matters when you compare
against autograd, see `parity_gates.md`.

`pharm_score_grad_se3_batch` is the surviving R-matrix-gradient variant. It is still exported by
dispatch but **no driver uses it**; do not model a new kernel on it.

### Combining channels

A combined mode is a weighted sum in the same quaternion space:

```
dO/dq = (1 - w) * (scale_shape * dQ_shape) + w * (scale_field * dQ_field)
```

Because projection is linear, scaling each channel's `dO/dq` and summing is identical to projecting
a combined rotation-space gradient, but cheaper and done in one place. This is why a blended mode
usually needs **no new kernel**: the driver blends `dQ`s that existing kernels already emit.

`vol_avoid` is the instructive exception — its descent gradient is
`-scale_s·dQ_shape + avoid_weight·dQ_avoid`, with the penalty raw and un-normalized (no Tanimoto
scale) and a **positive** sign, because the penalty is subtracted from the score.

## Dispatch (`accel/kernels/dispatch.py`)

Each kernel name is exported as a thin wrapper routing a *call* to the implementation matching the
**device of its first tensor argument**:

- CUDA tensors → the Triton kernel; CPU tensors → the numba kernel.
- The choice is per call, never frozen at import, so one process can run both. That is what lets
  `backend="numba"` run CPU tensors on a box where Triton also imports.
- Triton source modules are imported lazily — only the first time a CUDA tensor is dispatched — so
  importing `dispatch` on a CPU-only box never touches them.
- Resolution is cached in `_RESOLVED[(name, tag)]`, so dispatch costs one dict lookup after the
  first call. Tests assert both tags land there in one process.

The numba and Triton kernels must have **identical signatures**; the wrapper cannot adapt between
calling conventions.

One kernel deliberately bypasses dispatch: `vol_color_triton.vol_color_score_grad_se3_batch`, the
fused shape+colour single-kernel variant, is imported directly by `drivers/vol_color.py` and used
only when every pad is at or below `VOL_COLOR_FUSED_MAX_PAD` (32). It has no numba twin, which is
exactly why it is not in dispatch.

## The driver and the CUDA-graph fine loop

The batched driver runs the fine loop over all seeds and takes the per-pair maximum. The fine loop
is shared across modes in `drivers/_graphed.py`: capture the per-step kernel+optimizer update once
as a CUDA graph, then replay it.

```python
run_graphed(make, key, inputs, *, es_patience=0, es_tol=1e-5, es_seeds=0)
```

- `make` is a zero-arg factory for your `_GraphedFineBase` subclass, called **only on cache miss**.
- `key` is the cache key `(device, mode, shapes, P, steps, params)`.
- `es_seeds` is the **seed rows per pair**, and it is set on every call including cache hits,
  because it is a property of the call rather than of the cached graph. Pass the same value your
  driver uses to gather its own result.

Subclass hooks: `_step()`, `_load(*x)`, `_reset()`, `_result()`. Capture does three warmup steps on
a side stream, then `torch.cuda.graph(..., capture_error_mode="thread_local")`.

**Blocked, per-pair early stop.** The convergence test is per pair: a pair has converged when its
own best — the max over its own seeds — stops improving, and the loop breaks only once no pair has
improved for `es_patience` checks. Never test a maximum over the whole bucket: that lets one
converged pair halt every pair sharing it, silently truncating the search and making the mode read
faster than it is. The loop replays once to seed `prev`, then checks every `_GRAPH_ES_BLOCK` (5)
replays, with `_GRAPH_ES_MARGIN` (2) added to the eager patience so the graphed schedule neither
over- nor under-runs relative to eager. One host sync per block, not per step.

**Graph engagement is not uniform**, and the budget is per mode, not global:

| Budget | Modes |
|---|---|
| `3e8` (the `_GRAPH_WORK_BUDGET` default) | shape, ESP, and their Tversky variants |
| `3e7` | `vol_color`, `vol_lipo`, `vol_atomtype` and their Tversky variants |
| `1e7` | `pharm`, `vol_pharm` |
| `8e6` | `vol_and_surf_esp` and its Tversky variant |
| none | `vol_avoid` — eager only, by design |

Whether a bucket graphs is a function of its pad shape and that budget, via
`graph_cap(work, budget)` clamped between `_GRAPH_CAP_MIN` (2,000) and `_GRAPH_CAP_CEIL` (262,144).
It is deliberately **not** a function of allocator state: sub-batch sizing used to derive from
*free* device memory, which made the same screen return different scores run to run.

Graphs live in a bounded LRU of `_GRAPH_CACHE_MAX` (24) and **pin GPU buffers for the process
lifetime**; `reset_graph_cache()` releases them. An OOM during capture evicts the LRU and retries.

Two caveats worth knowing before you trust a graphed number:

- **Tversky modes forfeit the fused-CPU path but KEEP the CUDA graph.** `cpu_fused_shape`
  hardcodes the Tanimoto reduction, so no Tversky driver calls it; each one does have its own
  `_GraphedFine*` subclass and takes the graph under the usual budget. Do not remove a Tversky
  mode's graphed path on the strength of the fused-CPU exclusion.
- **`vol_and_surf_esp`'s graph and eager paths compute different algorithms.** The eager loop
  applies `_ESP_STRIDE` (5) and the captured step does not, so the graph scores every step and
  returns uniformly higher values. The screen runs eager. Do not treat the two as interchangeable.

Use the helpers in `drivers/_common.py` for seed generation, transforms and result extraction so
your driver stays small. Note `_update_best` returns **new** tensors and must not be used inside a
captured `_step`.

## Seeds

Three sources, and you need to know which one your mode gets.

**Per-molecule** — `_common.batched_seeds_torch(A, B, N_real, M_real, num_seeds, *,
ref_shared=False)`. The set is identity, then 4 PCA principal-axis-alignment quaternions, then up
to 6 **structured** ±90° rotations about each reference principal axis (the axis *swaps* that sign
flips miss), then a Fibonacci fill. Translations align centres of mass. This is **not** the
reference/JAX seeder's set, which has no structured seeds — which is why cross-backend scores are
not comparable.

The eigensolve is a closed-form Cardano solve in float64 (`_analytic_sym3x3_axes`), deliberately
not `torch.linalg.eigh`, which syncs and fails past ~8192 rows. Two dedups apply: the reference
side is solved once when `ref_shared=True`, which the caller must establish by **object identity**
(`a is b`), not by value; and the fit side is deduped at the first axis iteration.

**Constant / canonical-frame** — `_common.canonical_seed_quats(ref_points, n_real, num_seeds,
device)` returns one seed set for the whole screen, valid only against a canonical store where
every molecule is already in its own principal frame. One 3×3 solve per screen instead of one per
molecule. It is plumbed as `const_seeds`, accepted by `align_batch_vol_arrays` **alone** — the
object path's `_align_batch_vol` raises `TypeError` on it — and is gated to `mode == "vol"` with a
single query. It is not bit-identical to per-molecule seeds, by construction.

**Coarse-grid translations** — `legacy_seeds_with_translations_torch` and `build_coarse_grid`, used
only on the `trans_init` path.

Seed and step **counts** come from `MODE_SEEDS` / `MODE_STEPS` via `aligners._seeds_for` /
`_steps_for`. Note that `_align_batch_vol_color` and `_align_batch_vol_lipo` **accept and ignore**
`num_repeats`; their seed count comes from the table alone.

## The optimizer is not `torch.optim.Adam`

β₁=0.9, β₂=0.999, ε=1e-8 *inside* the sqrt, **no bias correction**, a quaternion tangent-space
projection fused into the kernel, and unit-quaternion renormalization every step. `lr` does not
mean what it means in `torch.optim.Adam`, and the drivers' internal default is `0.075`, not the
`0.1` the public API advertises. Early stopping is per mode and per pair: patience 2 for
`vol`/`surf`/`vol_color`, 5 for the ESP and pharmacophore modes, tolerance 1e-5, checked every 5
steps.

## Uploads: `_batch_upload` is keyed on the MOLECULE

`accel/batch/aligners.py:_batch_upload(pairs, attr, src_fn, dtype, device, *, key_fn=None)`.

It skips warm pairs entirely, infers `key_fn` from the attribute prefix (`_ref*` → the ref
molecule, `_fit*` → the fit molecule), picks one representative pair per distinct molecule
*object*, does one concatenate and one host-to-device copy, and clones once **per molecule**. An
all-vs-all workload draws K pairs from far fewer distinct molecules — 100,000 pairs from 317
compounds, each appearing ~632 times — so pair-keying uploaded every molecule hundreds of times.
This was 61.3% of a `vol_color` pairwise batch.

Two consequences for a new mode:

- Name your tensor attributes with the `_ref_` / `_fit_` prefix so the key inference works. An
  un-keyable attribute silently falls back to per-pair uploads.
- Molecules sharing an object share one tensor. That is safe only because these are **read-only**
  inputs; if you ever add an in-place op on a `_ref_*_t` / `_fit_*_t` tensor you break it.

The screen path never exercises this: `build_fit` pre-warms the attributes, so `cold` is empty.

## numba specifics

- Torch is pinned to one intra-op thread for a CPU batch and restored afterwards — the numba
  kernels own the cores, and an unpinned torch pool spin-waits against them.
- The numba kernels run `fastmath=True, parallel=True` over `NUMBA_NUM_THREADS` (all cores by
  default), which oversubscribes alongside a process pool. That is why `screen_parallel` pins it
  to 1 and pins each worker to a physical core.
- **SVML changes precision.** With `numba 0.59.1 + llvmlite 0.42 + icc_rt`, the shape and ESP inner
  loops switch to fp32 structure-of-arrays kernels (`cpu_soa.py`): values ~1e-6 relative, gradients
  ~1e-4. Without it they are correct but much slower, and emit a one-time `RuntimeWarning`.
  A CPU throughput number taken without SVML understates the library by a large factor.
- **The fused CPU loop is not applied uniformly.** `cpu_fused` is called by `shape`, `esp` and
  `vol_color` only; `surf_esp` is excluded above 100 padded points, and `pharm` is excluded
  entirely — its fused loop applies the Adam tail *before* the early-stop check where the eager
  loop interleaves the other way, which with 32 seeds relocates the optimum rather than perturbing
  it. `cpu_fused_pharm` is kept as dead-but-working code because it is what anyone fixing that
  ordering would need.

## Padding and masking

Pad per-pair inputs to a common width, but mask padded slots by the **real count**, not by a
sentinel type value — masking on a magic type index breaks the moment the type table is reordered.
Carry `N_real` / `M_real` and mask on those.

Bucketing is result-identical by construction: the kernels are one-CTA-per-pose and mask to the
real counts, and seeds key on real counts rather than pad width, so padding two different-sized
molecules into one bucket cannot change a score. Sub-batching bounds peak memory — on CUDA by a
fixed pose cap (`_FINE_CHUNK_POSES`), on CPU by `_CPU_CHUNK_PAIRS` (10,000).
