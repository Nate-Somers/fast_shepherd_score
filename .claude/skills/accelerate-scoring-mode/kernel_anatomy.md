# Kernel and engine anatomy

How the kernels, the dispatch, the term evaluators and the one fine loop fit together. Read this
alongside an existing family's kernel before writing your own.

## The value+gradient kernel

Each channel pair has one fused kernel computing, in a single launch, both the overlap value and
its gradient with respect to the SE(3) pose. It does not return an intermediate the host has to
differentiate — the analytic gradient is computed in-register.

- **Inputs**: padded batch tensors (reference and fit positions / charges / pharm types, anchors
  and vectors, as the term needs), the current unit quaternion `q` and translation `t`, and the
  real per-pair counts `N_real` / `M_real` used to mask padding.
- **Outputs**: the scalar overlap per pair, `dO/dq` (unit-quaternion space) and the translation
  gradient.
- **`NEED_GRAD`** is a flag on every kernel, so value-only passes share one implementation.

### Emit `dO/dq` in-register

Older code computed a gradient in rotation-matrix space and projected it back to the quaternion on
the host. Do not do that. Compute `dO/dq` inside the kernel, reusing the shape channel's validated
`dR/dq` tail (`_quat_grad_tail` in `shape_triton.py`):

- **Positional term**: force ⊗ fit-anchor, fed through the tail.
- **Directional term** (if the channel weights orientation vectors): Σ over vector pairs of the
  per-feature coefficient times (ref-vec ⊗ fit-vec), through the same tail.

`q` is kept unit each step (the Adam renormalizes), so pass a unit quaternion and the kernel need
not carry the normalization Jacobian in the hot loop. The kernels therefore emit a **raw** `dO/dq`
including the radial component, which the optimizer discards — this matters when you compare
against autograd, see `parity_gates.md`.

`pharm_score_grad_se3_batch` is the surviving R-matrix-gradient variant. It is still exported by
dispatch but **no term uses it**; do not model a new kernel on it.

### Combining channels is the ENGINE's job, not a kernel's

A blended mode is a weighted sum in the same quaternion space:

```
dO/dq = (1 - w) * (scale_shape * dQ_shape) + w * (scale_field * dQ_field)
```

Because projection is linear, scaling each term's `dO/dq` and summing is identical to projecting a
combined rotation-space gradient, but cheaper. `engine._step_generic` does exactly this, over an
arbitrary number of terms, from the spec's weights. **This is why a blended mode needs no new
kernel and no new driver**: it declares two `Term`s and the engine blends what existing kernels
already emit.

`vol_avoid` is the instructive exception: its descent gradient is `-scale_s·dQ_shape +
avoid_weight·dQ_avoid`, with the penalty raw and un-normalized (no Tanimoto scale) and a **positive**
sign, because the penalty is subtracted from the score. In the registry that is a term with
`reduction="raw"` and `weight="-avoid_weight"`; the engine's sign handling falls out of the weight.

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
fused shape+colour single-kernel variant, reached through `ModeSpec.fused_pair` and used only when
every pad is at or below `VOL_COLOR_FUSED_MAX_PAD` (32). It has no numba twin, which is exactly why
it is not in dispatch — the engine falls back to the two separate kernels on CPU or past that pad.

## Adding a kernel to the engine: three small branches

A new kernel is not wired by writing a driver. It is three branches:

1. `drivers/terms.py::evaluate` — one `if term.kernel == "<yours>":` returning `(V, dQ, dT)`.
2. `drivers/terms.py::self_overlap` — one branch, **only** if your reduction needs a self-overlap
   (`tanimoto`, `tversky` and `pharm_sim` do; `raw` and `agreement` do not).
3. `kernels/cpu_fused.py::_term_closure` — one closure marshalling the term's tensors to numpy
   once and returning `(q, t) -> (V, dQ, dT)` float32, so the mode keeps the fused CPU path.

Skip (3) and the mode still works — it just falls back to the eager loop on CPU. Skip (1) and it
raises `KeyError` at the first fine step.

## The fine loop (`drivers/engine.py`)

One body serves every mode. `align()`:

1. **`assemble`** — resolve the spec's channels under this call's keywords (`channel_switch`),
   centre the seed clouds if `center_clouds`, build each term's inputs, compute the pose-invariant
   self-overlaps each reduction needs, generate (or accept) the seeds, and expand everything into
   the per-pose layout.
2. **the graph path** — `run_graphed` on `_GraphedFineTerms`, gated by the spec's `graph_budget`.
3. **the fused CPU path** — `cpu_fused.run_fused`, gated by `cpu_fused` / `cpu_fused_max_pad`.
4. **the eager path** — the fall-back, and the reference all three agree with.
5. **gather** — each pair's best over its own seeds, with the pharm family's centring folded back
   into the returned translation.

All three loops run the **same per-step arithmetic in the same order**: evaluate every term, track
the best PRE-Adam pose on the blended score, form the blended descent gradient, then (after the
early-stop check) apply the tangent-projected Adam. That ordering is deliberate — the old fused CPU
loop applied Adam *before* the check, which relocated the optimum for multi-basin modes.

**Blocked, per-pair early stop.** The convergence test is per pair: a pair has converged when its
own best — the max over its own seeds — stops improving, and the loop breaks only once no pair has
improved for `patience` checks. Never test a maximum over the whole bucket: that lets one converged
pair halt every pair sharing it, silently truncating the search and making the mode read faster
than it is. The graph replays once to seed `prev`, then checks every `_GRAPH_ES_BLOCK` (5) replays,
with `_GRAPH_ES_MARGIN` (2) added to the eager patience. One host sync per block, not per step.

**Graph engagement is not uniform**, and the budget is per mode (`ModeSpec.graph_budget`):

| Budget | Modes |
|---|---|
| `3e8` (the `_GRAPH_WORK_BUDGET` default) | shape, ESP, and their Tversky variants |
| `3e7` | `vol_color`, `vol_lipo`, `vol_mr`, `vol_fukui`, `vol_atomtype` and their Tversky variants |
| `1e7` | `pharm`, `pharm_tversky`, `vol_pharm` |
| `8e6` | `vol_and_surf_esp` and its Tversky variant |
| `None` | `vol_avoid` — eager only, by design: its objective is piecewise-linear and kinked while the graph's early-stop schedule is tuned for smooth Gaussians |

Whether a bucket graphs is a function of its pad shape and that budget, via `graph_cap(work,
budget)` clamped between `_GRAPH_CAP_MIN` (2,000) and `_GRAPH_CAP_CEIL` (262,144). It is
deliberately **not** a function of allocator state: sub-batch sizing used to derive from *free*
device memory, which made the same screen return different scores run to run.

Graphs live in a bounded LRU of `_GRAPH_CACHE_MAX` (24) and **pin GPU buffers for the process
lifetime**; `reset_graph_cache()` releases them. An OOM during capture evicts the LRU and retries.

Two caveats worth knowing before you trust a graphed number:

- **Tversky modes keep BOTH the CUDA graph and the fused CPU loop.** The old fused CPU entry point
  hardcoded Tanimoto, which excluded them; the generic one takes the reduction from the spec, so
  they no longer forfeit it.
- **`vol_and_surf_esp`'s graph and eager paths compute different algorithms.** The eager loop
  applies `_ESP_STRIDE` (5) to its value-only ESP term and the captured step does not, so the graph
  scores every step and returns uniformly higher values. `graph_full_steps` records that the mode
  replays the full step count. The screen runs eager. Do not treat the two as interchangeable.

## Seeds

Three sources, and you need to know which one your mode gets.

**Per-molecule** — `_common.batched_seeds_torch(A, B, N_real, M_real, num_seeds, *,
ref_shared=False)`. The set is identity, then 4 PCA principal-axis-alignment quaternions, then up
to 6 **structured** ±90° rotations about each reference principal axis (the axis *swaps* that sign
flips miss), then a Fibonacci fill. Translations align centres of mass. This is **not** the
reference/JAX seeder's set, which has no structured seeds — which is why cross-backend scores are
not comparable.

The eigensolve is a closed-form Cardano solve in float64 (`_analytic_sym3x3_axes`), deliberately
not `torch.linalg.eigh`, which syncs and fails past ~8192 rows. Two dedups apply: the reference side
is solved once when `ref_shared=True`, which the caller must establish by **object identity**
(`a is b`) or structurally (the screen broadcasts one query into every row); and the fit side is
deduped at the first axis iteration.

**Constant / canonical-frame** — `_common.canonical_seed_quats(...)` returns one seed set for the
whole screen, valid only against a canonical store where every molecule is already in its own
principal frame. One 3×3 solve per screen instead of one per molecule. A mode gets it when its
`seed_channel` resolves to `atoms` or `heavy` — `CONST_SEED_MODES` is derived from exactly that, so
a new mode seeding from the atom cloud inherits it. It is not bit-identical to per-molecule seeds,
by construction.

**Coarse-grid translations** — `legacy_seeds_with_translations_torch` and `build_coarse_grid`, used
only on the `trans_init` path.

Seed and step **counts** come from the spec via `aligners._seeds_for` / `_steps_for`. `num_repeats`
reaches the seed count only for modes whose spec sets `honors_num_repeats` (the pharm family); the
others accept and ignore it.

## The optimizer is not `torch.optim.Adam`

β₁=0.9, β₂=0.999, ε=1e-8 *inside* the sqrt, **no bias correction**, a quaternion tangent-space
projection fused into the kernel, and unit-quaternion renormalization every step. `lr` does not mean
what it means in `torch.optim.Adam`, and the drivers' internal default is `0.075`, not the `0.1` the
public API advertises — which is why `ModeSpec.screen_lr` records per mode which one the screen
front end uses. Early stopping is per mode and per pair (`ModeSpec.patience`: 2 for the shape and
colour modes, 5 for the ESP and pharmacophore ones), tolerance 1e-5, checked every 5 steps.

## Uploads: `_batch_upload` is keyed on the MOLECULE

`accel/batch/aligners.py:_batch_upload(pairs, attr, src_fn, dtype, device, *, key_fn=None)`.

It skips warm pairs entirely, infers `key_fn` from the attribute prefix (`_ref*` → the ref molecule,
`_fit*` → the fit molecule), picks one representative pair per distinct molecule *object*, does one
concatenate and one host-to-device copy, and clones once **per molecule**. An all-vs-all workload
draws K pairs from far fewer distinct molecules — 100,000 pairs from 317 compounds, each appearing
~632 times — so pair-keying uploaded every molecule hundreds of times. This was 61.3% of a
`vol_color` pairwise batch.

Two consequences for a new channel:

- The `Channel.attr` you choose becomes `_ref_<attr>_t` / `_fit_<attr>_t`, so the key inference
  works automatically. Do not invent a name outside that shape.
- Molecules sharing an object share one tensor. That is safe only because these are **read-only**
  inputs; if you ever add an in-place op on a `_ref_*_t` / `_fit_*_t` tensor you break it.

The screen path never exercises this: the fit builders pre-warm the attributes, so `cold` is empty.

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
- **The fused CPU loop is now near-universal**: 19 of 21 modes take it. The pharmacophore family
  opts out via `cpu_fused=False`, measured — its float32 tail rounding alone (the schedule matches
  eager step for step) moved 5 of 12 screen scores by up to 35.7% relative, against an old-measured
  gain of only 1.025–1.046x. `surf_esp` keeps `cpu_fused_max_pad=100`.

## Padding and masking

Pad per-pair inputs to a common width, but mask padded slots by the **real count**, not by a
sentinel type value — masking on a magic type index breaks the moment the type table is reordered.
Carry `N_real` / `M_real` and mask on those. (`Channel.pad` exists so a padded slot holds something
harmless — the Dummy pharmacophore family, element 0 — not so anything masks on it.)

Bucketing is result-identical by construction: the kernels are one-CTA-per-pose and mask to the real
counts, and seeds key on real counts rather than pad width, so padding two different-sized molecules
into one bucket cannot change a score. Sub-batching bounds peak memory — on CUDA by a fixed pose cap
(`_FINE_CHUNK_POSES`, armed per mode through `_arrays._POSE_CAP_MODES`), on CPU by
`_CPU_CHUNK_PAIRS` (10,000).
