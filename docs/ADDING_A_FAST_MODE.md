# Adding a customization to fast_shepherd_score and getting it on the fast backend

A self-contained playbook for a coding agent with no prior knowledge of the repo.
All paths are under `fast_shepherd_score/shepherd_score/` unless noted. Derived from real
shipped changes: the **`vol_color`** mode addition (new combo mode, end-to-end), the
**`pharm`** in-register-dQ kernel upgrade (speeding up an existing mode), and the
**`esp_combo`** fused-ESP-channel + numba-backend completion.

**If you already have a working PyTorch autograd objective for your customization** (the
most common real starting point), go straight to [**Scenario C**](#scenario-c--you-already-have-a-working-autograd-objective-the-common-case)
— it turns that objective into your gradient oracle, gives you a no-kernel GPU speedup
first, and only then ports to a fused kernel.

---

## 0. Mental model (read first)

A *mode* is an SE(3) alignment objective optimized by coarse-to-fine Adam (rotation
quaternion `q` + translation `t`). The repo is layered:

1. **Scalar scoring math** — `score/*` in three parallel implementations: torch
   (`gaussian_overlap.py`, `electrostatic_scoring.py`, `pharmacophore_scoring.py`),
   numpy (`*_np.py`), jax (`*_jax.py`).
2. **Per-pair optimizers** — `alignment/_torch.py` (eager autograd),
   `alignment/_torch_analytical.py` (hand-derived gradients), `alignment/_jax.py`.
3. **Fast batched backend** — `accel/*`: optimizes thousands of pairs at once on fused
   **Triton (CUDA)** / **numba (CPU)** kernels.
4. **Container / public API** — `container/_core.py` (`MoleculePair`, per-pair) and
   `container/_batch.py` (`MoleculePairBatch`, batched) route a request to a backend.

### The four backends are chosen at TWO layers — this is the key thing to understand

- **Layer 1 — `backend=` string** (`container/_batch.py:_run_fast_or_fallthrough`,
  lines ~98–116). `backend='jax'` → the original sequential/sharded path; it **never
  enters** the fast `_align_batch_*` code. `backend` in `_TRITON_BACKENDS =
  ('triton','cuda','gpu')` or `_NUMBA_BACKENDS = ('numba','cpu')` → both call the **same**
  `_align_batch_<mode>` code; the only difference is `_prepare_numba()` first moves every
  tensor to CPU. Anything else raises `ValueError`. **Triton-vs-numba is NOT decided here.**
- **Layer 2 — tensor DEVICE** (`accel/kernels/dispatch.py`). Every kernel the drivers
  import is built by `dispatch._make(name, triton_tag)`. The wrapper inspects the device
  of the first tensor arg (`_args_on_cuda`): **CUDA → Triton** (`shape`/`esp`/`pharm`
  module), **CPU → numba** (`cpu.py`); the result is memoized in `_RESOLVED`. Triton
  modules import lazily, so a CPU-only box never touches them.

**Consequence:** a new kernel must ALWAYS exist in two forms with *identical signatures* —
Triton (`accel/kernels/*_triton.py`) and numba (`accel/kernels/cpu.py`) — registered once
via `_make`. You never branch on backend inside a driver. `backend='numba'` works on a
CUDA box precisely because `_prepare_numba` moves tensors to CPU and the kernel wrapper
then selects numba by device.

### Prefer the in-register dO/dq kernel family

Kernels come in two families:

- **q-in / dO/dq-out (preferred):** take the unit quaternion `q`, build `R(q)` in
  registers, emit `dO/dq` directly via the analytic `dR/dq` tail. The driver applies the
  Tanimoto/Tversky scalar scale **straight to `dO/dq`** and calls
  `fused_adam_qt_with_tangent_proj` — **no host matrix→quaternion projection**. (shape,
  esp, fused-surf, `pharm_grad_dq_se3_batch`, `pharm_color_score_grad_se3_batch`.)
- **R-in / grad_R-out (legacy):** emit `dO/dR (P,3,3)`, requiring a per-step host
  projection tail (`apply_*_chain_rule` + `project_grad_R_to_quaternion`). The only
  remaining instance is the directional `pharm_score_grad_se3_batch` (extended_points).

Each overlap kernel is **one tiled pass** that always accumulates the value and, gated by
the compile-time constexpr `NEED_GRAD`, also the gradient (dead-code-eliminated when
`False`, so value-only self-overlaps are free). Gaussians use `tl.exp2(x·log2 e)` on
Triton vs `math.exp` on numba (numerically equal in fp32, not bit-identical). Heavy
kernels are `@triton.autotune(key=['N_pad','M_pad'], cache_results=True)` → **requires
triton ≥ 3.6**.

### Two more shared mechanisms

- **Seeds:** all paths share one generator. Fast batch uses
  `accel/drivers/_common.py:batched_seeds_torch(A, B, N_real, M_real, num_seeds)` (identity
  + 4 PCA-alignment quats in float64 + optional ±90° axis-swap seeds via `FSS_STRUCT_SEEDS`
  + Fibonacci fill). Per-mode counts live in `accel/batch/aligners.py:_MODE_SEEDS`
  (verified: `{"vol":18,"surf":20,"esp":40,"vol_esp":40,"pharm":40,"vol_color":40}`,
  default 50) read by `_seeds_for(mode)` (env `FINE_NUM_SEEDS` overrides). Seed from the
  **shape/atom cloud**, never the pharmacophore anchors.
- **Padding:** `accel/batch/_pad.py:_band_key(n)` rounds `n` up to the next multiple of 16;
  same-band pairs share one padded workspace + one kernel launch. Real counts go in int32
  `N_real`/`M_real`; kernels mask padded slots. Pharm type slots pad with `Dummy = 8`
  (category 3, skipped). `_subbatched_align` splits into GPU-memory-safe chunks
  (halve-on-OOM).

---

## Decision guide — which path do you actually need?

| Your customization | Path |
|---|---|
| **You already have a working autograd objective** (a `loss.backward()` optimizer that produces correct alignments) | **Scenario C** below — the common case. Batch it over pairs for a no-kernel GPU speedup first; port to a fused kernel only if that's the measured bottleneck, using autograd as the gradient oracle. |
| New mode/term, need it fast now, don't want to write CUDA/Triton yet | **Minimum viable fast path** (below): reuse an existing fused kernel for the matching channel + score the new term in pure torch each step. How `vol_color`/`esp_combo` started. |
| New term is an **SE(3)-invariant multiplicative per-pair weight** (depends only on fixed per-point scalars like charge, not coordinates) | Fold it into the kernel as one more `exp2` factor, exactly like ESP folded `exp(-‖dq‖²/lam)` into the shape kernel. Gradient structure unchanged; reuse the whole shape gradient. Cheapest "real kernel" extension. |
| New **pharmacophore feature type** / orientation category (not new geometry) | No new kernel. Append to `P_TYPES`/`P_ALPHAS` (`constants.py`), extend `build_lookup_tables` (category + alpha/K) and the cat-based `w`-selection (`tl.where`) in the pharm kernels + jax `P_TYPE_CONFIG_MAP`. |
| **Joint combo** of two channels, both must steer the pose | Copy `accel/drivers/vol_color.py`: each channel emits `dO/dq` in the same quaternion space; sum `g_q=(1-w)·(-scale_a·dQ_a)+w·(-scale_b·dQ_b)`. If one channel's gradient doesn't matter, copy `esp_combo.py` (score with both, steer with one). |
| Existing mode is correct but **slow** because its kernel emits `grad_R` + a host projection tail | **Scenario B** below — move it onto an in-register dO/dq kernel (the pharm worked example, ~1.4–1.6×). |
| Genuinely new pairwise functional form with a **coordinate-dependent gradient** that dominates runtime | Write a dedicated value+grad kernel (Triton + numba twin). **Scenario A** below, full path. |
| Only ever a handful of pairs, not thousands | Stop at the per-pair layer: `optimize_<mode>_overlay` in `alignment/_torch.py` + `MoleculePair.align_with_<mode>`. Skip `accel/*` entirely. |

### Minimum viable fast path (no new kernel)

If your customization is a **re-blend of existing channels** or an **SE(3)-invariant
per-pair weight**, you do not need a new kernel:

- Reuse the fused kernel for whatever channel matches — shape via
  `overlap_score_grad_se3_batch`, esp via `overlap_score_grad_esp_se3_batch`, color via
  `pharm_color_score_grad_se3_batch`.
- **Score** your new term in pure torch each fine step and **steer** the pose with the
  existing channel's gradient. This is exactly how `esp_combo` still works (shape gradient
  only; ESP term scored in plain torch).
- Ship this first. Only write a dedicated kernel once the per-step torch score+grad is the
  *measured* bottleneck, or the new term needs its own coordinate-dependent gradient.

---

## SCENARIO C — you already have a working autograd objective (the common case)

You've written a PyTorch function that scores a pose and you optimize it with
`loss.backward()`. **That is the best possible starting point.** You already have the
per-pair layer *and* a gradient oracle for every layer below it — porting to the fast
backend becomes a sequence of validated, low-risk steps. Climb this ladder and **stop at the
first rung that's fast enough.**

### C0. Confirm your objective matches the repo's per-pair shape
A working autograd objective *is* `objective_<mode>_overlay` / `optimize_<mode>_overlay`
(step A3). The repo's contract (see `objective_ROCS_overlay` + `optimize_ROCS_overlay` in
[`alignment/_torch.py`](../shepherd_score/alignment/_torch.py)):
- `se3_params` has shape `(P, 7)` — `P` = number of seeds, cols 0–3 a quaternion `(r,i,j,k)`,
  cols 4–6 the translation. **One `backward()` covers all `P` seeds at once** (the objective
  returns `1 - score.mean()` over the batch).
- The pose is built by `get_SE3_transform(se3_params)`, which **`F.normalize`s the quaternion
  inside the autograd graph** and calls `quaternions_to_rotation_matrix`. Remember this — it is
  the crux of the convention-matching gotcha in C2.
- Optimize with `torch.optim.Adam([se3_params])`, `loss.backward()`, early-stop, then `argmax`
  the best seed.

If your objective differs, reshape it to this first, then wire it up (steps A4, A13–A14) so
`MoleculePair.align_with_<mode>` and `MoleculePairBatch.align_with_<mode>(backend='jax')` work.
You now have a correct, tested mode — and the ground truth everything below is checked against.

### C1. Rung 1 — batch the autograd objective over *pairs* (NO kernel; best bang-for-buck)
Your per-pair optimizer already batches over **seeds**. The cheapest large speedup is to
extend the *same* batching to the **pair** dimension and run it on the GPU — still pure
autograd, **zero kernel code**:
- Pad all pairs to a common size with `accel/batch/_pad.py:_band_key`, stack into
  `se3_params (B·P, 7)`, broadcast the ref/fit clouds to `(B·P, N_pad, 3)` / `(B·P, M_pad, 3)`,
  and carry int32 `N_real`/`M_real` masks (zero-weight padded rows in the score).
- Run the identical Adam loop, let **autograd** produce the gradient (`loss.backward()`), then
  `argmax` per pair. This is a driver in spirit, but autograd replaces the fused kernel.
- Typically **1–2 orders of magnitude** faster than the per-pair Python loop, on CUDA, for free.
- **Cost:** autograd materializes the full `(B·P, N, M)` pairwise graph every step (the backward
  ~doubles it) — memory-heavy and slower than a fused kernel, but often plenty. **Ship it,
  measure, and only continue if the per-step autograd time/memory is the bottleneck.** (This is
  the minimum-viable-fast-path, specialized to an autograd start.)

### C2. Rung 2 — write the fused kernel, with autograd as the gradient oracle
When C1's per-step autograd is the bottleneck, replace just the value+gradient with a kernel
(steps A7–A9). **A literal, annotated kernel + numba-twin + driver code template for this exact
step lives in [`accel/agent_prompt.md`](../shepherd_score/accel/agent_prompt.md) → "Recipe — turn
your autograd objective into a Triton+numba accel method"** — copy it and edit two lines (the value
term and the force coefficient). The autograd objective makes this low-risk because it *derives and
checks* the gradient for you:

- **You only implement the value + the *force*.** For any rigidly-transformed overlap, the
  quaternion gradient factorizes as
  `dO/dq = (∂O/∂(R·fit + t)) ⊗ (dR/dq of body-frame fit coords)`.
  The `dR/dq` tail is **mode-independent** — copy it verbatim from `_gauss_overlap_se3_tiled`
  (shape kernel). So porting your objective reduces to coding **(a)** the overlap *value* and
  **(b)** the *force* `∂O/∂(transformed fit coords)` of your pairwise term; the shared tail turns
  the force into `dO/dq`. (ESP did exactly this: same force structure × an `exp(-Δq²/λ)` factor.
  The color kernel: same tail, isotropic force.)
- **Get the reference gradients from autograd** (this is how you implement validation gate 1):
  - force oracle: `g = torch.autograd.grad(O.sum(), fit_coords_transformed)` — the per-atom force
    your kernel must accumulate;
  - dQ oracle: `dq_ref = torch.autograd.grad(O.sum(), q)` — what the kernel's `dQ` output must match.
- **Match the unit-quaternion convention (the #1 gotcha).** The kernel differentiates w.r.t. the
  **unit** quaternion (it builds `R(q)` assuming `|q|=1`). But `get_SE3_transform` `F.normalize`s
  `q` *inside the graph*, so `autograd.grad` w.r.t. raw `q` returns the **projection-included**
  gradient and will **not** equal the bare kernel `dQ`. For a clean comparison build the reference
  `R` from a **unit-norm leaf** `q` **without** normalizing — use
  `score/analytical_gradients/_torch.py:_rotation_matrix_from_unit_quat(q)` (NOT
  `get_SE3_transform`), in **float64** (the se3 utils also force float32). Then
  `autograd.grad(O.sum(), q)` is exactly the bare `dO/dq` the kernel emits, and they agree to
  ~1e-16. (Equivalently, apply the normalization Jacobian `dQ=(g−q·(q·g))/‖q‖` to the kernel
  output — which is precisely the Jacobian the driver keeps at run time; see pitfalls.)
- **Validate value, force, and dO/dq separately, float64, CPU/numba first**, then Triton==numba in
  fp32. The pharm and vol_color kernels were validated this way to ~1e-17.

### C3. Keep the similarity chain rule OUT of the kernel
Your objective optimizes a *similarity* `S = f(O)` (Tanimoto `O/(U−O)`, Tversky, a blend …).
**The kernel emits only the bare overlap `O` and `dO/dq`; the chain-rule factor `dS/dO` (the
`scale`) stays in the driver.** Therefore:
- the kernel's gradient oracle is `autograd.grad(O.sum(), q)` for the **bare overlap `O`**, NOT the
  full `1−S` loss;
- the driver computes `score = f(O)` and `scale = dS/dO` (e.g. `(VAA+VBB)/denom²` for Tanimoto)
  and applies `g_q = -scale·dQ`. For a blend, sum the per-channel `−scaleᵢ·dQᵢ` (the `vol_color`
  joint-gradient pattern, `vol_color.py:226-232`).
- the end-to-end **gate 4** (distinct == per-pair autograd) then confirms the driver's similarity +
  chain rule reproduce your original objective.

### C4. You can skip the analytical variant
`alignment/_torch_analytical.py` (+ `score/analytical_gradients/_torch.py`) is a *hand-derived*
gradient that reproduces autograd (~1.5–2.5× on the per-pair CPU path only). **If you trust
autograd, skip it** — go autograd-objective → kernel directly (`vol_color` has no analytical
variant). The kernel already replaces autograd on the hot path; the analytical per-pair variant is
worth it only if you specifically need a faster *per-pair CPU* path without a kernel.

**From here the back-half is Scenario A steps 7–17** (kernel → numba twin → dispatch → driver →
batch aligner → wiring → tests). If your new term is only an SE(3)-invariant weight or a re-blend
of existing channels, you may not need a kernel at all — see the decision guide and minimum-viable
fast path above.

---

## SCENARIO A — add a brand-new mode/term (full path)

Implement in dependency order. Canonical JOINT-combo template: `vol_color`. Cleanest
shape-only template: `surface.py`.

1. **Scalar scoring (torch)** — `score/<family>_scoring.py`. Add `get_overlap_<X>(...)`
   following `VAB_2nd_order → shape_tanimoto → get_overlap`. For a blend follow
   `get_pharm_combo_score`: `(1-w)·channel_a + w·channel_b`. For a per-pair multiplicative
   weight mirror `VAB_2nd_order_esp` (term `*= exp(-C²/lam)`); the weight **must be
   SE(3)-invariant**. Guard incompatible options and `raise` (vol_color raises if
   `directional=False` AND `precomputed_self_overlaps` is given).
2. **Scalar scoring (numpy)** — `score/<family>_scoring_np.py`. Mirror step 1 op-for-op
   (scipy `cdist`). Keeps `Objective._score` consistent. (Optional jax in `*_jax.py`.)
3. **Per-pair eager optimizer** — `alignment/_torch.py`. Add
   `objective_<mode>_overlay(se3_params, …)` (build SE(3) via `get_SE3_transform`, apply
   with `apply_SE3_transform` / `apply_SO3_transform` for vectors, return `1 - score`) and
   `optimize_<mode>_overlay(…)` (seed via `_initialize_se3_params`, `torch.optim.Adam`,
   precompute SE(3)-invariant self-overlaps once — **exception:** recompute every step if
   directionless like vol_color — early-stop, argmax-return best seed). **This is the
   ground truth your kernels are validated against.** *(If you already have a working autograd
   objective, this step is essentially done — start at [Scenario C](#scenario-c--you-already-have-a-working-autograd-objective-the-common-case).)*
4. **Export** — `alignment/__init__.py`: add both functions to the `from ._torch import`
   block and `__all__`.
5. *(Optional)* **Analytical variant** — `alignment/_torch_analytical.py` +
   `score/analytical_gradients/_torch.py`: `optimize_<mode>_overlay_analytical` (~1.5–2.5×).
   Skip if going straight to the kernel (vol_color has none).
6. **Lookup tables** (typed/pharmacophore terms only) —
   `score/analytical_gradients/_torch.py`: extend
   `build_lookup_tables_cached(device, dtype, directionless=…)` (returns alphas, Ks, cats;
   lru_cache key includes new flags). `directionless=True` collapses every real type to
   category 0 (isotropic, w=1) — this is what lets one kernel serve both directional
   `pharm` and directionless `vol_color`.
7. **Kernel — Triton** — `accel/kernels/<family>_triton.py`: `@triton.jit` + host wrapper,
   `@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad','M_pad'], cache_results=True)`,
   `grid=(P,)`. Take the **unit quaternion `q`** (not R), build `R(q)` in registers, gate
   the gradient block behind `NEED_GRAD: tl.constexpr`, emit `dO/dq` directly using the
   shape `dR/dq` tail (copy `dw/dxq/dyq/dzq` verbatim from `_gauss_overlap_se3_tiled`).
   Convention: **REF=axis0, FIT=axis1, `dx = ref - rot(fit)`**. `tl.exp2(x·1.4426950408889634)`.
8. **Kernel — numba twin** — `accel/kernels/cpu.py`: `@njit(parallel=True, fastmath=False,
   cache=True)`, **identical public signature**, `prange` over poses, `math.exp`, fp64
   accumulation. Copy the `dR/dq` tail verbatim, same `q=(w,x,y,z)` convention, same `dx`
   sign. Only allowed divergence: exp2 vs exp.
9. **Register** — `accel/kernels/dispatch.py`: one line
   `my_kernel = _make("my_kernel", "<shape|esp|pharm>")`. New source module → extend `_mod()`.
10. **Driver** — `accel/drivers/<mode>.py`, three functions:
    - `coarse_fine_<mode>_align_many(…)`: default-fill int32 `N_real`/`M_real`; seed via
      `batched_seeds_torch(shape_cloud_1, shape_cloud_2, N_real, M_real, num_seeds)`; expand
      per-pair tensors across P seeds; precompute pose-invariant self-overlaps **once** (per
      channel for combos). Fine loop: kernel → `VAB/dQ/dT`;
      `denom=VAA+VBB-VAB; score=VAB/denom; scale=(VAA+VBB)/denom²`; `_update_best`;
      every-5-step early stop with `(ES_PATIENCE_OVERRIDE or early_stop_patience)`;
      `fused_adam_qt_with_tangent_proj(q, t, -dQ·scale, -dT·scale, …)`. **Joint combo:**
      `g_q=(1-w)·(-scale_a·dQ_a)+w·(-scale_b·dQ_b)` (copy `vol_color.py:226-232`). Return
      via the `final_score.view(BATCH,P).argmax(1)` + gather tail.
    - `fast_optimize_<mode>_overlay_batch(…)`: precompute VAA/VBB via a
      `_self_overlap_<mode>_chunks` helper, call the worker, `apply_se3_transform`, return.
    - `fast_optimize_<mode>_overlay(…)`: single-pair; CPU fallback
      `if not check_gpu_available(): from ...alignment._torch import optimize_<mode>_overlay`.
11. **Batch aligner** — `accel/batch/aligners.py`: add `'<mode>': N` to `_MODE_SEEDS`
    (40 for multi-basin channel modes, 18–20 for pure shape). Add
    `_align_batch_<mode>(pairs, *, …)`: empty-guard; `_should_distribute`/`_run_distributed`;
    bucket by `_band_key` on **each** size dim (vol_color uses a 4-size-dim band key —
    ref/fit centers AND ref/fit pharm anchors); `_scatter_fill` padded workspaces (pad
    pharm slots with `_PHARM_PAD_TYPE=8`); call the driver in `_subbatched_align` with
    `num_seeds=_seeds_for('<mode>')`; write back to `p.transform_<mode>` / `p.sim_aligned_<mode>`.
12. **Re-export** — `accel/batch/__init__.py`: add `_align_batch_<mode>`.
13. **Bind on MoleculePair** — `container/_core.py`: `self.transform_<mode> = np.eye(4)`,
    `self.sim_aligned_<mode> = None` in `__init__`;
    `_align_batch_<mode> = staticmethod(_ba._align_batch_<mode>)`; add
    `align_with_<mode>(…)` mirroring `align_with_vol_color` (jax / analytical / eager
    branches + optional CUDA-gated `use_fast` seam that delegates a single pair UP into the
    batched driver, falling through on `ImportError`).
14. **Public batch API** — `container/_batch.py`: add `align_with_<mode>(…, backend='jax')`
    calling `self._run_fast_or_fallthrough(backend, MoleculePair._align_batch_<mode>,
    align_kwargs, 'sim_aligned_<mode>', 'transform_<mode>', '_fit_<…>_t', return_aligned,
    numba_ok=<True unless no numba kernel>)`, then the jax fallthrough.
15. *(Optional)* **Multi-GPU / CPU pool** — `accel/batch/_dispatch.py:_MODE_SPEC['<mode>']`
    for process-per-GPU (without it: single-GPU only, like vol_color/esp_combo);
    `accel/cpu_pool.py:POOL_MODES` for `num_workers>1` CPU sharding.
16. *(Optional)* **Screening front-end** — `screen.py`: attr maps, feature gate, mode
    branches, `_FAST_MODES`, `_NEEDS_ARRVIEW`.
17. **Tests** — `tests/test_<mode>.py` (validation gates below).

---

## SCENARIO B — speed up an existing mode (move it to an in-register-dQ kernel)

The `pharm` worked example. Use when a mode is correct but its kernel emits `grad_R
(P,3,3)` and the driver runs a per-step host projection tail.

1. **New Triton kernel** — `accel/kernels/<family>_triton.py`: q-in / dO/dq-out
   `@triton.jit` + wrapper (same shape as the old one but taking `q` not `R`, returning
   `(O, dQ, dT)`). Build `R(q)` in registers; reuse the same alphas/Ks/cats lookup and
   per-type `w`. Apply the shape `dR/dq` tail to the **positional** force; if the mode has
   a **directional weight** term, apply the tail a **second time** with the weight force
   crossed with the body-frame fit vector (this second term *is* the directional gradient —
   omitting it silently degrades to isotropic). `NEED_GRAD: tl.constexpr`, `tl.exp2`.
2. **numba twin** — `accel/kernels/cpu.py`: op-for-op mirror, identical signature, fp64.
3. **Register** — `accel/kernels/dispatch.py`: `new_kernel = _make("new_kernel", "<tag>")`.
4. **Driver** — `accel/drivers/<mode>.py`: add `use_kernel = (not extended_points)` (or your
   guard). Compute value-only self-overlaps via the **same** kernel with identity quaternion
   `(1,0,0,0)` and `NEED_GRAD=False`. In the fine loop, get `dO/dq` directly, compute the
   scalar `scale`, apply it straight: `sgrad_q = scale.unsqueeze(1) * dQ_raw` — **drop**
   `apply_*_chain_rule`, `project_grad_R_to_quaternion`, and the host R build. **KEEP** the
   unit→raw normalization Jacobian `dQ = (sgrad_q - q*(q*sgrad_q).sum()) / qn`. Leave the
   legacy R-out path for cases the new kernel doesn't cover (extended_points). Justification:
   matrix→quaternion projection is **linear**, so `project(scale·grad_R) == scale·dO/dq`.

---

## Validation gates (all four must pass)

1. **numba vs autograd:** numba kernel `dQ` matches torch autograd of the same objective to
   **~1e-17** (including any directional weight-gradient term). Run on CPU tensors. Build the
   autograd reference from a **unit-norm leaf `q`** via `_rotation_matrix_from_unit_quat` in
   **float64** — NOT `get_SE3_transform`, which `F.normalize`s in-graph (folding in the
   projection Jacobian) and forces float32 — and differentiate the **bare overlap `O`**, not the
   similarity (the chain-rule `scale` is the driver's job). See [Scenario C2/C3](#scenario-c--you-already-have-a-working-autograd-objective-the-common-case)
   for why, and for the `autograd.grad(O.sum(), fit_coords)` force oracle.
2. **Triton == numba:** match to **fp32** on a GPU box (exp2 vs exp is the only intentional
   divergence).
3. **self-copy recovers 1.000:** aligning a molecule to a copy of itself returns similarity
   1.000 for tanimoto AND tversky. Catches self-overlap convention bugs.
4. **distinct == per-pair autograd:** batched score on distinct pairs matches the per-pair
   eager `optimize_<mode>_overlay` ground truth.

Also confirm: kernel registered via `_make` (so both backends resolve with no `is_cuda`
branch); `_MODE_SEEDS` entry present; padded slots masked (int32 `N_real`/`M_real`, pharm
pad = Dummy 8, band key covers every size dim).

---

## Pitfalls (learned from the worked examples)

- **Docstrings lie about vol_color.** The header of `accel/drivers/vol_color.py` (and
  comments in `aligners.py`/`_batch.py`) describe a "shape-driven / FastROCS" or
  "project_grad_R_to_quaternion" path. The actual code (`vol_color.py:226-232`, verified)
  computes a **joint** gradient `g_q=(1-w)·(-scale_s·dQ_s)+w·(-scale_c·dQ_c)` with **no
  R→q projection**. Trust the code; don't copy the stale comment into a new mode.
- **Zeroing orientation vectors ≠ directionless.** A zero vector yields cosine weight
  `(0+2)/3=2/3` on cross terms but 1 on self terms, which doesn't cancel in the Tanimoto.
  You must **skip** the cosine path (route types through the isotropic `VAB_2nd_order`
  helper). See `pharmacophore_scoring.py` ~512-515.
- **Self-overlaps MUST use the same kernel as the cross-overlap** (identity quaternion,
  zero translation, `NEED_GRAD=False`). Mixing a different self-overlap path collapses the
  Tanimoto ratio.
- **Directional dO/dq applies the dR/dq tail twice** — positional force × body-frame fit
  **anchor**, and weight force × body-frame fit **vector** (different coefficients). The
  directionless color kernel deliberately has NO weight-force term.
- **Keep the unit→raw normalization Jacobian** when you drop the projection tail. Kernels
  differentiate w.r.t. the unit quaternion (Adam renormalizes each step) but Adam optimizes
  the raw quaternion. (`pharm.py` ~319-320.)
- **`pair_weights` must be SE(3)-invariant.** ESP reuses the shape kernel because
  `exp(-‖dq‖²/lam)` doesn't depend on the pose. A coordinate-dependent weight needs a new
  gradient derivation.
- **Axis convention differs between the two pharm kernels:** legacy R-in has FIT=axis0,
  REF=axis1; the new q-in flips to REF=axis0, FIT=axis1 with `dx=ref-rot(fit)` — deliberately,
  so the shape `dR/dq` tail can be reused. Match the shape-kernel convention in any new kernel.
- **`esp_combo` has no numba path (`numba_ok=False`) and no `_MODE_SPEC`; `vol_color` has
  numba but no `_MODE_SPEC`.** Numba twin and multi-GPU spec are independent opt-ins —
  decide each deliberately when copying a template.
- **Index order in `P_TYPES` (`constants.py`) is load-bearing** (used directly as the
  integer type index everywhere; `Dummy=8`, category 3). Appending is safe; reordering is not.
- **Seed from the shape/atom cloud, not the pharm anchors.** Carry separate real-count
  tensors for centers vs anchors throughout (vol_color does).
- **`FINE_FUSED_STEP` is a measured regression at large batch (default OFF)** and the numba
  `fused_surf_step_batch` is a `NotImplementedError` stub. The portable path is the eager
  fine loop with separate value+grad and `fused_adam_qt` calls.
- **`num_workers` is a CPU process-pool knob, NOT per-pair delegation.** `num_workers=1`
  runs one in-process batched kernel call. The per-pair `use_fast` seam is the opposite
  direction — it delegates a single pair UP into the batched driver.

---

## Environment / dependencies

- Triton kernels require **`triton ≥ 3.6`** (autotune `cache_results=True`).
- numba kernels require **`numpy ≤ 2.3`** (numba ABI).
- jax backend requires `pip install shepherd-score[jax]` (JAX ≥ 0.9). The triton/numba fast
  backend has **no** jax dependency.
- Speed/accuracy env levers (no code change): `FINE_NUM_SEEDS`, `FSS_STRUCT_SEEDS`,
  `FINE_ES_PATIENCE`, `FINE_CUDA_GRAPHS`, `FINE_FUSED_STEP`, `SUBBATCH_DEBUG`,
  `FSS_MGPU_BACKEND=process`.
