# CPU speed experiments — toward >2,000 pairs/s on ALL modes (no accuracy loss)

**Goal:** push the **CPU** alignment path (the default `backend="jax"` JAX/XLA path, plus the
per-pair torch CPU paths) past **2,000 aligned pairs/second on every mode** — `vol`, `vol_esp`,
`surf`, `esp`, `pharm` (and, if reachable, `esp_combo`) — **without sacrificing accuracy**:
self-copy score stays **1.000** (pharm ~0.999), and distinct-molecule-pair scores match the current
path (`max|Δ|` ≈ 0, or provably < 1e-5 float-level).

This is the CPU counterpart to the GPU work in [`SPEED_EXPERIMENTS.md`](SPEED_EXPERIMENTS.md). The
GPU fork hit 50k–180k pairs/s by **batching every pair into one kernel dispatch**. The headline
finding here is that **the same batched driver already runs on CPU** — it just needs one CPU compute
kernel dropped into it. The hard part is the surface modes, which are genuinely compute-bound and
need a structural kernel attack, not constant-tuning.

> **Accuracy bar (load-bearing, inherited from the GPU log).** Every "2×-class" lever that *shifts*
> distinct-pair scores — fewer seeds, fewer points, fewer steps, lower precision, bigger reduction
> tiles, the `‖x‖²+‖y‖²−2x·y` GEMM form of `cdist` — was **rejected** on the GPU side. The same bar
> holds here. Only **bit-identical** or **provably < 1e-5 (score-level, zero winner-changes)** levers
> ship.

---

## Environment

The dev box is **Windows, CPU-only, with NO `jax` and NO `open3d`** (torch 2.6 CPU, numba 0.61,
numpy 2.1, rdkit are present). So:

- The JAX CPU path and the surface/ESP builders (open3d) **cannot run on the Windows box** — the
  benchmark itself runs in the **WSL2 `SimModelEnv`** conda env (same env as the GPU benchmarks; see
  [`benchmarks/README.md`](benchmarks/README.md)). All `pairs/s` baselines and gate runs below are to
  be filled from there.
- **But the prime CPU lever is jax-free.** The batched optimizer (below) is plain torch, and the
  innermost overlap kernel is independently buildable/testable in **numba** — both run on the Windows
  box. Kernel microbenchmarks and parity checks can be developed locally; only the end-to-end
  `pairs/s` numbers need WSL2.

---

## Method

**Throughput:** `benchmarks/experiments/speedlab.py` (in-process paired best-of-N A/B) for levers,
and `benchmarks/benchmark.py` for the headline. Run in `SimModelEnv`.

**Accuracy gate — three tiers, every lever passes all three before it ships.** Reuse
`benchmarks/experiments/parity_scores.py` with `dev=torch.device("cpu")`, comparing each new CPU
path against the **already-passing batched reference** `MoleculePair._align_batch_{vol,surf,esp,pharm}`
(because `coarse_fine_align_many` keys off `pairs[0].device`, feeding CPU tensors exercises the new
CPU kernel through the *same* driver the GPU validates against — so the diff is vs a trusted
reference, not the slow legacy JAX loop).

- **T1 — kernel microbench (before wiring anything).** Compare the new kernel's `(VAB, dQ, dT)`
  against the reference analytical kernel `compute_overlap_and_grad_shape`
  (`score/analytical_gradients/_torch.py`) on ≥2048 random `(R, t, points)`.
- **T2 — component asserts.** Algebraic fusions (`exp(a)·exp(b) == exp(a+b)`) max|Δ| < 1e-6;
  cached pose-invariant `VAA/VBB` == recompute to 0 ulps; seed-tiling `tiled == untiled` on
  `final_score/best_q/best_t` element-wise.
- **T3 — end-to-end parity.** `parity_scores.py` per mode: **(a)** self-copy mean == 1.000
  (pharm ~0.999); **(b)** distinct 30-pair `max|Δ|` at 6-decimal vs the batched reference.

> **Gate the SCORE, not VAB; and count winner-changes.** (Correction from the adversarial audit.)
> A fused on-the-fly `r²` is *not* < 1e-5 against `torch.cdist(...)**2` at the **VAB** level
> (~3e-5 on a 200×200 fp32 tile) — but the Tanimoto `VAB/(VAA+VBB−VAB)` cancels that error to
> **~2e-7 at the score level** (provided VAA/VBB use the *same* kernel, making self-copy bit-exactly
> 1.0). So the T1/T3 gate is on the **score**, plus, for any non-bit-identical lever, a
> **zero-argmax-seed-change** check over a **large random distinct-pair cohort (≥2000, oversampling
> pseudo-symmetric/elongated molecules)** — a basin switch shifts one pair's score discontinuously
> and is invisible to a mean |Δ|. Any lever exceeding score `max|Δ| < 1e-5` **or** flipping a winning
> seed is rejected, or made per-mode opt-in with a code-enforced fallback to the strict kernel.

**Constants held bit-for-bit:** `alpha=0.81`, `K=(π/2α)^1.5`, `exp(−α/2·r²)`; ESP `lam=0.3·LAM_SCALING`
(surf) / `0.1` (vol); `num_repeats=50` (1 identity + 4 PCA + 45 `_get_45_fibo`); `lr`, `steps_fine`;
pharm `+1e-6` eps, `P_ALPHAS`, `(sim+2)/3`; **row-major NxM reduction order (NO GEMM `cdist`).**

---

## Baseline (to be measured on WSL2 `SimModelEnv`, num_repeats=50)

Per-mode CPU throughput at the **default `num_repeats=50`** (the accuracy-load-bearing config — the
surface modes need all 50 PCA/Fibonacci seeds to reach self-copy 1.000). The upstream
[`docs/performance/timings.md`](docs/performance/timings.md) figures are at `num_repeats=5`, so they
are an **optimistic reference** — at 50 seeds the optimization-bound modes run proportionally slower
and the real gap is *wider*. Fill the cells from a fresh `--no-original` CPU run.

| mode | JAX-batch 1-proc (ref, nr=5) | 8-cpu spawn (ref, nr=5) | measured nr=50 (TODO) | gap to 2k |
|---|--:|--:|--:|--:|
| vol      | ~110/s  | ~507/s | _TBD_ | ~4–18× |
| vol_esp  | ~124/s  | ~506/s | _TBD_ | ~4–16× |
| pharm    | ~67/s   | ~533/s | _TBD_ | ~4–30× |
| **surf** | **~6.6/s** | ~6.7/s (does not scale) | _TBD_ | **~300×** |
| **esp**  | **~2.7/s** | — | _TBD_ | **~740×** |
| esp_combo | unmeasured (per-pair autograd only) | — | _TBD_ | largest |

---

## 🎯 The structural finding — the CPU path is dispatch-bound, and the batched driver already runs on CPU

Two facts, established by reading the code:

**1. The default CPU path optimizes one pair per XLA dispatch.** In the `use_shmap=False` sequential
branch (`container/_batch.py`, the `for` loop ~line 356), every pair calls
`optimize_*_jax_mask` **once** → one whole `lax.while_loop` (up to 200 Adam steps) per pair. 10,000
pairs = 10,000 separate XLA executions. The 50 SE(3) seeds are already `vmap`-batched *within* a pair,
but there is **no batching across pairs**. For `vol`/`vol_esp`/`pharm` the NxM overlap grid is tiny
(~30×30 heavy atoms / features), so the kernel is *not* the bottleneck — **host/dispatch is**. This
is exactly the per-pair-overhead trap the GPU fork's F3 work beat by batching.

**2. The GPU batched driver is device-agnostic except for one kernel.** In
`coarse_fine_align_many` (`alignment/utils/fast_se3.py:349`), the fused and CUDA-graph fast paths
are both guarded by `torch.cuda.is_available()`. On CPU, execution falls through to the **eager
batched Adam loop (lines 408–473)**, which is pure device-agnostic torch:
`device = A_batch.device`, `torch.where` best-tracking, `fused_adam_qt_with_tangent_proj`,
`argmax` over seeds — **and it keeps the accurate patience-based early-stop** (lines 458–466), so a
CPU port does *not* inherit the JAX `lax.scan` fixed-step convergence risk. The **only** compute
dependency is `_overlap_in_chunks` (line 426), which calls the Triton overlap value+grad kernel.

> **Therefore:** drop a **CPU overlap value+grad kernel** behind `_overlap_in_chunks`, and the *entire*
> GPU batching design — banded seed-gen, `_self_overlap_in_chunks`, the batched Adam fine loop,
> best-tracking, top-k select — runs on CPU, collapsing 10,000 per-pair dispatches into one batched
> loop. `fused_adam_qt_with_tangent_proj` already executes in the non-CUDA branch, so the lever is
> essentially **one new function**. This is the enabler every other lever stacks on.

The surface modes need *more* than this: their ~200-point cloud (`num_surf_points = max(24, 3×heavy)`,
so ~75–150 pts) makes the NxM overlap ~6–44× heavier per step than `vol`. Batching removes dispatch
but not the tile — so `surf`/`esp` also need a **fused, single-pass numba kernel** + **seed-tiling**.

---

## Bottleneck table (per mode)

| mode | dominant cost | nature | headroom |
|---|---|---|---|
| `vol` | 10,000 per-pair `while_loop`s; inner ~30×30 grid is tiny | **dispatch-bound** | Large — one batched dispatch clears 2k/s; bit-identical |
| `vol_esp` | same dispatch + SE(3)-invariant charge weight (precomputed once, rides `pair_weights`) | dispatch-bound | Large — CPU kernel handles it for free; bit-identical |
| `pharm` | per-pair dispatch over tiny typed grids; batched analytical grad **already CPU-ready** | dispatch-bound | Large + **easiest** — driver-wiring only; bit-identical |
| `surf` | ~200×200 NxM `exp+sum` per step, ×50 seeds × ~100–200 steps | **compute-bound** | Moderate-hard — needs fused numba kernel + tiling + cores |
| `esp` | surf cost + a 2nd 200×200 charge grid (fusable into one `exp`) | compute-bound | Moderate-hard — rides every surf lever; worst gap (~740×) |
| `esp_combo` | NO batched CPU path; autograd-only; 3 point-sets/step; **nondeterministic** | unbatched | Hardest — needs a deterministic kernel + new driver first |

---

## Levers (sized by impact, ranked best-first)

Risk legend: **bit-identical** · **float<1e-5** (score-level, zero winner-changes) · **risky**
(can shift scores — gated, opt-in only).

### L1 — CPU overlap value+grad behind `_overlap_in_chunks` *(the prime lever)* — bit-identical · M
**Modes:** vol, vol_esp, surf, esp. **Mechanism:** write `overlap_score_grad_se3_batch_cpu` mirroring
`compute_overlap_and_grad_shape` (`score/analytical_gradients/_torch.py:430`) byte-for-byte —
`K=(π/2α)^1.5`, `fit_t=bmm(R,B)+t`, `dist_sq=cdist**2`, `E=exp(−α/2·dist_sq)[·pair_weights]`,
`O_AB=K·E.sum`, the three `bmm` gradient contractions, then `project_grad_R_to_quaternion` to return
`dQ(4)` matching the Triton signature. The pad mask **and** the ESP charge weight both enter the
existing `pair_weights` slot. Add a CPU branch in `_overlap_in_chunks` (drop the CUDA-only 65535
grid-z guard). **Expected:** vol/vol_esp 10,000 `while_loop`s → 1 batched dispatch, ~16–30× over
baseline → clears 2k/s on modest cores. surf/esp: removes dispatch but tile remains (~5–15× alone) —
*necessary, not sufficient*, see L2. **Gate:** T1 microbench (score-level), then T3 parity on cpu.
*Why #1:* the batched optimizer **already exists and already selects the eager path on CPU** — this
is one kernel-swap, reusing already-validated math.

### L2 — numba-prange fused single-pass surf/esp kernel *(the brutal-mode attack)* — float<1e-5 · L
**Modes:** surf, esp. **Mechanism:** the L1 torch kernel materializes a `(K·S, N, M)` tensor and makes
~5 passes; at 50 seeds × 200² surface points that thrashes (tens of GB). Replace the inner math with
`@njit(parallel=True)`: `prange` over the `K·S` pose rows (embarrassingly parallel — the exact CPU
analogue of Triton's one-CTA-per-pose), stream the NxM grid in register-resident M-tiles, compute
`r²` on the fly, accumulate `VAB` and the three gradient contractions in **one pass** in the **same
fit-outer/ref-inner order** as the torch reference. No `(B,N,M)` temp ever exists. Fold the two ESP
exponents into one `exp(−α/2·r² − C2/lam)`. Independently buildable/testable on the Windows box.
**Expected:** fused single-pass + all cores ~8–20× over naïve batched-torch surf; stacked on L1 targets
the ~300×/~740× gap. **Honest:** 2k/s on surf/esp likely needs ≥16 cores; ~1000–1500/s on 8.
**Gate (audit-corrected):** numba **without `fastmath`** (preserve IEEE add order); **gate on the
score** (`max|Δ| < 1e-5` vs `_align_batch_surf/esp`), **not VAB** (on-the-fly `r²` is ~3e-5 at VAB —
expected and fine, it cancels in Tanimoto); compute VAA/VBB with the **same** numba kernel so
self-copy is 1.0 by construction; **count argmax-seed flips over ≥2000 random pairs — require 0**;
strongly consider **fp64 accumulation** inside the njit kernel (matched the reference score to 1.8e-7
in the audit, removes the borderline gradient delta). *Why #2:* the only lever that structurally
attacks the compute-bound surf/esp tile; make-or-break for the brutal modes.

### L3 — cache pose-invariant VAA/VBB + tile the seed dim — bit-identical · S
**Modes:** all (load-bearing for surf/esp). **Mechanism:** (1) `VAA/VBB` are pose-invariant
(rotation/translation preserve intra-molecule distances); the batched driver already computes them
once per band via `_self_overlap_in_chunks` — the CPU port must **not** regress into per-step recompute
(the legacy JAX objective recomputes them every step = 2/3 redundancy). (2) Run the fine loop over
**seed-tiles** (8–12 seeds at a time, carry per-seed best) via the `seeds=` precompute hook
(`fast_se3.py:365`) so peak workspace stays L2/L3-friendly. **Expected:** caching banks the ~3× the GPU
driver already has (vs a naïve regression); tiling is the difference between OOM/thrash and sustaining
large surf batches (~300/s vs ~2000/s). **Gate:** caching bit-identical by construction (assert
`max|Δ|=0` vs recompute); tiling bit-identical because argmax over concatenated per-tile maxes ==
argmax over the whole (verify `tiled==untiled` element-wise on a 256-pair surf batch). *Why #3:* cheap,
bit-identical, and what stops L1+L2 from OOMing at surface scale — **ships alongside L2.**

### L4 — pharm via the already-batched analytical grad — bit-identical · M
**Modes:** pharm. **Mechanism:** pharm already ships a device-agnostic batched value+grad
(`compute_overlap_and_grad_pharm`, `_torch.py:212`). Build `coarse_fine_pharm_align_many` mirroring
`coarse_fine_align_many` — `batched_seeds_torch` on anchors, `DUMMY_TYPE=8` padding + type-match mask
exactly as `_align_batch_pharm` already pads, the batched Adam fine loop calling
`compute_overlap_and_grad_pharm` with the `(sim+2)/3` weight, per-row best, argmax over 50. Wire
`_align_batch_pharm` to it when CUDA is absent. **No new innermost kernel.** **Expected:** ~30–50× over
67/s → 2000–3000/s; beats the 533/s 8-proc spawn single-process. **Gate:** T3 parity vs
`_align_batch_pharm` on cpu (self ~0.999); preserve `+1e-6` eps, `P_ALPHAS`, `(sim+2)/3` bit-for-bit.
*Why #4:* pharm is wrongly assumed hard but is the **easiest** mode — a driver-wiring job, high-
confidence bit-identical; ranked below the surf/esp structural levers only because it isn't the gap.

### L5 — ESP exponent fusion + cached self-overlap — float<1e-5 / bit-identical · S
**Modes:** esp, vol_esp. **Mechanism:** `exp(−α/2·r²)·exp(−C2/lam) → exp(−α/2·r² − C2/lam)` (one `exp`,
not two; `C2/lam` is SE(3)-invariant, precomputed once). Plus honor cached VAA/VBB. **Expected:**
~1.5–1.8× on esp inner cost (removes the 2nd 200×200 `exp`) → esp throughput approaches surf's.
**Gate (audit-verified safe):** the fusion is genuinely bit-safe — measured **sum-level Δ = 0.0,
elementwise max 6e-8** on a 200×200 fp32 tile; it does *not* touch reduction order. Assert
`exp(a+b) vs exp(a)·exp(b)` < 1e-6 in isolation, keep the fit-outer/ref-inner sum order, assert cached
VAA/VBB byte-identical, then T3 esp parity. *Why #5:* cheapest bit-safe halving of the worst mode's
transcendentals; folds into L2's ESP path (listed separately so the fusion is gated explicitly).

### L6 — torch.compile the fixed-step fine body *(opportunistic, per-mode gated)* — float<1e-5 · M
**Modes:** vol, vol_esp, pharm (and maybe surf/esp). **Mechanism:** wrap the per-step body
(overlap+grad + Tanimoto + `torch.where` best-track + CPU Adam) in
`torch.compile(mode="max-autotune", dynamic=False)` keyed by `(N_pad, M_pad, batch)` so Inductor fuses
~10 ops/step into a few SIMD kernels; keep the every-5-steps early-stop `.item()` outside the compiled
region. **Expected:** vol/pharm ~1.5–3× (kills per-op dispatch); surf/esp ~1.5–2× but likely loses to
the hand-tiled L2 numba kernel. **Gate:** Inductor can reassociate reductions — compare
compiled-vs-eager `score` `max|Δ| < 1e-5` **per mode**, then T3; if any mode drifts, fall back to
eager/numba for that mode. *Why #6:* a genuine but secondary multiplier and a possible numba
alternative for small-grid modes; ships per-mode only after passing < 1e-5.

### L7 — closed-form Kabsch warm-seed — risky · M *(demoted by the audit)*
**Modes:** all. **Mechanism:** one weighted-Procrustes (SVD over `(K·S,3,3)`) refinement per seed before
Adam, to start closer to the basin optimum. **Audit verdict — likely-shifts-scores:** the warm-start
at *fixed* `steps_fine` converges to the same basin (pose Δ 4.6e-7) but buys **zero** speedup; the
claimed 1.4–1.6× comes entirely from then **reducing `steps_fine`**, which is the rejected step-count
lever — and in testing the warm-start did **not** monotonically reduce steps-to-converge (Adam momentum
re-warms from the jumped pose; warm scored *lower* than cold at 30 and 60 steps). **Gate if attempted:**
ship at unchanged `steps_fine` first (prove SVD numerics: score `max|Δ| < 1e-5`, zero argmax flips);
only then sweep steps down with full parity + a winner-change count + a "no seed still improving at
cutoff" check on the **slow-converging tail** (not the fast self-copy benchmark). *Default: do not cut
steps below the value already proven for the batched path.*

### L8 — coarse-to-fine surface downsampling — risky · L *(last resort, gated hard)*
**Modes:** surf, esp. **Mechanism:** optimize SE(3) on a deterministic farthest-point-sampled subset
(e.g. Nc=64, ~9.6× fewer NxM evals) for ~80% of steps, then a short fine phase on full points; FPS
index set is pose-invariant (commutes with SE(3)); the reported best is **re-scored on full points**.
**Audit verdict — likely-shifts-scores:** this is the surface analogue of "fewer points." Self-copy=1.0
holds (trivial), and the full-points polish fixes within-basin drift — but it **cannot climb out of a
wrong basin** the coarse phase commits to. The codebase **already removed** coarse-grid top-k pruning
(`coarse_fine_surface_align_many`) precisely because "pruning on raw overlap dropped the true basin for
**pseudo-symmetric molecules**" — the same pseudo-symmetry defeats coarse-surface optimization.
**Gate:** on a **large** distinct cohort (≥5000, oversampling symmetric/elongated mols), for each
Nc∈{48,64,96,128}: require **zero argmax-seed flips** and score `max|Δ| < 1e-5` vs full-points
`_align_batch_surf/esp`; lengthen the fine phase until no winner changes. A small mean Δ is **not**
sufficient — the winner-change count is the real gate. **Deploy only if L1–L3 fall short of 2k/s on
the available cores.**

### L9 — JAX-path systems levers (split the bundle) — mixed · M
**Modes:** vol, vol_esp, pharm (does nothing for surf/esp — they are compute-bound). The audit
**splits this into two independently-gated halves:**
- **L9a — persistent pre-warmed JAX worker pool + core-pinning** (`OMP_NUM_THREADS=1`/worker,
  compile-once-dispatch-many; applies the [[mgpu-process-backend-status]] "persistent worker >
  spawn-per-call" lesson). **bit-identical** (assert `max|Δ|=0` vs spawn) — **ship freely.** Today the
  spawn fallback re-imports JAX + re-JITs in *every* worker on *every* call.
- **L9b — make `shard_map(vmap(scan))` the single-process default** at physical-core virtual-device
  count. **risky:** replaces the data-dependent `while_loop` early-stop with a **fixed `lax.scan`** —
  over-runs (wasteful) or **under-runs the slow-converging tail** (shifts scores). Also,
  `--xla_force_host_platform_device_count` changes how vmapped reductions are partitioned and is **not
  guaranteed bit-identical across device counts**. **Gate:** raise the scan step count until parity
  matches the `while_loop` path on the cohort **and** a large distinct set with **zero argmax flips**
  (prove it covers the *slowest* pair, not the median); separately gate XLA reduction stability across
  device counts; keep `while_loop` as fallback. *Why last:* L1+L4 reach the same small-grid modes
  **bit-identically** and extend to surf/esp; L9 is a fallback for jax-only deployments.

---

## Assessment — is >2k/s on ALL modes reachable, accuracy-safe?

**CONDITIONAL YES for 5 of 6 modes; NO (as currently coded) for `esp_combo`.**

- **`vol`, `vol_esp`, `pharm` — HIGH confidence, bit-identical.** Dispatch-bound, so cross-pair
  batching (L1; pharm via L4) plus modest parallelism clears 2k/s on roughly a **16–24 physical-core**
  node (~18× vol, ~16× vol_esp, ~30× pharm at near-linear scaling once spawn/oversubscription are
  removed). pharm is the easiest — its batched analytical grad already runs on a CPU tensor.
- **`surf`, `esp` — reachable but TIGHT and core-budget-dependent, float<1e-5.** Compute-bound on the
  ~200×200 tile (~44× vol), so batching is necessary-not-sufficient. The numba fused kernel (L2) +
  seed-tiling (L3) + ESP fusion (L5) get them there on a **≥16–32-core** box at score-level < 1e-5,
  but likely only **~1000–1500/s on 8 cores**. If cores are fixed at 8, the gated downsampling (L8,
  < 1e-5 + zero winner-changes) is the lever that closes the last ~1.5–2×.
- **`esp_combo` — NOT demonstrably reachable accuracy-safe.** No batched CPU path
  (`fast_optimize_esp_combo_score_overlay_batch` hardcodes `device='cuda'`, falls back to the per-pair
  autograd `optimize_esp_combo_score_overlay`), no analytical gradient, three point-sets/step, and the
  combo kernel is **nondeterministic run-to-run** (per [[batch-align-host-overhead-f3]] — it's
  excluded from `parity_scores.py` for this reason). It needs a **deterministic kernel + a new batched
  CPU driver** before any speed lever applies, and cannot meet the bit-identical gate today. **Scope it
  separately; do not block the other five modes on it.**

**Assumptions:** ≥16 physical cores for surf/esp at 2k/s (8 → ~1000–1500/s); torch 2.6 CPU + numba
0.61 (jax-free, so L1/L2 are independently buildable on the Windows box); row-major reduction preserved
(no GEMM `cdist`); seed set, steps, `alpha`, `lam` held bit-for-bit.

---

## Recommended sequence

0. **Gate first.** Port `parity_scores.py` + `speedlab.py` to `dev=cpu` against `_align_batch_*`.
   Confirm it runs in WSL2 `SimModelEnv`. *Build the gate before any kernel — every lever is bit-checked
   against an already-passing reference.*
1. **L1 (unblocks everything).** Write `overlap_score_grad_se3_batch_cpu` + the CPU branch in
   `_overlap_in_chunks`. T1 microbench (score-level) before wiring. → should clear 2k/s on
   vol/vol_esp and give a working (slow) surf/esp.
2. **L4 (easy win).** Wire `_align_batch_pharm` to a CPU pharm driver reusing the batched analytical
   grad. → banks pharm cheaply.
3. **L2 + L5 (brutal-mode attack).** `@njit(parallel=True)` fused single-pass surf/esp kernel, ESP
   exponents fused. Microbench (score < 1e-5) + `exp(a+b)` check first.
4. **L3 (memory enabler — ship with step 3).** Seed-dim tiling + assert VAA/VBB cached. Verify
   `tiled==untiled`. Pin `torch.set_num_threads` to physical cores; BLAS single-threaded inside numba.
5. **Measure, then decide.** Benchmark surf/esp on the target core budget. ≥2k/s → done. Short →
   deploy L7/L8 **gated** (winner-change harness), and L6 per-mode where it passes < 1e-5.
6. **`esp_combo` (separate track).** Decide: (a) deterministic batched CPU combo kernel + driver, or
   (b) document out-of-scope for the bit-identical 2k/s target. Don't block the other five.
7. **(Optional) L9** for jax-only deployments — L9a (pool) freely, L9b (scan-default) gated.

---

## Experiment log

Baseline + every lever recorded here once measured on WSL2 `SimModelEnv`. A change ships only if it is
**faster AND passes the three-tier gate** (score-level, zero winner-changes).

| # | lever | mode | throughput Δ | accuracy (self / distinct max\|Δ\| / winner-flips) | verdict |
|---|---|---|---|---|---|
| 0 | baseline (CPU, nr=50, isolated) | all | — | 1.000 / 0 / 0 | reference (TODO: measure) |
| L1 | CPU `_overlap_in_chunks` kernel + batched driver | vol,vol_esp,surf,esp | _TBD_ | _TBD_ | pending |
| L4 | pharm batched analytical-grad driver | pharm | _TBD_ | _TBD_ | pending |
| L2 | numba fused single-pass surf/esp kernel | surf,esp | _TBD_ | _TBD_ | pending |
| L3 | VAA/VBB cache + seed-dim tiling | surf,esp,vol,vol_esp | _TBD_ | _TBD_ | pending |
| L5 | ESP exponent fusion | esp,vol_esp | _TBD_ | _TBD_ | pending |
| L6 | torch.compile fine body (per-mode) | vol,vol_esp,pharm | _TBD_ | _TBD_ | pending |
| L7 | Kabsch warm-seed (+ step cut) | all | _TBD_ | _TBD_ | pending (risky) |
| L8 | FPS surface downsampling | surf,esp | _TBD_ | _TBD_ | pending (risky) |
| L9a | persistent JAX pool + core-pin | vol,vol_esp,pharm | _TBD_ | _TBD_ | pending |
| L9b | shard_map-scan default | vol,vol_esp,pharm | _TBD_ | _TBD_ | pending (risky) |

---

## Open questions

1. **`esp_combo`:** build a deterministic batched CPU kernel + driver, or formally scope it out of the
   bit-identical 2k/s target? Its nondeterminism means it cannot meet the standard gate as written.
2. **Core budget?** The whole surf/esp verdict hinges on it — state the assumed physical-core count up
   front. vol/vol_esp/pharm clear 2k/s at ~16–24 cores; surf/esp need ~16–32 with L2+L3.
3. **Memory wall timing:** does the L1 torch eager path OOM at surface scale *before* L2 lands? If so,
   L3 (tiling) must ship *with* L1 for surf/esp — needs a quick `(B,50,200,200)` measurement.
4. **numba reduction parity:** will fit-outer/ref-inner `r²` (no fastmath, fp64 accumulate) hold
   score < 1e-5 with zero winner-flips on a real surf-sized batch? Measure before committing.
5. **Appetite for risky levers:** if the bit-identical path lands ~1500/s on the available cores, ship
   the gated L8 downsampling to close the gap, or accept < 2k/s on surf/esp?
6. **torch.compile (L6) on CPU:** does Inductor's reduction codegen stay < 1e-5 on the 200×200 surf
   sum, or reassociate enough to force numba-only for surf?
