# CPU speed experiments — toward >2,000 pairs/s on ALL modes (no accuracy loss)

**Goal:** push the **CPU** alignment path (the default `backend="jax"` JAX/XLA path, plus the
per-pair torch CPU paths) past **2,000 aligned pairs/second on every mode** — `vol`, `vol_esp`,
`surf`, `esp`, `pharm` (and, if reachable, `esp_combo`) — **without sacrificing accuracy**:
self-copy score stays **1.000** (pharm ~0.999), and distinct-molecule-pair scores match the current
path (`max|Δ|` ≈ 0, or provably < 1e-5 float-level).

> **The bar is 2,000 pairs/s PER SINGLE CORE — not aggregate.** Parallelism does **not** count toward
> the target; the per-core kernel itself must reach 2k/s, and 16 cores then give ~32k/s aggregate. This
> is a much harder bar than "2k/s using all cores," and it changes the verdict completely: the
> dispatch-bound modes (vol/vol_esp/pharm) can plausibly get there with a batched, SIMD-vectorized
> single-thread kernel, but the **compute-bound surface modes (surf/esp) are ~30–160× short of 2k/s on
> one core by a raw FLOP bound** — so for them, multicore is off the table and the only path is
> **accuracy-gated algorithmic work reduction** (fewer seeds/points/steps, provably < 1e-5 with zero
> winner-changes). That is the central creative tension of this effort.

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
  box. Kernel microbenchmarks, parity checks, **and single-core throughput probes** can be developed
  locally (`torch.set_num_threads(1)`); only the end-to-end real-molecule `pairs/s` numbers need WSL2.
- **16 physical cores available**, but the target is **2k/s per single core** (see Goal) — so the
  load-bearing measurement is single-thread kernel throughput, which this box can produce directly.

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
and the real gap is *wider*. **These JAX/XLA baselines also already use multiple CPU threads** (XLA
parallelizes the batched op), so the true **single-core** baseline is *lower* still and the
**per-core gap is the widest of all**. Fill the `nr=50` cells from a fresh `--no-original` CPU run
**pinned to one core** (and a 16-core aggregate cell for context).

| mode | JAX-batch 1-proc (ref, nr=5, multi-thread) | 8-cpu spawn (ref, nr=5) | single-core nr=50 (TODO) | gap to **2k/core** |
|---|--:|--:|--:|--:|
| vol      | ~110/s  | ~507/s | _TBD_ | ≥~18× |
| vol_esp  | ~124/s  | ~506/s | _TBD_ | ≥~16× |
| pharm    | ~67/s   | ~533/s | _TBD_ | ≥~30× |
| **surf** | **~6.6/s** | ~6.7/s (does not scale) | _TBD_ | **≥~300× (FLOP-bound ~30–160× over a SIMD kernel)** |
| **esp**  | **~2.7/s** | — | _TBD_ | **≥~740×** |
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

### Single-core FLOP budget — why surf/esp can't reach 2k/core bit-identically

At **2,000 pairs/s on one core** the budget is **0.5 ms/pair ≈ 1.5×10⁶ cycles/pair** (3 GHz). The
fine loop does `~50 seeds × ~80 effective Adam steps × (forward + analytical grad)` per pair:

- **vol/vol_esp/pharm** (~30×30 grid): `50 × 80 × ~30² × O(10)` ≈ a few ×10⁷ op-equiv/pair. Today
  it's dispatch-bound, not compute-bound — so a batched, SIMD-vectorized single-thread kernel that
  amortizes per-pair Python/launch overhead has a real shot at 2k/core. **Plausible; confirm by probe.**
- **surf/esp** (~100×100 grid, `exp` heavy): `50 × 80 × ~100² × O(20)` ≈ **8×10⁸ op-equiv/pair**.
  At 2k/core that demands ~1.6×10¹² op-equiv/s on one core; realistic sustained single-core
  (`exp`-bound, AVX2) is ~1–5×10¹⁰/s → **~30–160× short.** No amount of the bit-identical levers
  (L1–L3, L5) closes that on one core: they remove dispatch and memory traffic, not the arithmetic.

**Consequence.** For surf/esp the per-core 2k bar can only be met by **reducing the work itself** —
fewer seeds, decimated surface points, fewer steps — the exact levers the GPU log *rejected* for
shifting scores. So this effort must **re-litigate those levers under a strict accuracy gate**
(score `max|Δ| < 1e-5`, **zero argmax-seed-changes** on a large pseudo-symmetric cohort) and ship the
largest cut that provably holds. The bold, creative work is concentrated here: a smaller seed set that
still covers every basin, or a decimation + full-points re-score that never switches a winner. If none
passes, the honest outcome is **surf/esp hold accuracy but land below 2k/core** (while 16-core
aggregate still clears 32k/s — which is *not* the stated bar). This is measured, not assumed: the probe
below pins the real single-core ceiling.

### Measured single-core ceiling (this box — `benchmarks/experiments/cpu_singlecore_probe.py`, 2026-06-19)

Ran the **device-agnostic batched fine loop** (`compute_analytical_grad_se3_shape`, the exact
per-step CPU work) single-threaded (`set_num_threads(1)`, `OMP/MKL=1`) on synthetic clouds (timing is
independent of point distribution). **The numbers are far below the doc's earlier estimates and reset
the strategy:**

| size | µs / pose-step | quat-grad overhead | pairs/s/core @ nr=50, 100 steps | @ nr=50, 30 steps (early-stop) | gap to 2k/core |
|---|--:|--:|--:|--:|--:|
| vol (30×30)   | ~24  | ×1.06 | ~8   | ~27  | **~74×** |
| surf (75×75)  | ~129 | ×1.02 | ~1.6 | ~5.2 | **~385×** |
| surf (128×128)| ~361 | ×1.02 | ~0.6 | ~1.9 | **~1080×** |

Three findings that rewrite the plan:

1. **Bit-identical 2k/core is out of reach for EVERY mode**, not just surf/esp. Even `vol`'s bare
   kernel is ~74× short single-core at nr=50; L1–L3/L5 (which remove overhead, not arithmetic) cannot
   span that. The earlier "vol needs ~18×" was anchored to the *multi-thread, nr=5* JAX baseline — the
   true single-core nr=50 baseline is **~8 pairs/s**.
2. **The quaternion-gradient machinery is NOT the cost** (×1.02–1.10) — the overlap kernel is. Good: a
   fused kernel attacks the right thing.
3. **The naive torch kernel runs at <1 GFLOP/s** on small batched grids (24 µs for a ~20 kFLOP
   30×30 grid) — it's per-op-overhead/memory-bound across ~15 materializing passes. So the **fused
   single-pass numba kernel (L2) is the single biggest lever for ALL modes** (plausibly ~8–15×, not the
   ~3–6× I first guessed) — it was mis-scoped as surf-only.

**Revised lever arithmetic** (multiplicative, all gated): fused numba kernel **~10×** + early-stop
(100→~30 steps) **~3×** + **`num_repeats` 50→5 ~10×** (upstream calls 5 "adequate for non-surface
modes" — but it's accuracy-risky, must be gated) + (surf only) decimation Nc **~3–4×**.
- `vol`/`vol_esp`/`pharm`: 8 × 10 × 3 × 10 ≈ **2,400/core** — clears 2k, but **only with gated nr=5**
  (and fusion). **Not bit-identical.**
- `surf`/`esp`: 1.9 × 10 × 3 × 10 × 4 ≈ **2,280/core** for the 128² case *if every gated lever lands*,
  ~600–1000 if nr-reduction or decimation is held back — i.e. **genuinely on the edge, more likely
  ~2–4× short** when the accuracy gate trims the cuts. Honest expectation: surf/esp hold accuracy and
  land **below 2k/core**.

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

> **⚠️ Re-ranked by the single-core probe.** The original ranking assumed L1 (batching) was the prime
> lever — true for *aggregate*, but the probe shows the per-core kernel is so far from 2k that batching
> alone is nowhere near enough. The load-bearing per-core levers are now **L2 (fused kernel, ~10×, all
> modes)** and **L10 (`num_repeats` 50→5, ~10×, gated)** — without *both*, no mode clears 2k/core. L1
> remains the enabler (it's what lets a single dispatch feed the fused kernel and is needed for the
> aggregate 16-core number), but it is not sufficient on its own. Read L2/L10 first.

### L10 — `num_repeats` 50→5 *(now load-bearing for every mode; gated)* — risky · S
**Modes:** all. **Mechanism:** drop `num_repeats` from 50 to 5 (1 identity + 4 PCA seeds — the
RDKit/PubChem shape-color initialization count; upstream `docs/performance/timings.md` states "for
non-surface modes, five repeats are typically adequate"). A flat **10×** on every mode — the largest
accuracy-safe-*looking* lever, and per the probe arithmetic, **mandatory** to reach 2k/core. **Risk:**
this is the canonical rejected lever — fewer seeds dropped distinct-pair accuracy on the GPU side
(`SPEED_EXPERIMENTS.md` exp. #3: 50→35 already cost ~0.003–0.01; self stayed 1.0 only because the
identity seed wins self-copies). The 45 Fibonacci seeds are *coverage*; cutting them risks missing a
basin on distinct pairs, especially pseudo-symmetric/elongated molecules. **Gate (hard):** on a large
random distinct cohort (≥5000), require score `max|Δ| < 1e-5` **and zero argmax-seed changes** vs the
nr=50 reference — *per mode*; surface modes are the most seed-sensitive and likely fail at 5, so sweep
nr ∈ {50,35,25,15,10,5} and ship the smallest that passes (it may be >5 for surf/esp). **Creative
angle:** a *smarter* 5–10 seed set (e.g. PCA + a coarse-overlap-ranked subset of the Fibonacci sphere)
could keep basin coverage at lower count — design and gate it rather than blindly truncating.

### L1 — CPU overlap value+grad behind `_overlap_in_chunks` *(the enabler)* — bit-identical · M
**Modes:** vol, vol_esp, surf, esp. **Mechanism:** write `overlap_score_grad_se3_batch_cpu` mirroring
`compute_overlap_and_grad_shape` (`score/analytical_gradients/_torch.py:430`) byte-for-byte —
`K=(π/2α)^1.5`, `fit_t=bmm(R,B)+t`, `dist_sq=cdist**2`, `E=exp(−α/2·dist_sq)[·pair_weights]`,
`O_AB=K·E.sum`, the three `bmm` gradient contractions, then `project_grad_R_to_quaternion` to return
`dQ(4)` matching the Triton signature. The pad mask **and** the ESP charge weight both enter the
existing `pair_weights` slot. Add a CPU branch in `_overlap_in_chunks` (drop the CUDA-only 65535
grid-z guard). **Expected:** vol/vol_esp 10,000 `while_loop`s → 1 batched dispatch, ~16–30× over
baseline **single-thread** (overhead amortization, not parallelism) → plausibly clears 2k/**core** for
the small-grid modes. surf/esp: removes dispatch but the tile remains (~5–15× alone) —
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
**Expected (per-core framing):** the fused **single-pass, register-tiled, SIMD** kernel is the
*single-thread* win (no `(B,N,M)` materialization, no redundant passes, vectorized `exp`) — call it
~3–6× over the naïve batched-torch surf kernel on **one core**. The `prange` across pose rows is an
**aggregate** multiplier (×cores) that does **not** count toward the per-core bar. **Honest:** even
the best single-pass SIMD kernel is, per the FLOP budget, still ~5–30× short of 2k/**core** on surf —
so L2 alone does **not** reach the bar; it must stack with a gated work cut (L8 / seed reduction). L2
is still essential: it's what makes the work-reduced kernel fast enough that a *modest* cut suffices.
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
> **Aggregate only — does NOT count toward the per-core 2k bar.** Multiprocessing/`shard_map`/thread
> scaling multiplies *aggregate* throughput across the 16 cores; it cannot raise single-core
> throughput. Keep it for the 16-core aggregate number (~32k/s once per-core 2k is hit), but the
> per-core target must be met by the kernel/algorithm levers (L1–L8), not by L9.

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

**Per-core bar (2k/s/core), corrected by the single-core probe: NOT bit-identically reachable for ANY
mode; reachable for vol/vol_esp/pharm only WITH gated `num_repeats=5` + a fused kernel; surf/esp likely
land below 2k/core even with every gated lever; NO for `esp_combo`.**

- **`vol`, `vol_esp`, `pharm` — reachable but NOT bit-identical; needs the gated nr=5.** The measured
  single-core nr=50 baseline is **~8 pairs/s** (not the ~110 multi-thread figure), so the bare kernel is
  ~74× short. The bit-identical enablers (L1 batching, L3 caching, L6 fusion) plus early-stop get to
  only a few hundred/core; clearing 2k/core also requires **`num_repeats` 50→5** (a 10× cut upstream
  calls "adequate for non-surface modes," but it *shifts scores* unless the gate proves otherwise). So:
  reachable, **conditional on the nr=5 accuracy gate passing** (score < 1e-5, zero winner-changes), plus
  the fused numba kernel (L2) which is the biggest single lever (~10× — the naive torch kernel runs at
  <1 GFLOP/s). pharm easiest mechanically (L4).
- **`surf`, `esp` — likely below 2k/core even with the full gated stack.** Measured single-core nr=50
  is **~0.6–1.6 pairs/s** (~385–1080× short). Stacking *every* gated lever — fused kernel ×10,
  early-stop ×3, nr=5 ×10, decimation ×3–4 — lands ~600–2,300/core depending on how much the accuracy
  gate trims the seed and point cuts. Realistic expectation: surf/esp **hold accuracy and land below
  2k/core**; 16-core aggregate still clears ~10–32k/s (not the bar). Reaching 2k/core requires ALL of
  the risky cuts to pass simultaneously — possible, not likely. Do not promise it before the gate proves
  it pair-by-pair on a pseudo-symmetric cohort.
- **`esp_combo` — NOT demonstrably reachable accuracy-safe.** No batched CPU path
  (`fast_optimize_esp_combo_score_overlay_batch` hardcodes `device='cuda'`, falls back to the per-pair
  autograd `optimize_esp_combo_score_overlay`), no analytical gradient, three point-sets/step, and the
  combo kernel is **nondeterministic run-to-run** (per [[batch-align-host-overhead-f3]] — it's
  excluded from `parity_scores.py` for this reason). It needs a **deterministic kernel + a new batched
  CPU driver** before any speed lever applies, and cannot meet the bit-identical gate today. **Scope it
  separately; do not block the other five modes on it.**

**Assumptions:** the bar is **2k/s per single core** (16 cores available → ~32k/s aggregate, but
parallelism does not count toward the bar); torch 2.6 CPU + numba 0.61 (jax-free, so L1/L2 + the
single-core probe are buildable on the Windows box); for bit-identical levers, row-major reduction
preserved (no GEMM `cdist`), seed set / steps / `alpha` / `lam` held bit-for-bit; for surf/esp the
work-reduction levers are gated on score < 1e-5 **and** zero argmax-seed-changes over a large
pseudo-symmetric distinct cohort.

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
5. **Measure single-core, then decide.** Benchmark surf/esp **pinned to one core** (`set_num_threads(1)`).
   ≥2k/core → done. Short (expected) → deploy L7/L8 **gated** (winner-change harness) and re-measure;
   L6 per-mode where it passes < 1e-5. Track 16-core aggregate separately for context, not as the bar.
6. **`esp_combo` (separate track).** Decide: (a) deterministic batched CPU combo kernel + driver, or
   (b) document out-of-scope for the bit-identical 2k/s target. Don't block the other five.
7. **(Optional) L9** for jax-only deployments — L9a (pool) freely, L9b (scan-default) gated.

---

## Experiment log

Baseline + every lever recorded here once measured on WSL2 `SimModelEnv`. A change ships only if it is
**faster AND passes the three-tier gate** (score-level, zero winner-changes).

| # | lever | mode | throughput Δ | accuracy (self / distinct max\|Δ\| / winner-flips) | verdict |
|---|---|---|---|---|---|
| 0 | **measured single-core ceiling** (naive batched torch, nr=50, fixed 100 steps; `cpu_singlecore_probe.py`) | — | **vol ~8/s, surf-75 ~1.6/s, surf-128 ~0.6/s** (per-core) | n/a (timing) | reference — bar is ~74–1080× above this |
| L10 | `num_repeats` 50→5 (sweep, gated) | all | ~10× (if it passes) | _TBD_ (per-mode nr sweep) | pending (risky — likely fails at 5 for surf) |
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
2. **How large a work cut passes the gate?** *(Resolved input: bar = 2k/s per single core; 16 cores
   available for aggregate.)* The surf/esp verdict now hinges entirely on how much the seed/point/step
   count can be reduced while holding score < 1e-5 with zero winner-changes — measure the largest safe
   cut on a big pseudo-symmetric cohort. If the safe cut isn't ~30–160×, surf/esp hold accuracy below
   2k/core.
3. **Memory wall timing:** does the L1 torch eager path OOM at surface scale *before* L2 lands? If so,
   L3 (tiling) must ship *with* L1 for surf/esp — needs a quick `(B,50,200,200)` measurement.
4. **numba reduction parity:** will fit-outer/ref-inner `r²` (no fastmath, fp64 accumulate) hold
   score < 1e-5 with zero winner-flips on a real surf-sized batch? Measure before committing.
5. **Appetite for risky levers:** if the bit-identical path lands ~1500/s on the available cores, ship
   the gated L8 downsampling to close the gap, or accept < 2k/s on surf/esp?
6. **torch.compile (L6) on CPU:** does Inductor's reduction codegen stay < 1e-5 on the 200×200 surf
   sum, or reassociate enough to force numba-only for surf?
