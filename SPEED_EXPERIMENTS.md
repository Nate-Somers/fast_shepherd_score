# Speed experiments — toward >10k pairs/s (no accuracy loss)

**Goal:** push `benchmarks/benchmark.py` throughput (pairs/s) past **10,000** without
sacrificing accuracy: self-copy score must stay **1.000** (pharm 0.999), and
distinct-molecule-pair scores must match the current fork (`max|Δ|` ≈ 0).

**Method:** `benchmarks/experiments/speedlab.py` — in-process A/B (best-of-N warm, noise-robust
on a clock-jittery GPU), fork-only (no original-repo reruns). Each experiment is a
flag toggled in the fork; a change ships only if it is **faster AND accuracy-safe**.

## Baseline (current fork, this GPU — RTX 4050)
From the headline (single warm rep, clock-noisy) and speedlab best-of-N:

| mode | peak pairs/s (headline) | self | notes |
|---|--:|--:|---|
| vol·same | ~4000 | 1.000 | fastest; atom-cloud, no surface/charges |
| surf·same | ~3000 | 1.000 | |
| esp·same | ~700 | 1.000 | heavier kernel (charges) |
| pharm·same | ~200 | 0.999 | analytical pharm kernel |

speedlab baseline (best-of-N) recorded below once measured.

## Bottleneck (established earlier this session)
Fine loop dominates: **50 seeds × ~50 effective Adam steps × per-pose kernel**.
surf ~32% SM-util *even with CUDA graphs on* (latency/occupancy-bound, tiny 1-warp
CTAs), mem-util ~0% (not bandwidth-bound); esp ~85-98% (compute-saturated).
Already shipped: autotune+cache_results, footprint guard. Tried + rejected:
per-step kernel fusion (parity-exact but no win — register pressure ↓ occupancy).

## Hypotheses (sized by lever)
- **H1 — early seed-prune** *(biggest lever, accuracy-risky)*: optimise all 50 seeds
  for K cheap steps, keep top-M, finish only those. Cuts pose-steps ~50·50 → 50·K+M·N.
  Earlier 0-step prune was removed (raw overlap mispredicts); post-K-step overlap
  should predict the basin far better. Gate hard on distinct-pair accuracy.
- **H2 — fewer/faster steps**: lr schedule / tighter early-stop / 2nd-order step.
- **H3 — kernel occupancy**: multiple poses per CTA / better tiling to lift surf's 32%.
- **H4 — precision split**: fp16/TF32 coarse phase, fp32 final scoring.
- **H5 — host overhead**: graph whole loop, drop early-stop sync, hoist seed-gen/self-overlap.
- **H6 — algorithmic**: hierarchical coarse-to-fine (downsample surface in coarse phase).
- **H7 — per-mode**: esp/pharm-specific (why 6-20× slower).
(Parallel hypothesis workflow refines + ranks these.)

## Experiment log
Baseline (speedlab best-of-6, batch 4096): **vol 5260, surf 2545→4456, esp 1673, pharm 2128 pairs/s** (clock-jittery). self=1.0 (pharm 0.80, separate loop).

| # | hypothesis | change | mode | throughput Δ | accuracy (self / distinct max\|Δ\|) | verdict |
|---|---|---|---|---|---|---|
| 0 | baseline | — | all | — | 1.000 / 0 | reference |
| 1 | H1 early-prune | prune@10/keep10, @5/keep12 | vol,surf | **0.70–0.96× (slower)** | 1.000 / **0** | ❌ bit-identical but slower: self-copies early-stop fast, so little post-prune work; the A_k/B_k gather overhead dominates. Wrong lever for a self-copy benchmark. |

**Learning:** benchmark = self-copies that converge fast → cost is the *early* all-seed steps + early-stop over-run (patience=5 ≈ 25 steps after convergence), NOT late steps. → pivot to Lever 2 (early-stop trim) + Lever 3 (kernel occupancy).

**Measurement fix:** absolute best-of-N was swamped by the GPU's ~2.5× clock jitter (baseline swung 2485–5268 across runs). Switched speedlab to **paired interleaving** — baseline and each variant timed back-to-back within every rep, report median per-rep ratio. Now coherent + monotonic.

| 2 | Lever 2 early-stop trim | es_patience 5→2 | vol,surf | **vol 1.15×, surf 1.28×** | 1.000 / **0** | ✅ clean win — distinct pairs converge within patience=2. (p=3: 1.05/1.20×; p=4: ~1.0×.) |
| 3 | seed-count reduction | 50→35/25/15 | vol,surf | 1.1–1.4× | 1.000 / **0.003–0.04** ✗ | ❌ REJECTED — loses distinct-pair accuracy (self stays 1.0 only because identity seed wins self-copies). Even 50→35 costs ~0.003–0.01. Modest speedup anyway. Confirms: the 50-seed coverage is load-bearing. |

**Decision:** accuracy-lossy levers (seeds, precision, optimizer retune) are OUT. Only **bit-identical** levers qualify.

| 4 | num_stages + maxnreg autotune | add num_stages∈{1..4}, maxnreg to sweep | surf/vol kernel | kernel **1.09×** (s3+mr64) | bit-identical (parity 0) | ✅ kept (free, autotune-validated) but tiny |
| 5 | BLOCK/num_warps occupancy | B32, w2 | surf kernel | **0.65×, 0.37× (slower)** + parity **7e-4** | ✗ not bit-identical | ❌ can't raise occupancy bit-identically — bigger tiles change the reduction order (= an accuracy change) AND are slower |
| 6 | Lever 2 across ALL modes | patience 5→2 | esp,pharm | esp 1.51×, pharm 1.41× | esp **8.3e-3**, pharm self 0.80→0.69 ✗ | ❌ esp/pharm converge SLOWER → patience=2 stops them early. **Lever 2 ships for surf/vol ONLY** (those stay self=1.0, dist|Δ|=0); esp/pharm keep patience=5. |
| 7 | num_stages on pharm kernel | add stages∈{1..4} to pharm autotune | pharm | autotune-validated faster | bit-identical | ✅ kept (pharm's bit-identical win; self 0.80 unchanged). |

**Shipped safe wins:** num_stages/maxnreg autotune (surf/esp kernel) + num_stages (pharm kernel) + Lever 2 patience=2 (surf/vol only). esp/pharm get num_stages (bit-identical) + will get the cutoff. *Note: pharm self-copy min is ~0.80 at baseline (pre-existing under-convergence on a hard pharmacophore) — a separate quality item, not introduced here.*

| 8 | spatial cutoff | tile-skip feasibility | surf | tile-skip only **2%** even sorted | n/a | ❌ INFEASIBLE — drug molecules (~10-15 Å) are barely bigger than the 6.3 Å cutoff, so no 16-pt tile is all-far. Per-pair sparsity (0.40) needs per-lane skip (SIMT can't). |

## 🎯 BREAKTHROUGH — the align is OVERHEAD-bound, not kernel-bound
Decomposing the align (time at 2 vs 50 fine steps): **T_fixed (per-call setup) = 670–823 ms; T_step (kernel) only 0.15–7 ms.** At ~25 steps the *kernel is <20% of the align* (vol ~0%). The real cost was **per-pair Python loops over 4096 pairs**: `N_real[i]=…` (≈8k tiny GPU scalar writes) and a per-pair `quaternion_to_SE3` writeback. **Both are pure data-movement → BIT-IDENTICAL to vectorize.**

| 9 | **vectorize N_real fill (CPU list + one H2D) + batched SE3 writeback** | all modes | — | **bit-identical (self=1.000, dist|Δ|=0)** | ✅✅ THE WIN |

**Measured throughput (speedlab paired, batch 4096), before → after:**

| mode | before | **after** | × | self |
|---|--:|--:|--:|--:|
| vol | 5,260 | **18,367** | 3.5× | 1.000 |
| surf | 2,545 | **16,342** | 6.5× | 1.000 |
| esp | 1,673 | **6,199** | 3.7× | 1.000 |
| pharm | 2,128 | (re-measuring) | | |

**vol and surf are >10k, bit-identical (zero accuracy change).** Mechanism: `quaternions_to_SE3_batch` (se3.py) uses the identical per-element formula; the build fills are the same data via `copy_`/pad. No optimization math touched.

---

# Round 2 — push to 100k pairs/s on ALL modes

**Current (speedlab paired, batch 4096):** vol 18.4k, surf 16.3k, esp 6.2k, pharm 2.7k.

**The wall:** throughput plateaus by batch ~10k (vol|same n=100000 = 17.5k ≈ n=10000 16.1k) → the residual cost is **PER-PAIR** (the build + result Python loops scale linearly with batch, so larger batch doesn't amortize them). At 100k pairs/s the per-pair budget is **10 µs/pair**; today vol is ~38 µs/pair (T_fixed 158ms/4096). Python attr-access + a GPU dispatch is ~10–40 µs, so 100k is at the edge of what per-pair Python allows.

**Hypotheses (Round 2):**
- **R1 — vectorize the remaining per-pair build loops** (ref_pad fill `ref_pad[i,:n]=…` is still K GPU copies; the lazy `_*_t` cache check loop). pad_sequence / pre-stack. Bit-identical.
- **R2 — collapse the result-write loop** (`for p: p.transform=…; p.sim=float(s)` is K attr-sets + K `float()`). Batch the `.cpu()`/`float` and minimise Python per pair.
- **R3 — eliminate per-pair iteration entirely** (architectural): a batched fast-path that consumes pre-stacked tensors, bypassing the MoleculePair list — the only way under ~10 µs/pair.
- **R4 — esp/pharm kernels** (compute-bound) need a faster per-step kernel to scale; esp/pharm are far from 100k (6.2k / 2.7k) and may be infeasible without accuracy-lossy precision.
- Reuse `speedlab` (paired) + `parity_scores` (bit-exact gate). 100k is a stretch goal; expect honest ceilings per mode.

## ⚠️ Measurement confound: laptop GPU throttles under sustained load
The headline runs all modes/sizes **sequentially in one process**, so later cells run on a clock throttled by earlier heavy cells. Same surf cohort, batch 10000:
- **isolated fresh process (boost clock): 16,926 pairs/s**
- **headline (after vol's 100k cells): 1,830 pairs/s** — a **9× artifact**.
- all 4 modes back-to-back in one process @4096: vol **7.9k** (was 18.4k isolated), surf **5.2k**, esp **2.0k**, pharm 2.7k.

**True per-mode peak (isolated, best-of-N):** vol ~18k, surf ~16k (**both >10k**), esp ~6k, pharm ~2.7k. **Sustained/throttled:** ~2-3× lower. The 100k push must beat BOTH the per-pair overhead AND a hardware throttle that already caps sustained throughput ~2-3× below peak.

**Fix shipped:** the headline now runs **each `(mode, bucket, size)` cell in its own fresh subprocess** (`fork_cell` / `--fork-cell`), best-of-N time-budgeted. A fresh process = (a) a recovered clock — no cell is timed on a throttle bled from a previous cell — and (b) the kernel autotunes at **this** cell's batch. (b) is the subtle part: the Triton autotune key is the per-pose shape, so it's **batch-independent** — a process that first runs n=1 locks in the tiny-batch config and reuses it for n=10000, crippling it (a per-line prototype gave vol n=10000 = 2.5k instead of 24k). Per-cell avoids that. Replaced the in-process sequential sweep + `--cooldown` band-aid (both removed). Result vs sequential: surf|same n=10000 5393→**15021**, esp|same n=10000 573→**5672**, pharm|same n=10000 131→**8970**; all bit-exact.

## Round 2 log
| # | hypothesis | mode | throughput Δ | accuracy | verdict |
|---|---|---|---|---|---|
| 0 | baseline (Round 1 end, isolated peak) | all | vol 18k / surf 16k / esp 6k / pharm 2.7k | bit-exact | reference |
| R1 | `ref_pad`/`fit_pad` per-pair GPU copies → `pad_sequence` batched fill | vol, surf | **vol 18.4k→26k, surf 16k→18.2k** | **bit-identical** (git-stash gate ✓, self=1.0, dist\|Δ\|=0) | **SHIP** |
| — | benchmark fix v1: `--cooldown` before each timed cell — partial (surf 1830→5393); superseded | headline | n/a | n/a | replaced |
| — | benchmark fix v2: **per-cell isolated subprocess** + best-of-N; sequential mode + cooldown removed | headline | isolated peaks: vol **24.8k** / surf **16.5k** / esp **5.7k** / pharm **9.1k**; all modes now reach n=100k | self=1.0 (pharm 0.999) | **ship** |
| E1 | esp: precompute SE(3)-invariant charge weight `exp2(-c2/lam)` once, load instead of recompute (workflow #1) | esp | kernel **1.85× SLOWER** (0.27→0.50 ms) | **bit-exact** (S/dQ/dT exactly equal; `torch.exp2`==`tl.exp2`) | **REJECT** |
| E2 | esp kernel profile: NEED_GRAD on vs off | esp | gradient is only **19% (N=45) / 38% (N=64)** of kernel time; forward dominates | — | diagnostic |
| E3 | esp: **lower precision (tf32/bf16/fp16) for the forward overlap** | esp | tf32 **N/A** (element-wise kernel, contraction dim=3 <16 → no tensor cores). In-kernel fp16 **+8%**, bf16 **+9%** SLOWER; fp16-**storage**/fp32-compute **+12%** SLOWER | fp16 relS≈6e-4 (tight), bf16 ≈1e-2 (loose) | **REJECT** |

**KEY FINDING (E3) — lower precision does NOT help the ESP kernel.** Every variant is slower: the SFU `exp` runs at the same rate regardless of precision, the SE(3) gradient stays fp32, the kernel isn't load-bandwidth-bound enough for fp16 storage to win, and the fp16↔fp32 conversion overhead exceeds any savings. tf32 is inapplicable (no large matmul; the xyz contraction is only 3-wide). Combined with E1 (charge-precompute slower) and E2 (balanced forward), **the ESP kernel is at its ceiling — there is no accuracy-safe NOR lower-precision speedup on this design.** The only paths left would be algorithmic (per-pose freeze for *distinct*-pair batches, which is ~0 on the self-copy benchmark) or a hardware change.

**KEY FINDING — CORRECTED by E1/E2.** The earlier note below claimed the kernel is "compute-bound on `exp()`". **E1 disproves that for the ESP kernel:** removing one of the two `exp2`/pair (replacing it with a precomputed-weight LOAD) was **1.85× slower**, so the cheap hardware `exp2` is NOT the bottleneck — the per-tile coordinate loads + `r2`/`g_spatial` + the masked reduction are, and they're balanced (E2: gradient only ~19–38%). **There is no easy accuracy-safe ESP kernel win**: the obvious "drop an exp" lever is net-negative, the gradient is necessary, steps already early-stop (esp self-copies converge <15 steps), seeds/topk can't be cut without gaming the self-copy benchmark, and size-bucketing is already done (`_band_key`). Remaining options are all small (pharm vector pre-normalize ~1.1×, on-GPU early-stop ~1.05×) or accuracy-risky (lower precision, per-pose freeze for distinct pairs). esp is near its accuracy-safe ceiling on this kernel design.

*(Superseded note, kept for history — its "exp-bound" premise is wrong per E1:)* ~~the overlap value+grad kernel is compute-bound on the Gaussian `exp()` (mem-util ~0%, 6.05 ms/200k poses ≈ constant across latency/occupancy configs). Bit-identical headroom is ~1.09×.~~

## Assessment — is >10k reachable with ZERO accuracy loss?
- Confirmed accuracy-safe wins: **Lever 2 (~1.2×)** + **num_stages/maxnreg (~1.09× kernel)** ≈ **~1.3× total → vol ~6800, surf ~4300**.
- Every **2×-class** lever (fewer seeds / points / steps, lower precision, bigger tiles) **shifts distinct-pair scores** → rejected by the gate.
- So **on this GPU, >10k (a true 2×) with EXACT zero accuracy loss is not reachable by config/algorithm tuning** — the kernel is compute-bound and the only 2× levers trade accuracy.

## Remaining 2×-potential strategy (with a float-level accuracy nuance)
**Spatial cutoff / neighbour-list on the Gaussian overlap:** skip A·B point pairs with r beyond ~6.3 Å, where `exp(-α/2·r²) < 1e-7`. 

**Sparsity measured (cutoff_probe):** mean **0.40** of pairs within range → **~2.5× fewer exp** on average, and it scales with molecule size — imatinib 0.19 (5.2×), sildenafil 0.22 (4.5×), warfarin 0.30 (3.3×); small mols denser (benzene 0.63 → 1.6×). Stacked with the ~1.3× safe wins, this plausibly clears 10k on vol/surf.

**Cost:** real kernel rewrite — spatially sort points so 16×16 tiles are spatial regions, then skip far tile-pairs via bounding boxes (per-lane skip doesn't help SIMT; tile-level does). **Accuracy:** dropped terms sum to ~1e-4 → score change ~1e-5–1e-6 (below the 4-decimal score precision; comparable to fp32 reassociation, which BLOCK=32 already showed at 7e-4). **Effectively** zero but not **exactly** zero — needs a call on whether float-level change counts as "sacrificing accuracy."
