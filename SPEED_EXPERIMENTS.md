# Speed experiments — toward >10k pairs/s (no accuracy loss)

**Goal:** push `benchmarks/headline.py` throughput (pairs/s) past **10,000** without
sacrificing accuracy: self-copy score must stay **1.000** (pharm 0.999), and
distinct-molecule-pair scores must match the current fork (`max|Δ|` ≈ 0).

**Method:** `benchmarks/speedlab.py` — in-process A/B (best-of-N warm, noise-robust
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

**KEY FINDING (kernel microbench, clean paired):** the overlap value+grad kernel is **compute-bound on the Gaussian `exp()`** (mem-util ~0%, 6.05 ms/200k poses ≈ constant across latency/occupancy configs). **Bit-identical headroom is ~1.09×, full stop.** Occupancy (BLOCK/warps) cannot be raised without changing the float reduction (accuracy). So the multi-pose-per-CTA rewrite is also unlikely to help (latency hiding gave only 1.06×; the bottleneck is exp throughput, not stalls).

## Assessment — is >10k reachable with ZERO accuracy loss?
- Confirmed accuracy-safe wins: **Lever 2 (~1.2×)** + **num_stages/maxnreg (~1.09× kernel)** ≈ **~1.3× total → vol ~6800, surf ~4300**.
- Every **2×-class** lever (fewer seeds / points / steps, lower precision, bigger tiles) **shifts distinct-pair scores** → rejected by the gate.
- So **on this GPU, >10k (a true 2×) with EXACT zero accuracy loss is not reachable by config/algorithm tuning** — the kernel is compute-bound and the only 2× levers trade accuracy.

## Remaining 2×-potential strategy (with a float-level accuracy nuance)
**Spatial cutoff / neighbour-list on the Gaussian overlap:** skip A·B point pairs with r beyond ~6.3 Å, where `exp(-α/2·r²) < 1e-7`. 

**Sparsity measured (cutoff_probe):** mean **0.40** of pairs within range → **~2.5× fewer exp** on average, and it scales with molecule size — imatinib 0.19 (5.2×), sildenafil 0.22 (4.5×), warfarin 0.30 (3.3×); small mols denser (benzene 0.63 → 1.6×). Stacked with the ~1.3× safe wins, this plausibly clears 10k on vol/surf.

**Cost:** real kernel rewrite — spatially sort points so 16×16 tiles are spatial regions, then skip far tile-pairs via bounding boxes (per-lane skip doesn't help SIMT; tile-level does). **Accuracy:** dropped terms sum to ~1e-4 → score change ~1e-5–1e-6 (below the 4-decimal score precision; comparable to fp32 reassociation, which BLOCK=32 already showed at 7e-4). **Effectively** zero but not **exactly** zero — needs a call on whether float-level change counts as "sacrificing accuracy."
