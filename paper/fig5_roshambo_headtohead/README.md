# Figure 5 — head-to-head vs ROSHAMBO2 (the missing benchmark)

## Caption
**The closest open competitor, ROSHAMBO2 (GPU Gaussian-shape overlay), has only ever published
speed-ups against its own predecessor — no head-to-head against an external tool exists — so
running that controlled comparison is itself a contribution; we show `fast_shepherd_score` is
faster on the matched shape task while reaching the same answer.** ROSHAMBO2 was built from
source into its own environment; both tools read the **same conformers** (embedded once and
written to SDF) and run on the **same GPU** (NVIDIA L40S) over a real 3,986-molecule library,
averaged across 8 query molecules. The representation-matched comparison is fss `vol`
(atom-centred Gaussian volume) versus ROSHAMBO2 shape (also atom-centred Gaussian); one-time
featurisation (prep) is separated from the repeated alignment (compute), each timed with
warm-up, best-of-N and CUDA synchronisation. Because the two optimisers differ (fss: 50 SE(3)
seeds × 100 steps; ROSHAMBO2: start_mode = 2 × 100 steps), effort is pinned and reported, and
the fairness anchor is recovered self-overlap on rigid SE(3) self-copies (known optimum 1.0).
**(Left)** fss `vol` sustains 67.9k pairs/s versus ROSHAMBO2's 17.7k compute-only — **3.8×
faster** — and 1.5× end-to-end, despite fss performing more search per pair (50 vs 10 starts).
**(Right)** both tools recover a self-overlap Tanimoto of 1.000 on self-copies, confirming the
throughput is measured at matched, fully-solved alignment quality rather than by one tool
cutting corners. fss additionally offers the ESP and pharmacophore overlays ROSHAMBO2 lacks
(Figs 3, 4, 6).

**Claim defended:** on identical molecules and identical hardware, `fast_shepherd_score`
is competitive with / faster than **ROSHAMBO2** — the closest open-source GPU comparator
(Gaussian shape + pharmacophore "color", GPL-3.0). No such apples-to-apples benchmark
exists in the literature (ROSHAMBO2's headline ">200×" is vs its own v1, not vs an external
tool), so this is a genuine contribution.

## Result (NVIDIA L40S, 8 query actives × 3,986-molecule library)
| tool · mode | compute-only | end-to-end | recovered self-overlap |
|---|--:|--:|--:|
| **fss · vol** (atom Gaussian) | **67,904 pairs/s** | 14,268 pairs/s | **1.000** |
| **ROSHAMBO2 · shape** (atom Gaussian) | 17,699 pairs/s | 9,486 pairs/s | **1.000** |

**fss is 3.8× faster on the matched shape comparison (compute-only), 1.5× end-to-end, and
both tools recover the optimum (1.000) on rigid SE(3) self-copies** — so the speed is at
matched, solved alignment quality, not one tool cutting corners. fss does *more* search per
pair here (50 SE(3) seeds vs ROSHAMBO2's `start_mode=2` = 10 discrete starts) and is still
faster.

## Why `vol` is the comparison (not `surf`)
ROSHAMBO2 overlays **atom-centred Gaussian volumes**. The representation-matched fss mode is
therefore `vol` (atomic-Gaussian volume overlap), not the surface-point `surf` mode. fss's
`surf` is a heavier, different representation (sampled surface points) that *additionally*
carries the ESP and pharmacophore overlays ROSHAMBO2 lacks — that capability gap is the
subject of Figs 3, 4 and 6, so Fig 5 stays a clean shape-vs-shape throughput test.

## Design (fair, built to survive review)
- **Identical molecules:** both tools read the SAME conformers (RDKit ETKDG+MMFF, embedded
  once, written to `queries.sdf`/`dataset.sdf`; fss builds its molecules from those exact
  conformers). Library = a real, diverse drug-like set (DUD-E `fa10` decoys/actives), not a
  handful of molecules sampled with replacement.
- **Symmetric timing:** one-time PREP (load/featurize) is separated from the repeated
  COMPUTE (alignment); we report BOTH compute-only and end-to-end for each tool, with
  warmup + best-of-N + CUDA sync on both sides. (The earlier harness timed only fss's GPU
  loop while timing ROSHAMBO end-to-end incl. disk IO — fixed.)
- **Quality anchor:** rather than pretend the optimizers are identical (they are not — fss
  uses many random SE(3) seeds; ROSHAMBO2 uses a few discrete starts + a local optimizer),
  we pin and report each, and use recovered self-overlap on SE(3) self-copies (optimum=1.0)
  as the real fairness check.
- **Same GPU** for both tools (run sequentially in their own conda envs on one allocation).

## Reproduce (MIT Engaging)
```bash
# build ROSHAMBO2 once (its own env, sm_89+sm_90):
sbatch paper/_engaging/build_roshambo2.sbatch
# head-to-head on an L40S (or --gres=gpu:h100:1 for H100):
sbatch paper/_engaging/fig5_roshambo.sbatch
PYTHONPATH=. python paper/fig5_roshambo_headtohead/plot.py
```
`run.py` has four phases (`--prepare`, `--side fss`, `--side roshambo2`, `--combine`); the
two tools run in separate interpreters because both ship a package that cannot co-exist.

## Provenance / honesty
- ROSHAMBO2 built from source (`molecularinformatics/roshambo2`, GPL-3.0) — NOT vendor
  numbers. API verified at runtime: `Roshambo2(q, d, color=False).compute(backend="cuda",
  start_mode=2, color_scores=False, optim_mode="shape")`, `tanimoto_shape` column read back.
- fss numbers are on the SAME GPU as Fig 2's L40S column, so the figures reconcile.
- `surf` is intentionally excluded from the throughput bars (different/heavier
  representation); its slower rate and its lower self-copy recovery (a nondeterministic
  surface-resampling artifact on flexible molecules) would only muddy the matched comparison.
