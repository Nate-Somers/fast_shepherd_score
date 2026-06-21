# Figure 2 — GPU throughput & batch-size scaling

## Caption
**Virtual screening aligns millions of molecule pairs, so the figure of merit is sustained
throughput at scale; we show the batched GPU path reaches ~10⁵ alignments/second on a single
datacenter card and characterise honestly how that throughput is governed by batch fill rather
than raw GPU horsepower.** Each "pair" is a real drug aligned to a rigid SE(3) copy of itself
(known optimum = 1.0), so this isolates kernel + launch throughput from convergence
difficulty. For every (GPU, mode, batch-size) cell the batched Triton alignment is timed
best-of-N over 7 repetitions and reported as mean ± SD; all four GPUs (L40S, A100-80GB,
H100-80GB, H200) run the **same conda environment** (identical torch/CUDA), removing the
mixed-software confound of stitching together runs from different machines. **(Left)** surface-
mode throughput versus batch size rises as larger batches amortise fixed per-call host
overhead, then flattens at a saturation knee and the GPU curves converge. **(Right)** peak
throughput per mode per GPU, with SD error bars, reaches ≈190k pairs/s (H200, `vol`) and ≈107k
(L40S, `surf`). Critically, there is **no clean generational ordering** — the L40S exceeds the
H100 on both `vol` and `surf`, and the ranking reshuffles by mode — the expected signature of a
launch/host-bound workload at large batch. We therefore claim high absolute throughput and
batch-scaling, not "newer GPU = faster"; multi-GPU is sub-linear and not featured.

**Claim defended:** `fast_shepherd_score` delivers high alignment throughput that **scales
with batch size** (the batched Triton path amortizes per-call host overhead) and then
**saturates** (becomes launch/host-bound) at large batch. We do **not** claim it scales
cleanly across GPU generations — our own data don't support that.

## What it shows (measured here, one controlled environment)
- **Left:** pair-alignments/s vs batch size (surface mode), mean ± SD over 7 reps per cell,
  one curve per GPU. Throughput rises with batch then flattens — the saturation knee.
- **Right:** peak throughput per mode per GPU, with SD error bars.

`measure.py` runs the sweep on whichever GPU it lands on, in the **same conda env** on every
card (same torch/CUDA) — removing the mixed-environment confound of stitching saved runs from
different machines. Self-copy pairs (optimum = 1.0): this is kernel+launch throughput.

## Peak throughput (pairs/s, mean over 7 reps)
| GPU | vol | surf | esp | pharm |
|---|--:|--:|--:|--:|
| L40S | 149,920 | **107,262** | 10,363 | 19,047 |
| A100-SXM4-80GB | 134,368 | 48,992 | 11,355 | 3,910 |
| H100-80GB-HBM3 | 96,307 | 50,508 | 13,493 | 18,151 |
| H200-NVL | **189,739** | 67,110 | **27,840** | **32,703** |

## Honest notes
- **No clean generational ordering.** L40S beats H100 on both `vol` and `surf`; the cards
  cluster and even cross over by mode. At this workload the bottleneck is kernel-launch / host
  overhead, not raw FLOPs, so a "newer = faster" story is not supported and is not claimed.
- **Saturation is real:** throughput rises with batch then flattens (and `vol` can dip at the
  largest batch) — consistent with a launch-bound regime.
- **Self-copy pairs** (known optimum) measure throughput + that the optimizer recovers perfect
  overlap; retrieval quality is Fig 6, the ROSHAMBO2 head-to-head is Fig 5.

## Reproduce
```bash
# once per GPU, same env:
for g in l40s h100 a100 h200; do sbatch --gres=gpu:$g:1 --job-name=fss_fig2_$g paper/_engaging/fig2_measure.sbatch; done
PYTHONPATH=. python paper/fig2_throughput_scaling/plot.py
```
(The repo's broader `benchmarks/results/` set — incl. an RTX 4050 laptop and a Blackwell
RTX 6000 — remains available as the package benchmark; this figure uses the controlled-env
re-measurement for honest cross-GPU error bars.)
