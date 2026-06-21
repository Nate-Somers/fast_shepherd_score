# Figure 2 — GPU throughput & batch-size scaling

**Claim defended:** `fast_shepherd_score` delivers high alignment throughput on
commodity *and* datacenter GPUs, and throughput scales with batch size (the
batched Triton path amortizes per-call host overhead) and across GPU generations.

## What it shows
- **Left panel:** pair-alignments/s vs batch size (surface-shape mode, same-size
  bucket) for all six GPUs. Curves rise with batch size and plateau ~10³–10⁴
  pairs/call — below that the align is host/launch-bound, above it GPU-bound.
- **Right panel:** peak throughput per alignment mode (shape-atoms `vol`,
  shape-surface `surf`, shape+ESP `esp`, pharmacophore `pharm`) per GPU.

## Data provenance (no new compute)
Pure plotting from the repo's **existing, real** benchmark outputs at
`benchmarks/results/<gpu>/plot_data.json`, produced by `benchmarks/benchmark.py`
(real drug molecules → RDKit ETKDG conformers + Open3D surfaces + MMFF charges +
RDKit pharmacophores; each pair = a molecule aligned to a rigid SE(3) copy of
itself, optimum score 1.0; isolated best-of-N per cell). Hardware: RTX 4050
laptop, L40S (1 & 4 GPU), RTX PRO 6000 Blackwell, H100 NVL, H200.

## Peak throughput (pairs/s, max over buckets/sizes)
| GPU | vol | surf | esp | pharm |
|---|--:|--:|--:|--:|
| RTX 4050 laptop | 54,231 | 28,817 | 8,455 | 23,279 |
| L40S · 1 GPU | 160,497 | 81,695 | 32,133 | 57,316 |
| L40S · 4 GPU | 160,224 | 125,839 | 72,431 | 83,893 |
| RTX PRO 6000 Blackwell | 174,350 | 108,845 | 48,911 | 64,524 |
| H100 NVL | 177,623 | 65,685 | 28,467 | 67,756 |
| H200 | 165,737 | 70,557 | 31,776 | 74,836 |

## Honest notes / caveats
- **Multi-GPU is NOT featured as a scaling result.** L40S 1→4 GPU gives only
  ~1.0× on `vol` (already saturated/host-bound) up to ~2.25× on the
  compute-heavy `esp`, ~1.5× on average — sub-linear. The persistent-worker-pool
  fix that would give near-linear multi-GPU scaling is not yet shipped, so we do
  not claim multi-GPU scaling here.
- These are **self-copy** pairs (known optimum), measuring throughput + that the
  optimizer recovers the perfect overlap, not retrieval quality (see Fig 5).
- `vol` is fastest (atom point cloud, no surface/charges); `esp` is heaviest
  (surface points × electrostatics).

## Reproduce
```bash
PYTHONPATH=. python paper/fig2_throughput_scaling/plot.py
```
To regenerate the underlying numbers on a given GPU:
`python -m benchmarks.benchmark --no-original --tag <gpu>` (in the WSL GPU env).

Outputs: `fig2_throughput_scaling.{png,pdf}` (main),
`fig2_supp_scaling_by_mode.{png,pdf}` (per-mode supplementary).
