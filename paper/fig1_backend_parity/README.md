# Figure 1 — backend parity (fast kernels preserve the reference math)

**Claim defended:** the speed of the GPU path does not come at the cost of
accuracy — the fork's fast **Triton/GPU** kernels reproduce the **reference JAX**
implementation pair-for-pair.

## Design
Both paths are the *same* public API (`MoleculePairBatch.align_with_*`); only the
`backend=` differs: `"jax"` (reference JAX/XLA) vs `"triton"` (the fork's GPU
kernels). We align **all 105 distinct drug pairs** (C(15,2) over the repo's
curated drug set) with both backends, under an identical alignment config
(`num_repeats=16, steps=100, lr=0.1, alpha=0.81, lam=0.3`), and compare the
per-pair aligned Tanimoto for each of the four modes. Distinct pairs (optimum < 1)
are used on purpose so scores span a real range — a meaningful scatter, unlike
self-copies that all sit at 1.0.

## Result (measured on RTX 4050, this run)
| mode | n | mean&#124;Δ&#124; | max&#124;Δ&#124; | Spearman ρ |
|---|--:|--:|--:|--:|
| vol (shape, atoms) | 105 | 1.0e-03 | 2.2e-02 | 0.9996 |
| surf (shape, surface) | 105 | 3.9e-03 | 2.9e-02 | 0.9986 |
| esp (shape + ESP) | 105 | 4.2e-03 | 3.0e-02 | 0.9982 |
| pharm (pharmacophore) | 105 | 3.8e-03 | 1.1e-01 | 0.9960 |

Points lie on y = x in all four panels. The residual (mean|Δ| ≈ 0.001–0.004) is
**multi-start optimization noise** — the two backends draw independent random
restart poses and can settle in marginally different basins — **not kernel
error**. Scoring-level *bit-exactness* (identical pose → identical score) is gated
separately by `benchmarks/experiments/parity_scores.py`; this figure is the
stronger end-to-end check that the whole aligned-similarity pipeline agrees.

## Reproduce
```bash
PYTHONPATH=. python paper/fig1_backend_parity/run.py    # GPU env; writes parity.json
PYTHONPATH=. python paper/fig1_backend_parity/plot.py   # writes fig1_backend_parity.{png,pdf}
```
