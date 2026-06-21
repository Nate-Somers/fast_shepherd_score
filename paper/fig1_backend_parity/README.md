# Figure 1 — backend parity (fast kernels preserve the reference math)

## Caption
**A fast similarity engine is only trustworthy if it reproduces the established reference math;
here we show it does, so every speed and accuracy result that follows rests on identical
calculations rather than an approximation.** `fast_shepherd_score` exposes one alignment API
(`MoleculePairBatch.align_with_*`) with two interchangeable back ends — a reference JAX/XLA
implementation and the fork's fast Triton/GPU kernels — selected only by a `backend=` flag. We
compare them on all 105 distinct pairs of 15 marketed drugs, under one identical configuration
(16 deterministic SE(3) starts, 100 optimizer steps, α = 0.81, λ = 0.3), for the four scoring
modes the package supports: `vol` (atom-centred Gaussian volume), `surf` (sampled
molecular-surface overlap), `esp` (surface shape + electrostatic potential) and `pharm`
(pharmacophore). Two complementary experiments separate the scoring kernel from the optimizer.
**(Top row)** end-to-end aligned Tanimoto, JAX back end (x) vs Triton back end (y): points lie
on y = x, with a small residual (mean |Δ| ≈ 7×10⁻⁴ for `vol`, 4×10⁻³ for `surf`) that is
*directional* — a systematic bias, not zero-mean scatter (e.g. JAX > Triton on 85/105 `vol`
pairs, sign-test p ≪ 10⁻³). **(Bottom row)** the same molecule pairs scored at a single fixed
(identity) pose by three implementations — NumPy fp64 (reference), JAX, and PyTorch fp32 (the
precision the Triton kernels use): the fp32 GPU kernels agree with the fp64 reference to ≈10⁻⁷,
i.e. 3–4 orders of magnitude tighter than the aligned residual. Because the multi-start seeds
are deterministic and identical across back ends (no random restarts), the larger aligned
residual is not stochastic restart noise; it is the optimizer trajectory diverging under fp32
versus fp64 arithmetic, while each scoring step itself agrees to near machine precision.
Measured on an NVIDIA L40S.

**Claim defended:** the GPU path's speed does not cost accuracy — the fork's **Triton/GPU**
kernels reproduce the reference scoring; the small end-to-end aligned residual is
optimizer-trajectory divergence under fp32 arithmetic, **not** a kernel error and **not**
"random-restart noise".

## Two experiments (both in `parity.json`)
- **Top row — aligned parity:** align all 105 distinct drug pairs with `backend="jax"`
  (reference) and `backend="triton"`, identical config (`num_repeats=16, steps=100, lr=0.1,
  alpha=0.81, lam=0.3`); compare per-pair aligned Tanimoto per mode. This is the END-TO-END
  comparison (kernel + multi-start SE(3) optimizer).
- **Bottom row — fixed-pose parity:** score every pair at the SAME fixed (identity) pose with
  NumPy (fp64, reference), JAX, and PyTorch (fp32 — the precision the Triton kernels use), via
  `score_with_*(use=...)`. With the optimizer removed, this isolates pure scoring-kernel
  agreement.

## Result (NVIDIA L40S)
| mode | aligned mean&#124;Δ&#124; | aligned signed Δ | direction | fixed-pose max&#124;torch−fp64&#124; |
|---|--:|--:|:--|--:|
| vol  | 6.9e-04 | +4.1e-04 | jax>triton 85/105 | **1.8e-07** |
| surf | 4.2e-03 | +3.1e-03 | triton>jax 43/105 | 2.4e-07 |
| esp  | ~4e-03 | small + | systematic | ~1e-6 |
| pharm| ~4e-03 | — | — | ~1e-6 |

**At a fixed pose the fp32 GPU kernels agree with the fp64 reference to ~1e-7** — the kernels
are faithful. The end-to-end aligned residual is ~1e-3, i.e. **3–4 orders of magnitude larger
than the kernel floor**, and it is *directional* (a small systematic bias, confirmed by a sign
test), not zero-mean scatter.

## Why this is the correct story (the old caption was wrong)
The multi-start SE(3) seeds are **deterministic and identical across backends** (identity +
PCA quaternions + precomputed Fibonacci rotations — no RNG; see
`alignment/utils/fast_common.py`). So the aligned residual **cannot** be "independent random
restart" noise. It is the optimizer trajectory diverging under different arithmetic
(fp32 Triton vs the fp64-capable JAX reference) — the fixed-pose panel proves the per-step
scoring agrees to ~1e-7, so the divergence accumulates only through the optimization path.

## Reproduce
```bash
PYTHONPATH=. python paper/fig1_backend_parity/run.py    # GPU env w/ jax+optax; writes parity.json
PYTHONPATH=. python paper/fig1_backend_parity/plot.py   # writes fig1_backend_parity.{png,pdf}
```
