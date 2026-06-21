# Paper figures — `fast_shepherd_score`

Production figures + the experiments behind them, positioning the package against
prior art (see `../RELATED_WORK.md`). One subfolder per figure; each has a `run.py`
(gathers data), a `plot.py` (renders `figN_*.{png,pdf}`), and a `README.md`
(claim, design, fairness, provenance, reproduce). Shared helpers: `common.py`.

## Environment
Runs in the repo's GPU env (`SimModelEnv`, WSL2): torch 2.5.1+cu124, RDKit
2025.03, JAX, Triton, Open3D, xTB, plus `espsim` (`pip install espsim`). RDKit
provides the O3A + USRCAT baselines natively.
```bash
# from repo root, in the GPU env:
PYTHONPATH=. python paper/figN_.../run.py     # then plot.py
```
All figures use the **same real-drug molecule set** as `benchmarks/benchmark.py`
(RDKit ETKDG conformers + Open3D surfaces + MMFF/xTB charges + RDKit pharmacophores).

## Figure status & headline results
All figures are now produced on **MIT Engaging** (datacenter GPUs, env `fss`); figs 1–4
were revised to fix claims surfaced by an adversarial methods review (see git history).
**Manuscript-ready figure legends (high-level → why it matters → methods) are in
[`CAPTIONS.md`](CAPTIONS.md)**, and a `## Caption` section heads each figure's own README.

| # | Figure | Claim | Status | Headline |
|---|---|---|---|---|
| 1 | backend parity | fast kernels don't change the math | ✅ | fixed-pose fp32 GPU vs fp64 ref ≈1e-6; aligned residual ~1e-3 is **optimizer-trajectory divergence** (deterministic seeds) + a small systematic offset — not "random noise" |
| 2 | throughput & scaling | high GPU throughput, scales w/ batch | ✅ per-rep mean±SD | rises with batch then **saturates** (launch-bound); does **not** order by GPU generation |
| 3 | speed vs CPU baselines | unique: aligned shape+ESP+pharm **and** GPU | ✅ **same-machine** | fss the only ESP+pharm+GPU+open tool; ESP speedup vs ESP-Sim measured on one node |
| 4 | ESP carries orthogonal info | ESP ≠ shape | ✅ +UQ + dipole axis | separates shape-matched analogs by polarity — weight (λ)-dependent, quantified |
| 5 | head-to-head vs ROSHAMBO2 | competitive/faster than the open GPU SOTA | ✅ **run (L40S)** | **fss `vol` 3.8× ROSHAMBO2 shape (compute); both recover self-overlap 1.000** |
| 6 | enrichment + ESP ablation | retrieves actives; ESP improves retrieval | ✅ **run (DUD-E)** | **adding ESP to shape ↑ retrieval on charged pockets** (ACES ΔAUC +0.25, 8/8 queries); pocket-specific |

## Two findings the paper addresses head-on (both now RESOLVED)
1. **ESP is real, weight-sensitive (Fig 4), and demonstrably useful (Fig 6).** At the
   default `lam=0.3` the ESP-Tanimoto term is shape-dominated; it discriminates only at
   smaller `lam`. Fig 6 makes this the headline: across a λ sweep the *retrieval* gain from
   ESP grows as λ decreases and is concentrated on electrostatically-driven pockets (ACES
   ΔAUC +0.016→+0.25 as λ 0.3→0.003). **Expose `lam` and report the sweep.**
2. **The FSS-vs-ROSHAMBO2 benchmark now exists (Fig 5).** Built ROSHAMBO2 from source and
   ran the first apples-to-apples open-GPU shape comparison (same conformers, same L40S):
   fss `vol` is 3.8× faster on the matched representation, both recovering the optimum.

## Reproduce everything (MIT Engaging, env `fss`)
The `_engaging/` folder has unattended SLURM jobs (submit once, retrieve the result JSON,
run the local `plot.py`):
```bash
sbatch paper/_engaging/build_roshambo2.sbatch                 # build ROSHAMBO2 (Fig 5), once
sbatch paper/_engaging/figs134_refresh.sbatch                 # Figs 1, 3, 4 on one GPU node
for g in l40s h100 a100 h200; do sbatch --gres=gpu:$g:1 --job-name=fss_fig2_$g paper/_engaging/fig2_measure.sbatch; done
sbatch paper/_engaging/fig5_roshambo.sbatch                   # Fig 5 head-to-head
for t in aces fa10 hmdh ampc ada adrb2 andr fabp4; do sbatch --job-name=fss_fig6_$t --export=ALL,TARGET=$t paper/_engaging/fig6_enrichment.sbatch; done
# then, in each figN folder: PYTHONPATH=. python plot.py
```

## Provenance / honesty notes
- Every figure is **measured here**, no fabricated data. Figs 1–6 run in one conda env on
  Engaging datacenter GPUs (consistent torch/CUDA — removes the mixed-environment confound).
- **Fig 1** caption corrected: the residual is a small *systematic* fp32-vs-fp64 offset +
  optimizer-trajectory divergence (multi-start seeds are deterministic/identical across
  backends), not "random-restart noise"; a fixed-pose panel isolates kernel agreement (~1e-6).
- **Fig 2** claims batch-scaling + high absolute throughput with per-rep error bars; it does
  **not** claim generational scaling (our own data don't support it) and multi-GPU is not featured.
- **Fig 5** ROSHAMBO2 is **built from source and benchmarked head-to-head** (GPL-3.0), not cited.
- FastROCS/ROCS numbers remain vendor-published (proprietary; not runnable) — cited, not re-measured.
