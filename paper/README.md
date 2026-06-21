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
| # | Figure | Claim | Status | Headline |
|---|---|---|---|---|
| 1 | backend parity | fast kernels don't change the math | ✅ run (RTX 4050) | Triton vs JAX: mean&#124;Δ&#124; 1–4e-3, ρ≥0.996 over 105 pairs |
| 2 | throughput & scaling | high GPU throughput, scales w/ batch & hardware | ✅ from real saved 6-GPU data | up to ~178k pairs/s (H100, vol); scales to ~10⁴ batch |
| 3 | speed vs CPU baselines | unique: aligned shape+ESP+pharm **and** GPU | ✅ run (RTX 4050) | only tool with ESP overlay + pharm + GPU; L40S esp ≈28× ESP-Sim |
| 4 | ESP carries orthogonal info | ESP ≠ shape | ✅ run (xTB) | ESP separates shape-matched analogs by polarity — **weight (lam)-dependent** |
| 5 | head-to-head vs ROSHAMBO2 | competitive/faster than the open GPU SOTA | 🧩 harness ready (needs CUDA toolkit + datacenter GPU) | — |
| 6 | enrichment + ESP ablation | retrieves actives; ESP improves retrieval | 🧩 harness ready + smoke-tested (needs DUDE-Z/LIT-PCBA) | — |

✅ = real figure produced here · 🧩 = complete ready-to-run harness, not executed
(infeasible on this 6 GB laptop without a CUDA toolkit / large datasets — see each
README).

## Two findings the paper must address head-on
1. **ESP is real but weight-sensitive (Fig 4).** At the package default `lam=0.3`
   the ESP-Tanimoto score is shape-dominated (adds ~nothing); ESP becomes
   discriminative only at smaller `lam` (and with physical xTB charges). Decide how
   to present the ESP differentiator: justify a default `lam`, expose it, use the
   additive `esp_combo` (ShaEP) metric, and lean on enrichment (Fig 6) as the proof.
2. **No published FSS-vs-ROSHAMBO2 benchmark exists (Fig 5)** — running it (same
   hardware, same library) is a genuine contribution, but needs the ROSHAMBO CUDA
   build on a datacenter GPU.

## Reproduce everything
```bash
# Fig 1, 3, 4 (GPU env): run.py then plot.py in each folder
# Fig 2: plot.py only (reads benchmarks/results/*/plot_data.json)
# Fig 5: bash fig5_roshambo_headtohead/setup.sh   then run.py   (CUDA-toolkit machine)
# Fig 6: download DUDE-Z, then fig6_enrichment/run.py --actives ... --decoys ...
```

## Provenance / honesty notes
- Fig 2 reuses **real saved** benchmark runs (6 GPUs); no fabricated data.
- Multi-GPU scaling is **not** claimed (L40S 1→4 GPU is sub-linear; see fig2 README).
- FastROCS/ROCS numbers are vendor-published (proprietary; not runnable) — cited,
  not re-measured. ROSHAMBO2 appears in Fig 3's capability matrix from the
  literature (GPL; CUDA build not run here).
