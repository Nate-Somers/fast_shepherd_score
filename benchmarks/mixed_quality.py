"""
MIXED-molecule overlap QUALITY: fork Triton-GPU vs original JAX-CPU.

Unlike the self-SE(3)-copy parity test (optimum = 1.0), here ref and fit are
DIFFERENT drugs, so the best achievable overlap is < 1.0 and unknown. This
measures the thing that actually matters: does the fork find overlaps as good as
the original optimiser?  NOT a speed test.

Fairness: both methods get the SAME seed set -- _initialize_se3_params(num_repeats)
with num_repeats=50 (each method's real default; the fork's seeds-all is exactly
this 50-seed set) -- and the same 100 fine steps. The only residual difference is
each method's stock learning rate (JAX 0.10 vs fork 0.075) and the optimiser
backend (JAX Adam vs Triton Adam), which is precisely what we want to compare.
"""
import argparse
import numpy as np
import torch

from benchmarks.alignment_bench import jax_shmap
from benchmarks.alignment_bench import backends as B
from benchmarks.alignment_bench.workloads import PairSpec, Cohort
from benchmarks.real_workloads import _build_molecule, _count_for_mode, DRUGS
from shepherd_score.container import MoleculePair

# DIFFERENT-molecule pairs spanning small->large, chemically varied.
PAIRS = [
    ("benzene", "phenol"),
    ("phenol", "aniline"),
    ("paracetamol", "salicylic_acid"),
    ("aspirin", "salicylic_acid"),
    ("aspirin", "ibuprofen"),
    ("ibuprofen", "naproxen"),
    ("paracetamol", "paracetamol2"),
    ("caffeine", "paracetamol"),
    ("naproxen", "indomethacin"),
    ("warfarin", "diphenhydramine"),
    ("indomethacin", "warfarin"),
    ("sildenafil", "imatinib"),
]
SMI = {n: s for n, s, _ in DRUGS}


def build_cohort(mode):
    pairs = []
    for a, b in PAIRS:
        ref = _build_molecule(SMI[a]); fit = _build_molecule(SMI[b])
        pairs.append(PairSpec(ref=ref, fit=fit, R=np.eye(3), t=np.zeros(3),
                              n_ref=_count_for_mode(ref, mode),
                              n_fit=_count_for_mode(fit, mode)))
    return Cohort(name="mixed", mode=mode, pairs=pairs, size_kind="cross",
                  seed=0, noise=0.0, meta={})


def fork_scores(mode, cohort, cfg):
    dev = torch.device("cuda")
    pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in cohort.pairs]
    if mode == "surf":
        MoleculePair.align_batch_surf(pairs, alpha=cfg.alpha, steps_fine=cfg.steps_fine)
        return np.array([p.sim_aligned_surf for p in pairs], dtype=float)
    MoleculePair.align_batch_esp(pairs, alpha=cfg.alpha, lam=cfg.lam,
                                 num_repeats=cfg.num_repeats, topk=cfg.topk,
                                 steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
    return np.array([p.sim_aligned_esp for p in pairs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp"])
    args = ap.parse_args()
    cfg = B.BenchConfig(num_repeats=50, max_steps=100, steps_fine=100, topk=30)

    print("=" * 78)
    print("MIXED-PAIR OVERLAP QUALITY  (different molecules, optimum < 1.0; NOT speed)")
    print("fork Triton-GPU  vs  original JAX-CPU | same 50-seed init + 100 steps")
    print(f"JAX lr={cfg.lr_cpu}  fork lr={cfg.lr_fast}  alpha={cfg.alpha}  lam={cfg.lam}")
    print("=" * 78)

    for mode in args.modes:
        co = build_cohort(mode)
        js = jax_shmap.run(jax_shmap.prepare_inputs(mode, co, cfg), cfg)
        fs = fork_scores(mode, co, cfg)
        print(f"\n--- {mode} ---")
        print(f'{"ref":>14s} {"fit":>14s} {"nR":>4s} {"nF":>4s} | '
              f'{"JAX":>7s} {"fork":>7s} {"Δ(fork-JAX)":>11s}')
        deltas = []
        for (a, b), p, j, f in zip(PAIRS, co.pairs, js, fs):
            d = float(f - j); deltas.append(d)
            print(f'{a:>14s} {b:>14s} {p.n_ref:4d} {p.n_fit:4d} | '
                  f'{j:7.4f} {f:7.4f} {d:+11.4f}')
        deltas = np.array(deltas)
        nwin = int((fs >= js - 1e-4).sum())
        print(f'{"MEAN":>14s} {"":>14s} {"":>4s} {"":>4s} | '
              f'{js.mean():7.4f} {fs.mean():7.4f} {deltas.mean():+11.4f}')
        print(f'  fork >= JAX (tol 1e-4): {nwin}/{len(PAIRS)} | '
              f'worst Δ: {deltas.min():+.4f} | best Δ: {deltas.max():+.4f}')


if __name__ == "__main__":
    raise SystemExit(main())
