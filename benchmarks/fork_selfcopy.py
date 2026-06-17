"""
Fast fork-only parity signal: run the REAL production batch aligners
(MoleculePair.align_batch_*) on real-molecule self-SE(3)-copy cohorts and report
the drop from the known optimum (1.0). The original reference reaches 1.0 on
these (established by real_parity.py), so drop ~ 0 == parity with the original.

This is the quick iteration signal (no slow CPU reference); real_parity.py is the
authoritative original-vs-fork comparison for the final report.
"""
import argparse
import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair


def run_mode(mode, cohort, steps_fine):
    dev = torch.device("cuda")
    pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in cohort.pairs]
    if mode == "surf":
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=steps_fine)
        return np.array([p.sim_aligned_surf for p in pairs], dtype=float)
    if mode == "esp":
        MoleculePair.align_batch_esp(pairs, alpha=0.81, lam=0.3, steps_fine=steps_fine)
        return np.array([p.sim_aligned_esp for p in pairs], dtype=float)
    if mode == "pharm":
        MoleculePair.align_batch_pharm(pairs, steps_fine=steps_fine)
        return np.array([p.sim_aligned_pharm for p in pairs], dtype=float)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--steps-fine", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    print(f"FORK SELF-COPY PARITY (drop from known optimum 1.0; steps_fine={args.steps_fine})")
    hdr = f'{"mode":6s} {"bucket":6s} {"fork_mean":>9s} {"mean_drop":>9s} {"worst_drop":>10s} {"n>=0.99":>8s}'
    print(hdr); print("-" * len(hdr))
    for mode in args.modes:
        for bucket in ("same", "cross"):
            co = make_real_cohort(mode, n_pairs=args.n_pairs, bucket_kind=bucket, seed=args.seed)
            s = run_mode(mode, co, args.steps_fine)
            drop = np.clip(1.0 - s, 0, None)
            frac = float((s >= 0.99).mean())
            print(f'{mode:6s} {bucket:6s} {s.mean():9.4f} {drop.mean():9.4f} '
                  f'{drop.max():10.4f} {frac:8.2f}')


if __name__ == "__main__":
    raise SystemExit(main())
