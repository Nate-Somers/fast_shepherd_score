"""
Controlled in-process A/B of the fine-step count: time align_batch_* at
steps_fine=100 vs 50 on the SAME molecules, back-to-back, median over reps -> a
noise-robust measure of lever #1 (sub-batcher at memory-sized chunks; perf cap
reverted). Also prints the achieved scores at each step count (parity must hold).
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair


def _align(mode, pairs, steps):
    if mode == "surf":
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=steps)
        return np.array([p.sim_aligned_surf for p in pairs], dtype=float)
    if mode == "esp":
        MoleculePair.align_batch_esp(pairs, alpha=0.81, lam=0.3, num_repeats=16,
                                     topk=30, steps_fine=steps, lr=0.075)
        return np.array([p.sim_aligned_esp for p in pairs], dtype=float)
    MoleculePair.align_batch_pharm(pairs, num_repeats=16, topk=30, steps_fine=steps, lr=0.075)
    return np.array([p.sim_aligned_pharm for p in pairs], dtype=float)


def _time(mode, pairs, steps, reps):
    _align(mode, pairs, steps); torch.cuda.synchronize()          # warmup
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        s = _align(mode, pairs, steps); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024, 10000])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda")

    print("CONTROLLED A/B  steps_fine 100 vs 50  (same molecules, MIN over %d reps)" % args.reps)
    hdr = (f'{"mode":5s} {"batch":>6s} | {"t100 s":>8s} {"t50 s":>8s} {"speedup":>8s} | '
           f'{"align/s@50":>10s} | {"score100":>8s} {"score50":>8s}')
    print(hdr); print("-" * len(hdr))
    for mode in args.modes:
        for nb in args.batches:
            co = make_real_cohort(mode, n_pairs=nb, bucket_kind="same", seed=3)
            pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
            t100, s100 = _time(mode, pairs, 100, args.reps)
            t50, s50 = _time(mode, pairs, 50, args.reps)
            print(f'{mode:5s} {nb:6d} | {t100:8.3f} {t50:8.3f} {t100/t50:7.2f}x | '
                  f'{nb/t50:10.1f} | {s100:8.4f} {s50:8.4f}')
            del pairs, co
            import shepherd_score.container._core as _cc
            _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
            torch.cuda.empty_cache()
        print("-" * len(hdr))


if __name__ == "__main__":
    raise SystemExit(main())
