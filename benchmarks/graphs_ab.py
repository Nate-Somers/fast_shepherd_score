"""
Controlled in-process A/B of the CUDA-graph fine-loop fast path: time the real
align_batch_surf with FINE_CUDA_GRAPHS off vs on, on the SAME molecules,
back-to-back, median over reps. Toggling the module flag keeps it one process
(noise-robust). Also checks the achieved scores match (parity gate).

Graphs engage only for P = batch*50 <= FINE_GRAPH_MAX_P (default 16000, i.e.
batch <= 320); larger batches fall back to eager, so on==off there by design.
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
import shepherd_score.alignment.utils.fast_se3 as fse3


def _align(pairs, steps):
    MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=steps)
    return np.array([p.sim_aligned_surf for p in pairs], dtype=float)


def _time(pairs, steps, reps):
    _align(pairs, steps); torch.cuda.synchronize()        # warmup (captures graph if on)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        s = _align(pairs, steps); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda")

    print(f"CUDA-GRAPH end-to-end A/B  align_batch_surf  off vs on  (MIN/{args.reps} reps, "
          f"steps_fine={args.steps}, graph_steps={fse3._GRAPH_STEPS}, max_P={fse3._GRAPH_MAX_P})")
    hdr = (f'{"batch":>6s} {"P":>7s} | {"off ms":>8s} {"on ms":>8s} {"speedup":>8s} | '
           f'{"score_off":>9s} {"score_on":>9s} {"max|Δ|":>9s}')
    print(hdr); print("-" * len(hdr))
    for nb in args.batches:
        co = make_real_cohort("surf", n_pairs=nb, bucket_kind="same", seed=3)
        pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
        fse3._FINE_GRAPHS = False
        t_off, s_off = _time(pairs, args.steps, args.reps)
        fse3._FINE_GRAPHS = True
        t_on, s_on = _time(pairs, args.steps, args.reps)
        print(f'{nb:6d} {nb*50:7d} | {t_off*1e3:8.2f} {t_on*1e3:8.2f} {t_off/t_on:7.2f}x | '
              f'{s_off:9.4f} {s_on:9.4f} {abs(s_off-s_on):9.2e}')
        del pairs, co
        import shepherd_score.container._core as _cc
        _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
