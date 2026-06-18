"""
Phase breakdown of align_batch_surf's per-call cost, to locate the fixed overhead
that dominates end-to-end at small batch. Each stage is timed in isolation
(median over reps, CUDA-synced):
  prep   : build padded ref/fit tensors from the pairs (host loop + H2D copies)
  self   : VAA/VBB self-overlaps
  seed   : batched_seeds_torch (PCA eigh + Fibonacci seeds)
  fine   : the fine loop (= coarse_fine_align_many minus seed-gen)
  xfer   : scores/q/t host transfer (.cpu())
Reports ms and % of the summed stages, per batch size.
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.alignment.utils.fast_common import batched_seeds_torch
from shepherd_score.alignment.utils.fast_se3 import _self_overlap_in_chunks, coarse_fine_align_many


def _med(fn, reps, warmup=2):
    r = None
    for _ in range(warmup):
        r = fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda")

    print(f"align_batch_surf phase breakdown (surf same-bucket, MIN... median/{args.reps} reps, steps={args.steps})")
    hdr = (f'{"batch":>6s} {"Npad":>5s} | {"prep ms":>8s} {"self ms":>8s} {"seed ms":>8s} '
           f'{"fine ms":>8s} {"xfer ms":>8s} | {"sum ms":>8s} || dominant')
    print(hdr); print("-" * len(hdr))
    for nb in args.batches:
        co = make_real_cohort("surf", n_pairs=nb, bucket_kind="same", seed=3)
        pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
        # one real align to populate p._ref_surf_t / warm kernels
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=args.steps)
        torch.cuda.synchronize()
        K = len(pairs)
        N_pad = max(p._ref_surf_t.shape[0] for p in pairs)
        M_pad = max(p._fit_surf_t.shape[0] for p in pairs)

        def build_pads():
            ref = torch.zeros(K, N_pad, 3, device=dev)
            fit = torch.zeros(K, M_pad, 3, device=dev)
            Nr = torch.empty(K, dtype=torch.int32, device=dev)
            Mr = torch.empty(K, dtype=torch.int32, device=dev)
            for i, p in enumerate(pairs):
                n = p._ref_surf_t.shape[0]; m = p._fit_surf_t.shape[0]
                ref[i, :n] = p._ref_surf_t; fit[i, :m] = p._fit_surf_t
                Nr[i] = n; Mr[i] = m
            return ref, fit, Nr, Mr

        t_prep, (ref_pad, fit_pad, N_real, M_real) = _med(build_pads, args.reps)
        t_self, _ = _med(lambda: (_self_overlap_in_chunks(ref_pad, N_real, 0.81),
                                  _self_overlap_in_chunks(fit_pad, M_real, 0.81)), args.reps)
        VAA = _self_overlap_in_chunks(ref_pad, N_real, 0.81)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, 0.81)
        t_seed, _ = _med(lambda: batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=50),
                         args.reps)
        t_align, out = _med(lambda: coarse_fine_align_many(
            ref_pad, fit_pad, VAA, VBB, N_real=N_real, M_real=M_real,
            alpha=0.81, steps_fine=args.steps), args.reps)
        sc, q, t = out
        t_xfer, _ = _med(lambda: (sc.cpu(), q.cpu(), t.cpu()), args.reps)
        t_fine = max(0.0, t_align - t_seed)

        stages = {"prep": t_prep, "self": t_self, "seed": t_seed, "fine": t_fine, "xfer": t_xfer}
        ssum = sum(stages.values())
        dom = max(stages, key=stages.get)
        print(f'{nb:6d} {N_pad:5d} | {t_prep*1e3:8.2f} {t_self*1e3:8.2f} {t_seed*1e3:8.2f} '
              f'{t_fine*1e3:8.2f} {t_xfer*1e3:8.2f} | {ssum*1e3:8.2f} || '
              f'{dom} ({100*stages[dom]/ssum:.0f}%)')
        del pairs, co
        import shepherd_score.container._core as _cc
        _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
