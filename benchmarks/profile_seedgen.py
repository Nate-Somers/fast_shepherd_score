"""
Op-level torch.profiler of batched_seeds_torch (the seed-gen stage) to confirm
whether the PCA eigh is the dominant cost (the isolated phase profiler over-counts,
so this is the reliable check). Prints top ops by CUDA + CPU time.
"""
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.alignment.utils.fast_common import batched_seeds_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")

    co = make_real_cohort("surf", n_pairs=args.batch, bucket_kind="same", seed=3)
    pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
    MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=100)
    torch.cuda.synchronize()
    K = len(pairs)
    N_pad = max(p._ref_surf_t.shape[0] for p in pairs)
    ref = torch.zeros(K, N_pad, 3, device=dev); fit = torch.zeros(K, N_pad, 3, device=dev)
    Nr = torch.empty(K, dtype=torch.int32, device=dev); Mr = torch.empty(K, dtype=torch.int32, device=dev)
    for i, p in enumerate(pairs):
        n = p._ref_surf_t.shape[0]
        ref[i, :n] = p._ref_surf_t; fit[i, :n] = p._fit_surf_t; Nr[i] = n; Mr[i] = n

    for _ in range(5):
        batched_seeds_torch(ref, fit, Nr, Mr, num_seeds=50)
    torch.cuda.synchronize()

    print(f"profiling batched_seeds_torch  (batch={args.batch}, K={K}, N_pad={N_pad}, "
          f"{args.iters} iters)")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            batched_seeds_torch(ref, fit, Nr, Mr, num_seeds=50)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))


if __name__ == "__main__":
    raise SystemExit(main())
