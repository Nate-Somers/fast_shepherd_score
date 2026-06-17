"""Ad-hoc profiler for the fast alignment paths (run inside SimModelEnv).

Breaks down where wall-clock goes in the single and batch fast paths so we can
target real bottlenecks rather than guessing. Not committed as a test.
"""
import time
import numpy as np
import torch

from benchmarks.alignment_bench.workloads import make_cohort
from benchmarks.alignment_bench import backends as B


def sync():
    torch.cuda.synchronize()


def timeit(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter()
        fn(); sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
section("pharm single: fast vs torch reference, with sub-component breakdown")
# ---------------------------------------------------------------------------
from shepherd_score.alignment.utils import fast_pharm_se3 as FP
from shepherd_score.alignment.utils.fast_common import build_coarse_grid
import shepherd_score.alignment as A

co = make_cohort("pharm", n_pairs=1, size_kind="uniform", size=40, noise=0.0, seed=1)
p = co.pairs[0]
dev = torch.device("cuda")
ra = torch.tensor(p.ref.pharm_ancs, dtype=torch.float32, device=dev)
fa = torch.tensor(p.fit.pharm_ancs, dtype=torch.float32, device=dev)
rv = torch.tensor(p.ref.pharm_vecs, dtype=torch.float32, device=dev)
fv = torch.tensor(p.fit.pharm_vecs, dtype=torch.float32, device=dev)
rt = torch.tensor(p.ref.pharm_types, dtype=torch.int64, device=dev)
ft = torch.tensor(p.fit.pharm_types, dtype=torch.int64, device=dev)
rt_f = rt.to(torch.float32); ft_f = ft.to(torch.float32)

NR = 12; STEPS = 40; TOPK = 30
fast_full = lambda: FP.fast_optimize_pharm_overlay(rt, ft, ra, fa, rv, fv,
              num_repeats=NR, topk=TOPK, steps_fine=STEPS)
torch_full = lambda: A.optimize_pharm_overlay(rt_f, ft_f, ra, fa, rv, fv,
              num_repeats=NR, max_num_steps=STEPS, use_fast=False)
t_fast = timeit(fast_full); t_torch = timeit(torch_full)
print(f"fast_optimize_pharm_overlay : {t_fast*1000:8.2f} ms")
print(f"optimize_pharm_overlay(torch): {t_torch*1000:8.2f} ms   (fast/torch = {t_fast/t_torch:.2f}x)")

# sub-component: steps_fine=0 isolates coarse+seed cost
fast_coarse_only = lambda: FP.fast_optimize_pharm_overlay(rt, ft, ra, fa, rv, fv,
              num_repeats=NR, topk=TOPK, steps_fine=0)
t_coarse = timeit(fast_coarse_only)
print(f"  coarse+seed only (steps=0): {t_coarse*1000:8.2f} ms")
print(f"  => fine loop (~{STEPS} steps): {(t_fast - t_coarse)*1000:8.2f} ms "
      f"({(t_fast-t_coarse)/max(t_fast,1e-9)*100:.0f}% of total)")

# ---------------------------------------------------------------------------
section("vol/surf batch: total + seed-PCA-loop cost (CPU) + per-step sync cost")
# ---------------------------------------------------------------------------
from shepherd_score.alignment.utils import fast_se3 as FS

for mode, size_kind, sz in [("surf", "uniform", 40), ("surf", "mixed", None)]:
    co = make_cohort(mode, n_pairs=24, size_kind=size_kind, size=40,
                     size_range=(12, 72), noise=0.0, seed=2)
    fb = [b for b in B.all_backends() if b.name == "gpu_multi_batch"][0]
    cfg = B.BenchConfig(num_repeats=NR, steps_fine=STEPS, topk=TOPK)
    st = fb.prepare(co, cfg)
    t_batch = timeit(lambda: fb.run(st), n=5, warmup=2)
    print(f"\n[{mode}/{size_kind}] gpu_multi_batch total: {t_batch*1000:7.2f} ms "
          f"({len(co)/t_batch:.0f} pairs/s)")

    # time the seed loop in isolation: build padded tensors like align_batch does
    pairs = st[1]
    device = pairs[0].device
    surfs_r = [pp._ref_surf_t for pp in pairs] if hasattr(pairs[0], "_ref_surf_t") else None
    # Reconstruct padded blocks via the batch backend path is internal; instead
    # directly measure _legacy_seeds_torch over the cohort sizes.
    def seed_loop():
        for pp in pairs:
            r = torch.as_tensor(pp.ref_molec.surf_pos, dtype=torch.float32, device=device)
            f = torch.as_tensor(pp.fit_molec.surf_pos, dtype=torch.float32, device=device)
            FS._legacy_seeds_torch(r, f, num_repeats=NR)
    t_seed = timeit(seed_loop, n=3, warmup=1)
    print(f"    seed PCA loop (CPU, {len(pairs)} pairs): {t_seed*1000:7.2f} ms "
          f"({t_seed/t_batch*100:.0f}% of batch total)")

print("\nDONE")
