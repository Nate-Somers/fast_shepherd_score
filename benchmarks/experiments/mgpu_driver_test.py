"""Test the top-level data-parallel driver (shepherd_score.container.multi_gpu).

1. CORRECTNESS: align_multi_gpu must match the single-GPU align_with_* (same
   molecules, same kwargs, do_center=False) within tolerance, per mode.
2. SCALING: throughput vs single-GPU. Reports BOTH the end-to-end speedup (incl.
   the one-time spawn+build) AND the align-only speedup (the steady-state scaling
   the design targets ~Nx), so the fixed spawn cost is visible, not hidden.

Run on the 4xL40S node:
    python -m benchmarks.experiments.mgpu_driver_test --correctness --k 2000
    python -m benchmarks.experiments.mgpu_driver_test --speed --modes vol esp --k 100000
"""
import argparse
import time

import numpy as np
import torch

from shepherd_score.container import MoleculePair as MP, MoleculePairBatch
from shepherd_score.container.multi_gpu import align_multi_gpu, _SCORE_ATTR, _TRANSFORM_ATTR
from benchmarks.benchmark import make_real_cohort

# Public align_with_* kwargs (mirror benchmarks.benchmark._fork_align).
KW = {"vol": dict(no_H=True, alpha=0.81, max_num_steps=100),
      "surf": dict(alpha=0.81, max_num_steps=100),
      "esp": dict(alpha=0.81, lam=0.3, num_repeats=16, lr=0.1, max_num_steps=100),
      "pharm": dict(num_repeats=16, lr=0.1, max_num_steps=100)}


def _dispatch_local():
    try:
        from shepherd_score.container._batch_align import _DISPATCH_LOCAL
        return _DISPATCH_LOCAL
    except Exception:
        from shepherd_score.container._core import _DISPATCH_LOCAL
        return _DISPATCH_LOCAL


def _build_pairs(mode, k, seed, dev):
    co = make_real_cohort(mode, n_pairs=k, bucket_kind="same", seed=seed)
    return [MP(s.ref, s.fit, do_center=False, device=dev) for s in co.pairs]


def _single_gpu_align(pairs, mode):
    """Force the single-GPU path (no auto-shard) and return (scores, transforms)."""
    DL = _dispatch_local()
    DL.active = True
    try:
        getattr(MoleculePairBatch(pairs), "align_with_" + mode)(backend="triton", **KW[mode])
        torch.cuda.synchronize()
    finally:
        DL.active = False
    sc = np.array([float(getattr(p, _SCORE_ATTR[mode])) for p in pairs], dtype=np.float64)
    tf = np.stack([torch.as_tensor(getattr(p, _TRANSFORM_ATTR[mode])).detach().cpu().numpy()
                   .astype(np.float64) for p in pairs])
    return sc, tf


def correctness(modes, k, seed):
    dev0 = torch.device("cuda", 0)
    print(f"\n==== CORRECTNESS (k={k} per mode, ndev={torch.cuda.device_count()}) ====")
    for mode in modes:
        pairs = _build_pairs(mode, k, seed, dev0)
        ref_sc, ref_tf = _single_gpu_align(pairs, mode)
        got_sc, got_tf = align_multi_gpu(pairs, mode, write_back=False, **KW[mode])
        ds = float(np.abs(ref_sc - got_sc).max())
        dt = float(np.abs(ref_tf - got_tf).max())
        ok = ds < 1e-4 and dt < 1e-3
        print(f"  {mode:5s}: max|Δscore|={ds:.2e}  max|Δtransform|={dt:.2e}  "
              f"mean_score={got_sc.mean():.4f}  {'PASS' if ok else 'FAIL'}", flush=True)


def speed(modes, k, seed, reps, budget):
    dev0 = torch.device("cuda", 0)
    ndev = torch.cuda.device_count()
    print(f"\n==== SCALING (k={k}, ndev={ndev}) ====")
    for mode in modes:
        pairs = _build_pairs(mode, k, seed, dev0)
        # single-GPU best-of-N (warm)
        _single_gpu_align(pairs, mode)
        best = float("inf"); n = 0; tot = 0.0
        while n < reps and tot < budget:
            t0 = time.perf_counter(); _single_gpu_align(pairs, mode)
            dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
        single_tput = k / best

        # driver: end-to-end (one shot, incl spawn+build) + reported align-only
        t0 = time.perf_counter()
        _, _, tm = align_multi_gpu(pairs, mode, write_back=False, return_timing=True, **KW[mode])
        wall = time.perf_counter() - t0
        e2e_tput = k / wall
        align_tput = k / tm["align_max"]
        overhead = wall - tm["align_max"] - tm["build_max"]
        print(f"  {mode:5s}: single={single_tput:,.0f}/s | "
              f"driver end-to-end={e2e_tput:,.0f}/s ({e2e_tput/single_tput:.2f}x) | "
              f"driver ALIGN-only={align_tput:,.0f}/s ({align_tput/single_tput:.2f}x) | "
              f"build={tm['build_max']:.1f}s spawn/other={overhead:.1f}s", flush=True)


def pool_speed(modes, k, seed, reps, budget):
    """Persistent MultiGPUAligner: build+retain shards ONCE, then time warm,
    steady-state align() calls (the regime the ~Nx ceiling lives in)."""
    from shepherd_score.container.multi_gpu import MultiGPUAligner
    dev0 = torch.device("cuda", 0)
    ndev = torch.cuda.device_count()
    print(f"\n==== PERSISTENT POOL SCALING (k={k}, ndev={ndev}) ====")
    pairs = _build_pairs("vol", k, seed, dev0)        # pairs support every mode

    singles = {}
    for mode in modes:
        _single_gpu_align(pairs, mode)
        best = float("inf"); n = 0; tot = 0.0
        while n < reps and tot < budget:
            t0 = time.perf_counter(); _single_gpu_align(pairs, mode)
            dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
        singles[mode] = k / best

    t0 = time.perf_counter()
    pool = MultiGPUAligner(pairs)                      # spawn+import+build ONCE
    setup = time.perf_counter() - t0
    print(f"  pool setup (spawn+import+build, one-time) = {setup:.1f}s", flush=True)
    try:
        for mode in modes:
            pool.align(mode, **KW[mode])               # warm (autotune) -- not timed
            best = float("inf"); n = 0; tot = 0.0
            while n < reps and tot < budget:
                t0 = time.perf_counter(); pool.align(mode, **KW[mode])
                dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
            pool_tput = k / best
            print(f"  {mode:5s}: single={singles[mode]:,.0f}/s | "
                  f"pool warm={pool_tput:,.0f}/s | SPEEDUP={pool_tput/singles[mode]:.2f}x", flush=True)
    finally:
        pool.close()


def warmup_test(modes, k, seed, reps, budget):
    """Show the fork pool warms an N-GPU pool in ~the time of a 1-GPU pool, AND that
    fork preserves the ~Nx throughput. MUST run with CUDA un-initialized -> build
    pairs on CPU and create the pool BEFORE any GPU work."""
    from shepherd_score.container.multi_gpu import MultiGPUAligner
    tp_modes = [m for m in ("vol", "esp") if m in modes] or ["vol"]
    print(f"\n==== POOL WARMUP + THROUGHPUT (k={k}) ====")
    cpu_pairs = _build_pairs("vol", k, seed, torch.device("cpu"))   # CPU build, no CUDA
    print(f"  CUDA initialized after CPU build? {torch.cuda.is_initialized()} "
          f"(False -> fork available)", flush=True)

    # --- setup-time parity: 1-GPU vs 4-GPU pool (both fork) ---
    setups = {}
    for nd in (1, 4):
        t0 = time.perf_counter()
        pool = MultiGPUAligner(cpu_pairs, ndev=nd)
        setups[nd] = (time.perf_counter() - t0, pool.start_method)
        pool.close()
    for nd in (1, 4):
        print(f"  setup ndev={nd}: {setups[nd][0]:.1f}s  (start={setups[nd][1]})", flush=True)

    # --- throughput of the 4-GPU fork pool (parent stays CUDA-clean) ---
    pool = MultiGPUAligner(cpu_pairs, ndev=4)
    pool_tput = {}
    for mode in tp_modes:
        pool.align(mode, **KW[mode])                       # warm
        best = float("inf"); n = 0; tot = 0.0
        while n < reps and tot < budget:
            t0 = time.perf_counter(); pool.align(mode, **KW[mode])
            dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
        pool_tput[mode] = k / best
    pool.close()

    # --- single-GPU baseline (NOW it's ok to init parent CUDA) ---
    gpu_pairs = [MP(p.ref_molec, p.fit_molec, do_center=False, device=torch.device("cuda", 0))
                 for p in cpu_pairs]
    for mode in tp_modes:
        _single_gpu_align(gpu_pairs, mode)
        best = float("inf"); n = 0; tot = 0.0
        while n < reps and tot < budget:
            t0 = time.perf_counter(); _single_gpu_align(gpu_pairs, mode)
            dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
        single = k / best
        print(f"  {mode:5s}: single={single:,.0f}/s  pool(fork)={pool_tput[mode]:,.0f}/s  "
              f"SPEEDUP={pool_tput[mode]/single:.2f}x", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["vol", "surf", "esp", "pharm"])
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--budget", type=float, default=12.0)
    args = ap.parse_args()
    if not (args.correctness or args.speed or args.pool or args.warmup):
        args.correctness = True
    if args.warmup:                       # run FIRST: needs CUDA un-initialized
        warmup_test(args.modes, args.k, args.seed, args.reps, args.budget)
    if args.correctness:
        correctness(args.modes, args.k, args.seed)
    if args.speed:
        speed(args.modes, args.k, args.seed, args.reps, args.budget)
    if args.pool:
        pool_speed(args.modes, args.k, args.seed, args.reps, args.budget)


if __name__ == "__main__":
    main()
