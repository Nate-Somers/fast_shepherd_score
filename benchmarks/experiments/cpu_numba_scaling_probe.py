"""Why does the numba CPU aligner only scale ~3-4x on 8 cores? Measure where it's lost.

Fully synthetic (timing is shape-, not value-dependent), so it needs no open3d/molcache.
Four experiments, each isolating one suspect:

  A. kernel-only scaling   -- is the prange overlap kernel itself linear in threads?
  B. end-to-end scaling    -- coarse_fine_align_many at numba=t, torch=1 (the real driver,
                              incl. the per-step torch Adam/where bookkeeping = serial frac).
  C. oversubscription      -- numba=N,torch=1  vs  numba=N,torch=N (do the two pools fight?).
  D. process sharding      -- P single-threaded processes, started together via a Barrier
                              (the embarrassingly-parallel-over-pairs alternative to threads).

Run:  <env>/bin/python -m benchmarks.experiments.cpu_numba_scaling_probe
"""
from __future__ import annotations
import json, os, time
import numpy as np
import torch


def _batch(n_pairs, N, seed=0):
    """Synthetic padded (K,N,3) ref/fit + real counts; self-copy-ish geometry."""
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n_pairs, N, 3)) * 2.0).astype(np.float32)
    B = (A + 0.3 * rng.standard_normal((n_pairs, N, 3))).astype(np.float32)
    Nr = np.full(n_pairs, N, np.int32)
    return (torch.from_numpy(A), torch.from_numpy(B),
            torch.from_numpy(Nr), torch.from_numpy(Nr.copy()))


def _set_threads(numba_t, torch_t):
    import numba
    numba.set_num_threads(int(numba_t))
    torch.set_num_threads(int(torch_t))


# --- A: kernel only --------------------------------------------------------
def kernel_only(K, N, threads, reps=8):
    from shepherd_score.alignment.utils import cpu_overlap as CO
    _set_threads(threads, 1)
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.standard_normal((K, N, 3)) * 2, dtype=torch.float32)
    B = torch.tensor(rng.standard_normal((K, N, 3)) * 2, dtype=torch.float32)
    q = torch.tensor(rng.standard_normal((K, 4)), dtype=torch.float32); q /= q.norm(dim=1, keepdim=True)
    t = torch.tensor(rng.standard_normal((K, 3)), dtype=torch.float32)
    Nr = torch.full((K,), N, dtype=torch.int32)
    CO.overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Nr)  # warmup
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        CO.overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Nr)
        best = min(best, time.perf_counter() - t0)
    return best


# --- B/C: end-to-end driver ------------------------------------------------
def end_to_end(n_pairs, N, numba_t, torch_t, num_seeds=50, steps=100, reps=3):
    from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many
    from shepherd_score.alignment.utils.cpu_overlap import _batch_self_overlap
    _set_threads(numba_t, torch_t)
    A, B, Nr, Mr = _batch(n_pairs, N, seed=1)
    VAA = _batch_self_overlap(A, Nr, 0.81)
    VBB = _batch_self_overlap(B, Mr, 0.81)

    def run():
        coarse_fine_align_many(A, B, VAA, VBB, alpha=0.81, num_seeds=num_seeds,
                               steps_fine=steps, N_real=Nr, M_real=Mr)
    run()                                                      # warmup
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); run(); best = min(best, time.perf_counter() - t0)
    return best, n_pairs / best


# --- D: process sharding ---------------------------------------------------
def _proc_worker(barrier, q, n_pairs, N, num_seeds, steps, reps):
    """One single-threaded process: warm up, sync at the barrier, time `reps` aligns."""
    _set_threads(1, 1)
    from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many
    from shepherd_score.alignment.utils.cpu_overlap import _batch_self_overlap
    A, B, Nr, Mr = _batch(n_pairs, N, seed=1)
    VAA = _batch_self_overlap(A, Nr, 0.81); VBB = _batch_self_overlap(B, Mr, 0.81)

    def run():
        coarse_fine_align_many(A, B, VAA, VBB, alpha=0.81, num_seeds=num_seeds,
                               steps_fine=steps, N_real=Nr, M_real=Mr)
    run()                                                      # warmup (compile-cache load)
    barrier.wait()                                             # all procs start timed phase together
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    q.put(time.perf_counter() - t0)


def process_shard(P, n_pairs, N, num_seeds=50, steps=100, reps=3):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(P); q = ctx.Queue()
    procs = [ctx.Process(target=_proc_worker,
                         args=(barrier, q, n_pairs, N, num_seeds, steps, reps))
             for _ in range(P)]
    for p in procs: p.start()
    times = [q.get() for _ in range(P)]
    for p in procs: p.join()
    wall = max(times)                                          # straggler bounds the cohort
    return wall, (P * n_pairs * reps) / wall                  # aggregate pairs/s


# --- E: the REAL persistent pool (_cpu_pool.align_pairs, incl. queue/pickle overhead) -
class _M:
    def __init__(self, d): self.__dict__.update(d)


class _P:
    def __init__(self, N):
        rng = np.random.default_rng()
        a = (np.random.default_rng(0).standard_normal((N, 3)) * 2).astype(np.float32)
        self.ref_molec = _M(dict(atom_pos=a, surf_pos=a))
        self.fit_molec = _M(dict(atom_pos=a + 0.3, surf_pos=a + 0.3))
        self.device = __import__("torch").device("cpu")


def end_to_end_pool(mode, n_pairs, N, P, steps=100, reps=3):
    from shepherd_score.container import _cpu_pool
    pairs = [_P(N) for _ in range(n_pairs)]
    kw = dict(alpha=0.81, steps_fine=steps)
    _cpu_pool.align_pairs(mode, pairs, P, kw)                  # warmup (spawns the pool)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); _cpu_pool.align_pairs(mode, pairs, P, kw)
        best = min(best, time.perf_counter() - t0)
    return n_pairs / best


def main():
    ncpu = os.cpu_count() or 1
    grids = {"vol(N=30)": 30, "surf(N=75)": 75}
    thread_set = sorted({1, 2, 4, 6, 8, 11, ncpu})
    out = {"ncpu": ncpu, "grids": {}}
    for label, N in grids.items():
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        g = {}

        # A. kernel-only scaling
        K = 100 * 50                                           # 100 pairs x 50 seeds = 5000 poses
        base = kernel_only(K, N, 1)
        print(f"[A] kernel-only (K={K}):  1 thread = {base*1e3:7.2f} ms")
        ka = {1: 1.0}
        for t in thread_set:
            if t == 1: continue
            dt = kernel_only(K, N, t); ka[t] = base / dt
            print(f"    {t:2d} threads: {dt*1e3:7.2f} ms   scaling {base/dt:4.1f}x")
        g["kernel_scaling"] = ka

        # B. end-to-end driver scaling (numba=t, torch=1)
        npairs = 200
        bt1, thr1 = end_to_end(npairs, N, 1, 1)
        print(f"[B] end-to-end (n={npairs}): 1 thread = {thr1:7.1f} pairs/s")
        ee = {1: 1.0}
        for t in thread_set:
            if t == 1: continue
            _, thr = end_to_end(npairs, N, t, 1); ee[t] = thr / thr1
            print(f"    numba={t:2d},torch=1: {thr:7.1f} pairs/s   scaling {thr/thr1:4.1f}x")
        g["e2e_scaling_numba_only"] = ee

        # C. oversubscription: numba=N,torch=N vs numba=N,torch=1
        N8 = min(8, ncpu)
        _, thr_n1 = end_to_end(npairs, N, N8, 1)
        _, thr_nn = end_to_end(npairs, N, N8, N8)
        _, thr_1n = end_to_end(npairs, N, 1, N8)
        print(f"[C] @{N8} cores: numba={N8},torch=1 -> {thr_n1:7.1f} | "
              f"numba={N8},torch={N8} -> {thr_nn:7.1f} | numba=1,torch={N8} -> {thr_1n:7.1f} pairs/s")
        g["oversub"] = {"numba_only": thr_n1, "both": thr_nn, "torch_only": thr_1n}

        # D. process sharding (each proc single-threaded)
        _, thr_p1 = process_shard(1, npairs, N)
        print(f"[D] process-shard: 1 proc = {thr_p1:7.1f} pairs/s")
        ps = {1: 1.0}
        for P in [2, 4, 8, min(11, ncpu)]:
            _, thr = process_shard(P, npairs, N); ps[P] = thr / thr_p1
            print(f"    {P:2d} procs: {thr:7.1f} pairs/s   scaling {thr/thr_p1:4.1f}x")
        g["process_scaling"] = ps

        # E. the REAL persistent pool via _cpu_pool.align_pairs (mode from the grid label)
        pmode = label.split("(")[0]
        thr_pool1 = end_to_end_pool(pmode, npairs, N, 1)
        print(f"[E] _cpu_pool.align_pairs: 1 worker = {thr_pool1:7.1f} pairs/s "
              f"(vs thread-1core {thr1:7.1f})")
        # NOTE: compare pool-vs-threads ABSOLUTE numbers only in benchmark_cpu
        # (--numba-mode threads|pool), where both go through the same _align_batch_*;
        # this probe's [B]/[D]/[E] call paths differ, so only read SCALING here.
        pool = {1: 1.0}
        for P in [2, 4, 8, min(11, ncpu)]:
            thr = end_to_end_pool(pmode, npairs, N, P); pool[P] = thr / thr_pool1
            print(f"    {P:2d} workers: {thr:7.1f} pairs/s   scaling {thr/thr_pool1:4.1f}x")
        g["pool_scaling"] = pool
        out["grids"][label] = g

    print("\n" + json.dumps(out))


if __name__ == "__main__":
    main()
