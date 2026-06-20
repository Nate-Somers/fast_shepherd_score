"""Process-isolation CEILING probe for multi-GPU align.

WHY: the default `thread` multi-GPU backend drives all GPUs from ONE Python
process, so per-pair host/launch work is GIL-serialized and the GPUs starve --
on 4xL40S @100k it gives vol ~1.0x and esp ~2.6x (committed l40s_4gpu vs
l40s_1gpu). The opt-in `process` backend is bit-exact but respawns 4 fresh CUDA
contexts EVERY call, so it is ~33x slower. Neither tells us the *achievable*
ceiling.

WHAT THIS MEASURES: the ceiling a persistent one-process-per-GPU pool could
reach. We spawn ndev worker processes (each `set_device(rank)` -> its own
interpreter/GIL/CUDA context), each builds its own K-pair cohort and aligns it.
A Barrier makes every worker enter the timed window simultaneously, so they
contend on CPU / PCIe / memory-bw exactly as a real pool would -- but NOT on a
shared GIL. Aggregate pairs/s = (ndev*K) / max(worker_best_time).

Compare against a single-GPU K-pair run (same K, ndev=1): if aggregate ~= ndev x
single-GPU, process isolation delivers near-linear scaling and the fix is a
persistent process pool (not anything in the thread path). The gap between this
ceiling and the committed thread numbers is the GIL tax.

Run on the 4xL40S node (cluster), e.g.:
    python -m benchmarks.experiments.mgpu_ceiling --mode vol  --per-gpu 25000
    python -m benchmarks.experiments.mgpu_ceiling --mode esp  --per-gpu 25000
    python -m benchmarks.experiments.mgpu_ceiling --mode vol  --per-gpu 25000 --ndev 1   # baseline
"""
import argparse
import time

import torch
import torch.multiprocessing as mp

# Match benchmarks.benchmark._cfg_from_args defaults (the committed l40s runs use
# the CLI defaults: num_repeats=16, steps=100, lr=0.1, alpha=0.81, lam=0.3).
_CFG = dict(num_repeats=16, steps=100, lr=0.1, alpha=0.81, lam=0.3, surf_per_atom=3)


def _worker(rank, mode, bucket, k, reps, budget, seed, barrier, out_q):
    """One GPU's worker: build K pairs, warm up, then time best-of-N inside a
    Barrier-synchronized window so all GPUs contend simultaneously."""
    try:
        import time as _t
        import torch as _torch
        from shepherd_score.container import MoleculePair as _MP
        import shepherd_score.container._batch_align as _cc
        from benchmarks.benchmark import make_real_cohort, _fork_align

        _torch.cuda.set_device(rank)
        dev = _torch.device("cuda", rank)
        # This worker OWNS one GPU. device_count() still reports all GPUs, so without
        # this guard _should_distribute() would (at per_gpu >= 16384) make EACH worker
        # re-shard across ALL GPUs -> 4 workers fighting over 4 GPUs. Pin to single-GPU.
        _cc._DISPATCH_LOCAL.active = True

        # Distinct seed per rank -> each GPU aligns a DIFFERENT shard (no shared
        # work, mirrors a real round-robin shard of one big batch).
        co = make_real_cohort(mode, n_pairs=k, bucket_kind=bucket, seed=seed + rank)
        pairs = [_MP(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]

        _fork_align(mode, pairs, _CFG)                 # warmup: autotune + context warm
        _torch.cuda.synchronize()

        if barrier is not None:
            barrier.wait()                             # all GPUs start together
        best = float("inf"); n = 0; total = 0.0
        while n < reps and total < budget:
            _torch.cuda.synchronize(); t0 = _t.perf_counter()
            _fork_align(mode, pairs, _CFG)
            _torch.cuda.synchronize(); dt = _t.perf_counter() - t0
            best = min(best, dt); total += dt; n += 1
        out_q.put((rank, k, best, n))
    except Exception:                                  # noqa: BLE001 - relayed to parent
        import traceback
        out_q.put((rank, "__ERR__", traceback.format_exc(), 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="vol", choices=["vol", "surf", "esp", "pharm"])
    ap.add_argument("--bucket", default="same", choices=["same", "cross"])
    ap.add_argument("--per-gpu", type=int, default=25000,
                    help="pairs aligned PER GPU (4 GPUs x 25000 = 100k total)")
    ap.add_argument("--ndev", type=int, default=None,
                    help="number of GPUs/workers (default: all visible). Use 1 for the baseline.")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ndev = args.ndev or torch.cuda.device_count()
    k = args.per_gpu
    print(f"=== mgpu_ceiling | mode={args.mode} bucket={args.bucket} "
          f"per_gpu={k} ndev={ndev} ===", flush=True)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(ndev) if ndev > 1 else None
    out_q = ctx.Queue()
    procs = []
    t0 = time.perf_counter()
    for r in range(ndev):
        p = ctx.Process(target=_worker,
                        args=(r, args.mode, args.bucket, k, args.reps,
                              args.budget, args.seed, barrier, out_q))
        p.start(); procs.append(p)
    results = [out_q.get() for _ in range(ndev)]
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    errs = [r for r in results if r[1] == "__ERR__"]
    if errs:
        for r in errs:
            print(f"\n[rank {r[0]}] WORKER ERROR:\n{r[2]}", flush=True)
        raise SystemExit(1)

    results.sort()
    worst = max(r[2] for r in results)
    agg = ndev * k / worst
    per_gpu_best = {r[0]: k / r[2] for r in results}
    print(f"\n  per-GPU best pairs/s (in timed window): "
          f"{ {r: round(v) for r, v in per_gpu_best.items()} }", flush=True)
    print(f"  slowest worker best-time = {worst:.4f}s  ->  AGGREGATE "
          f"{agg:,.0f} pairs/s across {ndev} GPU(s)", flush=True)
    print(f"  (wall incl. build+warmup = {wall:.1f}s)", flush=True)
    print(f"\n  >>> RESULT mode={args.mode} ndev={ndev} per_gpu={k}: "
          f"AGG={agg:,.0f} pairs/s  (compare vs the same command with --ndev 1 "
          f"for the per-GPU baseline; ideal scaling = ndev x baseline)", flush=True)


if __name__ == "__main__":
    main()
