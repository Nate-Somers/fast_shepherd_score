"""Persistent one-process-per-GPU POOL probe -- measures the REAL end-to-end
multi-GPU align throughput of the actual shippable design: spawn ndev workers
ONCE (each pinned to a GPU, CPU threads capped), and per call scatter each shard
as CONTIGUOUS arrays (cat + per-pair sizes, NOT K tiny pickles) and gather
contiguous results. This is the missing measurement: the IPC-free ceiling probe
(mgpu_ceiling.py) showed ~3.8x with threads capped; here we pay the per-call IPC
handoff to see how much of that survives.

Baseline = single-GPU IN-PROCESS align of the whole batch (no pool, no IPC).
Pool = ndev persistent workers with contiguous handoff. The ratio is the true
user-facing 1->ndev speedup INCLUDING the IPC tax.

Run on 4xL40S node3615 (set the per-worker thread cap via env, cores/ndev):
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 python -m benchmarks.experiments.mgpu_pool_probe \
        --mode vol --total 100000 --ndev 4 --threads 6
"""
import argparse
import time

import numpy as np
import torch
import torch.multiprocessing as mp

# Direct _align_batch_* kwargs (authoritative: benchmarks/experiments/mgpu_parity.py
# uses these exact dicts and asserts bit-parity vs the public align_with_* path).
KW = {"vol": dict(alpha=0.81, steps_fine=100),
      "surf": dict(alpha=0.81, steps_fine=100),
      "esp": dict(alpha=0.81, lam=0.3, num_repeats=50, topk=30, steps_fine=100, lr=0.075),
      "pharm": dict(num_repeats=50, topk=30, steps_fine=100, lr=0.1)}


def _extract_contiguous(pairs, spec):
    """One concatenated array + per-pair sizes per extract entry (cheap to pickle)."""
    cats, sizes = [], []
    for (molattr, field) in spec["extract"]:
        arrs = [np.asarray(getattr(getattr(p, molattr), field)) for p in pairs]
        sizes.append(np.array([a.shape[0] for a in arrs], dtype=np.int64))
        cats.append(np.concatenate(arrs, axis=0))
    return cats, sizes


def _rebuild_standins(cats, sizes, tnames, SI, dev):
    """Inverse of _extract_contiguous, inside a worker: split each cat back to K
    per-pair tensors on the worker's GPU."""
    K = len(sizes[0])
    splits = [np.split(cat, np.cumsum(sz)[:-1]) for cat, sz in zip(cats, sizes)]
    standins = []
    for j in range(K):
        s = SI(dev)
        for (tname, dt), sp in zip(tnames, splits):
            setattr(s, tname, torch.as_tensor(sp[j], dtype=dt, device=dev))
        standins.append(s)
    return standins


def _pool_worker(rank, mode, cap, in_q, out_q):
    """Persistent worker: warm once, then loop scatter->align->gather until poisoned."""
    try:
        import numpy as _np
        import torch as _t
        from shepherd_score.container._core import (
            MoleculePair as _MP, _MODE_SPEC as _SPEC, _ProcStandIn as _SI,
            _DISPATCH_LOCAL as _DL)
        from benchmarks.experiments.mgpu_pool_probe import _rebuild_standins as _rb

        if cap:
            _t.set_num_threads(int(cap))
        _t.cuda.set_device(rank)
        _DL.active = True
        dev = _t.device("cuda", rank)
        spec = _SPEC[mode]; tnames = spec["tensors"]; tf_attr, sc_attr = spec["out"]
        align = getattr(_MP, "_align_batch_" + mode)
        _t.linalg.eigh(_t.eye(3, device=dev))            # warm cuSOLVER
        out_q.put(("READY", rank))

        while True:
            job = in_q.get()
            if job is None:
                break
            cats, sizes, kw = job
            standins = _rb(cats, sizes, tnames, _SI, dev)
            align(standins, **kw)
            _t.cuda.synchronize()
            scores = _np.array([float(getattr(s, sc_attr)) for s in standins], dtype=_np.float64)
            transforms = _np.stack([
                _t.as_tensor(getattr(s, tf_attr)).detach().cpu().numpy().astype(_np.float32)
                for s in standins])
            out_q.put(("RES", rank, scores, transforms))
    except Exception:                                    # noqa: BLE001
        import traceback
        out_q.put(("ERR", rank, traceback.format_exc()))


def _build(mode, total, bucket, seed, dev):
    from shepherd_score.container import MoleculePair as MP
    from benchmarks.benchmark import make_real_cohort
    co = make_real_cohort(mode, n_pairs=total, bucket_kind=bucket, seed=seed)
    return [MP(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="vol", choices=["vol", "surf", "esp", "pharm"])
    ap.add_argument("--bucket", default="same", choices=["same", "cross"])
    ap.add_argument("--total", type=int, default=100000, help="TOTAL pairs across all GPUs")
    ap.add_argument("--ndev", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6, help="CPU thread cap per worker (cores/ndev)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import shepherd_score.container._core as cc
    from shepherd_score.container import MoleculePair as MP

    ndev = args.ndev or torch.cuda.device_count()
    T = args.total
    spec = cc._MODE_SPEC[args.mode]
    kw = KW[args.mode]
    print(f"=== mgpu_pool_probe | mode={args.mode} total={T} ndev={ndev} "
          f"threads/worker={args.threads} ===", flush=True)

    # ---- baseline: single-GPU, in-process, no pool, no IPC ----------------------
    dev0 = torch.device("cuda", 0)
    base_pairs = _build(args.mode, T, args.bucket, args.seed, dev0)
    align = getattr(MP, "_align_batch_" + args.mode)
    cc._DISPATCH_LOCAL.active = True                     # force single-GPU (no auto-shard)
    align(base_pairs, **kw); torch.cuda.synchronize()    # warmup
    best = float("inf"); n = 0; tot = 0.0
    while n < args.reps and tot < args.budget:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        align(base_pairs, **kw); torch.cuda.synchronize()
        dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
    base_tput = T / best
    cc._DISPATCH_LOCAL.active = False
    del base_pairs
    print(f"  baseline 1-GPU in-process: {base_tput:,.0f} pairs/s (best {best:.3f}s)", flush=True)

    # ---- pool: ndev persistent workers, contiguous handoff ----------------------
    pool_pairs = _build(args.mode, T, args.bucket, args.seed, torch.device("cpu"))
    shards = [pool_pairs[i::ndev] for i in range(ndev)]  # round-robin
    shard_payload = [_extract_contiguous(sh, spec) for sh in shards]

    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(ndev)]
    out_q = ctx.Queue()
    workers = [ctx.Process(target=_pool_worker, args=(r, args.mode, args.threads, in_qs[r], out_q))
               for r in range(ndev)]
    for w in workers:
        w.start()
    ready = 0
    while ready < ndev:
        msg = out_q.get()
        if msg[0] == "ERR":
            print(f"[rank {msg[1]}] WORKER ERROR:\n{msg[2]}", flush=True)
            raise SystemExit(1)
        ready += 1

    def _one_call():
        for r in range(ndev):
            cats, sizes = shard_payload[r]
            in_qs[r].put((cats, sizes, kw))
        got = 0
        while got < ndev:
            msg = out_q.get()
            if msg[0] == "ERR":
                raise RuntimeError(f"[rank {msg[1]}]\n{msg[2]}")
            got += 1

    _one_call()                                          # warmup (autotune in workers)
    best = float("inf"); n = 0; tot = 0.0
    while n < args.reps and tot < args.budget:
        t0 = time.perf_counter()
        _one_call()
        dt = time.perf_counter() - t0; best = min(best, dt); tot += dt; n += 1
    pool_tput = T / best
    for r in range(ndev):
        in_qs[r].put(None)
    for w in workers:
        w.join(timeout=30)

    print(f"  pool {ndev}-GPU end-to-end (incl IPC): {pool_tput:,.0f} pairs/s (best {best:.3f}s)", flush=True)
    print(f"\n  >>> RESULT mode={args.mode} total={T} ndev={ndev}: "
          f"baseline={base_tput:,.0f}  pool={pool_tput:,.0f}  "
          f"SPEEDUP={pool_tput/base_tput:.2f}x", flush=True)


if __name__ == "__main__":
    main()
