"""Persistent single-threaded process pool for the CPU (``numba``) batched aligner.

Why this exists
---------------
The ``backend="numba"`` path parallelises *one* batch across **threads** -- the
``@njit(parallel=True)`` ``prange`` overlap kernel. On many-core CPUs that caps at
~5-6x (measured: ``benchmarks/experiments/cpu_numba_scaling_probe.py``): ``prange``
static-schedules the poses over a per-step barrier, so one slow core straggles every
step, and the numba + torch thread pools oversubscribe each other.

This module is the alternative that scales closer to *N*: shard the **pairs** across
*N* persistent worker **processes**, each running the unchanged batched aligner
**single-threaded**. Alignment pairs are independent (each score is its own max over
SE(3) seeds), so a sharded result is **bit-identical** to one big call -- it only
removes the per-step cross-core barrier, so heterogeneous cores no longer drag each
other and the aggregate approaches one-core-throughput x N.

Design
------
* **Persistent** workers (one pool per process, reused across calls): the numba
  JIT-cache load + heavy imports are paid once, not per call. This is the missing
  piece of the GPU ``FSS_MGPU_BACKEND=process`` path, which spawned per call and was
  therefore too slow to be the default.
* Reuses :data:`shepherd_score.container._batch_align._MODE_SPEC` and
  :class:`~shepherd_score.container._batch_align._ProcStandIn` -- the same per-mode
  ``extract`` / ``tensors`` / ``out`` declarations the GPU process path uses -- so only
  small numpy arrays cross the boundary (no RDKit / ``Molecule`` objects, no tensors).
* Opt-in via ``MoleculePairBatch.align_with_*(backend="numba", num_workers=N)`` with
  ``N > 1`` (mirrors the JAX path's ``num_workers``). ``N == 1`` keeps the original
  single-process thread path untouched.

Caveat: uses the ``spawn`` start method, so the importing program must be
``if __name__ == "__main__"``-guarded (the standard multiprocessing requirement).
"""
from __future__ import annotations

import atexit
import os

import numpy as np

# Modes with a `_MODE_SPEC` entry (declared in _batch_align for the GPU process path).
# vol_esp / esp_combo have no entry -> the caller falls back to the single-process path.
POOL_MODES = ("vol", "surf", "esp", "pharm")


# ---------------------------------------------------------------------------
# Worker side (runs in each persistent child process)
# ---------------------------------------------------------------------------
def _run_shard(bm, torch, mode, rows, kwargs):
    """Run the unchanged batched aligner on one shard of stand-ins.

    ``rows`` is a list of tuples-of-numpy, one per pair, ordered to match
    ``_MODE_SPEC[mode]['tensors']``. Returns numpy ``(scores (k,), transforms (k,4,4))``.
    """
    spec = bm._MODE_SPEC[mode]
    tnames = spec["tensors"]
    standins = []
    for row in rows:
        s = bm._ProcStandIn(torch.device("cpu"))
        for (tname, dt), arr in zip(tnames, row):
            setattr(s, tname, torch.as_tensor(np.asarray(arr), dtype=dt))
        standins.append(s)
    if standins:
        getattr(bm, "_align_batch_" + mode)(standins, **kwargs)
    tf_attr, sc_attr = spec["out"]
    scores = np.array([float(getattr(s, sc_attr)) for s in standins], dtype=np.float64)
    if standins:
        transforms = np.stack([
            torch.as_tensor(getattr(s, tf_attr)).detach().cpu().numpy().astype(np.float32)
            for s in standins])
    else:
        transforms = np.zeros((0, 4, 4), dtype=np.float32)
    return scores, transforms


def _worker_loop(task_q, res_q):
    """Persistent worker: pin to one core, then serve shard tasks until a ``None``
    sentinel. Task = ``(mode, rows, kwargs)``; reply = ``(scores, transforms)`` or
    ``("__ERR__", traceback)``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # pure CPU; set before torch import
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)                          # this worker IS one core of the budget
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass
    from shepherd_score.container import _batch_align as bm
    bm._DISPATCH_LOCAL.active = True                  # never re-distribute inside a worker
    while True:
        task = task_q.get()
        if task is None:
            break
        mode, rows, kwargs = task
        try:
            res_q.put(_run_shard(bm, torch, mode, rows, kwargs))
        except Exception:                             # noqa: BLE001 - relayed to parent
            import traceback
            res_q.put(("__ERR__", traceback.format_exc()))


# ---------------------------------------------------------------------------
# Pool (parent side)
# ---------------------------------------------------------------------------
class CpuAlignPool:
    """A fixed-size pool of persistent single-threaded worker processes.

    One task queue + one result queue per worker, so shard ``w`` always maps to worker
    ``w`` (round-robin over pairs) -- simple and order-preserving.
    """

    def __init__(self, num_workers: int):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.num_workers = int(num_workers)
        self._task_qs = [ctx.Queue() for _ in range(self.num_workers)]
        self._res_qs = [ctx.Queue() for _ in range(self.num_workers)]
        self._procs = [
            ctx.Process(target=_worker_loop, args=(self._task_qs[i], self._res_qs[i]),
                        daemon=True, name=f"cpu-align-{i}")
            for i in range(self.num_workers)]
        for p in self._procs:
            p.start()
        self._closed = False

    def align(self, mode, per_pair, kwargs):
        """Shard ``per_pair`` (list of tuples-of-numpy, original order) round-robin
        across workers and return ``(scores, transforms)`` reassembled in original
        order. Workers with an empty shard simply return nothing."""
        K = len(per_pair)
        shards = [list(range(w, K, self.num_workers)) for w in range(self.num_workers)]
        for w, idxs in enumerate(shards):
            self._task_qs[w].put((mode, [per_pair[i] for i in idxs], dict(kwargs)))
        scores = [0.0] * K
        transforms = [None] * K
        errs = []
        for w, idxs in enumerate(shards):
            res = self._res_qs[w].get()                # drain before join
            if isinstance(res[0], str) and res[0] == "__ERR__":   # success res[0] is an array
                errs.append(res[1])
                continue
            sc, tf = res
            for j, i in enumerate(idxs):
                scores[i] = float(sc[j])
                transforms[i] = tf[j]
        if errs:
            raise RuntimeError("CPU align pool worker failed:\n" + "\n".join(errs))
        return scores, transforms

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for q in self._task_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()


_POOL: CpuAlignPool | None = None


def get_pool(num_workers: int) -> CpuAlignPool:
    """Return the module-level persistent pool, rebuilt only if the worker count
    changes (so back-to-back calls at the same width reuse warm workers)."""
    global _POOL
    if _POOL is None or _POOL._closed or _POOL.num_workers != num_workers:
        if _POOL is not None:
            _POOL.close()
        _POOL = CpuAlignPool(num_workers)
    return _POOL


@atexit.register
def _shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


# ---------------------------------------------------------------------------
# Public entry (parent side)
# ---------------------------------------------------------------------------
def align_pairs(mode, pairs, num_workers, align_kwargs):
    """Align ``pairs`` (``MoleculePair``) across the persistent CPU pool, writing
    ``sim_aligned_*`` / ``transform_*`` back in-place -- bit-identical to the
    single-process numba path. Also caches the per-pair input tensors (``_*_t``) on
    each pair, so the caller's ``return_aligned`` path finds them just as the
    single-process path would. Results are in-place; returns nothing.
    """
    import torch
    from shepherd_score.container import _batch_align as bm
    spec = bm._MODE_SPEC[mode]
    extract, tnames = spec["extract"], spec["tensors"]
    tf_attr, sc_attr = spec["out"]

    per_pair = [tuple(np.asarray(getattr(getattr(p, m), a)) for (m, a) in extract)
                for p in pairs]
    scores, transforms = get_pool(num_workers).align(mode, per_pair, align_kwargs)

    for i, p in enumerate(pairs):
        for (tname, dt), arr in zip(tnames, per_pair[i]):
            setattr(p, tname, torch.as_tensor(arr, dtype=dt))
        setattr(p, tf_attr, torch.as_tensor(transforms[i], dtype=torch.float32))
        setattr(p, sc_attr, float(scores[i]))
