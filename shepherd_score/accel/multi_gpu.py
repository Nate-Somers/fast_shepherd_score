"""Top-level data-parallel multi-GPU driver for batch alignment.

WHY THIS EXISTS (see the multi-GPU scaling investigation): the alignment is
*host-bound*, not kernel-bound, so the in-library auto-shard paths can't reach
~Nx on N GPUs:
  - the `thread` backend drives all GPUs from one process -> GIL-serialized host
    work -> 1.0-2.6x;
  - a per-call process-scatter pays the bulk data handoff every call -> 0.6-1.2x.

The robust/simple/expected pattern that DOES scale (measured ~3.5-3.9x on 4xL40S
across vol/surf/esp/pharm) is plain data parallelism: ONE OS process per GPU,
each OWNING its shard (build + align end-to-end, data resident on its GPU), with
CPU threads capped to cores/ndev so the N workers don't oversubscribe the cores
(the un-capped default -- ~N x all-cores MKL/OMP threads -- was the hidden lever
that otherwise capped scaling to <1x).

Only lightweight `Molecule` objects cross the process boundary (once, at spawn);
no CUDA tensors are pickled. Each worker rebuilds `MoleculePair` on its own GPU
(passing a `Molecule` does NOT regenerate its surface, so this is cheap) and runs
the ordinary single-GPU `MoleculePairBatch.align_with_*` path.

This is a ONE-SHOT launcher: it spawns the workers, processes the whole batch in
one shot, and returns. The fixed spawn+context-init cost (~few seconds, paid once)
is negligible for a large screen but dominates a tiny batch -- so use it for big
workloads. For repeated query-vs-library screening, keep the workers warm with a
persistent pool instead (not implemented here).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Public per-mode result attributes written in-place by align_with_*. The multi-GPU process path
# supports exactly the registry's PROCESS_MODES (those with a _MODE_SPEC entry); derive both maps
# (and the validation set ``list(_SCORE_ATTR)``) from the registry so they can't drift from the
# canonical attribute names. Legacy aliases (esp/esp_combo) likewise come from the registry.
from ._modes import (MODE_ATTRS as _MODE_ATTRS, PROCESS_MODES as _PROCESS_MODES,
                     LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES)
_TRANSFORM_ATTR = {m: _MODE_ATTRS[m][0] for m in _PROCESS_MODES}
_SCORE_ATTR = {m: _MODE_ATTRS[m][1] for m in _PROCESS_MODES}


def _cap_threads(threads):
    """Cap a worker's CPU intra-op threads to ``threads`` at RUNTIME. With the
    ``fork`` start method the worker inherits the parent's already-sized MKL/OMP
    pools (env vars set after import don't resize them), so we cap via the runtime
    APIs: torch for ATen, threadpoolctl (if present) for MKL/OpenBLAS/OMP."""
    if not threads:
        return
    import torch
    torch.set_num_threads(int(threads))
    try:                                     # best-effort: resizes MKL/BLAS/OMP live
        import threadpoolctl
        threadpoolctl.threadpool_limits(int(threads))
    except Exception:
        pass


def _worker(rank, mode, backend, do_center, threads, align_kwargs, shard_mols, out_q):
    """One GPU's worker. Pins to ``cuda:rank`` and sets the dispatch-local
    ``active`` flag so the in-library auto-shard sees it's already inside a
    per-device shard and never re-distributes. Rebuilds MoleculePair on that GPU,
    aligns its shard, returns numpy results."""
    try:
        import time
        import torch
        from shepherd_score.container import MoleculePair, MoleculePairBatch
        # dispatch-local lives in _batch_align (refactored) or _core (committed).
        try:
            from shepherd_score.accel.batch import _DISPATCH_LOCAL
        except Exception:
            from shepherd_score.container._core import _DISPATCH_LOCAL

        _cap_threads(threads)
        torch.cuda.set_device(rank)
        _DISPATCH_LOCAL.active = True            # own ONE GPU; never re-distribute
        dev = torch.device("cuda", rank)
        t0 = time.perf_counter()
        pairs = [MoleculePair(ref, fit, do_center=do_center, device=dev)
                 for (ref, fit) in shard_mols]
        torch.cuda.synchronize()
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        getattr(MoleculePairBatch(pairs), "align_with_" + mode)(backend=backend, **align_kwargs)
        torch.cuda.synchronize()
        t_align = time.perf_counter() - t0

        sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
        scores = np.array([float(getattr(p, sc_attr)) for p in pairs], dtype=np.float64)
        transforms = np.stack([
            torch.as_tensor(getattr(p, tf_attr)).detach().cpu().numpy().astype(np.float64)
            for p in pairs]) if pairs else np.zeros((0, 4, 4))
        out_q.put((rank, scores, transforms, t_build, t_align))
    except Exception:                            # noqa: BLE001 - relayed to parent
        import traceback
        out_q.put((rank, "__ERR__", traceback.format_exc(), 0.0, 0.0))


def align_multi_gpu(pairs: Sequence,
                    mode: str,
                    *,
                    ndev: Optional[int] = None,
                    threads: Optional[int] = None,
                    backend: str = "triton",
                    do_center: bool = False,
                    write_back: bool = True,
                    return_timing: bool = False,
                    **align_kwargs):
    """Align ``pairs`` across ``ndev`` GPUs, one OS process per GPU.

    Parameters
    ----------
    pairs : list[MoleculePair]
        Pairs to align. Only their ``ref_molec`` / ``fit_molec`` (lightweight,
        picklable ``Molecule`` objects) cross the process boundary; each worker
        rebuilds the ``MoleculePair`` on its own GPU.
    mode : {"vol", "surf", "surf_esp", "pharm"}  (legacy "esp" accepted)
    ndev : int, optional
        Number of GPUs/processes (default: all visible CUDA devices).
    threads : int, optional
        CPU intra-op threads PER worker (default: cpu_cores // ndev). This cap is
        the lever that prevents the workers from oversubscribing the cores.
    backend : str
        Alignment backend forwarded to ``align_with_*`` (default "triton").
    do_center : bool
        Forwarded to ``MoleculePair`` construction (keep this matching how the
        single-GPU pairs were built, else results differ).
    write_back : bool
        If True, write ``sim_aligned_*`` / ``transform_*`` back onto the input
        ``pairs`` in order (matching the single-GPU API's in-place convention).
    **align_kwargs
        Forwarded verbatim to ``MoleculePairBatch.align_with_<mode>``.

    Returns
    -------
    (scores, transforms) : (np.ndarray (K,), np.ndarray (K, 4, 4))
        In input order. If ``return_timing`` also returns a dict of timings.
    """
    import torch
    import torch.multiprocessing as mp

    mode = _LEGACY_MODE_ALIASES.get(mode, mode)    # accept legacy esp / esp_combo
    if mode not in _SCORE_ATTR:
        raise ValueError(f"mode must be one of {list(_SCORE_ATTR)}, got {mode!r}")
    pairs = list(pairs)
    K = len(pairs)
    if K == 0:
        empty = (np.zeros((0,)), np.zeros((0, 4, 4)))
        return (*empty, {}) if return_timing else empty

    ndev = ndev or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    ndev = max(1, min(ndev, K))
    if threads is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or ndev
        threads = max(1, cores // ndev)

    # Lightweight, picklable specs -- NO CUDA tensors cross the boundary.
    specs = [(p.ref_molec, p.fit_molec) for p in pairs]
    # Contiguous balanced shards + their original indices (for in-order gather).
    bounds = np.linspace(0, K, ndev + 1).astype(int)
    shard_idx = [list(range(bounds[r], bounds[r + 1])) for r in range(ndev)]

    # Cap MKL/OMP for the CHILDREN: env is read at their (fresh) numpy/torch import,
    # so it must be set in the parent BEFORE spawning. Restore afterward.
    _saved = {k: os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in _saved:
        os.environ[k] = str(threads)
    try:
        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = []
        for r in range(ndev):
            shard = [specs[i] for i in shard_idx[r]]
            p = ctx.Process(target=_worker,
                            args=(r, mode, backend, do_center, threads,
                                  dict(align_kwargs), shard, out_q))
            p.start(); procs.append(p)

        results, errs = {}, []
        for _ in range(ndev):
            rank, a, b, t_build, t_align = out_q.get()   # drain before join
            if isinstance(a, str) and a == "__ERR__":
                errs.append((rank, b))
            else:
                results[rank] = (a, b, t_build, t_align)
        for p in procs:
            p.join()
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if errs:
        msg = "\n".join(f"[rank {r}]\n{tb}" for r, tb in errs)
        raise RuntimeError(f"multi-GPU align failed on {[r for r, _ in errs]}:\n{msg}")

    scores = np.empty(K, dtype=np.float64)
    transforms = np.empty((K, 4, 4), dtype=np.float64)
    for r in range(ndev):
        sc, tf, _, _ = results[r]
        for j, i in enumerate(shard_idx[r]):
            scores[i] = sc[j]
            transforms[i] = tf[j]

    if write_back:
        sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
        for i, p in enumerate(pairs):
            setattr(p, sc_attr, float(scores[i]))
            setattr(p, tf_attr, torch.as_tensor(transforms[i], dtype=torch.float32))

    if return_timing:
        timing = {"ndev": ndev, "threads": threads, "K": K,
                  "build_max": max(results[r][2] for r in results),
                  "align_max": max(results[r][3] for r in results),
                  "per_rank_align": {r: results[r][3] for r in results}}
        return scores, transforms, timing
    return scores, transforms


# ---------------------------------------------------------------------------
# Persistent pool: build+retain shards once, align resident data many times.
# ---------------------------------------------------------------------------
def _pool_worker(rank, threads, do_center, shard_mols, in_q, out_q):
    """Persistent worker: build+RETAIN this GPU's shard once, then align it
    in-place on every job. Only (mode, kwargs) come in and (scores, transforms)
    go out per call -- the bulk molecule data never recrosses the boundary, which
    is what preserves ~Nx scaling (vs re-shipping/rebuilding every call)."""
    try:
        import time
        import numpy as _np
        import torch
        from shepherd_score.container import MoleculePair, MoleculePairBatch
        try:
            from shepherd_score.accel.batch import _DISPATCH_LOCAL
        except Exception:
            from shepherd_score.container._core import _DISPATCH_LOCAL

        _cap_threads(threads)
        torch.cuda.set_device(rank)          # creates THIS worker's CUDA context
        _DISPATCH_LOCAL.active = True
        dev = torch.device("cuda", rank)
        pairs = [MoleculePair(ref, fit, do_center=do_center, device=dev)
                 for (ref, fit) in shard_mols]
        batch = MoleculePairBatch(pairs)
        torch.cuda.synchronize()
        out_q.put(("READY", rank))

        while True:
            job = in_q.get()
            if job is None:
                break
            mode, backend, kwargs = job
            t0 = time.perf_counter()
            getattr(batch, "align_with_" + mode)(backend=backend, **kwargs)
            torch.cuda.synchronize()
            t_align = time.perf_counter() - t0
            sc_attr, tf_attr = _SCORE_ATTR[mode], _TRANSFORM_ATTR[mode]
            scores = _np.array([float(getattr(p, sc_attr)) for p in pairs], dtype=_np.float64)
            transforms = _np.stack([
                torch.as_tensor(getattr(p, tf_attr)).detach().cpu().numpy().astype(_np.float64)
                for p in pairs]) if pairs else _np.zeros((0, 4, 4))
            out_q.put(("RES", rank, scores, transforms, t_align))
    except Exception:                            # noqa: BLE001
        import traceback
        out_q.put(("ERR", rank, traceback.format_exc()))


class MultiGPUAligner:
    """Persistent one-process-per-GPU pool that BUILDS and RETAINS its shard, so
    repeated :meth:`align` calls run on resident data at ~Nx (no per-call re-ship
    or rebuild). It is the right tool for repeated screening (several modes/params
    over the same pairs, or a resident library screened against many queries). For
    a single align of a huge batch, use the one-shot :func:`align_multi_gpu`.

    **Fast startup via ``fork``.** By default the pool forks its workers, so they
    inherit the parent's already-imported modules AND the molecule data via
    copy-on-write -- no per-worker re-import or pickling, making an N-GPU pool warm
    up in about the time of a single GPU (vs ~Nx with ``spawn``). ``fork`` is only
    CUDA-safe if the parent has NOT initialized CUDA yet, so for fast startup:
    **build ``pairs`` on CPU and create the pool BEFORE any GPU work.** If CUDA is
    already initialized the pool transparently falls back to ``spawn`` (slower).

    Usage::

        pairs = [MoleculePair(a, b, device="cpu") for a, b in mols]   # CPU build
        with MultiGPUAligner(pairs) as pool:                          # fork: fast
            scores, transforms = pool.align("vol", no_H=True, alpha=0.81)
            esp_scores, _      = pool.align("surf_esp", alpha=0.81, lam=0.3, num_repeats=16)
    """

    def __init__(self, pairs, *, ndev=None, threads=None, do_center=False,
                 start_method=None):
        import sys
        import warnings
        import torch
        import torch.multiprocessing as mp

        # Decide the start method up front. fork lets workers inherit the parent's
        # already-imported stack + data via COW -> an N-GPU pool warms up in about
        # the time of a single GPU (vs ~Nx with spawn, which re-imports per worker).
        # fork is only safe if the parent hasn't (a) initialized CUDA or (b) imported
        # Open3D -- both poison a subsequent fork+CUDA (Open3D's import does, even
        # though it never creates a CUDA context). Detect either and fall back to spawn.
        cuda_live = torch.cuda.is_initialized()
        o3d_live = "open3d" in sys.modules
        avail = mp.get_all_start_methods()
        unsafe = cuda_live or o3d_live
        if start_method is None:
            start_method = "fork" if ("fork" in avail and not unsafe) else "spawn"
            if unsafe and "fork" in avail:
                warnings.warn(
                    "MultiGPUAligner: "
                    + ("CUDA already initialized" if cuda_live else "Open3D already imported")
                    + " in the parent -> using slower 'spawn' startup. To get fast 'fork'"
                    " startup, create the pool before any GPU work / Open3D surface build"
                    " (e.g. from disk-cached molecules).")
        elif start_method == "fork" and unsafe:
            raise RuntimeError(
                "start_method='fork' needs the parent to have NOT initialized CUDA nor "
                "imported Open3D (both poison fork+CUDA). Create the pool earlier.")
        self.start_method = start_method

        pairs = list(pairs)
        self._K = len(pairs)
        if self._K == 0:
            raise ValueError("MultiGPUAligner needs at least one pair")
        ndev = ndev or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
        ndev = max(1, min(ndev, self._K))
        if threads is None:
            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:
                cores = os.cpu_count() or ndev
            threads = max(1, cores // ndev)
        self.ndev, self.threads = ndev, threads

        specs = [(p.ref_molec, p.fit_molec) for p in pairs]
        bounds = np.linspace(0, self._K, ndev + 1).astype(int)
        self._shard_idx = [list(range(bounds[r], bounds[r + 1])) for r in range(ndev)]

        # For spawn/forkserver, children re-import and read OMP/MKL env at import, so
        # cap it in the parent first. fork inherits live pools -> _cap_threads (runtime)
        # handles it in the worker instead, so the env dance is skipped.
        _saved = {}
        if start_method != "fork":
            _saved = {k: os.environ.get(k) for k in
                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
            for k in _saved:
                os.environ[k] = str(threads)
        try:
            ctx = mp.get_context(start_method)
            self._out_q = ctx.Queue()
            self._in_qs = [ctx.Queue() for _ in range(ndev)]
            self._procs = []
            for r in range(ndev):
                shard = [specs[i] for i in self._shard_idx[r]]
                p = ctx.Process(target=_pool_worker,
                                args=(r, threads, do_center, shard, self._in_qs[r], self._out_q))
                p.start(); self._procs.append(p)
            ready = 0
            while ready < ndev:
                msg = self._out_q.get()
                if msg[0] == "ERR":
                    self.close()
                    raise RuntimeError(f"worker {msg[1]} failed to start:\n{msg[2]}")
                ready += 1
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._closed = False

    def align(self, mode, *, backend="triton", return_timing=False, **align_kwargs):
        """Align the resident pairs with ``mode``; returns (scores, transforms) in
        the original input order. Cheap per call -- only params in, results out."""
        if self._closed:
            raise RuntimeError("MultiGPUAligner is closed")
        mode = _LEGACY_MODE_ALIASES.get(mode, mode)    # accept legacy esp / esp_combo
        if mode not in _SCORE_ATTR:
            raise ValueError(f"mode must be one of {list(_SCORE_ATTR)}, got {mode!r}")
        for r in range(self.ndev):
            self._in_qs[r].put((mode, backend, dict(align_kwargs)))
        results, errs = {}, []
        for _ in range(self.ndev):
            msg = self._out_q.get()
            if msg[0] == "ERR":
                errs.append((msg[1], msg[2]))
            else:
                _, rank, sc, tf, t_align = msg
                results[rank] = (sc, tf, t_align)
        if errs:
            raise RuntimeError("multi-GPU align failed:\n" +
                               "\n".join(f"[rank {r}]\n{tb}" for r, tb in errs))
        scores = np.empty(self._K, dtype=np.float64)
        transforms = np.empty((self._K, 4, 4), dtype=np.float64)
        for r in range(self.ndev):
            sc, tf, _ = results[r]
            for j, i in enumerate(self._shard_idx[r]):
                scores[i] = sc[j]
                transforms[i] = tf[j]
        if return_timing:
            return scores, transforms, {"align_max": max(results[r][2] for r in results)}
        return scores, transforms

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for q in self._in_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
