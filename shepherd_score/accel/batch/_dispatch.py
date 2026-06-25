# shepherd_score/accel/batch/_dispatch.py
"""Multi-GPU dispatch, plus the per-mode tensor spec shared by the multi-GPU
workers and the CPU process pool (``cpu_pool.py``).

Multi-GPU strategy (and why the transparent path is conservative)
----------------------------------------------------------------
The alignment is **host-bound**, not kernel-bound, so driving N GPUs from one
process serialises the per-pair host work on the GIL and tops out at ~1.0-2.3x.
That old thread-sharding default has been removed. The path that actually scales
is **one OS process per GPU** -- each worker owns a shard, rebuilds its tensors on
its own GPU, and runs the unmodified batched aligner so the host work parallelises
too: measured **3.50-3.79x on 4xL40S (node3615)**, bit-exact vs single-GPU.

The catch for a *transparent* ``align_with_*`` call: once the parent has CUDA
tensors it cannot ``fork`` (CUDA + fork is unsafe), so the workers must ``spawn``,
and ``spawn`` re-imports the caller's ``__main__`` module -- which silently breaks
any entry script lacking an ``if __name__ == '__main__':`` guard. A library
spawning behind the user's back is therefore a footgun. So:

* **Default** (``_run_distributed``): run on a **single GPU** and emit a one-time
  warning pointing at the explicit pool. Never spawns, never hangs.
* **The blessed multi-GPU path** is the explicit persistent pool
  :class:`shepherd_score.accel.multi_gpu.MultiGPUAligner` (builds each GPU's shard
  once, reuses it -> full steady-state scaling; the user opts into multiprocessing
  deliberately, so the guard is their call).
* **Opt-in** ``FSS_MGPU_BACKEND=process`` routes the transparent path through the
  process-per-GPU backend below (bit-exact; for guarded scripts that want
  transparent sharding). It is **hardened**: a worker that dies during startup
  (e.g. the missing ``__main__`` guard) is detected and the call falls back to a
  single-GPU run instead of blocking forever.
"""
from __future__ import annotations
import os
import threading as _threading

import numpy as np
import torch


# --- multi-GPU dispatch ------------------------------------------------------

_DISPATCH_LOCAL = _threading.local()
_WARNED_SINGLE_GPU = False          # emit the "transparent multi-GPU is off" notice once


def _dev_idx(device: torch.device) -> int:
    """Cache-key component so per-device workspaces/buffers never collide under
    the multi-GPU dispatcher. Constant 0 on a single GPU -> no behaviour change.

    A bare ``torch.device("cuda")`` has ``index is None``; it must still resolve to
    a concrete GPU index (the current device), NOT to the CPU sentinel -1 -- else a
    CUDA batch built with an indexless device shares a cache key with a CPU
    (backend="numba") batch, and the second reuses the first's wrong-device
    workspace (RuntimeError: tensors on cuda:0 and cpu)."""
    if device.type == "cuda":
        return device.index if device.index is not None else torch.cuda.current_device()
    return -1


# Minimum pairs PER DEVICE before multi-GPU sharding is even considered. Below this a
# single GPU is faster (sharding adds fixed per-call overhead). Calibrated on 4x L40S.
_MIN_SHARD_PER_DEVICE = 4096


def _should_distribute(pairs) -> bool:
    """True when `pairs` is a multi-GPU-sized batch on CUDA (used to gate the
    transparent dispatch in :func:`_run_distributed`)."""
    if getattr(_DISPATCH_LOCAL, "active", False):
        return False                       # already inside a per-device shard
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    if not pairs or pairs[0].device.type != "cuda":
        return False
    return len(pairs) >= _MIN_SHARD_PER_DEVICE * torch.cuda.device_count()


# --- per-mode tensor spec (shared: multi-GPU workers + cpu_pool.py) -----------
# Each mode declares how to (a) pull its per-pair inputs off the Molecule objects as
# picklable numpy arrays, (b) rebuild the cached device tensors inside a worker, and
# (c) read the results back. ``extract`` and ``tensors`` are positional-aligned.
# Modes absent here have no process path (e.g. ``vol_and_shape_esp``) and run single-GPU.
_MODE_SPEC = {
    "vol": {
        "extract": [("ref_molec", "atom_pos"), ("fit_molec", "atom_pos")],
        "tensors": [("_ref_xyz_t", torch.float32), ("_fit_xyz_t", torch.float32)],
        "out": ("transform_vol_noH", "sim_aligned_vol_noH"),
    },
    "surf": {
        "extract": [("ref_molec", "surf_pos"), ("fit_molec", "surf_pos")],
        "tensors": [("_ref_surf_t", torch.float32), ("_fit_surf_t", torch.float32)],
        "out": ("transform_surf", "sim_aligned_surf"),
    },
    "surf_esp": {                                  # canonical name for the legacy "esp" mode
        "extract": [("ref_molec", "surf_pos"), ("fit_molec", "surf_pos"),
                    ("ref_molec", "surf_esp"), ("fit_molec", "surf_esp")],
        "tensors": [("_ref_surf_t", torch.float32), ("_fit_surf_t", torch.float32),
                    ("_ref_surf_esp_t", torch.float32), ("_fit_surf_esp_t", torch.float32)],
        "out": ("transform_surf_esp", "sim_aligned_surf_esp"),
    },
    "pharm": {
        "extract": [("ref_molec", "pharm_types"), ("fit_molec", "pharm_types"),
                    ("ref_molec", "pharm_ancs"), ("fit_molec", "pharm_ancs"),
                    ("ref_molec", "pharm_vecs"), ("fit_molec", "pharm_vecs")],
        "tensors": [("_ref_pharm_types_t", torch.int64), ("_fit_pharm_types_t", torch.int64),
                    ("_ref_pharm_ancs_t", torch.float32), ("_fit_pharm_ancs_t", torch.float32),
                    ("_ref_pharm_vecs_t", torch.float32), ("_fit_pharm_vecs_t", torch.float32)],
        "out": ("transform_pharm", "sim_aligned_pharm"),
    },
}


class _ProcStandIn:
    """Minimal MoleculePair stand-in used inside a process-per-GPU worker. Carries
    only the cached device tensors the batched aligner reads (no RDKit / Molecule),
    so nothing heavy crosses the process boundary -- the worker rebuilds tensors from
    the numpy arrays it was handed. The aligner reads its inputs via the pre-set
    ``_*_t`` attributes and writes ``transform_*``/``sim_aligned_*`` back here."""
    def __init__(self, device):
        self.device = device


# --- transparent dispatch entry ----------------------------------------------

def _run_single_gpu(align_fn, pairs, **kwargs):
    """Run `align_fn` on the pairs' current device, in-process. Sets the re-entry
    guard so a nested ``_should_distribute`` returns False (no recursion)."""
    _DISPATCH_LOCAL.active = True
    try:
        return align_fn(pairs, **kwargs)
    finally:
        _DISPATCH_LOCAL.active = False


def _run_distributed(align_fn, pairs, **kwargs):
    """Transparent multi-GPU entry, called by the ``_align_batch_*`` hooks when
    :func:`_should_distribute` is true.

    Default: run on a **single GPU** (plus a one-time warning) -- a transparent
    library call must not silently ``spawn`` worker processes (``spawn`` re-imports
    the caller's ``__main__`` and breaks unguarded scripts). For real multi-GPU
    throughput use :class:`shepherd_score.accel.multi_gpu.MultiGPUAligner`.

    Opt-in ``FSS_MGPU_BACKEND=process`` routes here through the hardened
    process-per-GPU backend (for guarded scripts that want transparent sharding);
    it falls back to single-GPU if the workers can't start."""
    global _WARNED_SINGLE_GPU
    mode = align_fn.__name__.replace("_align_batch_", "")
    backend = os.environ.get("FSS_MGPU_BACKEND", "off").lower()

    if backend == "process" and mode in _MODE_SPEC:
        try:
            return _run_distributed_procs(align_fn, pairs, **kwargs)
        except _MGPUSpawnUnavailable as exc:
            import warnings
            warnings.warn(
                f"FSS_MGPU_BACKEND=process could not start workers ({exc}); running on a "
                "single GPU. Guard the entry script with `if __name__ == '__main__':`, or use "
                "shepherd_score.accel.multi_gpu.MultiGPUAligner.",
                RuntimeWarning, stacklevel=2)
            return _run_single_gpu(align_fn, pairs, **kwargs)

    if not _WARNED_SINGLE_GPU:
        import warnings
        warnings.warn(
            f"{len(pairs)} pairs on a {torch.cuda.device_count()}-GPU host: transparent "
            "multi-GPU sharding is disabled (it was GIL-bound and only ~1-2x). Running on a "
            "single GPU. For ~3.5-3.8x multi-GPU throughput use "
            "shepherd_score.accel.multi_gpu.MultiGPUAligner (a persistent process-per-GPU "
            "pool), or set FSS_MGPU_BACKEND=process for transparent process sharding.",
            RuntimeWarning, stacklevel=2)
        _WARNED_SINGLE_GPU = True
    return _run_single_gpu(align_fn, pairs, **kwargs)


# --- process-per-GPU backend (opt-in via FSS_MGPU_BACKEND=process) ------------

class _MGPUSpawnUnavailable(RuntimeError):
    """A process-per-GPU worker could not be started or died during startup -- most
    often because the entry module lacks an ``if __name__ == '__main__':`` guard,
    which ``spawn`` requires. The caller falls back to a single-GPU run."""


def _mgpu_proc_worker(gpu_id, mode, shard_arrays, kwargs, out_q):
    """Worker entry point (top-level so it is spawn-picklable). Pins itself to one
    physical GPU, rebuilds stand-ins from numpy, runs the unmodified batched aligner,
    and returns ``(scores, transforms)`` as numpy via the queue."""
    try:
        import numpy as _np
        import torch as _torch
        # Aligner free functions live in the sibling module; import lazily so the
        # spawned child resolves them in its own interpreter (avoids an import cycle
        # here, since aligners imports from this module).
        from shepherd_score.accel.batch import aligners as _al

        _torch.cuda.set_device(gpu_id)
        _DISPATCH_LOCAL.active = True          # this worker owns ONE GPU -> never re-distribute
        dev = _torch.device("cuda", gpu_id)
        spec = _MODE_SPEC[mode]
        tnames = spec["tensors"]
        standins = []
        for row in shard_arrays:                           # row: one numpy per extract entry
            s = _ProcStandIn(dev)
            for (tname, dt), arr in zip(tnames, row):
                setattr(s, tname, _torch.as_tensor(arr, dtype=dt, device=dev))
            standins.append(s)

        getattr(_al, "_align_batch_" + mode)(standins, **kwargs)
        _torch.cuda.synchronize()

        tf_attr, sc_attr = spec["out"]
        scores = _np.array([float(getattr(s, sc_attr)) for s in standins], dtype=_np.float64)
        transforms = _np.stack([
            _torch.as_tensor(getattr(s, tf_attr)).detach().cpu().numpy().astype(_np.float32)
            for s in standins])
        out_q.put((scores, transforms))
    except Exception:                                      # noqa: BLE001 - relayed to parent
        import traceback
        out_q.put(("__ERR__", traceback.format_exc()))


def _run_distributed_procs(align_fn, pairs, **kwargs):
    """Process-per-GPU multi-GPU: shard pairs round-robin, hand each shard's numpy
    inputs to a worker pinned to one GPU, and write results back in-place. Bit-exact
    vs single-GPU (``benchmarks/experiments/mgpu_parity.py``: max|Δscore|<1e-6).

    Hardened: each shard is drained with a liveness check, so a worker that dies
    during startup (missing ``__main__`` guard under ``spawn``) raises
    :class:`_MGPUSpawnUnavailable` (caught by :func:`_run_distributed` -> single-GPU)
    instead of blocking forever on a dead worker's queue. Genuine in-kernel errors
    are relayed as ``__ERR__`` and re-raised."""
    import queue as _queue
    import torch.multiprocessing as _mp

    mode = align_fn.__name__.replace("_align_batch_", "")
    spec = _MODE_SPEC[mode]
    ndev = torch.cuda.device_count()

    # Pull picklable per-pair inputs (numpy) once, in original order.
    per_pair = [tuple(np.asarray(getattr(getattr(p, m), a)) for (m, a) in spec["extract"])
                for p in pairs]

    ctx = _mp.get_context("spawn")
    jobs = []
    try:
        for gpu_id in range(ndev):
            idxs = list(range(gpu_id, len(pairs), ndev))   # round-robin
            if not idxs:
                continue
            q = ctx.Queue()
            shard = [per_pair[i] for i in idxs]
            pr = ctx.Process(target=_mgpu_proc_worker,
                             args=(gpu_id, mode, shard, dict(kwargs), q))
            pr.start()
            jobs.append((pr, idxs, q))
    except RuntimeError as exc:                             # e.g. spawn before __main__ guard
        for pr, _, _ in jobs:
            pr.terminate()
        raise _MGPUSpawnUnavailable(str(exc)) from exc

    ok, compute_errs, startup_dead = [], [], False
    for pr, idxs, q in jobs:
        res = None
        while True:
            try:
                res = q.get(timeout=5.0)                    # drain BEFORE join (large items)
                break
            except _queue.Empty:
                if not pr.is_alive():                       # died without delivering
                    break
        if res is None:
            startup_dead = True
            break
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) \
                and res[0] == "__ERR__":
            compute_errs.append(res[1])
            continue
        ok.append((idxs, res[0], res[1]))

    for pr, _, _ in jobs:                                  # reap / clean up
        if pr.is_alive():
            pr.join(timeout=10)
        if pr.is_alive():
            pr.terminate()

    if startup_dead:
        raise _MGPUSpawnUnavailable("a multi-GPU worker exited during startup")
    if compute_errs:
        raise RuntimeError("multi-GPU (process) align failed:\n" + "\n".join(compute_errs))
    tf_attr, sc_attr = spec["out"]
    for idxs, scores, transforms in ok:
        for j, i in enumerate(idxs):
            setattr(pairs[i], tf_attr, torch.as_tensor(transforms[j], dtype=torch.float32))
            setattr(pairs[i], sc_attr, float(scores[j]))
