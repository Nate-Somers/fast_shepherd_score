# shepherd_score/accel/batch/_dispatch.py
"""Multi-GPU dispatch, plus the per-mode tensor spec used by the CPU process pool
(``cpu_pool.py``).

Why the transparent path is single-GPU
--------------------------------------
The alignment is **host-bound**, not kernel-bound, so driving N GPUs from one process
serialises the per-pair host work on the GIL. The path that scales is **one OS process
per GPU**. But once the parent holds CUDA tensors it cannot ``fork`` (CUDA + fork is
unsafe), so workers must ``spawn``, and ``spawn`` re-imports the caller's ``__main__``
module -- which silently breaks any entry script lacking an
``if __name__ == '__main__':`` guard. A library must therefore not spawn behind the
user's back. Consequently:

* :func:`_run_distributed` (the transparent path) runs on a **single GPU** and emits a
  one-time warning. It never spawns and never hangs.
* The supported multi-GPU path is the explicit persistent pool
  :class:`shepherd_score.accel.multi_gpu.MultiGPUAligner` (builds each GPU's shard once
  and reuses it), where the user opts into multiprocessing deliberately, so the
  ``__main__`` guard is their call.
"""
from __future__ import annotations
import threading as _threading

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
# single GPU is faster (sharding adds fixed per-call overhead).
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


# --- per-mode tensor spec (consumed by the CPU process pool, cpu_pool.py) -----
# Each mode declares how to (a) pull its per-pair inputs off the Molecule objects as
# picklable numpy arrays, (b) rebuild the cached device tensors inside a worker, and
# (c) read the results back. ``extract`` and ``tensors`` are positional-aligned.
# This dict is the authority for ``accel/_modes.py:PROCESS_MODES`` (a test asserts
# ``tuple(_MODE_SPEC) == PROCESS_MODES``); modes absent here have no worker path
# (e.g. ``vol_and_surf_esp``) and run in-process.
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
    """Minimal MoleculePair stand-in used inside a worker process. Carries
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

    Runs on a **single GPU** (plus a one-time warning): a transparent library call must not
    silently ``spawn`` worker processes (``spawn`` re-imports the caller's ``__main__`` and
    breaks unguarded scripts). For real multi-GPU throughput use
    :class:`shepherd_score.accel.multi_gpu.MultiGPUAligner`."""
    global _WARNED_SINGLE_GPU
    if not _WARNED_SINGLE_GPU:
        import warnings
        warnings.warn(
            f"{len(pairs)} pairs on a {torch.cuda.device_count()}-GPU host: transparent "
            "multi-GPU sharding is disabled; running on a single GPU. For multi-GPU "
            "throughput use shepherd_score.accel.multi_gpu.MultiGPUAligner (a persistent "
            "process-per-GPU pool).",
            RuntimeWarning, stacklevel=2)
        _WARNED_SINGLE_GPU = True
    return _run_single_gpu(align_fn, pairs, **kwargs)
