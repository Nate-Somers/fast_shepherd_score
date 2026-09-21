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

from .._modes import SPECS
from ..channels import CHANNELS


# --- multi-GPU dispatch ------------------------------------------------------

_DISPATCH_LOCAL = _threading.local()
_WARNED_SINGLE_GPU = False          # emit the "transparent multi-GPU is off" notice once


def _dev_idx(device: torch.device) -> int:
    """Cache-key component so per-device workspaces/buffers never collide under
    the multi-GPU dispatcher. Constant 0 on a single GPU -> no behaviour change.

    A bare ``torch.device("cuda")`` has ``index is None``; it must still resolve to
    a concrete GPU index (the current device), NOT to the CPU sentinel -1."""
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
# Each mode declares how to (a) pull its per-pair inputs off the pair as picklable numpy
# arrays -- ``extract`` is a list of ``(side, reader)`` where ``side`` is ``"ref_molec"``,
# ``"fit_molec"`` or ``"pair"`` and ``reader`` a callable over that object -- (b) rebuild the
# cached device tensors inside a worker (``tensors``, positional with ``extract``), and (c) read
# the results back (``out``). DERIVED from each mode's channels, so every registry mode has a
# worker path; ``accel/_modes.py:PROCESS_MODES`` is the same set.
_DTYPES = {"float32": torch.float32, "int64": torch.int64}


def _spec_entry(spec):
    extract, tensors = [], []
    for name in spec.all_channels():
        ch = CHANNELS[name]
        dt = _DTYPES[ch.dtype]
        if ch.is_pair:
            extract.append(("pair", ch.read))
            tensors.append((ch.ref_attr, dt))
        else:
            extract.append(("ref_molec", ch.read))
            tensors.append((ch.ref_attr, dt))
            extract.append(("fit_molec", ch.read))
            tensors.append((ch.fit_attr, dt))
    return {"extract": extract, "tensors": tensors, "out": spec.attrs}


_MODE_SPEC = {m: _spec_entry(s) for m, s in SPECS.items() if s.process}


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
    silently ``spawn`` worker processes. For real multi-GPU throughput use
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
