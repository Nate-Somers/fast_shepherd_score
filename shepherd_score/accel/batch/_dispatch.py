# shepherd_score/accel/batch/_dispatch.py
"""Multi-GPU dispatch, plus the per-mode tensor spec used by the CPU process pool.

The transparent path runs on a single GPU. The alignment is host-bound, so driving several
GPUs from one process serialises on the GIL; the path that scales is one process per GPU. A
parent holding CUDA tensors cannot ``fork``, and ``spawn`` re-imports the caller's ``__main__``,
which breaks any entry script without an ``if __name__ == '__main__':`` guard, so a library
must not spawn behind the user's back. :func:`_run_distributed` therefore runs on one GPU with
a one-time warning; the supported multi-GPU path is the explicit persistent pool
:class:`shepherd_score.accel.multi_gpu.MultiGPUAligner`.
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
    """Cache-key component so per-device workspaces never collide under multi-GPU dispatch.

    A bare ``torch.device("cuda")`` has ``index is None`` and must resolve to the current
    device, not to the CPU sentinel -1."""
    if device.type == "cuda":
        return device.index if device.index is not None else torch.cuda.current_device()
    return -1


# Minimum pairs per device before multi-GPU sharding is considered; below this a single GPU
# is faster, since sharding adds a fixed per-call overhead.
_MIN_SHARD_PER_DEVICE = 4096


def _should_distribute(pairs) -> bool:
    """True when `pairs` is a multi-GPU-sized batch on CUDA; gates the transparent
    dispatch in :func:`_run_distributed`."""
    if getattr(_DISPATCH_LOCAL, "active", False):
        return False                       # already inside a per-device shard
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    if not pairs or pairs[0].device.type != "cuda":
        return False
    return len(pairs) >= _MIN_SHARD_PER_DEVICE * torch.cuda.device_count()


# --- per-mode tensor spec (consumed by the CPU process pool, cpu_pool.py) -----
# Each mode declares how to pull its per-pair inputs off the pair as picklable numpy arrays
# (``extract``: ``(side, reader)`` pairs, side in ref_molec / fit_molec / pair), how to rebuild
# the device tensors in a worker (``tensors``, positional with ``extract``) and which
# attributes to read back (``out``). Derived from each mode's channels; ``_modes.PROCESS_MODES``
# is the same set.
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
    """Minimal MoleculePair stand-in for a worker process: carries only the cached device
    tensors the batched aligner reads (no RDKit / Molecule), and receives the results."""
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
    :func:`_should_distribute` is true. Runs on a single GPU with a one-time warning; for real
    multi-GPU throughput use :class:`shepherd_score.accel.multi_gpu.MultiGPUAligner`."""
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
