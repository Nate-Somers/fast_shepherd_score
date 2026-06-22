# shepherd_score/accel/batch/_dispatch.py
"""Multi-GPU thread-sharding dispatch, plus the per-mode tensor spec used by
the CPU process pool (_cpu_pool.py)."""
from __future__ import annotations
import threading as _threading
import torch


# --- multi-GPU dispatch ------------------------------------------------------

_DISPATCH_LOCAL = _threading.local()


def _dev_idx(device: torch.device) -> int:
    """Cache-key component so per-device workspaces/buffers never collide under
    the multi-GPU dispatcher. Constant 0 on a single GPU -> no behaviour change."""
    return device.index if (device.type == "cuda" and device.index is not None) else -1


# Minimum pairs PER DEVICE before multi-GPU sharding pays off. Sharding adds fixed
# per-call overhead (thread spawn, per-device cuSOLVER warmup, result sync) and the
# per-pair host work is GIL-serialized across worker threads, so small/mid batches are
# faster on one GPU. Calibrated on 4x L40S (sub-linear; crossover a few k pairs/device).
_MIN_SHARD_PER_DEVICE = 4096


def _should_distribute(pairs) -> bool:
    """True when `pairs` should be sharded across multiple CUDA devices."""
    if getattr(_DISPATCH_LOCAL, "active", False):
        return False                       # already inside a per-device shard
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    if not pairs or pairs[0].device.type != "cuda":
        return False
    return len(pairs) >= _MIN_SHARD_PER_DEVICE * torch.cuda.device_count()



def _run_distributed(align_fn, pairs, **kwargs):
    """Shard `pairs` across ALL visible CUDA devices and run `align_fn` on each in
    parallel (CUDA ops release the GIL); results are written in-place to the pairs.
    The per-device workspace/footprint caches are device-keyed so concurrent
    shards never collide. NOTE: the single-GPU path (via _should_distribute) is
    validated; the multi-GPU concurrency path needs multi-GPU hardware to benchmark."""
    ndev = torch.cuda.device_count()
    # Warm thread-unsafe lazy CUDA init single-threaded BEFORE sharding: PyTorch cuSOLVER
    # handle init raises "lazy wrapper should be called at most once" if first-touched
    # concurrently from the worker threads below. Init eigh per device up front.
    for _d in range(ndev):
        with torch.cuda.device(_d):
            torch.linalg.eigh(torch.eye(3, device=torch.device("cuda", _d)))
    # CUDA-graph capture is not safe across the per-device worker threads (the RNG
    # generator-registration state races -> "graph should be registered to the state").
    # Use the eager fine loop for the multi-GPU path; per-pair results are unchanged.
    import shepherd_score.accel.drivers.shape as _fse3
    _fse3._FINE_GRAPHS = False
    shards = [sh for sh in (pairs[i::ndev] for i in range(ndev)) if sh]
    errs = {}

    def _worker(dev_idx, shard):
        _DISPATCH_LOCAL.active = True
        try:
            with torch.cuda.device(dev_idx):
                dev = torch.device("cuda", dev_idx)
                for p in shard:
                    p.device = dev          # align_fn moves this shard's tensors to `dev`
                align_fn(shard, **kwargs)
        except Exception as e:              # noqa: BLE001 - re-raised after join
            import traceback
            errs[dev_idx] = (repr(e), traceback.format_exc())
        finally:
            _DISPATCH_LOCAL.active = False

    threads = [_threading.Thread(target=_worker, args=(k, sh), daemon=True)
               for k, sh in enumerate(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise RuntimeError(f"multi-GPU align failed on device(s) {list(errs)}: {errs}")


# --- per-mode tensor spec used by the CPU process pool (_cpu_pool.py) ----
# Each mode declares how to (a) pull its per-pair
# inputs off the Molecule objects as picklable numpy arrays, (b) rebuild the cached
# device tensors inside a worker, and (c) read the results back. ``extract`` and
# ``tensors`` are positional-aligned. Modes absent here are not run via the CPU pool.
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
    "esp": {
        "extract": [("ref_molec", "surf_pos"), ("fit_molec", "surf_pos"),
                    ("ref_molec", "surf_esp"), ("fit_molec", "surf_esp")],
        "tensors": [("_ref_surf_t", torch.float32), ("_fit_surf_t", torch.float32),
                    ("_ref_surf_esp_t", torch.float32), ("_fit_surf_esp_t", torch.float32)],
        "out": ("transform_esp", "sim_aligned_esp"),
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
