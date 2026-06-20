"""
Batched, multi-GPU GPU/Triton aligners for :class:`MoleculePair`.

Extracted from ``_core.py`` (which kept growing) so that file stays readable.
Every function here is a *free function* that operates on duck-typed
``MoleculePair`` objects -- it only reads/writes their attributes. ``MoleculePair``
binds these as static methods, so the public seam is unchanged:
``MoleculePair._align_batch_vol(pairs, ...)`` still works exactly as before.

There is no runtime dependency on ``_core`` (only a TYPE_CHECKING import), so this
module imports cheaply and the spawn-based multi-GPU worker stays picklable.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import os
import threading as _threading

import numpy as np
import torch

from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE
from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

if TYPE_CHECKING:                     # annotations only; never imported at runtime
    from shepherd_score.container._core import MoleculePair


### BEGIN size_bucketing #####################################################
# Every heavy-atom count 3‒150 is mapped to a “band” of 8 atoms
# (   1-8, 9-16, 17-24, … ).  Pairs that fall in the same band
# share a common padded tensor size → one GPU launch.
_BAND = 16                     # change to 16/32 if you want larger bands

def _band_key(n: int) -> int:
    "return the *upper* bound of the 8-atom band this n falls into"
    return ((n + _BAND - 1) // _BAND) * _BAND
### END size_bucketing #######################################################

# ---- persistent, per-process caches (reused across calls) -------------------
_ALIGN_WORKSPACES: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
_INT_BUFFER_CACHE: dict[int, dict[str, torch.Tensor]] = {}

# Measured fine-loop footprint (bytes per pair) keyed by (mode, N_pad, M_pad,
# num_seeds). Lets the sub-batcher size each bucket's chunk to the GPU.
_PAIR_FOOTPRINT_BYTES: dict[tuple, int] = {}
# Set env SUBBATCH_DEBUG=1 to print the chosen chunk size per bucket.
_SUBBATCH_DEBUG = bool(os.environ.get("SUBBATCH_DEBUG"))

# Override the vol/surf seed count (default 50) for speed experiments. Fewer seeds
# = fewer poses = ~linear speedup, but coarser SO(3) coverage risks distinct-pair
# accuracy -- gated in benchmarks/experiments/speedlab.py. None -> 50.
_NUM_SEEDS = (lambda v: int(v) if v else None)(os.environ.get("FINE_NUM_SEEDS"))


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

# Multi-GPU execution backend (opt-in). "thread" (default) shards across one Python
# process with a worker THREAD per GPU -- simple, but the per-pair host work is
# GIL-serialized, so light/host-bound modes (vol/surf) barely scale past 1 GPU.
# "process" shards across one PROCESS per GPU (each its own interpreter/GIL), so the
# host work parallelises too -> near-linear scaling. Set via env FSS_MGPU_BACKEND.
_MGPU_BACKEND = os.environ.get("FSS_MGPU_BACKEND", "thread").lower()


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
    """Dispatch the multi-GPU align to the configured backend (``_MGPU_BACKEND``).

    Default ``"thread"`` keeps the original behaviour exactly. ``"process"`` routes to
    the process-per-GPU path for modes it supports (``_MODE_SPEC``), falling back to
    threads otherwise. Results are written in-place to the pairs either way."""
    if _MGPU_BACKEND == "process" and \
            align_fn.__name__.replace("_align_batch_", "") in _MODE_SPEC:
        return _run_distributed_procs(align_fn, pairs, **kwargs)
    return _run_distributed_threads(align_fn, pairs, **kwargs)


def _run_distributed_threads(align_fn, pairs, **kwargs):
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
    import shepherd_score.alignment.utils.fast_se3 as _fse3
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


# --- process-per-GPU multi-GPU path (opt-in via FSS_MGPU_BACKEND=process) ----
# Each mode that the process path supports declares how to (a) pull its per-pair
# inputs off the Molecule objects as picklable numpy arrays, (b) rebuild the cached
# device tensors inside a worker, and (c) read the results back. ``extract`` and
# ``tensors`` are positional-aligned. Modes absent here fall back to the thread path.
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


def _mgpu_proc_worker(gpu_id, mode, shard_arrays, kwargs, out_q):
    """Worker entry point (top-level so it is spawn-picklable). Pins itself to one
    physical GPU, rebuilds stand-ins from numpy, runs the unmodified batched aligner,
    and returns (scores, transforms) as numpy via the queue."""
    try:
        import numpy as _np
        import torch as _torch
        import shepherd_score.container._batch_align as _bm
        _SPEC = _bm._MODE_SPEC; _SI = _bm._ProcStandIn; _DL = _bm._DISPATCH_LOCAL

        _torch.cuda.set_device(gpu_id)
        _DL.active = True                  # this worker owns ONE GPU -> never re-distribute
        dev = _torch.device("cuda", gpu_id)
        spec = _SPEC[mode]
        tnames = spec["tensors"]
        standins = []
        for row in shard_arrays:                           # row: one numpy per extract entry
            s = _SI(dev)
            for (tname, dt), arr in zip(tnames, row):
                setattr(s, tname, _torch.as_tensor(arr, dtype=dt, device=dev))
            standins.append(s)

        getattr(_bm, "_align_batch_" + mode)(standins, **kwargs)
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
    inputs to a worker process pinned to one GPU, and write results back in-place.

    Each worker has its own interpreter/GIL, so the per-pair host work parallelises
    across GPUs (unlike the thread path). Only the small per-pair input arrays cross
    the process boundary -- no RDKit objects, no CUDA tensors."""
    import torch.multiprocessing as _mp

    mode = align_fn.__name__.replace("_align_batch_", "")
    spec = _MODE_SPEC[mode]
    ndev = torch.cuda.device_count()

    # Pull picklable per-pair inputs (numpy) once, in original order.
    per_pair = [tuple(np.asarray(getattr(getattr(p, m), a)) for (m, a) in spec["extract"])
                for p in pairs]

    ctx = _mp.get_context("spawn")
    jobs = []
    for gpu_id in range(ndev):
        idxs = list(range(gpu_id, len(pairs), ndev))       # round-robin (matches thread path)
        if not idxs:
            continue
        q = ctx.Queue()
        shard = [per_pair[i] for i in idxs]
        pr = ctx.Process(target=_mgpu_proc_worker,
                         args=(gpu_id, mode, shard, dict(kwargs), q))
        pr.start()
        jobs.append((pr, idxs, q))

    tf_attr, sc_attr = spec["out"]
    errs = []
    for pr, idxs, q in jobs:
        res = q.get()                                      # drain BEFORE join (large items)
        pr.join()
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) \
                and res[0] == "__ERR__":
            errs.append(res[1])
            continue
        scores, transforms = res
        for j, i in enumerate(idxs):
            setattr(pairs[i], tf_attr, torch.as_tensor(transforms[j], dtype=torch.float32))
            setattr(pairs[i], sc_attr, float(scores[j]))
    if errs:
        raise RuntimeError("multi-GPU (process) align failed:\n" + "\n".join(errs))


def _subbatched_align(process, K: int, *, key: tuple,
                      safety: float = 0.7, init_cap: int = 1024):
    """Drive ``process(start, count) -> (scores, q, t)`` over ``K`` independent
    pairs in GPU-memory-safe sub-batches and concatenate the per-pair results.

    Because pairs are independent (each result is its own max over seeds),
    chunking + concatenation is *exactly equivalent* to one big call -- it only
    bounds peak memory, so it never changes a score.

    Sizing is dynamic and per-bucket: bytes-per-pair is measured from the fine
    loop's peak allocation and cached per ``key=(mode, N_pad, M_pad, num_seeds)``
    (so a band-112 / pharm bucket -- whose footprint grows ~quadratically with
    pad size -- gets a much smaller chunk than a cheap band-32 surf bucket). Each
    chunk is sized so its peak stays under ``safety`` x (free device memory +
    torch's reusable cache). A previously-unseen shape starts at ``init_cap``
    pairs, then grows once calibrated (only chunks at least a quarter of the
    target size update the footprint, so a tiny trailing remainder cannot inflate
    it); an OOM halves the chunk and retries. Off CUDA (or if a single pair won't
    fit) it just calls ``process`` once.
    """
    if not torch.cuda.is_available():
        return process(0, K)

    key = (torch.cuda.current_device(),) + tuple(key)   # device-scope the footprint cache

    def _budget() -> float:
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return safety * (free + max(0, reusable))

    fp = _PAIR_FOOTPRINT_BYTES.get(key)
    need_resize = fp is None
    K_sub = max(1, min(K, int(_budget() // fp))) if fp else min(K, init_cap)
    if _SUBBATCH_DEBUG:
        print(f"[subbatch] key={key} K={K} init_fp={fp} K_sub0={K_sub} "
              f"free={torch.cuda.mem_get_info()[0]//(1024*1024)}MiB", flush=True)

    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    _nchunks = 0; _noom = 0; _ks = []                            # diag (SUBBATCH_DEBUG)
    while s < K:
        k = min(K_sub, K - s)
        try:
            torch.cuda.reset_peak_memory_stats()
            sc, q, t = process(s, k)
            peak = int(torch.cuda.max_memory_allocated())
            # Fold a chunk into the per-pair footprint only when it is large enough
            # that the fixed workspace overhead (seed/autotune scratch -- tens of MB,
            # independent of k) is amortised. peak/k = fixed/k + per_pair, so a tiny
            # trailing remainder (e.g. k=7) yields a wildly inflated bytes/pair that
            # max() would lock in, collapsing every later chunk to a fraction of its
            # right size (pharm was observed going 2 -> 16 -> 82 chunks this way). The
            # first chunk has k == K_sub so it always qualifies; calibration is never
            # starved.
            if k >= max(1, K_sub // 4):
                fp_meas = max(1, -(-peak // k))                  # ceil bytes/pair
                _PAIR_FOOTPRINT_BYTES[key] = max(_PAIR_FOOTPRINT_BYTES.get(key, 0), fp_meas)
            sc_parts.append(sc); q_parts.append(q); t_parts.append(t)
            s += k
            _nchunks += 1
            if _SUBBATCH_DEBUG:
                _ks.append(k)
            if need_resize:   # first success -> we now know the real footprint
                fp = _PAIR_FOOTPRINT_BYTES[key]
                remaining = K - s
                if remaining > 0:
                    K_sub = max(1, min(remaining, int(_budget() // fp)))
                need_resize = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Some OOMs surface as a plain RuntimeError; only treat those as OOM.
            if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                    and "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            _noom += 1
            if _SUBBATCH_DEBUG:
                print(f"[subbatch] OOM at k={k} (after {_nchunks} ok) "
                      f"free={torch.cuda.mem_get_info()[0]//(1024*1024)}MiB -> K_sub={max(1, k // 2)}",
                      flush=True)
            if k <= 1:
                raise
            K_sub = max(1, k // 2)
    if _SUBBATCH_DEBUG:
        print(f"[subbatch] DONE key={key} K={K} nchunks={_nchunks} noom={_noom} "
              f"ks={_ks} final_fp={_PAIR_FOOTPRINT_BYTES.get(key)}", flush=True)
    return torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts)


def _scatter_fill(out: torch.Tensor, tensors: list[torch.Tensor], sizes: list[int]) -> None:
    """Fill a pre-zeroed padded workspace ``out`` of shape ``(K, P_pad, *feat)`` so
    that ``out[i, :sizes[i]] = tensors[i]`` for each of the ``K`` per-pair tensors.

    Bit-identical to a per-pair ``out[i, :n] = t`` loop / ``pad_sequence`` fill, but
    it copies via ONE batched ``torch.cat`` + ONE vectorized scatter instead of ``K``
    launch-bound device copies. That fill is the dominant per-pair *host* cost at
    large batch -- on an RTX 4050 it drops a K=10000 (ref+fit) fill from ~100 ms to
    ~3 ms. ``out``'s padding rows are left untouched (the caller zeroes them), so the
    result is deterministic and exactly equal to the previous fill.
    """
    K, P_pad = out.shape[0], out.shape[1]
    device = out.device
    n = torch.as_tensor(sizes, device=device, dtype=torch.long)
    S = int(n.sum())
    if S == 0:
        return
    flat = torch.cat(tensors, dim=0)                       # (S, *feat)
    starts = torch.cumsum(n, 0) - n                        # (K,) first flat-row of each pair
    seg = torch.repeat_interleave(starts, n)               # (S,) segment start per flat row
    local = torch.arange(S, device=device) - seg           # (S,) within-pair row index
    dst = torch.repeat_interleave(torch.arange(K, device=device) * P_pad, n) + local
    out.view(K * P_pad, *out.shape[2:])[dst] = flat


def _align_batch_vol(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
    """
    Batched alignment with workspace reuse & reduced per-pair transfers.
    """

    global _ALIGN_WORKSPACES, _INT_BUFFER_CACHE
    
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol, pairs,
                                alpha=alpha, steps_fine=steps_fine)

    from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.alignment.utils.fast_common import batched_seeds_torch

    device = pairs[0].device
    # --- move coords once (skip if already there & right dtype) -------------
    for p in pairs:
        rx = p._ref_xyz_t
        fx = p._fit_xyz_t
        if rx.device != device:
            p._ref_xyz_t = rx.to(device, non_blocking=True)
        if fx.device != device:
            p._fit_xyz_t = fx.to(device, non_blocking=True)

    # --- result accumulators (GPU first; host copy only once) ---------------
    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # --- band bucketing -----------------------------------------------------
    buckets: dict[tuple[int, int], list[MoleculePair]] = {}
    for p in pairs:
        n_band = _band_key(p._ref_xyz_t.shape[0])
        m_band = _band_key(p._fit_xyz_t.shape[0])
        buckets.setdefault((n_band, m_band), []).append(p)

    for (N_pad, M_pad), bucket in buckets.items():
        K = len(bucket)

        # ---- integer buffers (reuse) ---------------------------------------
        ib_key = (_dev_idx(device), K)
        int_buf = _INT_BUFFER_CACHE.get(ib_key)
        if int_buf is None:
            int_buf = {
                'N': torch.empty(K, dtype=torch.int32, device=device),
                'M': torch.empty(K, dtype=torch.int32, device=device),
            }
            _INT_BUFFER_CACHE[ib_key] = int_buf
        N_real = int_buf['N']
        M_real = int_buf['M']

        # Fill once from CPU lists (one H2D each) -- was per-element GPU writes
        ref_ts = [p._ref_xyz_t for p in bucket]
        fit_ts = [p._fit_xyz_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ts]
        m_list = [t.shape[0] for t in fit_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        # ---- workspaces (reuse & grow) -------------------------------------
        ws_key = (_dev_idx(device), N_pad, M_pad)
        ws = _ALIGN_WORKSPACES.get(ws_key)
        if ws is None or ws['ref'].shape[0] < K:
            # allocate at least K; allow some headroom (optional)
            ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
            _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
        ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
        fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

        # We only write the valid prefix; no need to .zero_ entire array
        # but we do clear the padding slices for deterministic results.
        ref_pad.zero_()
        fit_pad.zero_()
        # Batched scatter pad-fill (was K per-pair GPU copies via pad_sequence).
        # cat+scatter into the zero-init prefix -> bit-identical, launch-count O(1)
        # in K instead of O(K) (R3: the pad_sequence fill was the top host cost).
        _scatter_fill(ref_pad, ref_ts, n_list)
        _scatter_fill(fit_pad, fit_ts, m_list)

        # ---- self-overlaps (reused kernel) ---------------------------------
        VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
        # memory-pressured chunking never re-pays the launch-bound seed-gen.
        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=(_NUM_SEEDS or 50))

        # ---- coarse + fine alignment, in GPU-memory-safe sub-batches -------
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(seeds_q[sl], seeds_t[sl]))
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("vol", N_pad, M_pad, 50))

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    # ---- final host transfer (single) --------------------------------------
    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched (was per-pair)
    scores_list = scores_cpu.tolist()                       # one C call (was K float())
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_noH = S
        p.sim_aligned_vol_noH = s

def _align_batch_surf(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
    """
    Batched alignment over *surface point clouds* using Gaussian-overlap
    surface similarity (ROCS-style), modeled after `_align_batch_vol`.

    Inputs
    ------
    pairs : list[MoleculePair]
        Each pair must provide surface point clouds for reference/fit:
        • prefer:   _ref_surf_t, _fit_surf_t  (torch.float32, (N/M, 3))
        • fallback: ref_molec.surf_pos, fit_molec.surf_pos (numpy, (N/M, 3))
    alpha : float
        Gaussian width parameter (same meaning as in `align_with_surf`).

    Side effects
    ------------
    Writes:
    • p.transform_surf      ← best SE(3) as 4×4 (via quaternion_to_SE3)
    • p.sim_aligned_surf    ← best Tanimoto surface score (float)
    """

    # Reuse the persistent, per-process workspace/int-buffer caches (same
    # ref/fit scratch-buffer layout as _align_batch_vol). The previous local
    # re-declarations here shadowed the module globals, so the surf path
    # never reused workspaces across calls. Buffers are zeroed before use,
    # so cross-call (and cross-modality) reuse is safe.
    global _ALIGN_WORKSPACES, _INT_BUFFER_CACHE

    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_surf, pairs,
                                alpha=alpha, steps_fine=steps_fine)

    from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.alignment.utils.fast_common import batched_seeds_torch

    device = pairs[0].device

    # --- ensure/prepare surface tensors on the right device --------------------
    for p in pairs:
        # Prefer already-prepared torch tensors
        r = getattr(p, "_ref_surf_t", None)
        f = getattr(p, "_fit_surf_t", None)

        if r is None or f is None:
            # Fallback: build from numpy surface arrays if present
            if not hasattr(p, "ref_molec") or not hasattr(p.ref_molec, "surf_pos"):
                raise ValueError(
                    "Surface points missing: MoleculePair must have _ref/_fit_surf_t "
                    "or ref_molec/fit_molec with .surf_pos."
                )
            r_np = p.ref_molec.surf_pos
            f_np = p.fit_molec.surf_pos
            if r_np is None or f_np is None:
                raise ValueError("Surface points are None; cannot run _align_batch_surf.")
            p._ref_surf_t = torch.as_tensor(r_np, dtype=torch.float32, device=device)
            p._fit_surf_t = torch.as_tensor(f_np, dtype=torch.float32, device=device)
        else:
            # move to target device if needed
            if r.device != device:
                p._ref_surf_t = r.to(device, non_blocking=True)
            if f.device != device:
                p._fit_surf_t = f.to(device, non_blocking=True)

    # --- result accumulators (GPU first; host copy only once) ------------------
    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # --- band bucketing (by padded N/M) ---------------------------------------
    buckets: dict[tuple[int, int], list[MoleculePair]] = {}
    for p in pairs:
        n_band = _band_key(p._ref_surf_t.shape[0])
        m_band = _band_key(p._fit_surf_t.shape[0])
        buckets.setdefault((n_band, m_band), []).append(p)

    for (N_pad, M_pad), bucket in buckets.items():
        K = len(bucket)

        # ---- integer buffers (reuse) ------------------------------------------
        ib_key = (_dev_idx(device), K)
        int_buf = _INT_BUFFER_CACHE.get(ib_key)
        if int_buf is None:
            int_buf = {
                'N': torch.empty(K, dtype=torch.int32, device=device),
                'M': torch.empty(K, dtype=torch.int32, device=device),
            }
            _INT_BUFFER_CACHE[ib_key] = int_buf
        N_real = int_buf['N']
        M_real = int_buf['M']

        # Fill once from CPU lists (one H2D each) instead of per-element GPU
        # scalar writes (which were ~8k tiny dispatches/sync per align).
        ref_ts = [p._ref_surf_t for p in bucket]
        fit_ts = [p._fit_surf_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ts]
        m_list = [t.shape[0] for t in fit_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        # ---- workspaces (reuse & grow) ----------------------------------------
        ws_key = (_dev_idx(device), N_pad, M_pad)
        ws = _ALIGN_WORKSPACES.get(ws_key)
        if ws is None or ws['ref'].shape[0] < K:
            ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
            _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
        ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
        fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

        # Clear padding slices for determinism; write valid prefixes
        ref_pad.zero_()
        fit_pad.zero_()
        # Batched scatter pad-fill (was K per-pair GPU copies via pad_sequence).
        # cat+scatter into the zero-init prefix -> bit-identical, launch-count O(1)
        # in K instead of O(K) (R3: the pad_sequence fill was the top host cost).
        _scatter_fill(ref_pad, ref_ts, n_list)
        _scatter_fill(fit_pad, fit_ts, m_list)

        # ---- self-overlaps on surface point clouds ----------------------------
        VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
        # memory-pressured chunking never re-pays the launch-bound seed-gen.
        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=(_NUM_SEEDS or 50))

        # ---- coarse + fine alignment (same engine as volumetric), processed in
        # GPU-memory-safe sub-batches sized per bucket (pairs are independent)
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(seeds_q[sl], seeds_t[sl]))
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("surf", N_pad, M_pad, 50))

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    # ---- final host transfer (single) -----------------------------------------
    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched (was per-pair)
    scores_list = scores_cpu.tolist()                       # one C call (was K float())
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_surf = S
        p.sim_aligned_surf = s

def _align_batch_esp(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam: float,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched ESP alignment using the fused ESP Triton kernel.

    Side effects
    ------------
    Writes:
    - p.transform_esp
    - p.sim_aligned_esp
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_esp, pairs,
                                alpha=alpha, lam=lam, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    from shepherd_score.score.constants import LAM_SCALING

    device = pairs[0].device
    lam_scaled = LAM_SCALING * lam

    # Ensure surface tensors (+ ESP) exist on device
    for p in pairs:
        r = getattr(p, "_ref_surf_t", None)
        f = getattr(p, "_fit_surf_t", None)
        rc = getattr(p, "_ref_surf_esp_t", None)
        fc = getattr(p, "_fit_surf_esp_t", None)

        if r is None or f is None or rc is None or fc is None:
            if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
                raise ValueError("Surface points are None; cannot run _align_batch_esp.")
            if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
                raise ValueError("Surface ESP is None; cannot run _align_batch_esp.")

            p._ref_surf_t = torch.as_tensor(p.ref_molec.surf_pos, dtype=torch.float32, device=device)
            p._fit_surf_t = torch.as_tensor(p.fit_molec.surf_pos, dtype=torch.float32, device=device)
            p._ref_surf_esp_t = torch.as_tensor(p.ref_molec.surf_esp, dtype=torch.float32, device=device)
            p._fit_surf_esp_t = torch.as_tensor(p.fit_molec.surf_esp, dtype=torch.float32, device=device)
        else:
            if r.device != device:
                p._ref_surf_t = r.to(device, non_blocking=True)
            if f.device != device:
                p._fit_surf_t = f.to(device, non_blocking=True)
            if rc.device != device:
                p._ref_surf_esp_t = rc.to(device, non_blocking=True)
            if fc.device != device:
                p._fit_surf_esp_t = fc.to(device, non_blocking=True)

        # Translation centers must be available on device for trans_init.
        if trans_init and getattr(p, "_ref_xyz_t", None) is None:
            p._ref_xyz_t = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=device)

    _esp_bucketed_align(
        pairs, alpha=alpha, lam_scaled=lam_scaled,
        ref_pts_attr="_ref_surf_t", fit_pts_attr="_fit_surf_t",
        ref_chg_attr="_ref_surf_esp_t", fit_chg_attr="_fit_surf_esp_t",
        out_tf_attr="transform_esp", out_sc_attr="sim_aligned_esp",
        subbatch_tag="esp", trans_init=trans_init,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk,
        steps_fine=steps_fine, lr=lr,
    )

def _esp_bucketed_align(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam_scaled: float,
    ref_pts_attr: str,
    fit_pts_attr: str,
    ref_chg_attr: str,
    fit_chg_attr: str,
    out_tf_attr: str,
    out_sc_attr: str,
    subbatch_tag: str,
    trans_init: bool,
    num_repeats_per_trans: int,
    topk: int,
    steps_fine: int,
    lr: float,
) -> None:
    """
    Shared bucket -> pad -> fused-ESP-kernel -> SE(3) writeback core for the
    ESP-weighted batch aligners. ``_align_batch_esp`` feeds surface points +
    surface ESP; ``_align_batch_vol_esp`` feeds (heavy-)atom coords + partial
    charges. The caller resolves ``lam_scaled`` (esp applies ``LAM_SCALING``,
    vol_esp uses raw lam) and supplies the cached-tensor attribute names + the
    output attrs. Translation centers are always the ref atom coords
    (``_ref_xyz_t``), matching every per-pair ESP optimizer.
    """
    from shepherd_score.alignment.utils.fast_esp_se3 import fast_optimize_ROCS_esp_overlay_batch

    device = pairs[0].device

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Bucket by padded point-cloud sizes; for translation-seeded mode, also bucket
    # by exact number of translation centers (legacy uses 10*P + 5 seeds).
    buckets: dict[tuple[int, int, int], list[MoleculePair]] = {}
    for p in pairs:
        n_band = _band_key(getattr(p, ref_pts_attr).shape[0])
        m_band = _band_key(getattr(p, fit_pts_attr).shape[0])
        tc = int(p._ref_xyz_t.shape[0]) if trans_init else 0
        buckets.setdefault((n_band, m_band, tc), []).append(p)

    # Workspace caches keyed by (N_pad, M_pad, K)
    workspaces: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
    int_buffers: dict[int, dict[str, torch.Tensor]] = {}

    for (N_pad, M_pad, tc), bucket in buckets.items():
        K = len(bucket)

        ib = int_buffers.get(K)
        if ib is None:
            ib = {
                "N": torch.empty(K, dtype=torch.int32, device=device),
                "M": torch.empty(K, dtype=torch.int32, device=device),
            }
            int_buffers[K] = ib
        N_real = ib["N"]
        M_real = ib["M"]

        ref_pts_ts = [getattr(p, ref_pts_attr) for p in bucket]
        fit_pts_ts = [getattr(p, fit_pts_attr) for p in bucket]
        ref_chg_ts = [getattr(p, ref_chg_attr) for p in bucket]
        fit_chg_ts = [getattr(p, fit_chg_attr) for p in bucket]
        n_list = [t.shape[0] for t in ref_pts_ts]
        m_list = [t.shape[0] for t in fit_pts_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        ws_key = (N_pad, M_pad, K)
        ws = workspaces.get(ws_key)
        if ws is None:
            ws = {
                "ref": torch.empty(K, N_pad, 3, device=device, dtype=torch.float32),
                "fit": torch.empty(K, M_pad, 3, device=device, dtype=torch.float32),
                "ref_c": torch.empty(K, N_pad, device=device, dtype=torch.float32),
                "fit_c": torch.empty(K, M_pad, device=device, dtype=torch.float32),
            }
            workspaces[ws_key] = ws

        ref_pad = ws["ref"]
        fit_pad = ws["fit"]
        ref_c_pad = ws["ref_c"]
        fit_c_pad = ws["fit_c"]

        ref_pad.zero_()
        fit_pad.zero_()
        ref_c_pad.zero_()
        fit_c_pad.zero_()
        # Batched scatter pad-fill (was a per-pair loop of 4*K device slice-copies).
        # cat+scatter into the zero-init prefix -> bit-identical, O(1) launches in K.
        _scatter_fill(ref_pad, ref_pts_ts, n_list)
        _scatter_fill(fit_pad, fit_pts_ts, m_list)
        _scatter_fill(ref_c_pad, ref_chg_ts, n_list)
        _scatter_fill(fit_c_pad, fit_chg_ts, m_list)

        trans_centers_batch = None
        trans_centers_real = None
        if trans_init:
            # NOTE: this bucket key uses exact translation center count (tc), so
            # the legacy seed count is identical for all pairs in this bucket.
            trans_centers_batch = torch.empty(K, tc, 3, device=device, dtype=torch.float32)
            for i, p in enumerate(bucket):
                trans_centers_batch[i] = p._ref_xyz_t
            trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

        # NOTE: seed-gen is intentionally NOT hoisted out of the sub-batch loop for
        # the ESP-family kernels. Unlike surf/vol (where hoisting cleanly helps under
        # memory pressure), the heavier per-chunk ESP footprint meant held full-band
        # seeds shaved enough free memory to tip the sub-batcher into OOM-retry thrash
        # (esp-same large-batch went 1912 -> 273 mol/s). Clean-process ESP is already
        # fast; the per-cell subprocess benchmark removes the pressure entirely.
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
            tcr = trans_centers_real[sl] if trans_centers_real is not None else None
            _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                ref_pad[sl], fit_pad[sl], ref_c_pad[sl], fit_c_pad[sl],
                alpha=alpha, lam=lam_scaled,
                N_real=N_real[sl], M_real=M_real[sl],
                trans_centers_batch=tcb, trans_centers_real=tcr,
                num_repeats_per_trans=num_repeats_per_trans,
                topk=topk, steps_fine=steps_fine, lr=lr,
            )
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=(subbatch_tag, N_pad, M_pad, 50))

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched (was per-pair)
    scores_list = scores_cpu.tolist()                       # one C call (was K float())
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        setattr(p, out_tf_attr, S)
        setattr(p, out_sc_attr, s)

def _align_batch_vol_esp(
    pairs: list["MoleculePair"],
    *,
    lam: float,
    alpha: float = 0.81,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched volumetric-ESP alignment: heavy-atom Gaussian overlap weighted by
    partial charge. Reuses the fused ESP Triton kernel via ``_esp_bucketed_align``,
    fed atom coords + heavy-atom partial charges instead of surface points + ESP.
    Heavy-atom only (mirrors ``_align_batch_vol``); ``lam`` is RAW (no
    ``LAM_SCALING``) to match the per-pair ``align_with_vol_esp``.

    Side effects
    ------------
    Writes:
    - p.transform_vol_esp_noH
    - p.sim_aligned_vol_esp_noH
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_esp, pairs,
                                lam=lam, alpha=alpha, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    device = pairs[0].device

    # Ensure heavy-atom coords (+ heavy-atom partial charges) exist on device.
    for p in pairs:
        if p.ref_molec.partial_charges is None or p.fit_molec.partial_charges is None:
            raise ValueError("Partial charges are None; cannot run _align_batch_vol_esp.")

        rx = getattr(p, "_ref_xyz_t", None)
        fx = getattr(p, "_fit_xyz_t", None)
        if rx is None:
            p._ref_xyz_t = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=device)
        elif rx.device != device:
            p._ref_xyz_t = rx.to(device, non_blocking=True)
        if fx is None:
            p._fit_xyz_t = torch.as_tensor(p.fit_molec.atom_pos, dtype=torch.float32, device=device)
        elif fx.device != device:
            p._fit_xyz_t = fx.to(device, non_blocking=True)

        rxe = getattr(p, "_ref_xyz_esp_t", None)
        if rxe is None:
            p._ref_xyz_esp_t = torch.as_tensor(
                p.ref_molec.partial_charges[p.ref_molec._nonH_atoms_idx],
                dtype=torch.float32, device=device)
        elif rxe.device != device:
            p._ref_xyz_esp_t = rxe.to(device, non_blocking=True)
        fxe = getattr(p, "_fit_xyz_esp_t", None)
        if fxe is None:
            p._fit_xyz_esp_t = torch.as_tensor(
                p.fit_molec.partial_charges[p.fit_molec._nonH_atoms_idx],
                dtype=torch.float32, device=device)
        elif fxe.device != device:
            p._fit_xyz_esp_t = fxe.to(device, non_blocking=True)

    _esp_bucketed_align(
        pairs, alpha=alpha, lam_scaled=lam,            # RAW lam (matches per-pair vol_esp)
        ref_pts_attr="_ref_xyz_t", fit_pts_attr="_fit_xyz_t",
        ref_chg_attr="_ref_xyz_esp_t", fit_chg_attr="_fit_xyz_esp_t",
        out_tf_attr="transform_vol_esp_noH", out_sc_attr="sim_aligned_vol_esp_noH",
        subbatch_tag="vol_esp", trans_init=trans_init,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk,
        steps_fine=steps_fine, lr=lr,
    )

def _align_batch_esp_combo(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam: float = 0.001,
    probe_radius: float = 1.0,
    esp_weight: float = 0.5,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched ESP-combo alignment (ShaEP-style) with padding-safe masks.

    Side effects
    ------------
    Writes:
    - p.transform_esp_combo
    - p.sim_aligned_esp_combo
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_esp_combo, pairs,
                                alpha=alpha, lam=lam, probe_radius=probe_radius,
                                esp_weight=esp_weight, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    from shepherd_score.alignment.utils.fast_esp_combo_se3 import fast_optimize_esp_combo_score_overlay_batch

    device = pairs[0].device

    # Ensure required tensors exist on device
    for p in pairs:
        if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
            raise ValueError("Surface points are None; cannot run _align_batch_esp_combo.")
        if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
            raise ValueError("Surface ESP is None; cannot run _align_batch_esp_combo.")

        def _ensure(p, attr, src, dtype):
            t = getattr(p, attr, None)
            if t is None:
                setattr(p, attr, torch.as_tensor(src, dtype=dtype, device=device))
            elif t.device != device:
                setattr(p, attr, t.to(device, non_blocking=True))

        _ensure(p, "_ref_surf_t", p.ref_molec.surf_pos, torch.float32)
        _ensure(p, "_fit_surf_t", p.fit_molec.surf_pos, torch.float32)
        _ensure(p, "_ref_surf_esp_t", p.ref_molec.surf_esp, torch.float32)
        _ensure(p, "_fit_surf_esp_t", p.fit_molec.surf_esp, torch.float32)
        _ensure(p, "_ref_centers_w_H_t",
                p.ref_molec.mol.GetConformer().GetPositions(), torch.float32)
        _ensure(p, "_fit_centers_w_H_t",
                p.fit_molec.mol.GetConformer().GetPositions(), torch.float32)
        _ensure(p, "_ref_partial_t", p.ref_molec.partial_charges, torch.float32)
        _ensure(p, "_fit_partial_t", p.fit_molec.partial_charges, torch.float32)
        _ensure(p, "_ref_radii_t", p.ref_molec.radii, torch.float32)
        _ensure(p, "_fit_radii_t", p.fit_molec.radii, torch.float32)

        if trans_init:
            _ensure(p, "_ref_xyz_t", p.ref_molec.atom_pos, torch.float32)
            _ensure(p, "_fit_xyz_t", p.fit_molec.atom_pos, torch.float32)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    buckets: dict[tuple[int, int, int, int, int, int, int], list[MoleculePair]] = {}
    for p in pairs:
        n_surf_band = _band_key(p._ref_surf_t.shape[0])
        m_surf_band = _band_key(p._fit_surf_t.shape[0])
        n_wH_band = _band_key(p._ref_centers_w_H_t.shape[0])
        m_wH_band = _band_key(p._fit_centers_w_H_t.shape[0])

        # Shape "centers" use volumetric atoms when alpha==0.81, else surface points.
        if alpha == 0.81:
            n_cent_band = _band_key(p._ref_xyz_t.shape[0])
            m_cent_band = _band_key(p._fit_xyz_t.shape[0])
        else:
            n_cent_band = n_surf_band
            m_cent_band = m_surf_band

        tc = int(p._ref_xyz_t.shape[0]) if trans_init else 0
        buckets.setdefault((n_wH_band, m_wH_band, n_cent_band, m_cent_band, n_surf_band, m_surf_band, tc), []).append(p)

    for (n_wH_pad, m_wH_pad, n_cent_pad, m_cent_pad, n_surf_pad, m_surf_pad, tc), bucket in buckets.items():
        K = len(bucket)

        # Allocate padded blocks
        centers_w_H_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
        centers_w_H_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)
        partial_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
        partial_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)
        radii_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
        radii_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)

        centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)

        points_1 = torch.zeros(K, n_surf_pad, 3, device=device, dtype=torch.float32)
        points_2 = torch.zeros(K, m_surf_pad, 3, device=device, dtype=torch.float32)
        point_charges_1 = torch.zeros(K, n_surf_pad, device=device, dtype=torch.float32)
        point_charges_2 = torch.zeros(K, m_surf_pad, device=device, dtype=torch.float32)

        N_real_atoms_w_H_1 = torch.empty(K, device=device, dtype=torch.int32)
        M_real_atoms_w_H_2 = torch.empty(K, device=device, dtype=torch.int32)
        N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        M_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        N_real_surf_1 = torch.empty(K, device=device, dtype=torch.int32)
        M_real_surf_2 = torch.empty(K, device=device, dtype=torch.int32)

        # Gather per-pair tensors once, then batched scatter-fill (was a per-pair
        # loop of ~10*K device slice-copies + per-element int scalar writes).
        ref_wH_ts = [p._ref_centers_w_H_t for p in bucket]
        fit_wH_ts = [p._fit_centers_w_H_t for p in bucket]
        ref_surf_ts = [p._ref_surf_t for p in bucket]
        fit_surf_ts = [p._fit_surf_t for p in bucket]
        n_wH_list = [t.shape[0] for t in ref_wH_ts]
        m_wH_list = [t.shape[0] for t in fit_wH_ts]
        n_surf_list = [t.shape[0] for t in ref_surf_ts]
        m_surf_list = [t.shape[0] for t in fit_surf_ts]

        N_real_atoms_w_H_1.copy_(torch.tensor(n_wH_list, dtype=torch.int32))
        M_real_atoms_w_H_2.copy_(torch.tensor(m_wH_list, dtype=torch.int32))
        N_real_surf_1.copy_(torch.tensor(n_surf_list, dtype=torch.int32))
        M_real_surf_2.copy_(torch.tensor(m_surf_list, dtype=torch.int32))

        _scatter_fill(centers_w_H_1, ref_wH_ts, n_wH_list)
        _scatter_fill(centers_w_H_2, fit_wH_ts, m_wH_list)
        _scatter_fill(partial_1, [p._ref_partial_t for p in bucket], n_wH_list)
        _scatter_fill(partial_2, [p._fit_partial_t for p in bucket], m_wH_list)
        _scatter_fill(radii_1, [p._ref_radii_t for p in bucket], n_wH_list)
        _scatter_fill(radii_2, [p._fit_radii_t for p in bucket], m_wH_list)
        _scatter_fill(points_1, ref_surf_ts, n_surf_list)
        _scatter_fill(points_2, fit_surf_ts, m_surf_list)
        _scatter_fill(point_charges_1, [p._ref_surf_esp_t for p in bucket], n_surf_list)
        _scatter_fill(point_charges_2, [p._fit_surf_esp_t for p in bucket], m_surf_list)

        # "centers" are volumetric atoms when alpha==0.81, else surface points
        # (constant for the whole call, so the branch is hoisted out of the bucket).
        if alpha == 0.81:
            ref_cent_ts = [p._ref_xyz_t for p in bucket]
            fit_cent_ts = [p._fit_xyz_t for p in bucket]
            n_cent_list = [t.shape[0] for t in ref_cent_ts]
            m_cent_list = [t.shape[0] for t in fit_cent_ts]
        else:
            ref_cent_ts, fit_cent_ts = ref_surf_ts, fit_surf_ts
            n_cent_list, m_cent_list = n_surf_list, m_surf_list
        _scatter_fill(centers_1, ref_cent_ts, n_cent_list)
        _scatter_fill(centers_2, fit_cent_ts, m_cent_list)
        N_real_centers.copy_(torch.tensor(n_cent_list, dtype=torch.int32))
        M_real_centers.copy_(torch.tensor(m_cent_list, dtype=torch.int32))

        trans_centers_batch = None
        trans_centers_real = None
        if trans_init:
            # All pairs in this bucket share exactly tc translation centers (bucket
            # key), so a single stack is equivalent to the per-pair fill.
            trans_centers_batch = torch.stack([p._ref_xyz_t for p in bucket])
            trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

        _, q_batch, t_batch, scores = fast_optimize_esp_combo_score_overlay_batch(
            centers_w_H_1,
            centers_w_H_2,
            centers_1,
            centers_2,
            points_1,
            points_2,
            partial_1,
            partial_2,
            point_charges_1,
            point_charges_2,
            radii_1,
            radii_2,
            alpha,
            lam=lam,
            probe_radius=probe_radius,
            esp_weight=esp_weight,
            N_real_atoms_w_H_1=N_real_atoms_w_H_1,
            M_real_atoms_w_H_2=M_real_atoms_w_H_2,
            N_real_centers=N_real_centers,
            M_real_centers=M_real_centers,
            N_real_surf_1=N_real_surf_1,
            M_real_surf_2=M_real_surf_2,
            trans_centers_batch=trans_centers_batch,
            trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans,
            topk=topk,
            steps_fine=steps_fine,
            lr=lr,
        )

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched (was per-pair quaternion_to_SE3)
    scores_list = scores_cpu.tolist()                       # one C call (was K float())
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_esp_combo = S
        p.sim_aligned_esp_combo = s

def _align_batch_pharm(
    pairs: list["MoleculePair"],
    *,
    similarity: _SIM_TYPE = "tanimoto",
    extended_points: bool = False,
    only_extended: bool = False,
    trans_init: bool = False,
    num_repeats: int = 50,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
):
    """
    Batched pharmacophore alignment using the fast GPU pathway when available.

    Writes per-pair:
    - p.transform_pharm (4x4)
    - p.sim_aligned_pharm (float)
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_pharm, pairs,
                                similarity=similarity, extended_points=extended_points,
                                only_extended=only_extended, trans_init=trans_init,
                                num_repeats=num_repeats, topk=topk,
                                steps_fine=steps_fine, lr=lr)

    if not torch.cuda.is_available():
        # Slow fallback: per-pair legacy optimizer
        for p in pairs:
            p.align_with_pharm(
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                trans_init=trans_init,
                num_repeats=num_repeats,
                lr=lr,
                max_num_steps=steps_fine,
                use_jax=False,
                verbose=False,
            )
        return

    try:
        from shepherd_score.alignment.utils.fast_pharm_se3 import fast_optimize_pharm_overlay_batch
    except ImportError:
        fast_optimize_pharm_overlay_batch = None

    if fast_optimize_pharm_overlay_batch is None:
        for p in pairs:
            p.align_with_pharm(
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                trans_init=trans_init,
                num_repeats=num_repeats,
                lr=lr,
                max_num_steps=steps_fine,
                use_jax=False,
                verbose=False,
            )
        return

    device = pairs[0].device

    # Ensure per-pair cached tensors exist on the correct device.
    for p in pairs:
        rt = getattr(p, "_ref_pharm_types_t", None)
        if rt is None:
            p._ref_pharm_types_t = torch.as_tensor(p.ref_molec.pharm_types, dtype=torch.int64, device=device)
        elif rt.device != device:
            p._ref_pharm_types_t = rt.to(device, non_blocking=True)
        ft = getattr(p, "_fit_pharm_types_t", None)
        if ft is None:
            p._fit_pharm_types_t = torch.as_tensor(p.fit_molec.pharm_types, dtype=torch.int64, device=device)
        elif ft.device != device:
            p._fit_pharm_types_t = ft.to(device, non_blocking=True)
        ra = getattr(p, "_ref_pharm_ancs_t", None)
        if ra is None:
            p._ref_pharm_ancs_t = torch.as_tensor(p.ref_molec.pharm_ancs, dtype=torch.float32, device=device)
        elif ra.device != device:
            p._ref_pharm_ancs_t = ra.to(device, non_blocking=True)
        fa = getattr(p, "_fit_pharm_ancs_t", None)
        if fa is None:
            p._fit_pharm_ancs_t = torch.as_tensor(p.fit_molec.pharm_ancs, dtype=torch.float32, device=device)
        elif fa.device != device:
            p._fit_pharm_ancs_t = fa.to(device, non_blocking=True)
        rv = getattr(p, "_ref_pharm_vecs_t", None)
        if rv is None:
            p._ref_pharm_vecs_t = torch.as_tensor(p.ref_molec.pharm_vecs, dtype=torch.float32, device=device)
        elif rv.device != device:
            p._ref_pharm_vecs_t = rv.to(device, non_blocking=True)
        fv = getattr(p, "_fit_pharm_vecs_t", None)
        if fv is None:
            p._fit_pharm_vecs_t = torch.as_tensor(p.fit_molec.pharm_vecs, dtype=torch.float32, device=device)
        elif fv.device != device:
            p._fit_pharm_vecs_t = fv.to(device, non_blocking=True)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    buckets: dict[tuple[int, int, int], list[MoleculePair]] = {}
    for p in pairs:
        n_band = _band_key(p._ref_pharm_ancs_t.shape[0])
        m_band = _band_key(p._fit_pharm_ancs_t.shape[0])
        tc = int(p._ref_pharm_ancs_t.shape[0]) if trans_init else 0
        buckets.setdefault((n_band, m_band, tc), []).append(p)

    for (N_pad, M_pad, tc), bucket in buckets.items():
        K = len(bucket)

        ref_types = torch.zeros(K, N_pad, device=device, dtype=torch.int64)
        fit_types = torch.zeros(K, M_pad, device=device, dtype=torch.int64)
        ref_ancs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_ancs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)
        ref_vecs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_vecs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)

        N_real = torch.empty(K, device=device, dtype=torch.int32)
        M_real = torch.empty(K, device=device, dtype=torch.int32)

        ref_ancs_ts = [p._ref_pharm_ancs_t for p in bucket]
        fit_ancs_ts = [p._fit_pharm_ancs_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ancs_ts]
        m_list = [t.shape[0] for t in fit_ancs_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))   # was per-element GPU writes
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))
        # Batched scatter pad-fill (was a per-pair loop of 6*K device slice-copies).
        _scatter_fill(ref_types, [p._ref_pharm_types_t for p in bucket], n_list)
        _scatter_fill(fit_types, [p._fit_pharm_types_t for p in bucket], m_list)
        _scatter_fill(ref_ancs, ref_ancs_ts, n_list)
        _scatter_fill(fit_ancs, fit_ancs_ts, m_list)
        _scatter_fill(ref_vecs, [p._ref_pharm_vecs_t for p in bucket], n_list)
        _scatter_fill(fit_vecs, [p._fit_pharm_vecs_t for p in bucket], m_list)

        trans_centers_batch = ref_ancs if trans_init else None
        trans_centers_real = N_real if trans_init else None

        # GPU-memory-safe sub-batching per bucket (independent pairs). Pharm's
        # analytical fine loop has the largest (~N_pad*M_pad) footprint, so
        # this is where the dynamic cap matters most.
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
            tcr = trans_centers_real[sl] if trans_centers_real is not None else None
            _, _, q, t, sc = fast_optimize_pharm_overlay_batch(
                ref_types[sl], fit_types[sl], ref_ancs[sl], fit_ancs[sl],
                ref_vecs[sl], fit_vecs[sl],
                similarity=similarity, extended_points=extended_points,
                only_extended=only_extended, num_repeats=num_repeats,
                trans_centers_batch=tcb, trans_centers_real=tcr,
                num_repeats_per_trans=10, N_real=N_real[sl], M_real=M_real[sl],
                topk=topk, steps_fine=steps_fine, lr=lr,
            )
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("pharm", N_pad, M_pad, num_repeats))

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched (was per-pair)
    scores_list = scores_cpu.tolist()                       # one C call (was K float())
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_pharm = S
        p.sim_aligned_pharm = s
