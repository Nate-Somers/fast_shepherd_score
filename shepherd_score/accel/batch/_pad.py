# shepherd_score/accel/batch/_pad.py
"""Size bucketing, GPU-memory-safe sub-batching, and batched scatter-fill
primitives shared by the batched aligners."""
from __future__ import annotations
import os
import torch


### BEGIN size_bucketing #####################################################
# Every point count is mapped up to a multiple of _BAND. Pairs in the same band share a
# common padded tensor size -> one kernel launch. This is the legacy fixed-band key; the
# adaptive planner in _bucket.py supersedes it and only snaps its pads to this grid.
_BAND = 16

def _band_key(n: int) -> int:
    "return the *upper* bound of the _BAND-sized band this n falls into"
    return ((n + _BAND - 1) // _BAND) * _BAND
### END size_bucketing #######################################################

# Measured fine-loop footprint (bytes per pair) keyed by (mode, N_pad, M_pad,
# num_seeds). Lets the sub-batcher size each bucket's chunk to the GPU.
_PAIR_FOOTPRINT_BYTES: dict[tuple, int] = {}


#: Fixed sub-batch size, in pairs. 0 (default) keeps the adaptive memory-derived schedule.
#: Set FSS_SUBBATCH_CHUNK=<n> to make the chunk boundaries deterministic -- see the comment in
#: _subbatched_align for why vol_and_surf_esp needs it.
_FIXED_CHUNK = int(os.environ.get("FSS_SUBBATCH_CHUNK", "0"))

#: Default upper bound, in POSES, on one fine-loop sub-batch. Applied only where a caller
#: passes ``pose_cap``, so every other call site keeps the purely memory-derived schedule.
#:
#: THE CAP IS PER MODE, NOT GLOBAL, and that is measured: each driver graphs below its own
#: ``graph_cap`` work budget (vol/vol_esp 3e8, vol_color 3e7, pharm 1e7), and the amount of
#: launch overhead a graph removes is tiny next to a heavy per-step kernel. Armed on every array
#: mode at N=20,000 (job 22598857) the answer differed by mode:
#:      vol       1.2985x     <- kept
#:      vol_esp   0.9981x     <- neutral; the ESP step is too heavy for launches to matter
#:      pharm     0.6496x     <- REGRESSION: its graph threshold is 9,765 poses, so this cap
#:                               only multiplied the chunk count without ever reaching a graph
#: So callers pass a cap they have measured, and most pass none.
#:
#: This is not a tuning knob, it is a CORRECTNESS-OF-MEASUREMENT one. The memory-derived chunk
#: decides WHICH FINE LOOP RUNS: ``drivers/_graphed.graph_cap`` refuses to capture past 262,144
#: poses, so a bucket that fits in one memory-sized chunk (a 96,850-molecule band at 10 seeds is
#: 968,500 poses) silently runs the eager loop instead of the CUDA graph. Profiling the same
#: N=100,000 vol screen twice showed both outcomes, because the answer depends on what the
#: allocator happened to be holding. The two paths are not equivalent: the graph replay adds
#: ``_GRAPH_ES_MARGIN`` blocks of early-stop patience, so it runs longer and scores differently
#: (measured: 71,736 of 100,000 scores move, max 6.5e-03, when the same screen takes the eager
#: path instead). Capping poses makes the graph path unconditional and the result reproducible.
#:
#: 81,920 = the measured optimum on an L40S, at BOTH library sizes and for all of vol's shapes
#: (job 22595747, aligns/s vs chunk in molecules at 10 seeds):
#:      N=1e5   4096:885,682  6144:939,470  8192:964,248  12288:903,760  24576:903,640
#:      N=1e6   4096:887,968  6144:940,391  8192:951,602  12288:903,819  24576:891,175
#: Every all-graphed setting above is BIT-IDENTICAL to every other (verified over all 100,000
#: and all 1,000,000 scores), so the size is free to choose on speed alone.
_FINE_CHUNK_POSES = int(os.environ.get("FSS_FINE_CHUNK_POSES", "81920"))


def _subbatched_align(process, K: int, *, key: tuple, device: torch.device,
                      safety: float = 0.7, init_cap: int = 1024, pose_cap: int = 0,
                      seeds: int = 1):
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
    if device.type != "cuda":
        # CPU (or any non-CUDA) tensors: memory-safe chunking is a GPU concern, so run
        # the whole batch in one call. Keys off the *data* device, not machine
        # capability, so a CUDA box driving CPU tensors (e.g. backend="numba") is CPU.
        return process(0, K)

    key = (torch.cuda.current_device(),) + tuple(key)   # device-scope the footprint cache

    def _budget() -> float:
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return safety * (free + max(0, reusable))

    # Pose cap: only ever SHRINKS a chunk, so the memory safety below is untouched. Applied
    # before the memory sizing so ``need_resize``'s later growth respects it too.
    cap_pairs = max(1, int(pose_cap) // max(1, int(seeds))) if pose_cap > 0 else K

    fp = _PAIR_FOOTPRINT_BYTES.get(key)
    need_resize = fp is None
    if _FIXED_CHUNK > 0:
        # DETERMINISTIC chunking: the schedule no longer depends on free memory, so a run is
        # reproducible across machines and memory states. Measured motivation: vol_and_surf_esp
        # is chunk-sensitive (it is multi-basin, and seed generation is batch-size dependent via
        # _masked_principal_axes), so the adaptive schedule made the SHIPPING object path diverge
        # from ITSELF -- 513 of 100,000 scores under a 12 GB memory hog, more than the array path
        # differed from it. An OOM still halves and retries below, so this bounds the
        # nondeterminism to genuine OOM rather than removing the safety net.
        K_sub = max(1, min(K, _FIXED_CHUNK))
        need_resize = False
    else:
        K_sub = max(1, min(K, int(_budget() // fp))) if fp else min(K, init_cap)
    K_sub = max(1, min(K_sub, cap_pairs))

    sc_parts, q_parts, t_parts = [], [], []
    s = 0
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
            if need_resize:   # first success -> we now know the real footprint
                fp = _PAIR_FOOTPRINT_BYTES[key]
                remaining = K - s
                if remaining > 0:
                    K_sub = max(1, min(remaining, int(_budget() // fp), cap_pairs))
                need_resize = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Some OOMs surface as a plain RuntimeError; only treat those as OOM.
            if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                    and "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            if k <= 1:
                raise
            K_sub = max(1, k // 2)
    return torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts)


def _scatter_fill(out: torch.Tensor, tensors: list[torch.Tensor], sizes: list[int]) -> None:
    """Fill a pre-zeroed padded workspace ``out`` of shape ``(K, P_pad, *feat)`` so
    that ``out[i, :sizes[i]] = tensors[i]`` for each of the ``K`` per-pair tensors.

    Bit-identical to a per-pair ``out[i, :n] = t`` loop / ``pad_sequence`` fill, but
    it copies via ONE batched ``torch.cat`` + ONE vectorized scatter instead of ``K``
    launch-bound device copies -- that fill is the dominant per-pair *host* cost at large
    batch. ``out``'s padding rows are left untouched (the caller zeroes them), so the
    result is deterministic and exactly equal to the previous fill.
    """
    K, P_pad = out.shape[0], out.shape[1]
    device = out.device
    # ``sizes`` is already a host list[int]; sum it on the host. The old
    # ``int(n.sum())`` on a device tensor forced a CUDA stream sync + scalar
    # copyback on EVERY scatter (2-10x per bucket for the multi-channel modes),
    # serializing the host against the GPU for a value we already know. Same result.
    S = sum(sizes)
    if S == 0:
        return
    n = torch.as_tensor(sizes, device=device, dtype=torch.long)
    flat = torch.cat(tensors, dim=0)                       # (S, *feat)
    starts = torch.cumsum(n, 0) - n                        # (K,) first flat-row of each pair
    seg = torch.repeat_interleave(starts, n)               # (S,) segment start per flat row
    local = torch.arange(S, device=device) - seg           # (S,) within-pair row index
    dst = torch.repeat_interleave(torch.arange(K, device=device) * P_pad, n) + local
    out.view(K * P_pad, *out.shape[2:])[dst] = flat
