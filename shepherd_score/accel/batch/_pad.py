# shepherd_score/accel/batch/_pad.py
"""Size bucketing, GPU-memory-safe sub-batching, and batched scatter-fill
primitives shared by the batched aligners."""
from __future__ import annotations
import os
import torch


### BEGIN size_bucketing #####################################################
# Every heavy-atom count 3‒150 is mapped to a “band” of 8 atoms
# (   1-8, 9-16, 17-24, … ).  Pairs that fall in the same band
# share a common padded tensor size → one GPU launch.
_BAND = 16                     # change to 16/32 if you want larger bands

def _band_key(n: int) -> int:
    "return the *upper* bound of the 8-atom band this n falls into"
    return ((n + _BAND - 1) // _BAND) * _BAND
### END size_bucketing #######################################################

# Measured fine-loop footprint (bytes per pair) keyed by (mode, N_pad, M_pad,
# num_seeds). Lets the sub-batcher size each bucket's chunk to the GPU.
_PAIR_FOOTPRINT_BYTES: dict[tuple, int] = {}
# Set env SUBBATCH_DEBUG=1 to print the chosen chunk size per bucket.
_SUBBATCH_DEBUG = bool(os.environ.get("SUBBATCH_DEBUG"))


def _subbatched_align(process, K: int, *, key: tuple, device: torch.device,
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
