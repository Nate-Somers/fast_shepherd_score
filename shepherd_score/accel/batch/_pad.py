# shepherd_score/accel/batch/_pad.py
"""Size bucketing, GPU-memory-safe sub-batching, and batched scatter-fill
primitives shared by the batched aligners."""
from __future__ import annotations
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


#: Default upper bound, in POSES, on one fine-loop sub-batch. Applied only where a caller
#: passes ``pose_cap``, so every other call site keeps the purely memory-derived schedule.
#:
#: THE CAP IS PER MODE, NOT GLOBAL, and that is measured: each driver graphs below its own
#: ``graph_cap`` work budget (vol/vol_esp 3e8, vol_color 3e7, pharm 1e7), and the amount of
#: launch overhead a graph removes is tiny next to a heavy per-step kernel. Armed on every array
#: mode at N=20,000 (job 22598857) the answer differed by mode:
#:      vol       1.2985x     <- kept
#:      vol_esp   0.9981x     <- neutral; the ESP step is too heavy for launches to matter
#:      pharm     0.6496x     <- REGRESSION, cause NOT established. The "graph threshold is
#:                               9,765 poses" reading was wrong: pharm hands graph_cap its
#:                               ANCHOR pads, not the shape band, so drug-like molecules give
#:                               work 256 -> cap 39,062 poses (job 22637452) and 9,765 needs
#:                               peptide-sized feature counts (job 22637626). A cap here CAN
#:                               reach a graph; why arming one cost 0.6496x is still open
#: So callers pass a cap they have measured, and most pass none.
#:
#: This is not a tuning knob, it is a CORRECTNESS-OF-MEASUREMENT one. The memory-derived chunk
#: decides WHICH FINE LOOP RUNS: ``drivers/_graphed.graph_cap`` refuses to capture past 262,144
#: poses, so a bucket that fits in one memory-sized chunk (a 96,850-molecule band at 10 seeds is
#: 968,500 poses) silently runs the eager loop instead of the CUDA graph. Profiling the same
#: N=100,000 vol screen twice showed both outcomes, because the answer depends on what the
#: allocator happened to be holding. The two paths CAN diverge, but only where the early-stop
#: margin is actually exercised: the graph replay adds ``_GRAPH_ES_MARGIN`` blocks of patience on
#: top of the eager schedule, so a run that early-stops runs longer under the graph and scores
#: differently. Both halves of that are now measured and they disagree BY MODE. It DOES diverge
#: for vol: 71,736 of 100,000 scores move, max 6.5e-03, when the same screen takes the eager path
#: instead (that run's job id was never recorded here). It does NOT for vol_color, vol_lipo or
#: pharm: forced graph-vs-eager at identical P is BIT-IDENTICAL at every point of jobs 22637452 +
#: 22637626 -- 9 forced P-points each for vol_color/vol_lipo and 10 for pharm, over two molecule
#: pools (drug-like and peptide), up to 16,384 pairs per point, plus all 128 budget-sweep points
#: of those two jobs (only vol_and_surf_esp moved there, for an unrelated reason -- see
#: drivers/esp_combo.py). The margin never fired in any of that: early_stop_frac = 0.000 in every
#: one of those cells, and in every mode of the 100k screen in job 22637463. So the divergence is
#: real but MODE- AND REGIME-SPECIFIC (vol, in a regime that early-stops); it is not a property
#: the graph path has unconditionally.
#: What the cap actually buys is REPRODUCIBILITY, not an unconditional graph, and
#: the arithmetic says so. The cap pins P at 81,920 poses (8,192 pairs x vol's 10 seeds) while
#: ``_graphed.graph_cap(work) = max(2000, min(262144, 300_000_000 // work))``, so a capped chunk
#: graphs only where N_pad*M_pad <= 3,662. Worked against the constants at HEAD: 48x48 -> cap
#: 130,208 (GRAPHS); 48x80 -> 78,125, 64x64 -> 73,242, 112x112 -> 23,915 (all EAGER). Wide
#: buckets keep taking the eager loop. The cap is also only an UPPER bound on P -- ``_budget() //
#: fp`` can shrink a chunk further on a busy device, which flips a bucket the other way, into the
#: graph. What the cap removes is the unbounded-P case that made a big band's path swing on
#: allocator state; it does not make the choice unconditional.
#:
#: 81,920 = the measured optimum on an L40S, at BOTH library sizes and for all of vol's shapes
#: (job 22595747, aligns/s vs chunk in molecules at 10 seeds):
#:      N=1e5   4096:885,682  6144:939,470  8192:964,248  12288:903,760  24576:903,640
#:      N=1e6   4096:887,968  6144:940,391  8192:951,602  12288:903,819  24576:891,175
#: Every all-graphed setting above is BIT-IDENTICAL to every other (verified over all 100,000
#: and all 1,000,000 scores), so the size is free to choose on speed alone.
_FINE_CHUNK_POSES = 81920


#: Pairs per call on the CPU path. Bounds peak host RSS on an out-of-core screen, where
#: screen.py's shard_size=100_000 default otherwise hands the fine loop one 100k-pair batch.
#: 10,000 measured indistinguishable from 1,000 on throughput (621.3 vs 621.5 aligns/s) while
#: making the chunk count 10x smaller, so it takes the top of the measured-flat decade.
_CPU_CHUNK_PAIRS = 10_000


def _concat_chunks(process, K: int, step: int):
    """Run ``process`` over ``K`` pairs in ``step``-sized chunks and concatenate.

    The GPU branch below does the same thing with a dynamic, memory-derived chunk; this is the
    fixed-size CPU twin, kept separate because none of the budget machinery (mem_get_info, the
    footprint cache, the OOM halve-and-retry) has a meaning off CUDA.
    """
    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    while s < K:
        k = min(step, K - s)
        sc, q, t = process(s, k)
        sc_parts.append(sc); q_parts.append(q); t_parts.append(t)
        s += k
    if len(sc_parts) == 1:
        return sc_parts[0], q_parts[0], t_parts[0]
    return (torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts))


def _subbatched_align(process, K: int, *, key: tuple, device: torch.device,
                      safety: float = 0.7, init_cap: int = 1024, pose_cap: int = 0,
                      seeds: int = 1):
    """Drive ``process(start, count) -> (scores, q, t)`` over ``K`` independent
    pairs in GPU-memory-safe sub-batches and concatenate the per-pair results.

    Chunking + concatenation bounds peak memory. It is score-identical only when
    every chunk runs the SAME number of fine steps. Pairs are independent WITHIN a
    step (each result is its own max over its own seeds), but the early-stop break
    is CHUNK-GLOBAL -- the ``if not improved.any(): ... break`` in drivers/shape.py,
    kernels/cpu_fused.py and drivers/_graphed.py all break only once every pair IN
    THE CHUNK has stalled -- so re-cutting the chunks can hand a pair a different
    step count. MEASURED over 100,000 vol pairs:
    0 of 100,000 scores move at chunks of 4 pairs and up; 19 of 100,000 move at ONE
    PAIR PER CHUNK (L40S, max 2.338e-02, all downward), and on CPU up to 2.203e-02 at
    a split boundary. ``early_stop_patience=0`` makes the two schedules bit-identical,
    which pins the cause on the early-stop break and nothing else.

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
        # CPU (or any non-CUDA) tensors. Memory-SAFETY chunking is a GPU concern -- there is no
        # device allocator to run out of -- so there is no budget loop here. But running the
        # whole batch in one call was NOT free: screen.py defaults shard_size=100_000, so an
        # out-of-core CPU screen hands this a single 100k-pair batch and every per-pair
        # intermediate is live at once.
        #
        # MEASURED (cluster node1611, 1 thread, numba 0.59.1 + SVML, pharm, library fixed at
        # N=1e5 with only the per-call batch varied):
        #     100 x 1,000   621.5 aligns/s     peak RSS   876 MB
        #      10 x 10,000  621.3 aligns/s
        #       1 x 100,000 575.0 aligns/s     peak RSS 6,383 MB
        # i.e. 1.081x wall (173.9 s -> 160.9 s) and a 7.3x cut in peak RSS, which is the larger
        # practical win -- 6.4 GB for one screen is what makes a CPU screen fall over on a shared
        # node. Any cap in the 1e3-1e4 decade behaves the same (the two above differ by 0.03%).
        #
        # BIT-IDENTICAL, measured over the full 100,000-element score vector across the
        # 1-shard / 10-shard / 100-shard partitions (0 moved, max|delta| 0.0). The guarantee is
        # CONDITIONAL and the condition was checked: _stats reported early_stop_frac = 0.0 with
        # steps_min == steps_max == steps_configured throughout, so every chunk ran its full
        # budget and the chunk-global early-stop break could not fire. A workload that DOES
        # early-stop can score differently under a different partition, for the reason
        # _subbatched_align's docstring gives above -- that is a property of the early stop, not
        # of this cap.
        #
        # Deliberately large: below _CPU_CHUNK_PAIRS nothing changes at all, so ordinary pairwise
        # batches (hundreds to a few thousand pairs) take exactly the path they took before.
        if K <= _CPU_CHUNK_PAIRS:
            return process(0, K)
        return _concat_chunks(process, K, _CPU_CHUNK_PAIRS)

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
    K_sub = max(1, min(K, int(_budget() // fp))) if fp else min(K, init_cap)
    K_sub = max(1, min(K_sub, cap_pairs))

    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    while s < K:
        k = min(K_sub, K - s)
        try:
            torch.cuda.reset_peak_memory_stats()
            base = int(torch.cuda.memory_allocated())
            sc, q, t = process(s, k)
            peak = int(torch.cuda.max_memory_allocated())
            # Charge the chunk for ITS OWN GROWTH, never for the device-wide high-water
            # mark. reset_peak_memory_stats() rebases the peak to whatever is ALREADY
            # resident, so without ``- base`` a chunk is billed for every byte any
            # unrelated object holds. MEASURED: with 4.495e10 B held elsewhere the same
            # shape recorded 1,879,004,971 B/pair against 112,631-173,764 B/pair on an
            # idle GPU -- 1.1e4x high. That collapses K_sub to 1 (predicted vs observed
            # chunk counts reconcile to three digits: 33.3 vs 33, and 159 vs 156), costs
            # ~100x throughput (0.227 s -> 21.96 s -> 60.1 s), and is the ONLY regime in
            # which any score moved (19 of 100,000, deterministic, all downward -- via
            # the chunk-global early stop the function docstring describes).
            growth = max(0, peak - base)
            # Fold a chunk into the per-pair footprint only when it is large enough
            # that the fixed workspace overhead (seed/autotune scratch -- tens of MB,
            # independent of k) is amortised. growth/k = fixed/k + per_pair, so a tiny
            # trailing remainder (e.g. k=7) yields a wildly inflated bytes/pair that
            # max() would lock in, collapsing every later chunk to a fraction of its
            # right size (pharm was observed going 2 -> 16 -> 82 chunks this way). The
            # first chunk has k == K_sub so it always qualifies; calibration is never
            # starved.
            if k >= max(1, K_sub // 4):
                fp_meas = max(1, -(-growth // k))                # ceil bytes/pair
                # The max() fold STAYS, even though ``- base`` removes the dominant source
                # of a poisoned reading. (a) Growth is legitimately non-stationary DOWNWARD:
                # the first chunk of a shape pays for the captured graph's persistent
                # buffers, which later chunks of the same P reuse out of
                # ``_graphed._FINE_GRAPH_CACHE``, so letting a later reading overwrite would
                # grow the chunk on a measurement that never paid for the buffers it is still
                # using. (b) No run has been measured WITHOUT the fold, and the failure it
                # would trade into -- an under-estimate sizing a chunk into an OOM -- costs
                # the halve-and-retry path below. Residual risk, unchanged by this fix: the
                # fold is monotone and the cache is never invalidated, so one over-estimate
                # from fixed in-call workspace is still permanent for the process.
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
