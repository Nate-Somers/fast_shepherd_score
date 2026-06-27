"""Generic CUDA-graph fine-loop: capture ONE Adam optimisation step, replay it N times.

The per-pose fine loop is launch-bound at small/medium batch (~10 kernel launches per
step for the single-channel modes, ~2x that for vol_color). We capture one in-place fine
step into a CUDA graph and replay it ``steps`` times -- replay carries loop state between
steps through persistent buffers, so N replays == N eager steps with ~zero per-step host
launch overhead. This is fss's analogue of ROSHAMBO2's in-kernel optimisation loop (one
launch drives the whole optimisation; the GPU never waits on the host between steps).

Originally this lived only in ``shape.py`` (``_GraphedFineSurf``), covering vol/surf. This
module lifts the mode-agnostic capture/replay machinery into a base class so every mode can
share it: a subclass only supplies its persistent buffers (``__init__``/``_alloc``), the
per-step body (``_step``), how a bucket's inputs load into the buffers (``_load``), the
loop-state reset (``_reset``), and the result (``_result``).

Why intra-capture kernel allocations are fine
---------------------------------------------
The value+grad kernels (``overlap_score_grad_se3_batch`` etc.) allocate their output
tensors internally each call. That is *not* a capture blocker: under ``torch.cuda.graph``
those allocations come from the graph's private memory pool and get fixed addresses reused
on every replay. The two things that DO break capture -- and that every ``_step`` must
avoid -- are (1) host syncs / data-dependent control flow inside the loop (the ``.item()``
early-stop: dropped here in favour of a fixed replay count) and (2) Python rebinding of
loop-carried tensors (the functional ``_update_best`` that returns new tensors: every
``_step`` instead updates ``best``/``best_q``/``best_t`` in place via ``where(..., out=)``).
Autotune/JIT must also be warmed BEFORE capture -- ``capture()`` does three warmup steps in
a side stream for exactly this.
"""
from __future__ import annotations

import os
from collections import OrderedDict
import torch

# Shared env knobs (single source of truth; shape.py and every other driver import these).
# Disable the whole graph fast path with FINE_CUDA_GRAPHS=0.
_FINE_GRAPHS = os.environ.get("FINE_CUDA_GRAPHS", "1") != "0"
# Max pose-rows P for a single captured graph. Above this the eager loop is used (it is
# compute-bound there, so the graph stops helping) UNLESS the mode opts into chunked replay
# via run_graphed_chunked (the launch-bound modes, where it keeps helping at large batch).
_GRAPH_MAX_P = int(os.environ.get("FINE_GRAPH_MAX_P", 16000))

# --- compute-aware P-cap ----------------------------------------------------------------
# The graph's win is launch-bound: its launch-savings are ~fixed per step, while its only
# large-P cost grows with the per-row kernel work. So the P at which the graph stops beating
# eager (the crossover) scales as work_budget / (per-row work). Measured crossovers (H200):
# vol (heavy-atom clouds, N_pad*M_pad ~ 256-1024) wins to P ~ 1M; surf (128-point clouds,
# ~16384) crosses at P ~ 16-18k -- and work*P is ~constant (~3e8) across both, because they
# share the gaussian kernel. So cap = clamp(BUDGET // (N_pad*M_pad), MIN, CEIL) graphs each
# mode/bucket exactly where it wins and falls to eager (the baseline -- never a regression)
# where it would lose. This makes ONE formula cover the vol/surf and vol_esp/surf_esp
# shared-code splits (light clouds -> high cap, heavy clouds -> low cap) automatically.
# CEIL bounds the per-graph buffer memory (important for the screen path's large buckets).
_GRAPH_WORK_BUDGET = int(os.environ.get("FINE_GRAPH_WORK_BUDGET", 300_000_000))
# Per-graph memory ceiling on P. 256k covers the 100k-library screen buckets (P~187k) for
# the light modes; their buffers are ~1KB/row so a single graph is ~270MB. Lower this on
# memory-constrained GPUs (the heavy modes are already work-budget-limited well below it).
_GRAPH_CAP_CEIL = int(os.environ.get("FINE_GRAPH_CAP_CEIL", 262144))
_GRAPH_CAP_MIN = int(os.environ.get("FINE_GRAPH_CAP_MIN", 2000))


def graph_cap(work, budget=None):
    """Max pose-rows P to graph for a bucket whose per-row kernel work is ``work`` (e.g.
    N_pad*M_pad). Below this the graph wins; above it the gating falls back to eager."""
    b = _GRAPH_WORK_BUDGET if budget is None else budget
    return max(_GRAPH_CAP_MIN, min(_GRAPH_CAP_CEIL, int(b) // max(int(work), 1)))
# Fixed replay cap when early-stop is disabled. Otherwise the graph replays UP TO steps_fine
# and stops early (see below), so this is mostly legacy.
_GRAPH_STEPS = int(os.environ.get("FINE_GRAPH_STEPS", 50))
# Blocked early-stop for the replay loop: check best.max() every _GRAPH_ES_BLOCK replays
# (one host sync per block, NOT per step) and stop on the same patience/tol schedule as the
# eager loop. This is essential -- without it the graph runs a FIXED step count and OVER-RUNS
# vs the eager early-stop, which regresses the COMPUTE-bound modes (surf, esp_combo, pharm at
# large batch) where the extra steps cost real kernel time. With it, the graph runs ~the same
# number of steps as eager, so the launch-bound modes keep their big win and the compute-bound
# modes stop regressing. Set _GRAPH_ES_BLOCK=0 to disable (fixed-step replay).
_GRAPH_ES_BLOCK = int(os.environ.get("FINE_GRAPH_ES_BLOCK", 5))
# Extra early-stop margin (in blocks) ADDED to the eager patience for the graph replay loop.
# The multi-basin modes (surf, esp) land in different near-equal optima under tiny numerical
# perturbations, so matching the eager step count EXACTLY exposes +/-0.4% basin noise (surf
# came in at -0.43% with no margin). One extra block of optimisation reliably keeps the
# graph's mean >= eager across that noise -- cheap for launch-bound modes, minor for surf.
_GRAPH_ES_MARGIN = int(os.environ.get("FINE_GRAPH_ES_MARGIN", 2))

# Process-wide LRU cache of captured graphs, keyed by (device, mode, shapes, P, steps, params).
# It is BOUNDED: each cached graph pins persistent GPU buffers (~P*buffers bytes), so an
# unbounded cache lets a long/large workload -- a big screen (many library-mol shapes), or a
# benchmark sweep over many sizes/modes -- accumulate GPU memory until fragmentation/OOM makes
# new captures fail and the driver silently falls back to eager (measured: later cells in a
# sweep degraded 12-27x, BELOW even eager). Evicting the least-recently-used graph frees its
# buffers; FINE_GRAPH_CACHE_MAX caps how many live at once.
_FINE_GRAPH_CACHE: "OrderedDict" = OrderedDict()
_GRAPH_CACHE_MAX = int(os.environ.get("FINE_GRAPH_CACHE_MAX", 24))


def _evict_lru():
    """Drop the least-recently-used cached graph and return its buffers to the allocator."""
    _, old = _FINE_GRAPH_CACHE.popitem(last=False)
    del old
    torch.cuda.empty_cache()


def reset_graph_cache():
    """Free every cached graph (and its persistent buffers) and return the blocks to the
    CUDA allocator. Use between independent measurements (e.g. benchmark cells) so each
    starts from a clean, unfragmented GPU state -- accumulated graph buffers + freed-block
    fragmentation otherwise degrade later cells. Leaves the Triton autotune cache intact
    (that is cheap compiled code, not the memory problem)."""
    _FINE_GRAPH_CACHE.clear()
    torch.cuda.empty_cache()


class _GraphedFineBase:
    """Capture one in-place fine step into a CUDA graph; ``run`` replays it ``steps`` times.

    Subclasses allocate all persistent buffers in their ``__init__`` (loop-carried state
    ``q``/``t``/Adam-moments/``best*`` plus the bucket inputs and per-step temporaries) and
    then call ``super().__init__(steps)``. They implement:

      * ``_step()``    -- one in-place optimisation step over the persistent buffers. Must use
                          ``out=``/in-place ops only (no Python rebinding of loop-carried
                          tensors, no host sync). Kernels may allocate their own outputs.
      * ``_load(*x)``  -- copy this bucket's inputs into the persistent input buffers.
      * ``_reset()``   -- reset q/t to the seeds, zero Adam moments, best=-inf, best*=seeds.
      * ``_result()``  -- return (best_score, best_q, best_t).

    The warmup -> capture -> replay logic below is identical for every mode.
    """

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.graph = None
        # Blocked early-stop schedule (set by run_graphed from the driver's eager params).
        # es_patience == 0 -> disabled (fixed-step replay).
        self.es_patience = 0
        self.es_tol = 1e-5
        self.es_block = _GRAPH_ES_BLOCK

    # --- subclass hooks (no-ops here so a mis-specified subclass fails loudly) ---
    def _step(self):        raise NotImplementedError
    def _load(self, *x):    raise NotImplementedError
    def _reset(self):       raise NotImplementedError
    def _result(self):      raise NotImplementedError

    def capture(self, *inputs):
        """Warm autotune/JIT off the capture stream, then capture exactly one ``_step``."""
        self._load(*inputs); self._reset()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                       # warmup -> compile/autotune
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self._step()

    def run(self, *inputs):
        """Load this bucket, reset loop state, replay the captured step (with blocked
        early-stop when enabled), and return the best pose."""
        self._load(*inputs); self._reset()
        if self.es_patience and self.es_block:
            # Mirror the eager early-stop schedule EXACTLY. Eager checks best.max() at steps
            # 0, 5, 10, ... and seeds its baseline `prev` with the STEP-0 value (the best of
            # the seed poses). best is max-over-past-poses, so after one replay it equals
            # eager's step-0 best. So: replay one step, seed `prev`, then check every es_block
            # replays. Matching the *schedule* (not just patience/tol) is essential -- seeding
            # `prev` a block late made the graph stop ~one block early, which regressed
            # slow-converging surf_esp by ~1%. With this, the graph stops at the same step as
            # eager, so best is bit-equivalent.
            self.graph.replay()
            done = 1
            prev = self.best.max().item(); no_improve = 0
            while done < self.steps:
                k = min(self.es_block, self.steps - done)
                for _ in range(k):
                    self.graph.replay()
                done += k
                cur = self.best.max().item()           # one host sync per block, not per step
                if cur - prev < self.es_tol:
                    no_improve += 1
                    if no_improve >= self.es_patience:
                        break
                else:
                    no_improve = 0; prev = cur
        else:
            for _ in range(self.steps):
                self.graph.replay()
        return self._result()


def run_graphed(make, key, inputs, *, es_patience=0, es_tol=1e-5):
    """Fetch (or build+capture) the graph for ``key`` and run this bucket through it.

    ``make`` is a zero-arg factory for the subclass instance (called only on cache miss);
    ``inputs`` is the tuple forwarded to ``capture``/``run`` (and thence ``_load``). One
    captured graph serves every bucket of the same key (same shapes/P/steps/params).
    ``es_patience``/``es_tol`` set the replay-loop blocked early-stop to match the driver's
    eager early-stop (0 -> run a fixed ``steps`` replays).
    """
    gf = _FINE_GRAPH_CACHE.get(key)
    if gf is not None:
        _FINE_GRAPH_CACHE.move_to_end(key)       # mark most-recently-used
        return gf.run(*inputs)
    gf = make()
    # Add the margin only when early-stop is enabled (es_patience > 0).
    gf.es_patience = (int(es_patience) + _GRAPH_ES_MARGIN) if es_patience else 0
    gf.es_tol = float(es_tol)
    # Capture; if it OOMs, free LRU graphs and retry (rather than failing -> eager forever).
    while True:
        try:
            gf.capture(*inputs)
            break
        except Exception as e:
            if "out of memory" not in str(e).lower() or not _FINE_GRAPH_CACHE:
                raise                             # non-OOM, or nothing left to free -> propagate
            _evict_lru()
    _FINE_GRAPH_CACHE[key] = gf
    while len(_FINE_GRAPH_CACHE) > _GRAPH_CACHE_MAX:
        _evict_lru()
    return gf.run(*inputs)


def run_graphed_chunked(make, key_for, inputs, *, P, max_p, es_patience=0, es_tol=1e-5):
    """Cap-lift: drive an arbitrarily large P through graphs of size <= ``max_p``.

    Poses (rows) are independent, so slicing the pose dimension into chunks and running each
    chunk through its own (cached) graph is exactly equivalent to running all P at once -- as
    long as results are concatenated back in row order. Every element of ``inputs`` is a
    per-pose tensor (dim 0 == P), so slicing is generic (``x[s:e]``) and the three outputs
    (best, best_q, best_t) concatenate along dim 0. This bounds the per-graph buffer memory
    (essential for the screen path, where P can be huge) while keeping the launch-bound win;
    at most two distinct chunk sizes occur (``max_p`` and the remainder), so at most two
    graphs are captured per key.

      * ``make(p_chunk)``    -> factory for a chunk-sized subclass instance
      * ``key_for(p_chunk)`` -> cache key for a chunk of that size
    """
    if P <= max_p:
        return run_graphed(lambda: make(P), key_for(P), inputs,
                           es_patience=es_patience, es_tol=es_tol)
    outs = []
    for s in range(0, P, max_p):
        e = min(s + max_p, P); pc = e - s
        sub = tuple(x[s:e].contiguous() for x in inputs)
        outs.append(run_graphed(lambda pc=pc: make(pc), key_for(pc), sub,
                                es_patience=es_patience, es_tol=es_tol))
    return tuple(torch.cat([o[i] for o in outs], 0) for i in range(len(outs[0])))
