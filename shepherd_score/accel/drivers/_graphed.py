"""Generic CUDA-graph fine-loop: capture ONE Adam optimisation step, replay it N times.

The per-pose fine loop is launch-bound at small/medium batch. One in-place fine step is
captured into a CUDA graph and replayed ``steps`` times -- replay carries loop state between
steps through persistent buffers, so N replays == N eager steps, with ~zero per-step host
launch overhead.

The capture/replay machinery is mode-agnostic and lives in the base class here: a subclass
supplies only its persistent buffers (``__init__``), the per-step body (``_step``), how a
bucket's inputs load into the buffers (``_load``), the loop-state reset (``_reset``), and the
result (``_result``).

Why intra-capture kernel allocations are fine
---------------------------------------------
The value+grad kernels (``overlap_score_grad_se3_batch`` etc.) allocate their output
tensors internally each call. That is *not* a capture blocker: under ``torch.cuda.graph``
those allocations come from the graph's private memory pool and get fixed addresses reused
on every replay. The two things that DO break capture -- and that every ``_step`` must
avoid -- are (1) host syncs / data-dependent control flow inside the captured step (the
blocked early-stop's ``.item()`` sync therefore happens OUTSIDE the graph, between replays)
and (2) Python rebinding of loop-carried tensors (so a ``_step`` must not use the functional
``_update_best``, which returns new tensors; it updates ``best``/``best_q``/``best_t`` in
place via ``where(..., out=)``). Autotune/JIT must also be warmed BEFORE capture --
``capture()`` does three warmup steps in a side stream for exactly this.
"""
from __future__ import annotations

from collections import OrderedDict
import torch

# --- compute-aware P-cap ----------------------------------------------------------------
# The graph's win is launch-bound: its per-step launch savings are ~fixed, while its cost
# grows with the per-row kernel work. So the P at which it stops beating eager scales as
# budget / (per-row work): cap = clamp(BUDGET // (N_pad*M_pad), MIN, CEIL) graphs each
# mode/bucket only where it wins and falls back to the eager loop (the baseline -- never a
# regression) where it would lose. One formula therefore covers the light-cloud and
# heavy-cloud splits that share a kernel (vol/surf, vol_esp/surf_esp) automatically.
# CEIL is a fixed per-graph memory ceiling on P (buffers are ~1KB/row, so 256k rows is a
# ~270MB graph); lower it on memory-constrained GPUs.
_GRAPH_WORK_BUDGET = 300_000_000
_GRAPH_CAP_CEIL = 262144
_GRAPH_CAP_MIN = 2000


def graph_cap(work, budget=None):
    """Max pose-rows P to graph for a bucket whose per-row kernel work is ``work`` (e.g.
    N_pad*M_pad). Below this the graph wins; above it the gating falls back to eager."""
    b = _GRAPH_WORK_BUDGET if budget is None else budget
    return max(_GRAPH_CAP_MIN, min(_GRAPH_CAP_CEIL, int(b) // max(int(work), 1)))
# Blocked early-stop for the replay loop: check best.max() every _GRAPH_ES_BLOCK replays (one
# host sync per block, NOT per step) and stop on the same patience/tol schedule as the eager
# loop. The blocked early-stop MUST reproduce the eager schedule: otherwise the graph runs a
# fixed step count, OVER-RUNS the eager early-stop, and the compute-bound modes (surf,
# esp_combo, pharm at large batch) regress on the extra kernel time.
_GRAPH_ES_BLOCK = 5
# Extra early-stop margin, in blocks, ADDED to the eager patience for the graph replay loop.
# The multi-basin modes (surf, surf_esp) land in different near-equal optima under tiny
# numerical perturbations, so they need this margin for the graph never to score below eager.
_GRAPH_ES_MARGIN = 2

# Process-wide LRU cache of captured graphs, keyed by (device, mode, shapes, P, steps, params).
# It MUST stay bounded: each cached graph pins persistent GPU buffers (~P*buffers bytes), so an
# unbounded cache lets a long/large workload accumulate GPU memory until fragmentation/OOM makes
# new captures fail and the driver silently falls back to eager. Evicting the least-recently-used
# graph frees its buffers; _GRAPH_CACHE_MAX caps how many live at once.
_FINE_GRAPH_CACHE: "OrderedDict" = OrderedDict()
_GRAPH_CACHE_MAX = 24


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
        if self.es_patience:
            # Mirror the eager early-stop SCHEDULE exactly, not just its patience/tol. Eager
            # checks best.max() at steps 0, 5, 10, ... and seeds its baseline `prev` with the
            # STEP-0 value (the best of the seed poses). `best` is a max over past poses, so
            # after one replay it equals eager's step-0 best: replay one step, seed `prev`
            # from it, then check every es_block replays. The graph then stops at the same
            # step as eager and `best` is equivalent.
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
