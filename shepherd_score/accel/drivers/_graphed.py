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
import torch

# Shared env knobs (single source of truth; shape.py and every other driver import these).
# Disable the whole graph fast path with FINE_CUDA_GRAPHS=0.
_FINE_GRAPHS = os.environ.get("FINE_CUDA_GRAPHS", "1") != "0"
# Max pose-rows P for a single captured graph. Above this the eager loop is used (it is
# compute-bound there, so the graph stops helping) UNLESS the mode opts into chunked replay
# via run_graphed_chunked (the launch-bound modes, where it keeps helping at large batch).
_GRAPH_MAX_P = int(os.environ.get("FINE_GRAPH_MAX_P", 16000))
# Fixed replay count. The eager loop early-stops ~50 steps on the fast-converging benchmark;
# a fixed-50 capture is parity-safe for surf/vol. Slower modes may raise this (validated).
_GRAPH_STEPS = int(os.environ.get("FINE_GRAPH_STEPS", 50))

# Process-wide cache of captured graphs, keyed by (device, mode, shapes, P, steps, params).
_FINE_GRAPH_CACHE: dict = {}


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
        """Load this bucket, reset loop state, replay the captured step ``steps`` times."""
        self._load(*inputs); self._reset()
        for _ in range(self.steps):
            self.graph.replay()
        return self._result()


def run_graphed(make, key, inputs):
    """Fetch (or build+capture) the graph for ``key`` and run this bucket through it.

    ``make`` is a zero-arg factory for the subclass instance (called only on cache miss);
    ``inputs`` is the tuple forwarded to ``capture``/``run`` (and thence ``_load``). One
    captured graph serves every bucket of the same key (same shapes/P/steps/params).
    """
    gf = _FINE_GRAPH_CACHE.get(key)
    if gf is None:
        gf = make()
        gf.capture(*inputs)
        _FINE_GRAPH_CACHE[key] = gf
    return gf.run(*inputs)


def run_graphed_chunked(make, key_for, inputs, *, P, max_p, slice_inputs, cat_results):
    """Cap-lift: drive an arbitrarily large P through graphs of size <= ``max_p``.

    Rows (poses) are independent, so slicing the pose dimension into chunks and running each
    chunk through its own (cached) graph is exactly equivalent to running all P at once, as
    long as results are concatenated back in row order. At most two distinct chunk sizes
    occur (``max_p`` and the remainder), so at most two graphs are captured per key.

      * ``key_for(p_chunk)`` -> cache key for a chunk of that size
      * ``make(p_chunk)``    -> factory for a chunk-sized subclass instance
      * ``slice_inputs(inputs, s, e)`` -> the inputs tuple sliced to rows [s, e)
      * ``cat_results(list_of_(best,bq,bt))`` -> the concatenated (best, bq, bt)
    """
    if P <= max_p:
        return run_graphed(lambda: make(P), key_for(P), inputs)
    out = []
    for s in range(0, P, max_p):
        e = min(s + max_p, P)
        pc = e - s
        sub = slice_inputs(inputs, s, e)
        out.append(run_graphed(lambda pc=pc: make(pc), key_for(pc), sub))
    return cat_results(out)
