"""GPU/CPU acceleration for batch molecular alignment.

Layered, bottom-up:

* :mod:`~shepherd_score.accel.kernels` -- the raw compute cores: hand-written
  Triton GPU kernels and their op-for-op ``numba`` CPU mirrors, behind a
  per-call device dispatcher (:mod:`~shepherd_score.accel.kernels.dispatch`).
* :mod:`~shepherd_score.accel.drivers` -- batched coarse-to-fine SE(3)
  optimizers, one module per alignment mode.
* :mod:`~shepherd_score.accel.batch` / :mod:`~shepherd_score.accel.cpu_pool` /
  :mod:`~shepherd_score.accel.multi_gpu` -- orchestration (size bucketing,
  GPU-memory sub-batching, multi-core CPU pool, multi-GPU data parallelism).

The primary public entry point is the ``backend=`` argument on
:meth:`shepherd_score.container.MoleculePairBatch.align_with_*` (``"jax"`` default,
``"triton"`` GPU, ``"numba"`` CPU). For large multi-GPU screens, the explicit
data-parallel driver is exported here (and from
:mod:`shepherd_score.container`) as :func:`align_multi_gpu` / :class:`MultiGPUAligner`.
"""
from .kernels.dispatch import has_triton

__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner", "clear_caches"]


def clear_caches() -> None:
    """Free the process-global accel caches and return their memory.

    The batched aligners retain a few module-level caches for the process lifetime:
    padded per-``(device, N_pad, M_pad)`` workspaces, integer index buffers, the learned
    per-pair GPU-memory footprint table, and (on CUDA) the captured-CUDA-graph LRU. They
    are keyed on tensor shapes, so a long-lived process that sees many distinct molecule
    sizes accumulates device memory it never releases. Call this between independent
    workloads (e.g. successive large screens) to reclaim it. Cheap to call; the caches
    simply repopulate on the next alignment.
    """
    from .batch import aligners as _al
    _al._ALIGN_WORKSPACES.clear()
    _al._INT_BUFFER_CACHE.clear()
    from .batch import _pad
    _pad._PAIR_FOOTPRINT_BYTES.clear()
    from .drivers._graphed import reset_graph_cache
    reset_graph_cache()


def __getattr__(name):
    # Lazy re-export of the multi-GPU driver. Kept lazy so that importing the
    # acceleration internals (e.g. ``shepherd_score.accel.batch`` during
    # ``container`` import) never eagerly pulls in ``multi_gpu`` -> ``container``.
    if name in ("align_multi_gpu", "MultiGPUAligner"):
        from . import multi_gpu
        return getattr(multi_gpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
