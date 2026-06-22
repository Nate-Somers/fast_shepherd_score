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

__all__ = ["has_triton", "align_multi_gpu", "MultiGPUAligner"]


def __getattr__(name):
    # Lazy re-export of the multi-GPU driver. Kept lazy so that importing the
    # acceleration internals (e.g. ``shepherd_score.accel.batch`` during
    # ``container`` import) never eagerly pulls in ``multi_gpu`` -> ``container``.
    if name in ("align_multi_gpu", "MultiGPUAligner"):
        from . import multi_gpu
        return getattr(multi_gpu, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
