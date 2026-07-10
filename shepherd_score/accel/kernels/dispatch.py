"""Device-driven kernel dispatch for the batched coarse-to-fine aligners.

Each value+gradient kernel the ``fast_*_se3`` drivers call has two interchangeable
implementations with identical signatures:

  * a **Triton** kernel (in ``shepherd_score.score.*_triton``) that runs on **CUDA**
    tensors, and
  * a **numba** kernel (in :mod:`cpu_overlap`) that runs on **CPU** tensors.

Historically each driver chose one implementation *once*, at import time
(``try: import <triton kernel>; except ImportError: import <numba kernel>``), and froze
that choice for the life of the process. That made the kernel a property of the
*machine* rather than of the *data*: a process that successfully imported the Triton
kernels could never run the numba path, even for CPU tensors -- so ``backend="numba"``
had to refuse on a CUDA+Triton box.

This module removes that freeze. It exposes the same kernel names, but each one is a
thin wrapper that routes a *call* to the implementation matching the **device of its
tensor arguments**. The drivers import these wrappers in place of the raw kernels and
are otherwise unchanged. Kernel choice is now a pure function of where the data lives,
which is also the only valid choice (a Triton kernel requires CUDA tensors; the numba
kernel requires CPU tensors). So CUDA tensors dispatch to Triton and CPU tensors to
numba **within the same process**.

The Triton source modules are imported **lazily** -- only the first time a CUDA tensor
is dispatched -- so importing this module on a CPU-only box (where Triton is not
installed) never touches them.
"""
from __future__ import annotations

import torch


def has_triton() -> bool:
    """True if the Triton package is importable (i.e. the GPU kernels can be loaded)."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


# Informational flag for external consumers (tests, diagnostics). Kernel selection no
# longer depends on it -- it is now per-call and device-driven (see module docstring).
# Exposed LAZILY via module __getattr__ so merely importing this module (which every
# accel/container import does, and every spawned CPU-pool / multi-GPU worker) does NOT
# eagerly ``import triton`` -- that cost is only paid the first time a CUDA tensor is
# actually dispatched (or someone reads ``dispatch._HAS_TRITON``). Keeps CPU-only / numba
# runs triton-free, as the module docstring promises.
def __getattr__(name):
    if name == "_HAS_TRITON":
        return has_triton()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Lazily-imported, cached source modules, keyed by a short tag.
_MODS: dict = {}


def _mod(tag: str):
    m = _MODS.get(tag)
    if m is None:
        if tag == "cpu":
            from . import cpu as m
        elif tag == "shape":
            from . import shape_triton as m
        elif tag == "esp":
            from . import esp_triton as m
        elif tag == "pharm":
            from . import pharm_triton as m
        else:  # pragma: no cover - defensive
            raise KeyError(tag)
        _MODS[tag] = m
    return m


def _args_on_cuda(args, kwargs) -> bool:
    """True if the first ``torch.Tensor`` among the call's arguments lives on CUDA."""
    for a in args:
        if torch.is_tensor(a):
            return a.is_cuda
    for a in kwargs.values():
        if torch.is_tensor(a):
            return a.is_cuda
    # No tensor argument found -> fall back to machine capability. The kernels below
    # all take tensors as their leading arguments, so this branch is not expected.
    return torch.cuda.is_available()


# Cache of resolved (name, source-tag) -> concrete function, so dispatch costs one dict
# lookup after the first call for each (kernel, device) combination.
_RESOLVED: dict = {}


def _make(name: str, triton_tag: str):
    """Build a wrapper for kernel ``name`` that dispatches to the Triton module
    ``triton_tag`` for CUDA tensors and to ``cpu_overlap`` for CPU tensors."""

    def _call(*args, **kwargs):
        tag = triton_tag if _args_on_cuda(args, kwargs) else "cpu"
        key = (name, tag)
        fn = _RESOLVED.get(key)
        if fn is None:
            fn = getattr(_mod(tag), name)
            _RESOLVED[key] = fn
        return fn(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = name
    _call.__doc__ = (
        f"Device-dispatched kernel '{name}': Triton ({triton_tag}) on CUDA tensors, "
        f"numba (cpu_overlap) on CPU tensors."
    )
    return _call


# --- shape kernels (gaussian_overlap_triton <-> cpu_overlap) ------------------
overlap_score_grad_se3_batch = _make("overlap_score_grad_se3_batch", "shape")
fused_adam_qt = _make("fused_adam_qt", "shape")
fused_adam_qt_with_tangent_proj = _make("fused_adam_qt_with_tangent_proj", "shape")
_batch_self_overlap = _make("_batch_self_overlap", "shape")
fused_surf_step_batch = _make("fused_surf_step_batch", "shape")

# --- ESP kernels (gaussian_overlap_esp_triton <-> cpu_overlap) ----------------
overlap_score_grad_esp_se3_batch = _make("overlap_score_grad_esp_se3_batch", "esp")
_batch_self_overlap_esp = _make("_batch_self_overlap_esp", "esp")
# ShaEP ESP surface-comparison (esp_combo), value-only fused reduction.
esp_comparison_batch = _make("esp_comparison_batch", "esp")

# --- pharmacophore kernel (pharmacophore_grad_triton <-> cpu_overlap) ---------
pharm_score_grad_se3_batch = _make("pharm_score_grad_se3_batch", "pharm")
# Directional pharm value+QUATERNION-grad kernel (pharm mode, in-register dQ).
pharm_grad_dq_se3_batch = _make("pharm_grad_dq_se3_batch", "pharm")
# Directionless "color" value+quaternion-grad kernel (vol_color).
pharm_color_score_grad_se3_batch = _make("pharm_color_score_grad_se3_batch", "pharm")
