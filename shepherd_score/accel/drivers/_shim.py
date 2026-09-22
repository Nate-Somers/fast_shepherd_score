"""Helpers for the per-mode driver modules, which are thin entry points over ``engine``.

The driver modules keep their ``coarse_fine_*`` / ``fast_optimize_*_batch`` signatures for
callers that hand padded tensors straight to a driver; each builds the engine's channel dict
and calls :func:`run`.
"""
from __future__ import annotations

import torch

from .._modes import SPECS
from .engine import Batch, align


def batch(ref, fit, n_real=None, m_real=None) -> Batch:
    """A :class:`Batch` with real counts defaulting to the padded widths."""
    B = ref.shape[0]
    if n_real is None:
        n_real = ref.new_full((B,), ref.shape[1], dtype=torch.int32)
    if m_real is None and fit is not None:
        m_real = fit.new_full((B,), fit.shape[1], dtype=torch.int32)
    return Batch(ref, fit, n_real, m_real)


def run(mode, chans, *, num_seeds=None, steps_fine=100, lr=0.075, early_stop_patience=None,
        early_stop_tol=1e-5, seeds=None, trans_centers=None, trans_centers_real=None,
        num_repeats_per_trans=10, topk=30, **params):
    """Run the engine for ``mode`` on ``chans`` with the driver-level defaults
    (``num_seeds=50, steps_fine=100, lr=0.075``, the mode's patience)."""
    spec = SPECS[mode]
    p = dict(spec.params)
    p.update(params)
    p["lr"] = float(lr)
    return align(spec, chans, params=p,
                 num_seeds=int(spec.seeds if num_seeds is None else num_seeds),
                 steps_fine=int(steps_fine), lr=float(lr),
                 early_stop_patience=int(spec.patience if early_stop_patience is None
                                         else early_stop_patience),
                 early_stop_tol=float(early_stop_tol), seeds=seeds,
                 trans_centers=trans_centers, trans_centers_real=trans_centers_real,
                 num_repeats_per_trans=num_repeats_per_trans, topk=topk)


def se3_of(q, t):
    """(4,4) SE(3) matrix from one unit quaternion and translation."""
    from ._common import quaternion_to_rotation_matrix
    SE3 = torch.eye(4, device=q.device, dtype=q.dtype)
    SE3[:3, :3] = quaternion_to_rotation_matrix(q)
    SE3[:3, 3] = t
    return SE3
