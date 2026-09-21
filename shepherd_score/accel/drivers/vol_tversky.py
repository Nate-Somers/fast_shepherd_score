"""``vol_tversky`` / ``surf_tversky`` driver entry points: the shape overlap under the asymmetric
Tversky reduction ``AB / (k*AB + C)``, ``C = ta*AA + tb*BB``, ``k = 1 - ta - tb``."""
from __future__ import annotations

from .shape import _overlap_in_chunks, _self_overlap_in_chunks  # noqa: F401  (re-export)
from ._shim import batch, run


def coarse_fine_align_many_tversky(A_batch, B_batch, VAA=None, VBB=None, *, alpha: float = 0.81,
                                   tversky_alpha: float = 0.95, tversky_beta: float = 0.05,
                                   num_seeds: int = 50, steps_fine: int = 100, lr: float = 0.075,
                                   N_real=None, M_real=None, early_stop_patience: int = 2,
                                   early_stop_tol: float = 1e-5, seeds=None,
                                   mode: str = "vol_tversky"):
    ch = "surf" if mode == "surf_tversky" else "atoms"
    return run(mode, {ch: batch(A_batch, B_batch, N_real, M_real)}, alpha=alpha,
               tversky_alpha=tversky_alpha, tversky_beta=tversky_beta, num_seeds=num_seeds,
               steps_fine=steps_fine, lr=lr, early_stop_patience=early_stop_patience,
               early_stop_tol=early_stop_tol, seeds=seeds)
