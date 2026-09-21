"""``vol_esp_tversky`` / ``surf_esp_tversky`` driver entry points: the charge-weighted overlap
under the Tversky reduction. ``lam`` is passed to the kernel as given (the surface aligner
applies ``LAM_SCALING`` before calling)."""
from __future__ import annotations

from .esp import _overlap_in_chunks_esp, _self_overlap_esp_chunks  # noqa: F401  (re-export)
from ._shim import batch, run


def coarse_fine_esp_tversky_align_many(A_batch, B_batch, CA_batch, CB_batch, VAA=None, VBB=None, *,
                                       alpha: float = 0.81, lam: float = 0.1,
                                       tversky_alpha: float = 0.95, tversky_beta: float = 0.05,
                                       num_seeds: int = 50, steps_fine: int = 100, lr: float = 0.075,
                                       N_real=None, M_real=None, early_stop_patience: int = 5,
                                       early_stop_tol: float = 1e-5, seeds=None,
                                       mode: str = "vol_esp_tversky"):
    b = batch(A_batch, B_batch, N_real, M_real)
    if mode == "surf_esp_tversky":
        chans = {"surf": b, "surf_esp": batch(CA_batch, CB_batch, b.n_real, b.m_real)}
    else:
        chans = {"heavy": b, "charges": batch(CA_batch, CB_batch, b.n_real, b.m_real)}
    return run(mode, chans, alpha=alpha, lam=lam, tversky_alpha=tversky_alpha,
               tversky_beta=tversky_beta, num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
               early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol, seeds=seeds)
