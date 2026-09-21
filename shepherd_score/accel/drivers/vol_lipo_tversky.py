"""``vol_lipo_tversky`` driver entry points (vol_lipo with Tversky on both channels)."""
from __future__ import annotations

from ._common import apply_se3_transform
from .vol_lipo import coarse_fine_vol_lipo_align_many


def coarse_fine_vol_lipo_tversky_align_many(
        centers_1, centers_2, lipo_pos_1, lipo_pos_2, lipo_1, lipo_2, VAA=None, VBB=None, *,
        alpha=0.81, lam=0.1, lipo_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05, num_seeds=50,
        steps_fine=100, lr=0.075, N_real_centers=None, M_real_centers=None, N_real_lipo=None,
        M_real_lipo=None, early_stop_patience=2, early_stop_tol=1e-5, seeds=None):
    return coarse_fine_vol_lipo_align_many(
        centers_1, centers_2, lipo_pos_1, lipo_pos_2, lipo_1, lipo_2, alpha=alpha, lam=lam,
        lipo_weight=lipo_weight, num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers, N_real_lipo=N_real_lipo,
        M_real_lipo=M_real_lipo, early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol, seeds=seeds, mode="vol_lipo_tversky",
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)


def fast_optimize_vol_lipo_tversky_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, *, alpha=0.81, lam=0.1, lipo_weight=0.5,
        tversky_alpha=0.95, tversky_beta=0.05, N_real_centers=None, M_real_centers=None,
        N_real_lipo=None, M_real_lipo=None, topk=30, steps_fine=100, lr=0.075, num_seeds=50,
        seeds=None):
    scores, q, t = coarse_fine_vol_lipo_tversky_align_many(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta, num_seeds=num_seeds,
        steps_fine=steps_fine, lr=lr, N_real_centers=N_real_centers,
        M_real_centers=M_real_centers, N_real_lipo=N_real_lipo, M_real_lipo=M_real_lipo,
        seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
