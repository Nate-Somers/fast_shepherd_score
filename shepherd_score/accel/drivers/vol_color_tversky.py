"""``vol_color_tversky`` driver entry points (vol_color with Tversky on both channels)."""
from __future__ import annotations

from ._common import apply_se3_transform
from .vol_color import coarse_fine_vol_color_align_many, _color_overlap, _PHARM_PAD_TYPE  # noqa: F401


def coarse_fine_vol_color_tversky_align_many(
        centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, VAA=None, VBB=None, *,
        alpha=0.81, color_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05, num_seeds=50,
        steps_fine=100, lr=0.075, N_real_centers=None, M_real_centers=None, N_real_pharm=None,
        M_real_pharm=None, early_stop_patience=2, early_stop_tol=1e-5, tables=None, seeds=None):
    return coarse_fine_vol_color_align_many(
        centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, alpha=alpha,
        color_weight=color_weight, num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers, N_real_pharm=N_real_pharm,
        M_real_pharm=M_real_pharm, early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol, seeds=seeds, mode="vol_color_tversky",
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)


def fast_optimize_vol_color_tversky_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, *, alpha=0.81, color_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05,
        N_real_centers=None, M_real_centers=None, N_real_pharm=None, M_real_pharm=None, topk=30,
        steps_fine=100, lr=0.075, num_seeds=50, seeds=None):
    scores, q, t = coarse_fine_vol_color_tversky_align_many(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, alpha=alpha, color_weight=color_weight, tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta, num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers, N_real_pharm=N_real_pharm,
        M_real_pharm=M_real_pharm, seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
