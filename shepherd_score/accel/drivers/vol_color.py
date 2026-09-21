"""``vol_color`` driver entry points (shape + directionless pharmacophore colour, joint gradient)."""
from __future__ import annotations

import torch

from ..kernels.dispatch import pharm_color_score_grad_se3_batch
from ._common import apply_se3_transform, quaternion_to_rotation_matrix  # noqa: F401
from ._shim import batch, run
from ..channels import CHANNELS

#: Padding type for pharmacophore slots (the 'Dummy' family, kernel category 3).
_PHARM_PAD_TYPE = CHANNELS["pharm_types"].pad


@torch.no_grad()
def _color_overlap(q, t, types_1, types_2, anchors_1, anchors_2, tables, N_real_ph, M_real_ph):
    """Directionless colour overlap (value only)."""
    al, Ks, cats = tables
    O, _, _ = pharm_color_score_grad_se3_batch(anchors_1, anchors_2, q, t, types_1, types_2,
                                               al, Ks, cats, N_real=N_real_ph, M_real=M_real_ph,
                                               NEED_GRAD=False)
    return O


def _chans(centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, N_real_centers,
           M_real_centers, N_real_pharm, M_real_pharm):
    c = batch(centers_1, centers_2, N_real_centers, M_real_centers)
    a = batch(anchors_1, anchors_2, N_real_pharm, M_real_pharm)
    return {"atoms": c, "pharm_ancs": a, "pharm_types": batch(ptype_1, ptype_2, a.n_real, a.m_real)}


def coarse_fine_vol_color_align_many(
        centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, VAA=None, VBB=None, *,
        alpha: float = 0.81, color_weight: float = 0.5, num_seeds: int = 50, trans_centers=None,
        trans_centers_real=None, num_repeats_per_trans: int = 10, topk: int = 30,
        steps_fine: int = 100, lr: float = 0.075, N_real_centers=None, M_real_centers=None,
        N_real_pharm=None, M_real_pharm=None, seeds=None, early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5, mode: str = "vol_color", **extra):
    chans = _chans(centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, N_real_centers,
                   M_real_centers, N_real_pharm, M_real_pharm)
    return run(mode, chans, alpha=alpha, color_weight=color_weight, num_seeds=num_seeds,
               steps_fine=steps_fine, lr=lr, early_stop_patience=early_stop_patience,
               early_stop_tol=early_stop_tol, seeds=seeds, trans_centers=trans_centers,
               trans_centers_real=trans_centers_real, num_repeats_per_trans=num_repeats_per_trans,
               topk=topk, **extra)


def fast_optimize_vol_color_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, *, alpha: float = 0.81, color_weight: float = 0.5, N_real_centers=None,
        M_real_centers=None, N_real_pharm=None, M_real_pharm=None, trans_centers_batch=None,
        trans_centers_real=None, num_repeats_per_trans: int = 10, topk: int = 30,
        steps_fine: int = 100, lr: float = 0.075, num_seeds: int = 50, seeds=None):
    """Batched vol_color alignment: ``(aligned_fit_centers, q, t, scores)``."""
    scores, q, t = coarse_fine_vol_color_align_many(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, alpha=alpha, color_weight=color_weight, num_seeds=num_seeds,
        trans_centers=trans_centers_batch, trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers, N_real_pharm=N_real_pharm,
        M_real_pharm=M_real_pharm, seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
