"""``vol_pharm`` driver entry points (shape + DIRECTIONAL pharmacophore overlap, joint gradient)."""
from __future__ import annotations

import torch

from ..kernels.dispatch import pharm_grad_dq_se3_batch
from ._common import apply_se3_transform
from ._shim import batch, run
from .vol_color import _PHARM_PAD_TYPE  # noqa: F401  (same pad type)


@torch.no_grad()
def _pharm_self_overlap(anc, vec, labels, N_real, tables):
    """Pose-invariant DIRECTIONAL pharmacophore self-overlap via the pharm kernel at identity."""
    al, Ks, cats = tables
    P = anc.shape[0]
    q0 = torch.zeros(P, 4, device=anc.device, dtype=anc.dtype); q0[:, 0] = 1.0
    z = torch.zeros(P, 3, device=anc.device, dtype=anc.dtype)
    O, _, _ = pharm_grad_dq_se3_batch(q0, z, labels, labels, anc, anc, vec, vec, al, Ks, cats,
                                      N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return O


def coarse_fine_vol_pharm_align_many(
        centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
        VAA=None, VBB=None, *, alpha=0.81, color_weight=0.5, num_seeds=50, steps_fine=100,
        lr=0.075, N_real_centers=None, M_real_centers=None, N_real_pharm=None, M_real_pharm=None,
        early_stop_patience=2, early_stop_tol=1e-5, tables=None, seeds=None):
    c = batch(centers_1, centers_2, N_real_centers, M_real_centers)
    a = batch(anchors_1, anchors_2, N_real_pharm, M_real_pharm)
    chans = {"atoms": c, "pharm_ancs": a, "pharm_vecs": batch(vectors_1, vectors_2, a.n_real, a.m_real),
             "pharm_types": batch(ptype_1, ptype_2, a.n_real, a.m_real)}
    return run("vol_pharm", chans, alpha=alpha, color_weight=color_weight, num_seeds=num_seeds,
               steps_fine=steps_fine, lr=lr, early_stop_patience=early_stop_patience,
               early_stop_tol=early_stop_tol, seeds=seeds)


def fast_optimize_vol_pharm_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, ref_vecs_batch, fit_vecs_batch, *, alpha=0.81, color_weight=0.5,
        N_real_centers=None, M_real_centers=None, N_real_pharm=None, M_real_pharm=None, topk=30,
        steps_fine=100, lr=0.075, num_seeds=50, seeds=None):
    scores, q, t = coarse_fine_vol_pharm_align_many(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch, ref_ancs_batch,
        fit_ancs_batch, ref_vecs_batch, fit_vecs_batch, alpha=alpha, color_weight=color_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr, N_real_centers=N_real_centers,
        M_real_centers=M_real_centers, N_real_pharm=N_real_pharm, M_real_pharm=M_real_pharm,
        seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
