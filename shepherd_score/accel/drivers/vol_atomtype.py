"""``vol_atomtype`` driver entry points (shape + element-identity categorical overlap on the
strict-heavy centres, through the directionless colour kernel with element tables)."""
from __future__ import annotations

import torch

from ..kernels.dispatch import pharm_color_score_grad_se3_batch
from ..channels import ATOMTYPE_PAD as _ATOMTYPE_PAD  # noqa: F401  (re-export)
from ._common import apply_se3_transform
from ._shim import batch, run
from .terms import build_element_tables  # noqa: F401  (re-export)


@torch.no_grad()
def _atomtype_self_overlap(pos, labels, N_real, tables):
    """Pose-invariant element-identity self-overlap via the colour kernel at the identity pose."""
    al, Ks, cats = tables
    P = pos.shape[0]
    eye = torch.tensor([[1., 0., 0., 0.]], device=pos.device).expand(P, 4)
    zero = torch.zeros(P, 3, device=pos.device)
    O, _, _ = pharm_color_score_grad_se3_batch(pos, pos, eye, zero, labels, labels, al, Ks, cats,
                                               N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return O


def coarse_fine_vol_atomtype_align_many(
        centers_1, centers_2, type_pos_1, type_pos_2, labels_1, labels_2, VAA=None, VBB=None, *,
        alpha=0.81, atomtype_weight=0.5, num_seeds=50, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None, N_real_type=None, M_real_type=None,
        early_stop_patience=2, early_stop_tol=1e-5, tables=None, seeds=None):
    c = batch(centers_1, centers_2, N_real_centers, M_real_centers)
    tp = batch(type_pos_1, type_pos_2, N_real_type, M_real_type)
    chans = {"atoms": c, "type_pos": tp, "atomlabels": batch(labels_1, labels_2, tp.n_real, tp.m_real)}
    return run("vol_atomtype", chans, alpha=alpha, atomtype_weight=atomtype_weight,
               num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
               early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol, seeds=seeds)


def fast_optimize_vol_atomtype_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_type_pos_batch, fit_type_pos_batch,
        ref_labels_batch, fit_labels_batch, *, alpha=0.81, atomtype_weight=0.5,
        N_real_centers=None, M_real_centers=None, N_real_type=None, M_real_type=None, topk=30,
        steps_fine=100, lr=0.075, num_seeds=50, seeds=None):
    scores, q, t = coarse_fine_vol_atomtype_align_many(
        ref_centers_batch, fit_centers_batch, ref_type_pos_batch, fit_type_pos_batch,
        ref_labels_batch, fit_labels_batch, alpha=alpha, atomtype_weight=atomtype_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr, N_real_centers=N_real_centers,
        M_real_centers=M_real_centers, N_real_type=N_real_type, M_real_type=M_real_type,
        seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
