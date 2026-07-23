# shepherd_score/accel/drivers/vol_atomtype.py
# Fast batched ``vol_atomtype`` alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  element-identity CATEGORICAL overlap.
#
#   score = (1-w)*shape_Tanimoto + w*atomtype_Tanimoto      (w = atomtype_weight)
#
# NO new kernel: the element-identity channel REUSES the directionless colour kernel
# ``pharm_color_score_grad_se3_batch`` -- the same kernel vol_color uses -- fed ELEMENT-indexed
# lookup tables (every element is an isotropic point Gaussian, category 0) and the per-heavy-atom
# ATOMIC NUMBER as its "type". The kernel contributes an overlap term only for pairs of the SAME
# type (``ityp == jtyp``), i.e. same-element atoms, exactly the categorical overlap of the reference
# ``score.atomtype_scoring.get_overlap_atomtype``. Two INDEPENDENT point sets like vol_lipo: the
# shape channel on ``atom_pos`` (RemoveHs) and the identity channel on the TRUE-heavy centres
# (own counts). Both fit sets transform rigidly under the same quaternion; the combined descent
# gradient is the per-channel-scaled weighted sum. Matches the per-pair reference
# ``alignment._torch.optimize_vol_atomtype_overlay``.

from __future__ import annotations

import math
import torch
from typing import Optional, Tuple

from ..kernels.dispatch import (
    fused_adam_qt_with_tangent_proj, pharm_color_score_grad_se3_batch)
from ._common import (
    batched_seeds_torch, apply_se3_transform, quaternion_to_rotation_matrix, _update_best)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp_combo import _overlap_in_chunks_volumetric, _self_overlap_chunks

# Padding label for the element channel: atomic number 0 (no real element), given category 3 so
# the colour kernel skips it -- the analogue of vol_color's 'Dummy' pad type.
_ATOMTYPE_PAD = 0


def build_element_tables(device, dtype, max_z: int = 100, alpha: float = 0.81):
    """Element-indexed (alphas, Ks, cats) for the colour kernel: every element is an isotropic
    point Gaussian of width ``alpha`` (category 0); the pad label (Z=0) is category 3 (skipped).
    Indexed directly by atomic number. K = (pi/(2a))^1.5 mirrors ``build_lookup_tables`` (a common
    factor across types under a uniform alpha, so it cancels in the Tanimoto)."""
    n = max_z + 1
    alphas = torch.full((n,), float(alpha), device=device, dtype=dtype)
    Ks = torch.full((n,), (math.pi / (2.0 * alpha)) ** 1.5, device=device, dtype=dtype)
    cats = torch.zeros(n, device=device, dtype=torch.long)
    cats[_ATOMTYPE_PAD] = 3
    return alphas, Ks, cats


@torch.no_grad()
def _atomtype_self_overlap(pos, labels, N_real, tables):
    """Pose-invariant element-identity self-overlap via the colour kernel at the identity pose."""
    al, Ks, cats = tables
    P = pos.shape[0]
    eye = torch.tensor([[1., 0., 0., 0.]], device=pos.device).expand(P, 4)
    zero = torch.zeros(P, 3, device=pos.device)
    O, _, _ = pharm_color_score_grad_se3_batch(
        pos, pos, eye, zero, labels, labels, al, Ks, cats,
        N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return O


def _vat_overlaps(A, B, TA, TB, LA, LB, q, t, tables, alpha, Nc, Mc, Na, Ma):
    """Both channels' value+grad: shape overlap on (A,B) + element-identity colour overlap on the
    heavy centres (TA,TB) with atomic-number labels (LA,LB). Always the two separate kernels (the
    fused vol_color kernel is pharm-table-specific)."""
    al, Ks, cats = tables
    VAB_s, dQ_s, dT_s = _overlap_in_chunks_volumetric(A, B, q, t, alpha=alpha, N_real=Nc, M_real=Mc)
    O_c, dQ_c, dT_c = pharm_color_score_grad_se3_batch(
        TA, TB, q, t, LA, LB, al, Ks, cats, N_real=Na, M_real=Ma, NEED_GRAD=True)
    return VAB_s, dQ_s, dT_s, O_c, dQ_c, dT_c


class _GraphedFineVolAtomtype(_GraphedFineBase):
    """CUDA-graph fine loop for vol_atomtype (shape + element-identity colour, Tanimoto each)."""

    def __init__(self, N_pad, M_pad, F_pad, G_pad, P, steps, alpha, atomtype_weight, lr,
                 tables, device):
        self.alpha = float(alpha); self.w = float(atomtype_weight); self.lr = float(lr)
        self.al, self.Ks, self.cats = tables
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        il = lambda *s: torch.empty(*s, device=device, dtype=torch.int64)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.TA = f(P, F_pad, 3); self.TB = f(P, G_pad, 3)
        self.LA = il(P, F_pad); self.LB = il(P, G_pad)        # atomic-number labels
        self.Nr = i(P); self.Mr = i(P); self.Na = i(P); self.Ma = i(P)
        self.norm_s = f(P); self.norm_c = f(P)
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        self.denom_s = f(P); self.d2s = f(P); self.shape_sim = f(P); self.scale_s = f(P)
        self.denom_c = f(P); self.d2c = f(P); self.color_sim = f(P); self.scale_c = f(P)
        self.score = f(P); self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3); self.tmpq = f(P, 4); self.tmpt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB, dQs, dTs, O_c, dQc, dTc = _vat_overlaps(
            self.A, self.B, self.TA, self.TB, self.LA, self.LB, self.q, self.t,
            (self.al, self.Ks, self.cats), self.alpha, self.Nr, self.Mr, self.Na, self.Ma)
        torch.sub(self.norm_s, VAB, out=self.denom_s)
        torch.div(VAB, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.norm_s, self.d2s, out=self.scale_s)
        torch.sub(self.norm_c, O_c, out=self.denom_c)
        torch.div(O_c, self.denom_c, out=self.color_sim)
        torch.mul(self.denom_c, self.denom_c, out=self.d2c)
        torch.div(self.norm_c, self.d2c, out=self.scale_c)
        torch.mul(self.shape_sim, 1.0 - self.w, out=self.score)
        self.score.add_(self.color_sim, alpha=self.w)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        torch.mul(dQs, self.scale_s.unsqueeze(1), out=self.gq); self.gq.mul_(-(1.0 - self.w))
        torch.mul(dQc, self.scale_c.unsqueeze(1), out=self.tmpq); self.tmpq.mul_(-self.w)
        self.gq.add_(self.tmpq)
        torch.mul(dTs, self.scale_s.unsqueeze(1), out=self.gt); self.gt.mul_(-(1.0 - self.w))
        torch.mul(dTc, self.scale_c.unsqueeze(1), out=self.tmpt); self.tmpt.mul_(-self.w)
        self.gt.add_(self.tmpt)
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, TA, TB, LA, LB, Nr, Mr, Na, Ma, norm_s, norm_c, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.TA.copy_(TA); self.TB.copy_(TB); self.LA.copy_(LA.to(torch.int64)); self.LB.copy_(LB.to(torch.int64))
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Na.copy_(Na.to(torch.int32)); self.Ma.copy_(Ma.to(torch.int32))
        self.norm_s.copy_(norm_s); self.norm_c.copy_(norm_c)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_vol_atomtype(A_k, B_k, TA_k, TB_k, LA_k, LB_k, N_k, M_k, Na_k, Ma_k,
                              norm_s, norm_c, q_seed, t_seed, tables, alpha, atomtype_weight, lr,
                              steps, N_pad, M_pad, F_pad, G_pad, P, es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_atomtype", N_pad, M_pad, F_pad, G_pad, P, steps,
           round(float(alpha), 4), round(float(atomtype_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolAtomtype(N_pad, M_pad, F_pad, G_pad, P, steps, alpha,
                                        atomtype_weight, lr, tables, A_k.device),
        key, (A_k, B_k, TA_k, TB_k, LA_k, LB_k, N_k, M_k, Na_k, Ma_k,
              norm_s, norm_c, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_atomtype_align_many(
        centers_1, centers_2, type_pos_1, type_pos_2, labels_1, labels_2, VAA, VBB, *,
        alpha=0.81, atomtype_weight=0.5, num_seeds=50, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None, N_real_type=None, M_real_type=None,
        early_stop_patience=2, early_stop_tol=1e-5, tables=None):
    """Batched vol_atomtype alignment (seed path). ``VAA``/``VBB`` are shape self-overlaps; the
    element-identity self-overlaps are computed here via the colour kernel."""
    device = centers_1.device
    dtype = centers_1.dtype
    BATCH = centers_1.shape[0]
    N_pad_cent, M_pad_cent = centers_1.shape[1], centers_2.shape[1]
    F_pad, G_pad = type_pos_1.shape[1], type_pos_2.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_type is None:
        N_real_type = type_pos_1.new_full((BATCH,), F_pad, dtype=torch.int32)
    if M_real_type is None:
        M_real_type = type_pos_2.new_full((BATCH,), G_pad, dtype=torch.int32)
    if tables is None:
        tables = build_element_tables(device, dtype, alpha=alpha)

    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_k = quats.clone().reshape(-1, 4).contiguous()
    t_k = t_seeds.clone().reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_cent, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_cent, 3)
    type_pos_1_k = type_pos_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, F_pad, 3)
    type_pos_2_k = type_pos_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, G_pad, 3)
    labels_1_k = labels_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, F_pad)
    labels_2_k = labels_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, G_pad)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    Na_k = N_real_type.repeat_interleave(P)
    Ma_k = M_real_type.repeat_interleave(P)
    VAA_plus_VBB = (VAA + VBB).repeat_interleave(P)
    VAA_c = _atomtype_self_overlap(type_pos_1_k, labels_1_k, Na_k, tables)
    VBB_c = _atomtype_self_overlap(type_pos_2_k, labels_2_k, Ma_k, tables)
    norm_c = VAA_c + VBB_c

    PK = q_k.shape[0]
    best_score = best_q = best_t = None
    if (centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=30_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_atomtype(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                type_pos_1_k.contiguous(), type_pos_2_k.contiguous(),
                labels_1_k, labels_2_k, N_k, M_k, Na_k, Ma_k,
                VAA_plus_VBB, norm_c, q_k, t_k, tables, alpha, atomtype_weight, lr,
                steps_fine, N_pad_cent, M_pad_cent, F_pad, G_pad, PK,
                es_patience=early_stop_patience, es_tol=early_stop_tol)
        except Exception:
            best_score = None

    if best_score is None:
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
        best_score = torch.full((PK,), -float('inf'), device=device)
        best_q = q_k.clone(); best_t = t_k.clone()
        prev_max_score = -float('inf'); no_improve_count = 0

        for step in range(steps_fine):
            VAB, dQ_s, dT_s, O_c, dQ_c, dT_c = _vat_overlaps(
                centers_1_k, centers_2_k, type_pos_1_k, type_pos_2_k, labels_1_k, labels_2_k,
                q_k, t_k, tables, alpha, N_k, M_k, Na_k, Ma_k)
            denom_s = VAA_plus_VBB - VAB
            shape_sim = VAB / denom_s
            scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)
            denom_c = norm_c - O_c
            color_sim = O_c / denom_c
            scale_c = (norm_c / (denom_c * denom_c)).unsqueeze(1)
            score = (1.0 - atomtype_weight) * shape_sim + atomtype_weight * color_sim
            g_q = (1.0 - atomtype_weight) * (-scale_s * dQ_s) + atomtype_weight * (-scale_c * dQ_c)
            g_t = (1.0 - atomtype_weight) * (-scale_s * dT_s) + atomtype_weight * (-scale_c * dT_c)
            best_score, best_q, best_t = _update_best(score, q_k, t_k, best_score, best_q, best_t)
            if step % 5 == 0:
                current_max = best_score.max().item()
                if current_max - prev_max_score < early_stop_tol:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_patience:
                        break
                else:
                    no_improve_count = 0
                    prev_max_score = current_max
            fused_adam_qt_with_tangent_proj(q_k, t_k, g_q, g_t, m_q, v_q, m_t, v_t, lr)

    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_vol_atomtype_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_type_pos_batch, fit_type_pos_batch,
        ref_labels_batch, fit_labels_batch, *,
        alpha=0.81, atomtype_weight=0.5,
        N_real_centers=None, M_real_centers=None, N_real_type=None, M_real_type=None,
        topk=30, steps_fine=100, lr=0.075, num_seeds=50):
    """Batched vol_atomtype alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_atomtype_align_many(
        ref_centers_batch, fit_centers_batch, ref_type_pos_batch, fit_type_pos_batch,
        ref_labels_batch, fit_labels_batch, VAA, VBB,
        alpha=alpha, atomtype_weight=atomtype_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_type=N_real_type, M_real_type=M_real_type)
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores
