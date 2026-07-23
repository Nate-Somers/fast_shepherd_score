# shepherd_score/accel/drivers/vol_pharm.py
# Fast batched ``vol_pharm`` alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  DIRECTIONAL pharmacophore overlap.
#
#   score = (1-w)*shape_Tanimoto + w*pharm_Tanimoto         (w = color_weight)
#
# This is the directional counterpart of vol_color: the pharmacophore channel keeps the
# orientation-vector cosine weighting (HBD/HBA/aromatic/halogen must also point compatibly),
# so it REUSES the ``pharm`` mode's directional in-register kernel ``pharm_grad_dq_se3_batch``
# (which takes the anchor positions AND the direction vectors) instead of vol_color's
# directionless colour kernel. Lookup tables are the DIRECTIONAL set
# (build_lookup_tables(directionless=False)). Both channels emit dO/dq in the same quaternion
# space (validated to ~1e-16 vs autograd for each kernel individually), so the combined descent
# gradient is the per-channel-scaled weighted sum, exactly like vol_color. Matches the per-pair
# reference ``alignment._torch.optimize_vol_color_overlay`` with ``directionless=False``.

from __future__ import annotations

import torch
from typing import Optional, Tuple

from ..kernels.dispatch import fused_adam_qt_with_tangent_proj, pharm_grad_dq_se3_batch
from ._common import (
    batched_seeds_torch, apply_se3_transform, quaternion_to_rotation_matrix, _update_best)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp_combo import _overlap_in_chunks_volumetric, _self_overlap_chunks
from ...score.analytical_gradients._torch import build_lookup_tables
from .vol_color import _PHARM_PAD_TYPE  # noqa: F401 (same pad type)


@torch.no_grad()
def _pharm_self_overlap(anc, vec, labels, N_real, tables):
    """Pose-invariant DIRECTIONAL pharmacophore self-overlap via the pharm kernel at identity."""
    al, Ks, cats = tables
    P = anc.shape[0]
    q0 = torch.zeros(P, 4, device=anc.device, dtype=anc.dtype); q0[:, 0] = 1.0
    z = torch.zeros(P, 3, device=anc.device, dtype=anc.dtype)
    O, _, _ = pharm_grad_dq_se3_batch(
        q0, z, labels, labels, anc, anc, vec, vec, al, Ks, cats,
        N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return O


def _vp_overlaps(c1, c2, a1, a2, v1, v2, q, t, pt1, pt2, tables, alpha, Nc, Mc, Na, Ma):
    """Shape overlap on (c1,c2) + DIRECTIONAL pharm overlap on (a1,a2) with vectors (v1,v2)."""
    al, Ks, cats = tables
    VAB_s, dQ_s, dT_s = _overlap_in_chunks_volumetric(c1, c2, q, t, alpha=alpha, N_real=Nc, M_real=Mc)
    O_c, dQ_c, dT_c = pharm_grad_dq_se3_batch(
        q, t, pt1, pt2, a1, a2, v1, v2, al, Ks, cats, N_real=Na, M_real=Ma)
    return VAB_s, dQ_s, dT_s, O_c, dQ_c, dT_c


class _GraphedFineVolPharm(_GraphedFineBase):
    """CUDA-graph fine loop for vol_pharm (shape + DIRECTIONAL pharm, Tanimoto each)."""

    def __init__(self, N_pad, M_pad, P_pad, Q_pad, P, steps, alpha, color_weight, lr,
                 tables, device):
        self.alpha = float(alpha); self.w = float(color_weight); self.lr = float(lr)
        self.al, self.Ks, self.cats = tables
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.anc1 = f(P, P_pad, 3); self.anc2 = f(P, Q_pad, 3)
        self.vec1 = f(P, P_pad, 3); self.vec2 = f(P, Q_pad, 3)
        self.pt1 = i(P, P_pad); self.pt2 = i(P, Q_pad)
        self.Nr = i(P); self.Mr = i(P); self.Npr = i(P); self.Mpr = i(P)
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
        VAB, dQs, dTs, O_c, dQc, dTc = _vp_overlaps(
            self.A, self.B, self.anc1, self.anc2, self.vec1, self.vec2, self.q, self.t,
            self.pt1, self.pt2, (self.al, self.Ks, self.cats), self.alpha,
            self.Nr, self.Mr, self.Npr, self.Mpr)
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

    def _load(self, A, B, anc1, anc2, vec1, vec2, pt1, pt2, Nr, Mr, Npr, Mpr, norm_s, norm_c, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.anc1.copy_(anc1); self.anc2.copy_(anc2); self.vec1.copy_(vec1); self.vec2.copy_(vec2)
        self.pt1.copy_(pt1.to(torch.int32)); self.pt2.copy_(pt2.to(torch.int32))
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Npr.copy_(Npr.to(torch.int32)); self.Mpr.copy_(Mpr.to(torch.int32))
        self.norm_s.copy_(norm_s); self.norm_c.copy_(norm_c)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_vol_pharm(A_k, B_k, anc1_k, anc2_k, vec1_k, vec2_k, pt1_k, pt2_k,
                           N_k, M_k, Np_k, Mp_k, norm_s, norm_c, q_seed, t_seed, tables,
                           alpha, color_weight, lr, steps, N_pad, M_pad, P_pad, Q_pad, P,
                           es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_pharm", N_pad, M_pad, P_pad, Q_pad, P, steps,
           round(float(alpha), 4), round(float(color_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolPharm(N_pad, M_pad, P_pad, Q_pad, P, steps, alpha,
                                     color_weight, lr, tables, A_k.device),
        key, (A_k, B_k, anc1_k, anc2_k, vec1_k, vec2_k, pt1_k, pt2_k, N_k, M_k, Np_k, Mp_k,
              norm_s, norm_c, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_pharm_align_many(
        centers_1, centers_2, ptype_1, ptype_2, anchors_1, anchors_2, vectors_1, vectors_2,
        VAA, VBB, *,
        alpha=0.81, color_weight=0.5, num_seeds=50, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None, N_real_pharm=None, M_real_pharm=None,
        early_stop_patience=2, early_stop_tol=1e-5, tables=None):
    """Batched vol_pharm alignment (seed path). ``VAA``/``VBB`` are shape self-overlaps; directional
    pharm self-overlaps computed here via the pharm kernel."""
    device = centers_1.device
    dtype = centers_1.dtype
    BATCH = centers_1.shape[0]
    N_pad_cent, M_pad_cent = centers_1.shape[1], centers_2.shape[1]
    P_pad, Q_pad = anchors_1.shape[1], anchors_2.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_pharm is None:
        N_real_pharm = anchors_1.new_full((BATCH,), P_pad, dtype=torch.int32)
    if M_real_pharm is None:
        M_real_pharm = anchors_2.new_full((BATCH,), Q_pad, dtype=torch.int32)
    if tables is None:
        tables = build_lookup_tables(device, dtype, directionless=False)

    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_k = quats.clone().reshape(-1, 4).contiguous()
    t_k = t_seeds.clone().reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_cent, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_cent, 3)
    anchors_1_k = anchors_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, P_pad, 3)
    anchors_2_k = anchors_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, Q_pad, 3)
    vectors_1_k = vectors_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, P_pad, 3)
    vectors_2_k = vectors_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, Q_pad, 3)
    ptype_1_k = ptype_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, P_pad)
    ptype_2_k = ptype_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, Q_pad)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    Np_k = N_real_pharm.repeat_interleave(P)
    Mp_k = M_real_pharm.repeat_interleave(P)
    VAA_plus_VBB = (VAA + VBB).repeat_interleave(P)
    VAA_c = _pharm_self_overlap(anchors_1_k, vectors_1_k, ptype_1_k, Np_k, tables)
    VBB_c = _pharm_self_overlap(anchors_2_k, vectors_2_k, ptype_2_k, Mp_k, tables)
    norm_c = VAA_c + VBB_c

    PK = q_k.shape[0]
    best_score = best_q = best_t = None
    if (centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=10_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_pharm(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                anchors_1_k.contiguous(), anchors_2_k.contiguous(),
                vectors_1_k.contiguous(), vectors_2_k.contiguous(),
                ptype_1_k, ptype_2_k, N_k, M_k, Np_k, Mp_k, VAA_plus_VBB, norm_c, q_k, t_k,
                tables, alpha, color_weight, lr,
                steps_fine, N_pad_cent, M_pad_cent, P_pad, Q_pad, PK,
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
            VAB, dQ_s, dT_s, O_c, dQ_c, dT_c = _vp_overlaps(
                centers_1_k, centers_2_k, anchors_1_k, anchors_2_k, vectors_1_k, vectors_2_k,
                q_k, t_k, ptype_1_k, ptype_2_k, tables, alpha, N_k, M_k, Np_k, Mp_k)
            denom_s = VAA_plus_VBB - VAB
            shape_sim = VAB / denom_s
            scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)
            denom_c = norm_c - O_c
            color_sim = O_c / denom_c
            scale_c = (norm_c / (denom_c * denom_c)).unsqueeze(1)
            score = (1.0 - color_weight) * shape_sim + color_weight * color_sim
            g_q = (1.0 - color_weight) * (-scale_s * dQ_s) + color_weight * (-scale_c * dQ_c)
            g_t = (1.0 - color_weight) * (-scale_s * dT_s) + color_weight * (-scale_c * dT_c)
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


def fast_optimize_vol_pharm_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch,
        ref_ancs_batch, fit_ancs_batch, ref_vecs_batch, fit_vecs_batch, *,
        alpha=0.81, color_weight=0.5,
        N_real_centers=None, M_real_centers=None, N_real_pharm=None, M_real_pharm=None,
        topk=30, steps_fine=100, lr=0.075, num_seeds=50):
    """Batched vol_pharm alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_pharm_align_many(
        ref_centers_batch, fit_centers_batch, ref_types_batch, fit_types_batch,
        ref_ancs_batch, fit_ancs_batch, ref_vecs_batch, fit_vecs_batch, VAA, VBB,
        alpha=alpha, color_weight=color_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_pharm=N_real_pharm, M_real_pharm=M_real_pharm)
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores
