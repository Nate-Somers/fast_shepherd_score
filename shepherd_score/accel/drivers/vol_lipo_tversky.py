# shepherd_score/accel/drivers/vol_lipo_tversky.py
# Fast batched ``vol_lipo_tversky`` alignment: the two-channel vol_lipo mode (shape + per-atom
# lipophilicity) scored with an asymmetric TVERSKY reduction on BOTH channels instead of Tanimoto.
#
#   score = (1-w)*shape_Tversky + w*lipo_Tversky            (w = lipo_weight)
#   channel_Tversky = VAB / (k*VAB + C),  C = tα*VAA + tβ*VBB,  k = 1 - tα - tβ
#   d(Tversky)/dVAB = C / (k*VAB + C)^2
#
# This is exactly ``vol_lipo``'s driver with the Tanimoto reduction (``VAB/(VAA+VBB-VAB)``,
# ``scale = (VAA+VBB)/denom^2``) swapped for the Tversky reduction proven in ``vol_tversky``'s
# ``_tversky_adam_tail``. No new kernel: the same shape + ESP value+grad kernels emit dO/dq; only
# the per-channel reduction (and hence the gradient scale) changes. The self-overlaps VAA/VBB are
# kept SEPARATE (not summed) so C = tα*VAA + tβ*VBB can be formed. Matches the per-pair reference
# ``alignment._torch.optimize_vol_lipo_tversky_overlay``.

from __future__ import annotations

import torch
from typing import Optional, Tuple

from ..kernels.dispatch import fused_adam_qt_with_tangent_proj
from ._common import (
    batched_seeds_torch, apply_se3_transform, quaternion_to_rotation_matrix, _update_best)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp_combo import _self_overlap_chunks
from .esp import _self_overlap_esp_chunks
from .vol_lipo import _vl_overlaps


class _GraphedFineVolLipoTversky(_GraphedFineBase):
    """CUDA-graph fine loop for vol_lipo_tversky. Two value+grad kernels per step (shape + lipo
    ESP); each channel reduced by Tversky ``VAB/(k*VAB+C)`` with ``C = tα*VAA+tβ*VBB``. Pairs with
    no lipo centres (has_lipo False) get lipo_sim = 0 and no lipo gradient via the guarded denom."""

    def __init__(self, N_pad, M_pad, F_pad, G_pad, P, steps, alpha, lam,
                 lipo_weight, ta, tb, lr, device):
        self.alpha = float(alpha); self.lam = float(lam)
        self.w = float(lipo_weight); self.lr = float(lr)
        self.ta = float(ta); self.tb = float(tb); self.k = 1.0 - float(ta) - float(tb)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.LA = f(P, F_pad, 3); self.LB = f(P, G_pad, 3)
        self.CA = f(P, F_pad); self.CB = f(P, G_pad)
        self.Nr = i(P); self.Mr = i(P); self.Nl = i(P); self.Ml = i(P)
        self.Cs = f(P); self.Cl = f(P)                        # tversky self-overlap combos
        self.has_lipo = torch.empty(P, device=device, dtype=torch.bool)
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        self.denom_s = f(P); self.d2s = f(P); self.shape_sim = f(P); self.scale_s = f(P)
        self.denom_l = f(P); self.d2l = f(P); self.lipo_sim = f(P); self.scale_l = f(P)
        self.score = f(P); self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3); self.tmpq = f(P, 4); self.tmpt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB_s, dQs, dTs, VAB_l, dQl, dTl = _vl_overlaps(
            self.A, self.B, self.LA, self.LB, self.CA, self.CB, self.q, self.t,
            self.alpha, self.lam, self.Nr, self.Mr, self.Nl, self.Ml)
        # shape Tversky: denom = k*VAB + Cs ; sim = VAB/denom ; scale = Cs/denom^2
        torch.mul(VAB_s, self.k, out=self.denom_s); self.denom_s.add_(self.Cs)
        torch.div(VAB_s, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.Cs, self.d2s, out=self.scale_s)
        # lipo Tversky (guard no-lipo pairs: Cl==0 -> denom 0 -> 0/0)
        torch.mul(VAB_l, self.k, out=self.denom_l); self.denom_l.add_(self.Cl)
        self.denom_l.masked_fill_(~self.has_lipo, 1.0)
        torch.div(VAB_l, self.denom_l, out=self.lipo_sim)
        self.lipo_sim.masked_fill_(~self.has_lipo, 0.0)
        torch.mul(self.denom_l, self.denom_l, out=self.d2l)
        torch.div(self.Cl, self.d2l, out=self.scale_l)
        self.scale_l.masked_fill_(~self.has_lipo, 0.0)
        # score = (1-w)*shape + w*lipo
        torch.mul(self.shape_sim, 1.0 - self.w, out=self.score)
        self.score.add_(self.lipo_sim, alpha=self.w)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        torch.mul(dQs, self.scale_s.unsqueeze(1), out=self.gq); self.gq.mul_(-(1.0 - self.w))
        torch.mul(dQl, self.scale_l.unsqueeze(1), out=self.tmpq); self.tmpq.mul_(-self.w)
        self.gq.add_(self.tmpq)
        torch.mul(dTs, self.scale_s.unsqueeze(1), out=self.gt); self.gt.mul_(-(1.0 - self.w))
        torch.mul(dTl, self.scale_l.unsqueeze(1), out=self.tmpt); self.tmpt.mul_(-self.w)
        self.gt.add_(self.tmpt)
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, LA, LB, CA, CB, Nr, Mr, Nl, Ml, Cs, Cl, has_lipo, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.LA.copy_(LA); self.LB.copy_(LB); self.CA.copy_(CA); self.CB.copy_(CB)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Nl.copy_(Nl.to(torch.int32)); self.Ml.copy_(Ml.to(torch.int32))
        self.Cs.copy_(Cs); self.Cl.copy_(Cl); self.has_lipo.copy_(has_lipo)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_vol_lipo_tversky(A_k, B_k, LA_k, LB_k, CA_k, CB_k, N_k, M_k, Nl_k, Ml_k,
                                  Cs, Cl, has_lipo, q_seed, t_seed, alpha, lam,
                                  lipo_weight, ta, tb, lr, steps, N_pad, M_pad, F_pad, G_pad, P,
                                  es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_lipo_tversky", N_pad, M_pad, F_pad, G_pad, P, steps,
           round(float(alpha), 4), round(float(lam), 6), round(float(lipo_weight), 4),
           round(float(ta), 4), round(float(tb), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolLipoTversky(N_pad, M_pad, F_pad, G_pad, P, steps, alpha,
                                           lam, lipo_weight, ta, tb, lr, A_k.device),
        key, (A_k, B_k, LA_k, LB_k, CA_k, CB_k, N_k, M_k, Nl_k, Ml_k,
              Cs, Cl, has_lipo, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_lipo_tversky_align_many(
        centers_1, centers_2, lipo_pos_1, lipo_pos_2, lipo_1, lipo_2, VAA, VBB, *,
        alpha=0.81, lam=0.1, lipo_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05,
        num_seeds=50, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None, N_real_lipo=None, M_real_lipo=None,
        early_stop_patience=2, early_stop_tol=1e-5):
    """Batched vol_lipo_tversky alignment. ``VAA``/``VBB`` are the shape self-overlaps (kept
    separate to form the Tversky combo C = tα*VAA + tβ*VBB); lipo self-overlaps recomputed here."""
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad_cent, M_pad_cent = centers_1.shape[1], centers_2.shape[1]
    F_pad, G_pad = lipo_pos_1.shape[1], lipo_pos_2.shape[1]
    ta, tb = float(tversky_alpha), float(tversky_beta); k = 1.0 - ta - tb

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_lipo is None:
        N_real_lipo = lipo_pos_1.new_full((BATCH,), F_pad, dtype=torch.int32)
    if M_real_lipo is None:
        M_real_lipo = lipo_pos_2.new_full((BATCH,), G_pad, dtype=torch.int32)

    VAA_l = _self_overlap_esp_chunks(lipo_pos_1, lipo_1, N_real_lipo, alpha, lam)
    VBB_l = _self_overlap_esp_chunks(lipo_pos_2, lipo_2, M_real_lipo, alpha, lam)

    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_k = quats.clone().reshape(-1, 4).contiguous()
    t_k = t_seeds.clone().reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_cent, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_cent, 3)
    lipo_pos_1_k = lipo_pos_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, F_pad, 3)
    lipo_pos_2_k = lipo_pos_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, G_pad, 3)
    lipo_1_k = lipo_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, F_pad)
    lipo_2_k = lipo_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, G_pad)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    Nl_k = N_real_lipo.repeat_interleave(P)
    Ml_k = M_real_lipo.repeat_interleave(P)
    Cs = (ta * VAA + tb * VBB).repeat_interleave(P)
    Cl = (ta * VAA_l + tb * VBB_l).repeat_interleave(P)
    has_lipo = (Nl_k > 0) & (Ml_k > 0)

    PK = q_k.shape[0]
    best_score = best_q = best_t = None

    if (centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=30_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_lipo_tversky(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                lipo_pos_1_k.contiguous(), lipo_pos_2_k.contiguous(),
                lipo_1_k.contiguous(), lipo_2_k.contiguous(),
                N_k, M_k, Nl_k, Ml_k, Cs, Cl, has_lipo, q_k, t_k,
                alpha, lam, lipo_weight, ta, tb, lr,
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
            VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l = _vl_overlaps(
                centers_1_k, centers_2_k, lipo_pos_1_k, lipo_pos_2_k, lipo_1_k, lipo_2_k,
                q_k, t_k, alpha, lam, N_k, M_k, Nl_k, Ml_k)

            denom_s = k * VAB_s + Cs
            shape_sim = VAB_s / denom_s
            scale_s = (Cs / (denom_s * denom_s)).unsqueeze(1)

            denom_l = k * VAB_l + Cl
            denom_l = torch.where(has_lipo, denom_l, torch.ones_like(denom_l))
            lipo_sim = torch.where(has_lipo, VAB_l / denom_l, torch.zeros_like(VAB_l))
            scale_l = torch.where(has_lipo, Cl / (denom_l * denom_l),
                                  torch.zeros_like(denom_l)).unsqueeze(1)

            score = (1.0 - lipo_weight) * shape_sim + lipo_weight * lipo_sim
            g_q = (1.0 - lipo_weight) * (-scale_s * dQ_s) + lipo_weight * (-scale_l * dQ_l)
            g_t = (1.0 - lipo_weight) * (-scale_s * dT_s) + lipo_weight * (-scale_l * dT_l)

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


def fast_optimize_vol_lipo_tversky_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, *,
        alpha=0.81, lam=0.1, lipo_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05,
        N_real_centers=None, M_real_centers=None, N_real_lipo=None, M_real_lipo=None,
        topk=30, steps_fine=100, lr=0.075, num_seeds=50):
    """Batched vol_lipo_tversky alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_lipo_tversky_align_many(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, VAA, VBB,
        alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_lipo=N_real_lipo, M_real_lipo=M_real_lipo)
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores
