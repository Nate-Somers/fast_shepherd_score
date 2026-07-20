# shepherd_score/accel/drivers/vol_lipo.py
# Fast batched ``vol_lipo`` alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  per-atom LIPOPHILICITY overlap.
#
# This is a two-channel JOINT-gradient mode (driver modeled on vol_color): BOTH
# channels steer the SE(3) pose, and the score is the weighted sum of the two Tanimotos
#   score = (1-w)*shape_Tanimoto + w*lipo_Tanimoto        (w = lipo_weight)
# matching the per-pair reference alignment._torch.optimize_vol_lipo_overlay.
#
# KERNEL REUSE -- NO new kernel is written here:
#   * SHAPE  -> overlap_score_grad_se3_batch (the same Gaussian-volume kernel as vol/vol_esp),
#              scored on the RemoveHs shape centres. Emits dO_s/dq directly.
#   * LIPO   -> overlap_score_grad_esp_se3_batch (the SAME fused ESP kernel used by
#              vol_esp/surf_esp), fed the TRUE-heavy atom centres as its coordinates
#              and the per-atom Crippen atomic logP as its "charges", matched by value
#              (get_overlap_esp with the logP where the ESP kernel expects charges, lam=0.1 RAW
#              / atom-centred, NOT LAM_SCALING-scaled). Emits dO_l/dq directly.
#
# TWO INDEPENDENT point sets: the shape channel runs on ``atom_pos`` (the RemoveHs coordinate
# set, own N count) and the lipo channel on the TRUE-heavy centres
# ``mol.GetConformer().GetPositions()[_nonH_atoms_idx]`` (own M count). These CAN DIFFER IN
# LENGTH on isotope-labelled molecules (RemoveHs retains e.g. deuterium, so the shape set is
# longer than the strict-heavy lipo set). The two paddings are kept SEPARATE -- exactly as the
# reference passes ``ref_centers``/``fit_centers`` apart from ``ref_lipo_pos``/``fit_lipo_pos``
# -- so the length divergence never desyncs a channel. Both fit sets transform rigidly under the
# SAME quaternion each step. Because both kernels emit the overlap gradient in the same
# unit-quaternion space, the combined descent gradient is a per-channel-scaled weighted sum:
#   g_q = (1-w)*(-scale_s*dQ_s) + w*(-scale_l*dQ_l)
# (no R->q projection tail; only the unit->raw normalization in fused_adam_qt_with_tangent_proj).
#
# M=0 handling: a pair with no lipo centres on either side has lipo_Tanimoto = 0 (shape-only,
# score = (1-w)*shape_T) and contributes NO lipo gradient -- matching the reference, which
# zeroes the lipo channel when either point set is empty. The lipo self-overlap norm is 0 for
# such pairs, so the Tanimoto denominator is guarded to avoid a 0/0.

from __future__ import annotations

import torch
from typing import Optional, Tuple

from ..kernels.dispatch import (
    fused_adam_qt_with_tangent_proj,
    overlap_score_grad_se3_batch,
    overlap_score_grad_esp_se3_batch,
)
from ._common import (
    batched_seeds_torch,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp_combo import _overlap_in_chunks_volumetric, _self_overlap_chunks
from .esp import _overlap_in_chunks_esp, _self_overlap_esp_chunks


def _vl_overlaps(A, B, LA, LB, CA, CB, q, t, alpha, lam, Nc, Mc, Nl, Ml):
    """Both channels' value+grad in one place: (VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l).
    Shape channel on the RemoveHs atom clouds (A, B); lipo channel on the TRUE-heavy centres
    (LA, LB) with the per-atom logP (CA, CB) as charges. Both channels share the SAME Gaussian
    width ``alpha`` (matching the reference _vol_lipo_overlap). Drop-in for the two kernels in
    the fine loop and the graph step."""
    VAB_s, dQ_s, dT_s = _overlap_in_chunks_volumetric(
        A, B, q, t, alpha=alpha, N_real=Nc, M_real=Mc)
    VAB_l, dQ_l, dT_l = _overlap_in_chunks_esp(
        LA, LB, CA, CB, q, t, alpha=alpha, lam=lam, N_real=Nl, M_real=Ml)
    return VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l


class _GraphedFineVolLipo(_GraphedFineBase):
    """CUDA-graph fine loop for vol_lipo (TWO value+grad kernels per step: shape overlap +
    per-atom lipophilicity ESP overlap). Both run inside the captured step; the combined-objective
    gradient g = (1-w)*(-scale_s*dO_s/dq) + w*(-scale_l*dO_l/dq) is built in-place, mirroring
    the eager loop exactly. Pairs with no lipo centres (norm_l == 0) get lipo_sim = 0 and no
    lipo gradient via the guarded denominator (has_lipo mask baked into norm_l)."""

    def __init__(self, N_pad, M_pad, F_pad, G_pad, P, steps, alpha, lam,
                 lipo_weight, lr, device):
        self.alpha = float(alpha)
        self.lam = float(lam); self.w = float(lipo_weight); self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        # shape atom clouds + lipo-centre clouds (persistent; loaded per bucket)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.LA = f(P, F_pad, 3); self.LB = f(P, G_pad, 3)
        self.CA = f(P, F_pad); self.CB = f(P, G_pad)          # per-atom logP (as charges)
        self.Nr = i(P); self.Mr = i(P)                        # shape-atom counts
        self.Nl = i(P); self.Ml = i(P)                        # lipo-centre counts
        self.norm_s = f(P); self.norm_l = f(P)                # shape / lipo self-overlap sums
        self.has_lipo = torch.empty(P, device=device, dtype=torch.bool)
        # loop-carried state
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        # per-step temporaries (out= targets)
        self.denom_s = f(P); self.d2s = f(P); self.shape_sim = f(P); self.scale_s = f(P)
        self.denom_l = f(P); self.d2l = f(P); self.lipo_sim = f(P); self.scale_l = f(P)
        self.score = f(P); self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3); self.tmpq = f(P, 4); self.tmpt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB_s, dQs, dTs, VAB_l, dQl, dTl = _vl_overlaps(
            self.A, self.B, self.LA, self.LB, self.CA, self.CB, self.q, self.t,
            self.alpha, self.lam, self.Nr, self.Mr, self.Nl, self.Ml)
        # shape Tanimoto + d/dO_s scale
        torch.sub(self.norm_s, VAB_s, out=self.denom_s)
        torch.div(VAB_s, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.norm_s, self.d2s, out=self.scale_s)
        # lipo Tanimoto + d/dO_l scale; guard the denom for no-lipo-centre pairs (norm_l==0).
        torch.sub(self.norm_l, VAB_l, out=self.denom_l)
        self.denom_l.masked_fill_(~self.has_lipo, 1.0)         # avoid 0/0 (lipo_sim forced 0)
        torch.div(VAB_l, self.denom_l, out=self.lipo_sim)
        self.lipo_sim.masked_fill_(~self.has_lipo, 0.0)
        torch.mul(self.denom_l, self.denom_l, out=self.d2l)
        torch.div(self.norm_l, self.d2l, out=self.scale_l)
        self.scale_l.masked_fill_(~self.has_lipo, 0.0)         # no lipo gradient contribution
        # score = (1-w)*shape_sim + w*lipo_sim
        torch.mul(self.shape_sim, 1.0 - self.w, out=self.score)
        self.score.add_(self.lipo_sim, alpha=self.w)
        # best-pose tracking (in-place)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        # combined descent grad g = (1-w)*(-scale_s*dQs) + w*(-scale_l*dQl)
        torch.mul(dQs, self.scale_s.unsqueeze(1), out=self.gq); self.gq.mul_(-(1.0 - self.w))
        torch.mul(dQl, self.scale_l.unsqueeze(1), out=self.tmpq); self.tmpq.mul_(-self.w)
        self.gq.add_(self.tmpq)
        torch.mul(dTs, self.scale_s.unsqueeze(1), out=self.gt); self.gt.mul_(-(1.0 - self.w))
        torch.mul(dTl, self.scale_l.unsqueeze(1), out=self.tmpt); self.tmpt.mul_(-self.w)
        self.gt.add_(self.tmpt)
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, LA, LB, CA, CB, Nr, Mr, Nl, Ml, norm_s, norm_l, has_lipo, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.LA.copy_(LA); self.LB.copy_(LB); self.CA.copy_(CA); self.CB.copy_(CB)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Nl.copy_(Nl.to(torch.int32)); self.Ml.copy_(Ml.to(torch.int32))
        self.norm_s.copy_(norm_s); self.norm_l.copy_(norm_l); self.has_lipo.copy_(has_lipo)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_vol_lipo(A_k, B_k, LA_k, LB_k, CA_k, CB_k, N_k, M_k, Nl_k, Ml_k,
                          norm_s, norm_l, has_lipo, q_seed, t_seed, alpha, lam,
                          lipo_weight, lr, steps, N_pad, M_pad, F_pad, G_pad, P,
                          es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_lipo", N_pad, M_pad, F_pad, G_pad, P, steps,
           round(float(alpha), 4), round(float(lam), 6),
           round(float(lipo_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolLipo(N_pad, M_pad, F_pad, G_pad, P, steps, alpha,
                                    lam, lipo_weight, lr, A_k.device),
        key, (A_k, B_k, LA_k, LB_k, CA_k, CB_k, N_k, M_k, Nl_k, Ml_k,
              norm_s, norm_l, has_lipo, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_lipo_align_many(
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        lipo_pos_1: torch.Tensor,
        lipo_pos_2: torch.Tensor,
        lipo_1: torch.Tensor,
        lipo_2: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_lipo: Optional[torch.Tensor] = None,
        M_real_lipo: Optional[torch.Tensor] = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized vol_lipo alignment over a batch of pairs (coarse-to-fine SE(3)).

    Seeds from the SHAPE atom clouds (identity + PCA + Fibonacci, COM-aligned), fine-optimises
    ALL seeds and takes the per-pair max -- NO coarse-grid + top-k pruning (matching the vol
    driver). ``VAA`` / ``VBB`` are the shape self-overlaps; lipo self-overlaps are
    recomputed here from the lipo centres + logP."""
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad_cent = centers_1.shape[1]
    M_pad_cent = centers_2.shape[1]
    F_pad = lipo_pos_1.shape[1]
    G_pad = lipo_pos_2.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_lipo is None:
        N_real_lipo = lipo_pos_1.new_full((BATCH,), F_pad, dtype=torch.int32)
    if M_real_lipo is None:
        M_real_lipo = lipo_pos_2.new_full((BATCH,), G_pad, dtype=torch.int32)

    # Lipo self-overlaps (pose-invariant). Zero for pairs with no lipo centres.
    VAA_l = _self_overlap_esp_chunks(lipo_pos_1, lipo_1, N_real_lipo, alpha, lam)
    VBB_l = _self_overlap_esp_chunks(lipo_pos_2, lipo_2, M_real_lipo, alpha, lam)

    # ------------------------------------------------------------------
    # 1) pose hypotheses (seed from the SHAPE atom clouds, like vol/vol_color)
    # ------------------------------------------------------------------
    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_best = quats.clone()
    t_best = t_seeds.clone()

    # ------------------------------------------------------------------
    # 2) fine optimization over ALL P poses (JOINT shape+lipo gradient)
    # ------------------------------------------------------------------
    q_k = q_best.reshape(-1, 4).contiguous()
    t_k = t_best.reshape(-1, 3).contiguous()

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
    VAA_k = VAA.repeat_interleave(P)
    VBB_k = VBB.repeat_interleave(P)
    VAA_plus_VBB = VAA_k + VBB_k
    norm_l = (VAA_l + VBB_l).repeat_interleave(P)
    has_lipo = (Nl_k > 0) & (Ml_k > 0)

    PK = q_k.shape[0]
    best_score = best_q = best_t = None

    # --- CUDA-graph fast path: capture the 2-kernel (shape+lipo) step, replay it. Gated to
    # the launch-bound small/medium-P CUDA fp32 regime; large P / capture failure fall back to
    # the eager loop below. Two value+grad kernels per step -> the lower vol_color-style budget.
    if (centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=30_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_lipo(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                lipo_pos_1_k.contiguous(), lipo_pos_2_k.contiguous(),
                lipo_1_k.contiguous(), lipo_2_k.contiguous(),
                N_k, M_k, Nl_k, Ml_k, VAA_plus_VBB, norm_l, has_lipo, q_k, t_k,
                alpha, lam, lipo_weight, lr,
                steps_fine, N_pad_cent, M_pad_cent, F_pad, G_pad, PK,
                es_patience=early_stop_patience, es_tol=early_stop_tol)
        except Exception:
            best_score = None                              # capture failed -> eager

    if best_score is None:
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

        best_score = torch.full((PK,), -float('inf'), device=device)
        best_q = q_k.clone()
        best_t = t_k.clone()

        prev_max_score = -float('inf')
        no_improve_count = 0

        for step in range(steps_fine):
            # Shape + lipo value+grad. Both channels emit dO/dq in the same quaternion space.
            VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l = _vl_overlaps(
                centers_1_k, centers_2_k, lipo_pos_1_k, lipo_pos_2_k, lipo_1_k, lipo_2_k,
                q_k, t_k, alpha, lam, N_k, M_k, Nl_k, Ml_k)

            denom_s = VAA_plus_VBB - VAB_s
            shape_sim = VAB_s / denom_s
            scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)   # d shape_T / d O_s

            denom_l = norm_l - VAB_l
            denom_l = torch.where(has_lipo, denom_l, torch.ones_like(denom_l))  # guard 0/0
            lipo_sim = torch.where(has_lipo, VAB_l / denom_l,
                                   torch.zeros_like(VAB_l))
            scale_l = torch.where(has_lipo, norm_l / (denom_l * denom_l),
                                  torch.zeros_like(denom_l)).unsqueeze(1)   # d lipo_T / d O_l

            score = (1.0 - lipo_weight) * shape_sim + lipo_weight * lipo_sim

            # Combined descent gradient: (1-w)*(-scale_s*dQ_s) + w*(-scale_l*dQ_l)
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

            fused_adam_qt_with_tangent_proj(
                q_k, t_k, g_q, g_t, m_q, v_q, m_t, v_t, lr)

    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_vol_lipo_overlay_batch(
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        ref_lipo_pos_batch: torch.Tensor,
        fit_lipo_pos_batch: torch.Tensor,
        ref_lipo_batch: torch.Tensor,
        fit_lipo_batch: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_lipo: Optional[torch.Tensor] = None,
        M_real_lipo: Optional[torch.Tensor] = None,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched vol_lipo alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_lipo_align_many(
        ref_centers_batch, fit_centers_batch,
        ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch,
        VAA, VBB,
        alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_lipo=N_real_lipo, M_real_lipo=M_real_lipo,
    )
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores


def fast_optimize_vol_lipo_overlay(
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        ref_lipo_pos: torch.Tensor,
        fit_lipo_pos: torch.Tensor,
        ref_lipo: torch.Tensor,
        fit_lipo: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        num_repeats: int = 50,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pair fast vol_lipo alignment (drop-in for optimize_vol_lipo_overlay).

    Returns (aligned_fit_centers, SE3_4x4, score) on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    rl = ref_lipo_pos.to(torch.float32).unsqueeze(0)
    fl = fit_lipo_pos.to(torch.float32).unsqueeze(0)
    rlv = ref_lipo.to(torch.float32).unsqueeze(0)
    flv = fit_lipo.to(torch.float32).unsqueeze(0)

    aligned, q_best, t_best, score = fast_optimize_vol_lipo_overlay_batch(
        rc, fc, rl, fl, rlv, flv,
        alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        N_real_lipo=torch.tensor([ref_lipo_pos.shape[0]], device=device, dtype=torch.int32),
        M_real_lipo=torch.tensor([fit_lipo_pos.shape[0]], device=device, dtype=torch.int32),
        topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=num_repeats,
    )
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]
    return aligned[0].cpu(), SE3.cpu(), score[0].cpu()
