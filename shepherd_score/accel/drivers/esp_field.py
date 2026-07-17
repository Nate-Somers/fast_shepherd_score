# shepherd_score/accel/drivers/esp_field.py
# Fast batched Cresset-style ``esp_field`` alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  signed ESP FIELD-POINT overlap.
#
# This is a two-channel JOINT-gradient mode (driver modeled on vol_color): BOTH channels
# steer the SE(3) pose, and the score is the weighted sum of the two Tanimotos
#   score = (1-w)*shape_Tanimoto + w*field_Tanimoto     (w = field_weight)
# matching the per-pair reference alignment._torch.optimize_esp_field_overlay.
#
# KERNEL REUSE -- NO new kernel is written here:
#   * SHAPE  -> overlap_score_grad_se3_batch (same Gaussian-volume kernel as vol/vol_esp),
#              scored on the heavy-atom centres. Emits dO_s/dq directly.
#   * FIELD  -> overlap_score_grad_esp_se3_batch (the SAME fused ESP kernel used by
#              vol_esp/surf_esp), fed the FIELD POINTS as its coordinates and the field-point
#              SIGNS (+-1) as its "charges". Emits dO_f/dq directly.
# The two point sets are INDEPENDENT: the atoms have their own (N) count, the field points
# their own (M) count (variable, and possibly 0). Both fit point sets transform rigidly under
# the SAME quaternion each step. Because both kernels emit the overlap gradient in the same
# unit-quaternion space, the combined descent gradient is a per-channel-scaled weighted sum:
#   g_q = (1-w)*(-scale_s*dQ_s) + w*(-scale_f*dQ_f)
# (no R->q projection tail; only the unit->raw normalization in fused_adam_qt_with_tangent_proj).
#
# M=0 handling: a pair with no field points on either side has field_Tanimoto = 0 (shape-only,
# score = (1-w)*shape_T) and contributes NO field gradient -- matching the reference, which
# zeroes the field channel when either point set is empty. The field self-overlap norm is 0 for
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


def _ef_overlaps(A, B, FA, FB, CA, CB, q, t, alpha, alpha_field, lam,
                 Nc, Mc, Nf, Mf):
    """Both channels' value+grad in one place: (VAB_s, dQ_s, dT_s, VAB_f, dQ_f, dT_f).
    Shape channel on the atom clouds (A, B); field channel on the field points (FA, FB) with
    the field-point signs (CA, CB) as charges. Drop-in for the two kernels in the fine loop
    and the graph step."""
    VAB_s, dQ_s, dT_s = _overlap_in_chunks_volumetric(
        A, B, q, t, alpha=alpha, N_real=Nc, M_real=Mc)
    VAB_f, dQ_f, dT_f = _overlap_in_chunks_esp(
        FA, FB, CA, CB, q, t, alpha=alpha_field, lam=lam, N_real=Nf, M_real=Mf)
    return VAB_s, dQ_s, dT_s, VAB_f, dQ_f, dT_f


class _GraphedFineEspField(_GraphedFineBase):
    """CUDA-graph fine loop for esp_field (TWO value+grad kernels per step: shape overlap +
    field-point ESP overlap). Both run inside the captured step; the combined-objective
    gradient g = (1-w)*(-scale_s*dO_s/dq) + w*(-scale_f*dO_f/dq) is built in-place, mirroring
    the eager loop exactly. Pairs with no field points (norm_f == 0) get field_sim = 0 and no
    field gradient via the guarded denominator (has_fp mask baked into norm_f)."""

    def __init__(self, N_pad, M_pad, F_pad, G_pad, P, steps, alpha, alpha_field, lam,
                 field_weight, lr, device):
        self.alpha = float(alpha); self.alpha_field = float(alpha_field)
        self.lam = float(lam); self.w = float(field_weight); self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        # shape atom clouds + field-point clouds (persistent; loaded per bucket)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.FA = f(P, F_pad, 3); self.FB = f(P, G_pad, 3)
        self.CA = f(P, F_pad); self.CB = f(P, G_pad)          # field-point signs (as charges)
        self.Nr = i(P); self.Mr = i(P)                        # shape-atom counts
        self.Nf = i(P); self.Mf = i(P)                        # field-point counts
        self.norm_s = f(P); self.norm_f = f(P)                # shape / field self-overlap sums
        self.has_fp = torch.empty(P, device=device, dtype=torch.bool)
        # loop-carried state
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        # per-step temporaries (out= targets)
        self.denom_s = f(P); self.d2s = f(P); self.shape_sim = f(P); self.scale_s = f(P)
        self.denom_f = f(P); self.d2f = f(P); self.field_sim = f(P); self.scale_f = f(P)
        self.score = f(P); self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3); self.tmpq = f(P, 4); self.tmpt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB_s, dQs, dTs, VAB_f, dQf, dTf = _ef_overlaps(
            self.A, self.B, self.FA, self.FB, self.CA, self.CB, self.q, self.t,
            self.alpha, self.alpha_field, self.lam, self.Nr, self.Mr, self.Nf, self.Mf)
        # shape Tanimoto + d/dO_s scale
        torch.sub(self.norm_s, VAB_s, out=self.denom_s)
        torch.div(VAB_s, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.norm_s, self.d2s, out=self.scale_s)
        # field Tanimoto + d/dO_f scale; guard the denom for no-field-point pairs (norm_f==0).
        torch.sub(self.norm_f, VAB_f, out=self.denom_f)
        self.denom_f.masked_fill_(~self.has_fp, 1.0)           # avoid 0/0 (field_sim forced 0)
        torch.div(VAB_f, self.denom_f, out=self.field_sim)
        self.field_sim.masked_fill_(~self.has_fp, 0.0)
        torch.mul(self.denom_f, self.denom_f, out=self.d2f)
        torch.div(self.norm_f, self.d2f, out=self.scale_f)
        self.scale_f.masked_fill_(~self.has_fp, 0.0)           # no field gradient contribution
        # score = (1-w)*shape_sim + w*field_sim
        torch.mul(self.shape_sim, 1.0 - self.w, out=self.score)
        self.score.add_(self.field_sim, alpha=self.w)
        # best-pose tracking (in-place)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        # combined descent grad g = (1-w)*(-scale_s*dQs) + w*(-scale_f*dQf)
        torch.mul(dQs, self.scale_s.unsqueeze(1), out=self.gq); self.gq.mul_(-(1.0 - self.w))
        torch.mul(dQf, self.scale_f.unsqueeze(1), out=self.tmpq); self.tmpq.mul_(-self.w)
        self.gq.add_(self.tmpq)
        torch.mul(dTs, self.scale_s.unsqueeze(1), out=self.gt); self.gt.mul_(-(1.0 - self.w))
        torch.mul(dTf, self.scale_f.unsqueeze(1), out=self.tmpt); self.tmpt.mul_(-self.w)
        self.gt.add_(self.tmpt)
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, FA, FB, CA, CB, Nr, Mr, Nf, Mf, norm_s, norm_f, has_fp, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.FA.copy_(FA); self.FB.copy_(FB); self.CA.copy_(CA); self.CB.copy_(CB)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Nf.copy_(Nf.to(torch.int32)); self.Mf.copy_(Mf.to(torch.int32))
        self.norm_s.copy_(norm_s); self.norm_f.copy_(norm_f); self.has_fp.copy_(has_fp)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_esp_field(A_k, B_k, FA_k, FB_k, CA_k, CB_k, N_k, M_k, Nf_k, Mf_k,
                           norm_s, norm_f, has_fp, q_seed, t_seed, alpha, alpha_field, lam,
                           field_weight, lr, steps, N_pad, M_pad, F_pad, G_pad, P,
                           es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "esp_field", N_pad, M_pad, F_pad, G_pad, P, steps,
           round(float(alpha), 4), round(float(alpha_field), 4), round(float(lam), 6),
           round(float(field_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineEspField(N_pad, M_pad, F_pad, G_pad, P, steps, alpha, alpha_field,
                                     lam, field_weight, lr, A_k.device),
        key, (A_k, B_k, FA_k, FB_k, CA_k, CB_k, N_k, M_k, Nf_k, Mf_k,
              norm_s, norm_f, has_fp, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_esp_field_align_many(
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        fp_pos_1: torch.Tensor,
        fp_pos_2: torch.Tensor,
        fp_sign_1: torch.Tensor,
        fp_sign_2: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float = 0.81,
        alpha_field: float = 0.81,
        lam: float = 0.1,
        field_weight: float = 0.5,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_fp: Optional[torch.Tensor] = None,
        M_real_fp: Optional[torch.Tensor] = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized esp_field alignment over a batch of pairs (coarse-to-fine SE(3)).

    Seeds from the SHAPE atom clouds (identity + PCA + Fibonacci, COM-aligned), fine-optimises
    ALL seeds and takes the per-pair max -- NO coarse-grid + top-k pruning (matching the vol /
    ESP drivers). ``VAA`` / ``VBB`` are the shape self-overlaps; field self-overlaps are
    recomputed here from the field points."""
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad_cent = centers_1.shape[1]
    M_pad_cent = centers_2.shape[1]
    F_pad = fp_pos_1.shape[1]
    G_pad = fp_pos_2.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_fp is None:
        N_real_fp = fp_pos_1.new_full((BATCH,), F_pad, dtype=torch.int32)
    if M_real_fp is None:
        M_real_fp = fp_pos_2.new_full((BATCH,), G_pad, dtype=torch.int32)

    # Field self-overlaps (pose-invariant). Zero for pairs with no field points.
    VAA_f = _self_overlap_esp_chunks(fp_pos_1, fp_sign_1, N_real_fp, alpha_field, lam)
    VBB_f = _self_overlap_esp_chunks(fp_pos_2, fp_sign_2, M_real_fp, alpha_field, lam)

    # ------------------------------------------------------------------
    # 1) pose hypotheses (seed from the SHAPE atom clouds, like vol/vol_color)
    # ------------------------------------------------------------------
    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_best = quats.clone()
    t_best = t_seeds.clone()

    # ------------------------------------------------------------------
    # 2) fine optimization over ALL P poses (JOINT shape+field gradient)
    # ------------------------------------------------------------------
    q_k = q_best.reshape(-1, 4).contiguous()
    t_k = t_best.reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_cent, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_cent, 3)
    fp_pos_1_k = fp_pos_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, F_pad, 3)
    fp_pos_2_k = fp_pos_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, G_pad, 3)
    fp_sign_1_k = fp_sign_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, F_pad)
    fp_sign_2_k = fp_sign_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, G_pad)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    Nf_k = N_real_fp.repeat_interleave(P)
    Mf_k = M_real_fp.repeat_interleave(P)
    VAA_k = VAA.repeat_interleave(P)
    VBB_k = VBB.repeat_interleave(P)
    VAA_plus_VBB = VAA_k + VBB_k
    norm_f = (VAA_f + VBB_f).repeat_interleave(P)
    has_fp = (Nf_k > 0) & (Mf_k > 0)

    PK = q_k.shape[0]
    best_score = best_q = best_t = None

    # --- CUDA-graph fast path: capture the 2-kernel (shape+field) step, replay it. Gated to
    # the launch-bound small/medium-P CUDA fp32 regime; large P / capture failure fall back to
    # the eager loop below. Two value+grad kernels per step -> the lower vol_color-style budget.
    if (centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=30_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_esp_field(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                fp_pos_1_k.contiguous(), fp_pos_2_k.contiguous(),
                fp_sign_1_k.contiguous(), fp_sign_2_k.contiguous(),
                N_k, M_k, Nf_k, Mf_k, VAA_plus_VBB, norm_f, has_fp, q_k, t_k,
                alpha, alpha_field, lam, field_weight, lr,
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
            # Shape + field value+grad. Both channels emit dO/dq in the same quaternion space.
            VAB_s, dQ_s, dT_s, VAB_f, dQ_f, dT_f = _ef_overlaps(
                centers_1_k, centers_2_k, fp_pos_1_k, fp_pos_2_k, fp_sign_1_k, fp_sign_2_k,
                q_k, t_k, alpha, alpha_field, lam, N_k, M_k, Nf_k, Mf_k)

            denom_s = VAA_plus_VBB - VAB_s
            shape_sim = VAB_s / denom_s
            scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)   # d shape_T / d O_s

            denom_f = norm_f - VAB_f
            denom_f = torch.where(has_fp, denom_f, torch.ones_like(denom_f))  # guard 0/0
            field_sim = torch.where(has_fp, VAB_f / denom_f,
                                    torch.zeros_like(VAB_f))
            scale_f = torch.where(has_fp, norm_f / (denom_f * denom_f),
                                  torch.zeros_like(denom_f)).unsqueeze(1)   # d field_T / d O_f

            score = (1.0 - field_weight) * shape_sim + field_weight * field_sim

            # Combined descent gradient: (1-w)*(-scale_s*dQ_s) + w*(-scale_f*dQ_f)
            g_q = (1.0 - field_weight) * (-scale_s * dQ_s) + field_weight * (-scale_f * dQ_f)
            g_t = (1.0 - field_weight) * (-scale_s * dT_s) + field_weight * (-scale_f * dT_f)

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


def fast_optimize_esp_field_overlay_batch(
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        ref_fp_pos_batch: torch.Tensor,
        fit_fp_pos_batch: torch.Tensor,
        ref_fp_sign_batch: torch.Tensor,
        fit_fp_sign_batch: torch.Tensor,
        *,
        alpha: float = 0.81,
        alpha_field: float = 0.81,
        lam: float = 0.1,
        field_weight: float = 0.5,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_fp: Optional[torch.Tensor] = None,
        M_real_fp: Optional[torch.Tensor] = None,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched esp_field alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_esp_field_align_many(
        ref_centers_batch, fit_centers_batch,
        ref_fp_pos_batch, fit_fp_pos_batch,
        ref_fp_sign_batch, fit_fp_sign_batch,
        VAA, VBB,
        alpha=alpha, alpha_field=alpha_field, lam=lam, field_weight=field_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_fp=N_real_fp, M_real_fp=M_real_fp,
    )
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores


def fast_optimize_esp_field_overlay(
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        ref_fp_pos: torch.Tensor,
        fit_fp_pos: torch.Tensor,
        ref_fp_sign: torch.Tensor,
        fit_fp_sign: torch.Tensor,
        *,
        alpha: float = 0.81,
        alpha_field: float = 0.81,
        lam: float = 0.1,
        field_weight: float = 0.5,
        num_repeats: int = 50,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pair fast esp_field alignment (drop-in for optimize_esp_field_overlay).

    Returns (aligned_fit_centers, SE3_4x4, score) on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    rf = ref_fp_pos.to(torch.float32).unsqueeze(0)
    ff = fit_fp_pos.to(torch.float32).unsqueeze(0)
    rs = ref_fp_sign.to(torch.float32).unsqueeze(0)
    fs = fit_fp_sign.to(torch.float32).unsqueeze(0)

    aligned, q_best, t_best, score = fast_optimize_esp_field_overlay_batch(
        rc, fc, rf, ff, rs, fs,
        alpha=alpha, alpha_field=alpha_field, lam=lam, field_weight=field_weight,
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        N_real_fp=torch.tensor([ref_fp_pos.shape[0]], device=device, dtype=torch.int32),
        M_real_fp=torch.tensor([fit_fp_pos.shape[0]], device=device, dtype=torch.int32),
        topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=num_repeats,
    )
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]
    return aligned[0].cpu(), SE3.cpu(), score[0].cpu()
