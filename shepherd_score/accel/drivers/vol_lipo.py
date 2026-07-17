# shepherd_score/accel/drivers/vol_lipo.py
# Fast batched vol_lipo alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  LIPOPHILICITY-FIELD overlap
#   (RDKit Crippen per-atom logP contributions scored through the ESP-style kernel).
#
# STRUCTURE == vol_color: a weighted two-channel combo
#   (1-w)*shape_Tanimoto + w*lipo_Tanimoto
# with BOTH channels steering the SE(3) pose (JOINT gradient). The only difference from
# vol_color is the FIELD channel physics: vol_lipo's field is the ESP overlap
# (score.electrostatic_scoring.get_overlap_esp) of a signed per-atom scalar, exactly like
# vol_esp -- so it REUSES the existing fused ESP value+grad kernel
# (overlap_score_grad_esp_se3_batch), fed lipophilicity where vol_esp feeds partial charge.
# There is NO new kernel: the shape kernel (overlap_score_grad_se3_batch) and the ESP kernel
# (overlap_score_grad_esp_se3_batch) both already emit dO/dq directly in quaternion space, so
# each channel's Tanimoto chain-rule scale is applied straight to its dO/dq and the per-channel
# descent gradients are summed:
#   g_q = (1-w)*(-scale_s*dQ_s) + w*(-scale_l*dQ_l)
# (identical projection convention as vol_color; validated vs autograd on optimize_vol_lipo_overlay).
#
# Unlike vol_color (shape=atom centers, color=pharm anchors -- two different point sets), BOTH
# vol_lipo channels share the SAME heavy-atom centers, so there is one N_real/M_real and one
# padded (A, B) cloud; only the per-atom lipophilicity scalar (CA, CB) is extra.

from __future__ import annotations

import torch
from typing import Optional, Tuple

# Device-driven kernel dispatch (Triton on CUDA, numba on CPU); see kernels.dispatch.
from ..kernels.dispatch import (
    fused_adam_qt_with_tangent_proj,
    overlap_score_grad_se3_batch,
    overlap_score_grad_esp_se3_batch,
)
from ._common import (
    batched_seeds_torch,
    build_coarse_grid,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp import _overlap_in_chunks_esp, _self_overlap_esp_chunks
from .esp_combo import _overlap_in_chunks_volumetric, _self_overlap_chunks


def _vl_overlaps(A, B, CA, CB, q, t, *, alpha, lam, N_real, M_real):
    """Both channels' value+grad on the SHARED centers: (VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l).
    Shape via the volumetric overlap kernel, lipo field via the ESP overlap kernel -- both
    emit dO/dq in the same quaternion space (no R->q projection tail)."""
    VAB_s, dQ_s, dT_s = _overlap_in_chunks_volumetric(
        A, B, q, t, alpha=alpha, N_real=N_real, M_real=M_real)
    VAB_l, dQ_l, dT_l = _overlap_in_chunks_esp(
        A, B, CA, CB, q, t, alpha=alpha, lam=lam, N_real=N_real, M_real=M_real)
    return VAB_s, dQ_s, dT_s, VAB_l, dQ_l, dT_l


class _GraphedFineVolLipo(_GraphedFineBase):
    """CUDA-graph fine loop for vol_lipo: TWO value+grad kernels per step (shape
    overlap_score_grad_se3_batch + ESP overlap_score_grad_esp_se3_batch on the shared
    centers). Both run inside the captured step. The combined-objective descent gradient
    g = (1-w)*(-scale_s*dO_s/dq) + w*(-scale_l*dO_l/dq) is built in place, mirroring the
    eager loop exactly. Modeled on drivers.vol_color._GraphedFineVolColor, with the ESP
    kernel (+ lam, + shared CA/CB charge buffers) in place of the color kernel."""

    def __init__(self, N_pad, M_pad, P, steps, alpha, lam, lipo_weight, lr, device):
        self.alpha = float(alpha); self.lam = float(lam)
        self.w = float(lipo_weight); self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        # shared shape/lipo input clouds + per-atom lipophilicity (persistent; loaded per bucket)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.CA = f(P, N_pad); self.CB = f(P, M_pad)
        self.Nr = i(P); self.Mr = i(P)                     # shared heavy-atom counts
        self.norm_s = f(P); self.norm_l = f(P)             # shape / lipo self-overlap sums
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
        VAB, dQs, dTs, O_l, dQl, dTl = _vl_overlaps(
            self.A, self.B, self.CA, self.CB, self.q, self.t,
            alpha=self.alpha, lam=self.lam, N_real=self.Nr, M_real=self.Mr)
        # shape Tanimoto + d/dO_s scale
        torch.sub(self.norm_s, VAB, out=self.denom_s)
        torch.div(VAB, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.norm_s, self.d2s, out=self.scale_s)
        # lipo Tanimoto + d/dO_l scale
        torch.sub(self.norm_l, O_l, out=self.denom_l)
        torch.div(O_l, self.denom_l, out=self.lipo_sim)
        torch.mul(self.denom_l, self.denom_l, out=self.d2l)
        torch.div(self.norm_l, self.d2l, out=self.scale_l)
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

    def _load(self, A, B, CA, CB, Nr, Mr, norm_s, norm_l, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.CA.copy_(CA); self.CB.copy_(CB)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.norm_s.copy_(norm_s); self.norm_l.copy_(norm_l)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_vol_lipo(A_k, B_k, CA_k, CB_k, N_k, M_k, norm_s, norm_l, q_seed, t_seed,
                          alpha, lam, lipo_weight, lr, steps, N_pad, M_pad, P,
                          es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_lipo", N_pad, M_pad, P, steps,
           round(float(alpha), 4), round(float(lam), 6), round(float(lipo_weight), 4),
           round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolLipo(N_pad, M_pad, P, steps, alpha, lam, lipo_weight, lr,
                                    A_k.device),
        key, (A_k, B_k, CA_k, CB_k, N_k, M_k, norm_s, norm_l, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_lipo_align_many(
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        charges_1: torch.Tensor,
        charges_2: torch.Tensor,
        VAA_s: torch.Tensor,
        VBB_s: torch.Tensor,
        VAA_l: torch.Tensor,
        VBB_l: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        num_seeds: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized vol_lipo alignment over a batch of pairs (coarse-to-fine SE(3))."""
    device = centers_1.device
    dtype = centers_1.dtype
    BATCH = centers_1.shape[0]
    N_pad = centers_1.shape[1]
    M_pad = centers_2.shape[1]

    if N_real is None:
        N_real = centers_1.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = centers_2.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) pose hypotheses (seed from the shared atom clouds, like vol/vol_color)
    # ------------------------------------------------------------------
    if trans_centers is None:
        quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real, M_real,
                                             num_seeds=num_seeds)
        P = quats.size(1)
        q_best = quats.clone()
        t_best = t_seeds.clone()
    else:
        q_grid, t_grid = build_coarse_grid(
            centers_1, centers_2, N_real, M_real, num_seeds=num_seeds,
            trans_centers_batch=trans_centers, trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans,
        )
        G = q_grid.size(1)
        ORI_CHUNK = 5_000
        coarse_score = torch.empty(BATCH, G, device=device, dtype=dtype)
        for o0 in range(0, G, ORI_CHUNK):
            o1 = min(o0 + ORI_CHUNK, G)
            g_len = o1 - o0
            q_rep = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
            t_rep = t_grid[:, o0:o1].reshape(-1, 3).contiguous()
            c1 = centers_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad, 3)
            c2 = centers_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad, 3)
            ca = charges_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, N_pad)
            cb = charges_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, M_pad)
            Nc_e = N_real.repeat_interleave(g_len)
            Mc_e = M_real.repeat_interleave(g_len)
            VAA_s_e = VAA_s.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VBB_s_e = VBB_s.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VAA_l_e = VAA_l.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VBB_l_e = VBB_l.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VAB_s, _, _ = _overlap_in_chunks_volumetric(
                c1, c2, q_rep, t_rep, alpha=alpha, N_real=Nc_e, M_real=Mc_e, NEED_GRAD=False)
            VAB_l, _, _ = _overlap_in_chunks_esp(
                c1, c2, ca, cb, q_rep, t_rep, alpha=alpha, lam=lam,
                N_real=Nc_e, M_real=Mc_e, NEED_GRAD=False)
            shape_sim = VAB_s / (VAA_s_e + VBB_s_e - VAB_s)
            lipo_sim = VAB_l / (VAA_l_e + VBB_l_e - VAB_l)
            sc = (1.0 - lipo_weight) * shape_sim + lipo_weight * lipo_sim
            coarse_score[:, o0:o1] = sc.view(BATCH, g_len)
        best_idx = coarse_score.topk(k=topk, dim=1).indices
        q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
        t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
        P = topk

    # ------------------------------------------------------------------
    # 2) fine optimization over ALL P poses (JOINT shape+lipo gradient)
    # ------------------------------------------------------------------
    q_k = q_best.reshape(-1, 4).contiguous()
    t_k = t_best.reshape(-1, 3).contiguous()

    A_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad, 3)
    B_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad, 3)
    CA_k = charges_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, N_pad)
    CB_k = charges_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, M_pad)

    N_k = N_real.repeat_interleave(P)
    M_k = M_real.repeat_interleave(P)
    VAA_s_k = VAA_s.repeat_interleave(P) + VBB_s.repeat_interleave(P)
    VAA_l_k = VAA_l.repeat_interleave(P) + VBB_l.repeat_interleave(P)

    PK = q_k.shape[0]
    best_score = best_q = best_t = None

    # --- CUDA-graph fast path: capture the 2-kernel (shape+lipo) step, replay it. Gated to the
    # launch-bound small/medium-P CUDA fp32 regime (large P / capture failure fall back to the
    # eager loop below). Two kernels per step, so the same lower budget as vol_color. See
    # drivers/_graphed. Validated separately on a GPU box (this machine has no CUDA).
    if (A_k.is_cuda
            and PK <= graph_cap(N_pad * M_pad, budget=30_000_000)
            and A_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_lipo(
                A_k.contiguous(), B_k.contiguous(), CA_k.contiguous(), CB_k.contiguous(),
                N_k, M_k, VAA_s_k, VAA_l_k, q_k, t_k,
                alpha, lam, lipo_weight, lr, steps_fine, N_pad, M_pad, PK,
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
            # Shape + lipo value+grad on the shared centers; both emit dO/dq in the same
            # quaternion space (no R->q projection tail).
            VAB, dQ_s, dT_s, O_l, dQ_l, dT_l = _vl_overlaps(
                A_k, B_k, CA_k, CB_k, q_k, t_k,
                alpha=alpha, lam=lam, N_real=N_k, M_real=M_k)
            denom_s = VAA_s_k - VAB
            shape_sim = VAB / denom_s
            scale_s = (VAA_s_k / (denom_s * denom_s)).unsqueeze(1)      # d shape_T / d O_s
            denom_l = VAA_l_k - O_l
            lipo_sim = O_l / denom_l
            scale_l = (VAA_l_k / (denom_l * denom_l)).unsqueeze(1)      # d lipo_T / d O_l

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

            fused_adam_qt_with_tangent_proj(q_k, t_k, g_q, g_t, m_q, v_q, m_t, v_t, lr)

    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_vol_lipo_overlay_batch(
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        ref_lipo_batch: torch.Tensor,
        fit_lipo_batch: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        trans_centers_batch: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched vol_lipo alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real is None:
        N_real = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real is None:
        M_real = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    # Shape self-overlaps (volumetric) and lipo-field self-overlaps (ESP), on the shared centers.
    VAA_s = _self_overlap_chunks(ref_centers_batch, N_real, alpha)
    VBB_s = _self_overlap_chunks(fit_centers_batch, M_real, alpha)
    VAA_l = _self_overlap_esp_chunks(ref_centers_batch, ref_lipo_batch, N_real, alpha, lam)
    VBB_l = _self_overlap_esp_chunks(fit_centers_batch, fit_lipo_batch, M_real, alpha, lam)

    scores, q_best, t_best = coarse_fine_vol_lipo_align_many(
        ref_centers_batch, fit_centers_batch, ref_lipo_batch, fit_lipo_batch,
        VAA_s, VBB_s, VAA_l, VBB_l,
        alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        num_seeds=num_seeds,
        trans_centers=trans_centers_batch, trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk, steps_fine=steps_fine, lr=lr,
        N_real=N_real, M_real=M_real,
    )
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores


def fast_optimize_vol_lipo_overlay(
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        ref_lipo: torch.Tensor,
        fit_lipo: torch.Tensor,
        *,
        alpha: float = 0.81,
        lam: float = 0.1,
        lipo_weight: float = 0.5,
        num_repeats: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pair fast vol_lipo alignment (drop-in for optimize_vol_lipo_overlay).

    Returns (aligned_fit_centers, SE3_4x4, score) on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    rl = ref_lipo.to(torch.float32).unsqueeze(0)
    fl = fit_lipo.to(torch.float32).unsqueeze(0)

    tcb = None; tcr = None
    if trans_centers is not None:
        tc = trans_centers.to(torch.float32)
        tcb = tc.unsqueeze(0)
        tcr = torch.tensor([tc.shape[0]], device=device, dtype=torch.int32)

    aligned, q_best, t_best, score = fast_optimize_vol_lipo_overlay_batch(
        rc, fc, rl, fl,
        alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        N_real=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        trans_centers_batch=tcb, trans_centers_real=tcr,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=num_repeats,
    )
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]
    return aligned[0].cpu(), SE3.cpu(), score[0].cpu()
