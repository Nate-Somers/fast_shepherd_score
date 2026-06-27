# shepherd_score/accel/drivers/vol_color.py
# Fast batched ROCS/ROSHAMBO-style vol_color alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  directionless pharmacophore COLOR overlap.
#
# BOTH channels run on fused value(+gradient) kernels (Triton on CUDA, numba on CPU, via
# kernel dispatch) — exactly like the other modes — so vol_color runs at comparable speed:
#   * SHAPE  -> overlap_score_grad_se3_batch (same kernel as vol/esp_combo); returns the
#              overlap gradient w.r.t. the quaternion (dO_s/dq) directly.
#   * COLOR  -> pharm_color_score_grad_se3_batch with DIRECTIONLESS lookup tables
#              (build_lookup_tables(directionless=True) -> every real type is category 0, an
#              isotropic point Gaussian). This is an IN-REGISTER dO/dq kernel: it takes the
#              unit quaternion q, builds R(q) in registers, and returns dO_c/dq directly --
#              the SAME quaternion-space convention as the shape kernel, with NO R->q
#              projection tail. Value+gradient each step (NEED_GRAD=True -- the SAME launch).
#
# JOINT gradient (ROSHAMBO2 `combination` mode): the SE(3) step descends on
#   d/dpose [ (1-w)*shape_Tanimoto + w*color_Tanimoto ]
# i.e. BOTH channels steer the pose, not just shape. Because both kernels already emit the
# overlap gradient in the SAME quaternion space, each channel's Tanimoto chain-rule scale is
# applied straight to its dO/dq and the per-channel descent gradients are summed (see the
# fine loop below):
#   g_q = (1-w)*(-scale_s*dQ_s) + w*(-scale_c*dQ_c)
# There is NO apply_tanimoto_chain_rule / project_grad_R_to_quaternion host projection tail --
# the color kernel was moved in-register (like the `pharm` driver), so only the unit->raw
# quaternion normalization handled by fused_adam_qt_with_tangent_proj remains. Validated to
# ~1e-16 vs autograd, so the weighted sum is the exact combined-objective gradient. Matches
# the per-pair torch path (alignment._torch.optimize_vol_color_overlay), recovers self-copy 1.0.

from __future__ import annotations

import os
import torch
from typing import Optional, Tuple

# Use the fused single-kernel shape+color overlap (one launch instead of two) where it is a
# win -- i.e. small clouds (all pads <= VOL_COLOR_FUSED_MAX_PAD=32); larger molecules fall
# back to the two separate kernels (the fused kernel's two-channel register footprint blows
# occupancy at BLOCK=64). Disable with FINE_VOL_COLOR_FUSED=0.
_FINE_VOL_COLOR_FUSED = os.environ.get("FINE_VOL_COLOR_FUSED", "1") != "0"

from ..kernels.dispatch import (
    fused_adam_qt_with_tangent_proj, pharm_color_score_grad_se3_batch,
    overlap_score_grad_se3_batch,
)
from ._common import (
    check_gpu_available,
    batched_seeds_torch,
    build_coarse_grid,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
    ES_PATIENCE_OVERRIDE,
)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap, _FINE_GRAPHS, _GRAPH_MAX_P, _GRAPH_STEPS
from .esp_combo import _overlap_in_chunks_volumetric, _self_overlap_chunks
from ...score.analytical_gradients._torch import build_lookup_tables

# Padding type for pharmacophore slots: P_TYPES index 8 == 'Dummy' (lookup category 3 ->
# skipped by the kernel). Padded slots are also masked out by N_real, so padding is free.
_PHARM_PAD_TYPE = 8


@torch.no_grad()
def _color_overlap(q: torch.Tensor,
                   t: torch.Tensor,
                   types_1: torch.Tensor,
                   types_2: torch.Tensor,
                   anchors_1: torch.Tensor,
                   anchors_2: torch.Tensor,
                   tables: tuple,
                   N_real_ph: torch.Tensor,
                   M_real_ph: torch.Tensor) -> torch.Tensor:
    """Directionless color overlap O_AB (value-only) via the fused color kernel; the kernel
    takes the quaternion q directly (assumes |q|=1) and applies (R(q), t) internally."""
    al, Ks, cats = tables
    O, _, _ = pharm_color_score_grad_se3_batch(
        anchors_1, anchors_2, q, t, types_1, types_2,
        al, Ks, cats, N_real=N_real_ph, M_real=M_real_ph, NEED_GRAD=False)
    return O


def _vc_overlaps(c1, c2, a1, a2, q, t, pt1, pt2, tables, alpha, Nc, Mc, Na, Ma):
    """Both channels' value+grad: (VAB, dQ_s, dT_s, O_c, dQ_c, dT_c). Uses the FUSED single
    kernel (one launch, shared R(q)) when enabled, on CUDA, and every cloud fits the fused
    tile; otherwise the two separate kernels (which the fused kernel beats only for small
    clouds). Drop-in for the shape + color kernel pair in the fine loop and the graph step."""
    al, Ks, cats = tables
    if _FINE_VOL_COLOR_FUSED and c1.is_cuda:
        from ..kernels.vol_color_triton import (
            vol_color_score_grad_se3_batch, VOL_COLOR_FUSED_MAX_PAD)
        if max(c1.shape[1], c2.shape[1], a1.shape[1], a2.shape[1]) <= VOL_COLOR_FUSED_MAX_PAD:
            return vol_color_score_grad_se3_batch(
                c1, c2, a1, a2, q, t, pt1, pt2, al, Ks, cats, alpha=alpha,
                N_real_cent=Nc, M_real_cent=Mc, N_real_anc=Na, M_real_anc=Ma)
    VAB, dQ_s, dT_s = _overlap_in_chunks_volumetric(c1, c2, q, t, alpha=alpha, N_real=Nc, M_real=Mc)
    O_c, dQ_c, dT_c = pharm_color_score_grad_se3_batch(
        a1, a2, q, t, pt1, pt2, al, Ks, cats, N_real=Na, M_real=Ma, NEED_GRAD=True)
    return VAB, dQ_s, dT_s, O_c, dQ_c, dT_c


class _GraphedFineVolColor(_GraphedFineBase):
    """CUDA-graph fine loop for vol_color -- the worst host-overhead mode (TWO value+grad
    kernels per step: shape overlap_score_grad_se3_batch + directionless color
    pharm_color_score_grad_se3_batch). Both run inside the captured step, so the per-step
    host gap (and the second launch's host cost) disappears on replay. The combined-objective
    gradient g = (1-w)*(-scale_s*dO_s/dq) + w*(-scale_c*dO_c/dq) is built in-place, mirroring
    the eager loop exactly. Color lookup tables are a capture-time constant (directionless,
    identical across buckets) so they are held by reference, not reloaded."""

    def __init__(self, N_pad, M_pad, P_pad, Q_pad, P, steps, alpha, color_weight, lr,
                 tables, device):
        self.alpha = float(alpha); self.w = float(color_weight); self.lr = float(lr)
        self.al, self.Ks, self.cats = tables               # constant directionless color tables
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        # shape + color input clouds (persistent; loaded per bucket)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.anc1 = f(P, P_pad, 3); self.anc2 = f(P, Q_pad, 3)
        self.pt1 = i(P, P_pad); self.pt2 = i(P, Q_pad)     # pharm type indices (pre-cast int32)
        self.Nr = i(P); self.Mr = i(P)                     # shape-atom counts
        self.Npr = i(P); self.Mpr = i(P)                   # pharm-feature counts
        self.norm_s = f(P); self.norm_c = f(P)             # shape / color self-overlap sums
        # loop-carried state
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        # per-step temporaries (out= targets)
        self.denom_s = f(P); self.d2s = f(P); self.shape_sim = f(P); self.scale_s = f(P)
        self.denom_c = f(P); self.d2c = f(P); self.color_sim = f(P); self.scale_c = f(P)
        self.score = f(P); self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3); self.tmpq = f(P, 4); self.tmpt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB, dQs, dTs, O_c, dQc, dTc = _vc_overlaps(
            self.A, self.B, self.anc1, self.anc2, self.q, self.t, self.pt1, self.pt2,
            (self.al, self.Ks, self.cats), self.alpha, self.Nr, self.Mr, self.Npr, self.Mpr)
        # shape Tanimoto + d/dO_s scale
        torch.sub(self.norm_s, VAB, out=self.denom_s)
        torch.div(VAB, self.denom_s, out=self.shape_sim)
        torch.mul(self.denom_s, self.denom_s, out=self.d2s)
        torch.div(self.norm_s, self.d2s, out=self.scale_s)
        # color Tanimoto + d/dO_c scale
        torch.sub(self.norm_c, O_c, out=self.denom_c)
        torch.div(O_c, self.denom_c, out=self.color_sim)
        torch.mul(self.denom_c, self.denom_c, out=self.d2c)
        torch.div(self.norm_c, self.d2c, out=self.scale_c)
        # score = (1-w)*shape_sim + w*color_sim
        torch.mul(self.shape_sim, 1.0 - self.w, out=self.score)
        self.score.add_(self.color_sim, alpha=self.w)
        # best-pose tracking (in-place)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        # combined descent grad g = (1-w)*(-scale_s*dQs) + w*(-scale_c*dQc)
        torch.mul(dQs, self.scale_s.unsqueeze(1), out=self.gq); self.gq.mul_(-(1.0 - self.w))
        torch.mul(dQc, self.scale_c.unsqueeze(1), out=self.tmpq); self.tmpq.mul_(-self.w)
        self.gq.add_(self.tmpq)
        torch.mul(dTs, self.scale_s.unsqueeze(1), out=self.gt); self.gt.mul_(-(1.0 - self.w))
        torch.mul(dTc, self.scale_c.unsqueeze(1), out=self.tmpt); self.tmpt.mul_(-self.w)
        self.gt.add_(self.tmpt)
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, anc1, anc2, pt1, pt2, Nr, Mr, Npr, Mpr, norm_s, norm_c, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.anc1.copy_(anc1); self.anc2.copy_(anc2)
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


def _run_graphed_vol_color(A_k, B_k, anc1_k, anc2_k, pt1_k, pt2_k, N_k, M_k, Np_k, Mp_k,
                           norm_s, norm_c, q_seed, t_seed, tables, alpha, color_weight, lr,
                           steps, N_pad, M_pad, P_pad, Q_pad, P, es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_color", N_pad, M_pad, P_pad, Q_pad, P, steps,
           round(float(alpha), 4), round(float(color_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineVolColor(N_pad, M_pad, P_pad, Q_pad, P, steps, alpha,
                                     color_weight, lr, tables, A_k.device),
        key, (A_k, B_k, anc1_k, anc2_k, pt1_k, pt2_k, N_k, M_k, Np_k, Mp_k,
              norm_s, norm_c, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_vol_color_align_many(
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        ptype_1: torch.Tensor,
        ptype_2: torch.Tensor,
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float = 0.81,
        color_weight: float = 0.5,
        num_seeds: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_pharm: Optional[torch.Tensor] = None,
        M_real_pharm: Optional[torch.Tensor] = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized vol_color alignment over a batch of pairs (coarse-to-fine SE(3))."""
    # Honor the FINE_ES_PATIENCE env override (as pharm does); default patience=2 matches the
    # shape/surf drivers (validated bit-identical + accuracy-safe for fast-converging modes).
    early_stop_patience = ES_PATIENCE_OVERRIDE or early_stop_patience
    device = centers_1.device
    dtype = centers_1.dtype
    BATCH = centers_1.shape[0]
    N_pad_cent = centers_1.shape[1]
    M_pad_cent = centers_2.shape[1]
    P_pad = anchors_1.shape[1]
    Q_pad = anchors_2.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_cent, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_cent, dtype=torch.int32)
    if N_real_pharm is None:
        N_real_pharm = anchors_1.new_full((BATCH,), P_pad, dtype=torch.int32)
    if M_real_pharm is None:
        M_real_pharm = anchors_2.new_full((BATCH,), Q_pad, dtype=torch.int32)

    tables = build_lookup_tables(device, dtype, directionless=True)

    # ------------------------------------------------------------------
    # 1) pose hypotheses (seed from the SHAPE atom clouds, like vol)
    # ------------------------------------------------------------------
    if trans_centers is None:
        quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                             M_real_centers, num_seeds=num_seeds)
        P = quats.size(1)
        q_best = quats.clone()
        t_best = t_seeds.clone()
    else:
        q_grid, t_grid = build_coarse_grid(
            centers_1, centers_2, N_real_centers, M_real_centers, num_seeds=num_seeds,
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
            cent1 = centers_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad_cent, 3)
            cent2 = centers_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad_cent, 3)
            anc1 = anchors_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, P_pad, 3)
            anc2 = anchors_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, Q_pad, 3)
            pt1 = ptype_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, P_pad)
            pt2 = ptype_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, Q_pad)
            VAA_e = VAA.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VBB_e = VBB.unsqueeze(1).expand(-1, g_len).reshape(-1)
            Nc_e = N_real_centers.repeat_interleave(g_len)
            Mc_e = M_real_centers.repeat_interleave(g_len)
            Np_e = N_real_pharm.repeat_interleave(g_len)
            Mp_e = M_real_pharm.repeat_interleave(g_len)
            eye_e = torch.tensor([[1., 0., 0., 0.]], device=device).expand(cent1.shape[0], 4)
            zero_e = torch.zeros(cent1.shape[0], 3, device=device)
            cent2_t = apply_se3_transform(cent2, q_rep, t_rep)
            VAB_s, _, _ = _overlap_in_chunks_volumetric(
                cent1, cent2_t, eye_e, zero_e,
                alpha=alpha, N_real=Nc_e, M_real=Mc_e, NEED_GRAD=False)
            shape_sim = VAB_s / (VAA_e + VBB_e - VAB_s)
            O_c = _color_overlap(q_rep, t_rep, pt1, pt2, anc1, anc2, tables, Np_e, Mp_e)
            VAA_c_e = _color_overlap(eye_e, zero_e, pt1, pt1, anc1, anc1, tables, Np_e, Np_e)
            VBB_c_e = _color_overlap(eye_e, zero_e, pt2, pt2, anc2, anc2, tables, Mp_e, Mp_e)
            color_sim = O_c / (VAA_c_e + VBB_c_e - O_c)
            sc = (1.0 - color_weight) * shape_sim + color_weight * color_sim
            coarse_score[:, o0:o1] = sc.view(BATCH, g_len)
        best_idx = coarse_score.topk(k=topk, dim=1).indices
        q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
        t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
        P = topk

    # ------------------------------------------------------------------
    # 2) fine optimization over ALL P poses (shape gradient drives SE(3))
    # ------------------------------------------------------------------
    q_k = q_best.reshape(-1, 4).contiguous()
    t_k = t_best.reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_cent, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_cent, 3)
    anchors_1_k = anchors_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, P_pad, 3)
    anchors_2_k = anchors_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, Q_pad, 3)
    ptype_1_k = ptype_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, P_pad)
    ptype_2_k = ptype_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, Q_pad)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    Np_k = N_real_pharm.repeat_interleave(P)
    Mp_k = M_real_pharm.repeat_interleave(P)
    VAA_k = VAA.repeat_interleave(P)
    VBB_k = VBB.repeat_interleave(P)
    VAA_plus_VBB = VAA_k + VBB_k

    # Color self-overlaps are pose-invariant -> compute once via the same kernel.
    PK = q_k.shape[0]
    eye_q = torch.tensor([[1., 0., 0., 0.]], device=device).expand(PK, 4)
    zero_t = torch.zeros(PK, 3, device=device)
    al, Ks, cats = tables
    VAA_c = _color_overlap(eye_q, zero_t, ptype_1_k, ptype_1_k, anchors_1_k, anchors_1_k,
                           tables, Np_k, Np_k)
    VBB_c = _color_overlap(eye_q, zero_t, ptype_2_k, ptype_2_k, anchors_2_k, anchors_2_k,
                           tables, Mp_k, Mp_k)
    VAA_c_plus_VBB_c = VAA_c + VBB_c

    best_score = best_q = best_t = None

    # --- CUDA-graph fast path: capture the 2-kernel (shape+color) step, replay it. This is
    # the worst host-overhead mode (2 launches/step); graphing removes the per-step host gap.
    # Gated to the launch-bound small/medium-P CUDA fp32 regime (large P / capture failure
    # fall back to the eager loop below). See drivers/_graphed.
    # vol_color runs a 2nd (color) kernel per step, so it crosses over sooner than vol -> an
    # explicit work budget keeps its cap (~120k) below its ~150k crossover regardless of CEIL.
    if (_FINE_GRAPHS and centers_1_k.is_cuda
            and PK <= graph_cap(N_pad_cent * M_pad_cent, budget=30_000_000)
            and centers_1_k.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_vol_color(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                anchors_1_k.contiguous(), anchors_2_k.contiguous(),
                ptype_1_k, ptype_2_k, N_k, M_k, Np_k, Mp_k,
                VAA_plus_VBB, VAA_c_plus_VBB_c, q_k, t_k,
                (al, Ks, cats), alpha, color_weight, lr,
                steps_fine, N_pad_cent, M_pad_cent, P_pad, Q_pad, PK,
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
            # --- Shape + color value+grad: one fused kernel where it wins, else two. Both
            # channels emit dO/dq in the same quaternion space (no R->q projection tail). ---
            VAB, dQ_s, dT_s, O_c, dQ_c, dT_c = _vc_overlaps(
                centers_1_k, centers_2_k, anchors_1_k, anchors_2_k, q_k, t_k,
                ptype_1_k, ptype_2_k, (al, Ks, cats), alpha, N_k, M_k, Np_k, Mp_k)
            denom_s = VAA_plus_VBB - VAB
            shape_sim = VAB / denom_s
            scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)   # d shape_T / d O_s
            denom_c = VAA_c_plus_VBB_c - O_c
            color_sim = O_c / denom_c
            scale_c = (VAA_c_plus_VBB_c / (denom_c * denom_c)).unsqueeze(1)   # d color_T / d O_c

            score = (1.0 - color_weight) * shape_sim + color_weight * color_sim

            # --- Combined descent gradient: (1-w)*(-scale_s*dQ_s) + w*(-scale_c*dQ_c) ---
            # Both kernels emit dO/dq in the same quaternion space (validated vs autograd to ~1e-16);
            # -scale*dO/dq = -d(Tanimoto)/dq is the per-channel descent gradient.
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

            fused_adam_qt_with_tangent_proj(
                q_k, t_k, g_q, g_t, m_q, v_q, m_t, v_t, lr)

    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_vol_color_overlay_batch(
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        ref_types_batch: torch.Tensor,
        fit_types_batch: torch.Tensor,
        ref_ancs_batch: torch.Tensor,
        fit_ancs_batch: torch.Tensor,
        *,
        alpha: float = 0.81,
        color_weight: float = 0.5,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_pharm: Optional[torch.Tensor] = None,
        M_real_pharm: Optional[torch.Tensor] = None,
        trans_centers_batch: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched vol_color alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    device = ref_centers_batch.device
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_color_align_many(
        ref_centers_batch, fit_centers_batch,
        ref_types_batch, fit_types_batch,
        ref_ancs_batch, fit_ancs_batch,
        VAA, VBB,
        alpha=alpha, color_weight=color_weight,
        num_seeds=num_seeds,
        trans_centers=trans_centers_batch, trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_pharm=N_real_pharm, M_real_pharm=M_real_pharm,
    )
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores


def fast_optimize_vol_color_overlay(
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        ref_types: torch.Tensor,
        fit_types: torch.Tensor,
        ref_ancs: torch.Tensor,
        fit_ancs: torch.Tensor,
        *,
        alpha: float = 0.81,
        color_weight: float = 0.5,
        num_repeats: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pair fast vol_color alignment (drop-in for optimize_vol_color_overlay).

    Returns (aligned_fit_centers, SE3_4x4, score) on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    rt = ref_types.to(torch.int64).unsqueeze(0)
    ft = fit_types.to(torch.int64).unsqueeze(0)
    ra = ref_ancs.to(torch.float32).unsqueeze(0)
    fa = fit_ancs.to(torch.float32).unsqueeze(0)

    N_ph = torch.tensor([ref_ancs.shape[0]], device=device, dtype=torch.int32)
    M_ph = torch.tensor([fit_ancs.shape[0]], device=device, dtype=torch.int32)

    tcb = None; tcr = None
    if trans_centers is not None:
        tc = trans_centers.to(torch.float32)
        tcb = tc.unsqueeze(0)
        tcr = torch.tensor([tc.shape[0]], device=device, dtype=torch.int32)

    aligned, q_best, t_best, score = fast_optimize_vol_color_overlay_batch(
        rc, fc, rt, ft, ra, fa,
        alpha=alpha, color_weight=color_weight,
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        N_real_pharm=N_ph, M_real_pharm=M_ph,
        trans_centers_batch=tcb, trans_centers_real=tcr,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=num_repeats,
    )
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]
    return aligned[0].cpu(), SE3.cpu(), score[0].cpu()
