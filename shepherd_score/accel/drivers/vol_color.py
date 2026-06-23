# shepherd_score/accel/drivers/vol_color.py
# Fast batched ROCS/ROSHAMBO-style vol_color alignment:
#   atom-centred Gaussian SHAPE (volume) overlap  +  directionless pharmacophore COLOR overlap.
#
# BOTH channels run on fused value(+gradient) kernels (Triton on CUDA, numba on CPU, via
# kernel dispatch) — exactly like the other modes — so vol_color runs at comparable speed:
#   * SHAPE  -> overlap_score_grad_se3_batch (same kernel as vol/esp_combo); drives the SE(3)
#              gradient (scaled by 1-color_weight), Tanimoto chain rule.
#   * COLOR  -> pharm_score_grad_se3_batch with DIRECTIONLESS lookup tables
#              (build_lookup_tables(directionless=True) -> every real type is category 0, an
#              isotropic point Gaussian). Evaluated value+gradient each step (NEED_GRAD=True --
#              the SAME launch, no extra kernel dispatch).
#
# JOINT gradient (ROSHAMBO2 `combination` mode): the SE(3) step descends on
#   d/dpose [ (1-w)*shape_Tanimoto + w*color_Tanimoto ]
# i.e. BOTH channels steer the pose, not just shape. The shape kernel returns the overlap
# gradient w.r.t. the quaternion directly; the color kernel returns it w.r.t. the rotation
# matrix, which is converted to the quaternion via apply_tanimoto_chain_rule ->
# project_grad_R_to_quaternion -> normalization Jacobian (the same path the `pharm` driver
# uses). Both land in the same quaternion space (validated to ~1e-16 vs autograd), so the
# weighted sum is the exact combined-objective gradient. This matches the per-pair torch path
# (alignment._torch.optimize_vol_color_overlay) and recovers self-copy 1.0.

from __future__ import annotations

import torch
from typing import Optional, Tuple

from ..kernels.dispatch import fused_adam_qt_with_tangent_proj, pharm_color_score_grad_se3_batch
from ._common import (
    check_gpu_available,
    batched_seeds_torch,
    build_coarse_grid,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
    ES_PATIENCE_OVERRIDE,
)
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

    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

    best_score = torch.full((PK,), -float('inf'), device=device)
    best_q = q_k.clone()
    best_t = t_k.clone()

    prev_max_score = -float('inf')
    no_improve_count = 0

    for step in range(steps_fine):
        # --- Shape: value + quaternion gradient (fused kernel) ---
        VAB, dQ_s, dT_s = _overlap_in_chunks_volumetric(
            centers_1_k, centers_2_k, q_k, t_k, alpha=alpha, N_real=N_k, M_real=M_k)
        denom_s = VAA_plus_VBB - VAB
        shape_sim = VAB / denom_s
        scale_s = (VAA_plus_VBB / (denom_s * denom_s)).unsqueeze(1)   # d shape_T / d O_s

        # --- Color: value + QUATERNION gradient directly from the fused color kernel ---
        # (dQ_c = dO_c/dq, same convention as the shape kernel; no R->q projection tail)
        O_c, dQ_c, dT_c = pharm_color_score_grad_se3_batch(
            anchors_1_k, anchors_2_k, q_k, t_k, ptype_1_k, ptype_2_k,
            al, Ks, cats, N_real=Np_k, M_real=Mp_k, NEED_GRAD=True)
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
