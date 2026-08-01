# shepherd_score/accel/drivers/avoid.py
# Fast batched ``vol_avoid`` alignment:
#   atom-centred Gaussian SHAPE (volume) Tanimoto  MINUS  a linear-hard-sphere EXCLUDED-VOLUME
#   penalty against a fixed set of avoid points.
#
#   score = shape_Tanimoto  -  avoid_weight * A_pen                      (matches the per-pair
#   A_pen = sum_a sum_b relu((d0 - ||avoid_a - fit'_b||)/d0)             reference
#                                                                        alignment._torch.optimize_ROCS_overlay
#                                                                        with avoid_points set)
#
# This is the ONE mode whose second channel is NOT another Tanimoto overlap: the avoid term is a
# RAW, un-normalized penalty that is SUBTRACTED. So unlike vol_lipo/vol_color (two blended
# Tanimotos), here the shape channel keeps its FULL weight and the penalty's gradient enters with a
# POSITIVE sign and NO d(Tanimoto)/dO scale:
#   g_q = -scale_s * dQ_shape  +  avoid_weight * dQ_avoid          (scale_s = U/(U-VAB)^2 as usual)
#
# THIRD point cloud: ``avoid_points`` (K,3) is fixed in the reference frame and is NEITHER molecule.
# It rides the avoid kernel's A (ref) slot -- which the kernel never transforms -- while the
# fit-avoid cloud (defaulting to the fit SHAPE centres) rides the B (transformed) slot under the
# same SE(3) as the shape channel. A new hard-sphere value+grad kernel
# (overlap_score_grad_avoid_se3_batch, numba + Triton) supplies A_pen and dA_pen/dq.
#
# Scope: eager coarse-to-fine only (no CUDA-graph fine loop). The graph loop's blocked early-stop is
# tuned for smooth Gaussians; this objective is piecewise-linear (kinked at d=d0), so the eager loop
# -- which dispatches to the Triton kernels on CUDA and numba on CPU either way -- is the safe path.
# Graph capture can be added later as a pure throughput optimization.

from __future__ import annotations

import torch
from typing import Optional, Tuple

from ..kernels.dispatch import (
    fused_adam_qt_with_tangent_proj,
    overlap_score_grad_se3_batch,
    overlap_score_grad_avoid_se3_batch,
)
from ._common import (
    batched_seeds_torch,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
)
from .esp_combo import _self_overlap_chunks


def coarse_fine_vol_avoid_align_many(
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        avoid_points: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float = 0.81,
        avoid_min_dist: float = 2.0,
        avoid_weight: float = 1.0,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        K_real_avoid: Optional[torch.Tensor] = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized vol_avoid alignment over a batch of pairs (coarse-to-fine SE(3)).

    Seeds from the SHAPE atom clouds (identity + PCA + Fibonacci, COM-aligned), fine-optimises ALL
    seeds and takes the per-pair max (matching the vol driver). ``VAA``/``VBB`` are the shape
    self-overlaps; the avoid penalty has no self-overlap normalization. The fit-avoid cloud is the
    fit shape centres (``centers_2``), transformed under the same SE(3) as the shape channel;
    ``avoid_points`` stays fixed in the reference frame."""
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad = centers_1.shape[1]
    M_pad = centers_2.shape[1]
    K_pad = avoid_points.shape[1]

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad, dtype=torch.int32)
    if K_real_avoid is None:
        K_real_avoid = avoid_points.new_full((BATCH,), K_pad, dtype=torch.int32)

    # 1) pose hypotheses (seed from the SHAPE atom clouds, like vol)
    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)

    # 2) fine optimization over ALL P poses (shape Tanimoto - w * avoid penalty)
    q_k = quats.reshape(-1, 4).contiguous()
    t_k = t_seeds.reshape(-1, 3).contiguous()

    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad, 3)
    avoid_k = avoid_points.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, K_pad, 3)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    K_k = K_real_avoid.repeat_interleave(P)
    U = (VAA + VBB).repeat_interleave(P)
    has_avoid = (K_k > 0)
    w_a = torch.where(has_avoid, torch.full_like(U, float(avoid_weight)),
                      torch.zeros_like(U)).unsqueeze(1)      # zero penalty grad on no-avoid pairs

    PK = q_k.shape[0]
    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
    best_score = torch.full((PK,), -float('inf'), device=device)
    best_q = q_k.clone(); best_t = t_k.clone()
    prev_max_score = -float('inf'); no_improve_count = 0

    for step in range(steps_fine):
        # shape channel (Gaussian volume Tanimoto)
        VAB_s, dQ_s, dT_s = overlap_score_grad_se3_batch(
            centers_1_k, centers_2_k, q_k, t_k, alpha=alpha, N_real=N_k, M_real=M_k)
        # avoid channel (linear hard-sphere penalty). A = fixed avoid points, B = fit shape centres
        # (transformed by the SAME q,t). N_real = avoid count, M_real = fit-centre count.
        A_pen, dQ_a, dT_a = overlap_score_grad_avoid_se3_batch(
            avoid_k, centers_2_k, q_k, t_k, min_dist=avoid_min_dist, N_real=K_k, M_real=M_k)
        A_pen = torch.where(has_avoid, A_pen, torch.zeros_like(A_pen))

        denom_s = U - VAB_s
        shape_sim = VAB_s / denom_s
        scale_s = (U / (denom_s * denom_s)).unsqueeze(1)          # d shape_T / d O_s

        # score = shape_Tanimoto - w * A_pen  (raw penalty, NOT normalized)
        score = shape_sim - avoid_weight * A_pen

        # descent gradient of (1 - score): -scale_s*dQ_s + w*dQ_avoid  (penalty grad sign is +,
        # because the penalty is SUBTRACTED in the score)
        g_q = -scale_s * dQ_s + w_a * dQ_a
        g_t = -scale_s * dT_s + w_a * dT_a

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


def fast_optimize_vol_avoid_overlay_batch(
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        avoid_points_batch: torch.Tensor,
        *,
        alpha: float = 0.81,
        avoid_min_dist: float = 2.0,
        avoid_weight: float = 1.0,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        K_real_avoid: Optional[torch.Tensor] = None,
        steps_fine: int = 100,
        lr: float = 0.075,
        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched vol_avoid alignment. Returns (aligned_fit_centers, q_best, t_best, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_vol_avoid_align_many(
        ref_centers_batch, fit_centers_batch, avoid_points_batch, VAA, VBB,
        alpha=alpha, avoid_min_dist=avoid_min_dist, avoid_weight=avoid_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers, K_real_avoid=K_real_avoid,
    )
    aligned = apply_se3_transform(fit_centers_batch, q_best, t_best)
    return aligned, q_best, t_best, scores


def fast_optimize_vol_avoid_overlay(
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        avoid_points: torch.Tensor,
        *,
        alpha: float = 0.81,
        avoid_min_dist: float = 2.0,
        avoid_weight: float = 1.0,
        num_repeats: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-pair fast vol_avoid alignment (drop-in for optimize_ROCS_overlay with avoid_points).

    Returns (aligned_fit_centers, SE3_4x4, score) on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    av = avoid_points.to(torch.float32).unsqueeze(0)

    aligned, q_best, t_best, score = fast_optimize_vol_avoid_overlay_batch(
        rc, fc, av,
        alpha=alpha, avoid_min_dist=avoid_min_dist, avoid_weight=avoid_weight,
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        K_real_avoid=torch.tensor([avoid_points.shape[0]], device=device, dtype=torch.int32),
        steps_fine=steps_fine, lr=lr, num_seeds=num_repeats,
    )
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]
    return aligned[0].cpu(), SE3.cpu(), score[0].cpu()
