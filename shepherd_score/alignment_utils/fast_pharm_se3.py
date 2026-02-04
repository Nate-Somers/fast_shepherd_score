# shepherd_score/alignment_utils/fast_pharm_se3.py
# Fast GPU-accelerated pharmacophore alignment.

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..score.pharmacophore_overlap_triton import (
    batch_pharm_tanimoto,
    batch_pharm_overlap_with_transform,
    batch_pharm_self_overlap
)
from ..score.gaussian_overlap_triton import fused_adam_qt
from .fast_common import (
    check_gpu_available,
    legacy_seeds_torch,
    build_coarse_grid,
    quat_mul,
    apply_se3_transform,
    apply_so3_transform,
    quaternion_to_rotation_matrix
)


def coarse_fine_pharm_align_many(
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        vectors_1: torch.Tensor,
        vectors_2: torch.Tensor,
        types_1: torch.Tensor,
        types_2: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized pharmacophore alignment over a batch of (A, B) pairs.

    Uses coarse-to-fine strategy:
    1. Build 500 pose hypotheses (250 rotations × 2 translations)
    2. Evaluate all poses (coarse scoring)
    3. Select top-k poses
    4. Fine optimization with Adam

    Parameters
    ----------
    anchors_1, anchors_2 : torch.Tensor (B, N_pad, 3) / (B, M_pad, 3)
        Pharmacophore anchor positions
    vectors_1, vectors_2 : torch.Tensor (B, N_pad, 3) / (B, M_pad, 3)
        Pharmacophore direction vectors
    types_1, types_2 : torch.Tensor (B, N_pad) / (B, M_pad)
        Pharmacophore type indices
    VAA, VBB : torch.Tensor (B,)
        Pre-computed self-overlaps
    topk : int
        Number of top poses to refine
    steps_fine : int
        Number of fine optimization steps
    lr : float
        Learning rate for Adam optimizer
    N_real, M_real : torch.Tensor (B,)
        Optional true feature counts

    Returns
    -------
    final_score : torch.Tensor (B,)
        Best Tanimoto scores
    q_best : torch.Tensor (B, 4)
        Best quaternions
    t_best : torch.Tensor (B, 3)
        Best translations
    """
    device = anchors_1.device
    BATCH, N_pad, _ = anchors_1.shape
    _, M_pad, _ = anchors_2.shape

    if N_real is None:
        N_real = anchors_1.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = anchors_2.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) Build coarse grid of 500 poses
    # ------------------------------------------------------------------
    q_grid, t_grid = build_coarse_grid(anchors_1, anchors_2, N_real, M_real, num_seeds=50)
    G = q_grid.size(1)  # 500 orientations

    # ------------------------------------------------------------------
    # 2) Coarse evaluation
    # ------------------------------------------------------------------
    ORI_CHUNK = 10_000  # Smaller chunks due to pharmacophore complexity
    coarse_score = torch.empty(BATCH, G, device=device, dtype=anchors_1.dtype)

    for o0 in range(0, G, ORI_CHUNK):
        o1 = min(o0 + ORI_CHUNK, G)
        g_len = o1 - o0

        # Expand data for this chunk
        q_rep = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
        t_rep = t_grid[:, o0:o1].reshape(-1, 3).contiguous()

        # Reshape for batch processing
        anchors_1_exp = anchors_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad, 3)
        anchors_2_exp = anchors_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad, 3)
        vectors_1_exp = vectors_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad, 3)
        vectors_2_exp = vectors_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad, 3)
        types_1_exp = types_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, N_pad)
        types_2_exp = types_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, M_pad)

        N_rep = N_real.repeat_interleave(g_len)
        M_rep = M_real.repeat_interleave(g_len)
        VAA_exp = VAA.repeat_interleave(g_len)
        VBB_exp = VBB.repeat_interleave(g_len)

        # Compute overlap with transform
        VAB_slice, _, _ = batch_pharm_overlap_with_transform(
            anchors_1_exp, anchors_2_exp,
            vectors_1_exp, vectors_2_exp,
            types_1_exp, types_2_exp,
            q_rep, t_rep,
            N_real=N_rep, M_real=M_rep)

        # Tanimoto score
        denom = VAA_exp + VBB_exp - VAB_slice
        scores_slice = torch.where(denom > 1e-8, VAB_slice / denom, torch.zeros_like(VAB_slice))

        coarse_score[:, o0:o1] = scores_slice.view(BATCH, g_len)

    # ------------------------------------------------------------------
    # 3) Select top-k orientations
    # ------------------------------------------------------------------
    best_idx = coarse_score.topk(k=topk, dim=1).indices
    q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
    t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()

    # ------------------------------------------------------------------
    # 4) Fine optimization with Adam
    # ------------------------------------------------------------------
    q_k = q_best.view(-1, 4)
    t_k = t_best.view(-1, 3)

    # Expand data for topk
    anchors_1_k = anchors_1.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, N_pad, 3)
    anchors_2_k = anchors_2.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, M_pad, 3)
    vectors_1_k = vectors_1.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, N_pad, 3)
    vectors_2_k = vectors_2.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, M_pad, 3)
    types_1_k = types_1.unsqueeze(1).expand(-1, topk, -1).reshape(-1, N_pad)
    types_2_k = types_2.unsqueeze(1).expand(-1, topk, -1).reshape(-1, M_pad)

    N_k = N_real.repeat_interleave(topk)
    M_k = M_real.repeat_interleave(topk)
    VAA_k = VAA.repeat_interleave(topk)
    VBB_k = VBB.repeat_interleave(topk)

    # Adam state
    m_q = torch.zeros_like(q_k)
    v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k)
    v_t = torch.zeros_like(t_k)

    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q = q_k.clone()
    best_t = t_k.clone()

    prev_max_score = -float('inf')
    no_improve_count = 0

    for step in range(steps_fine):
        # Compute overlap with current transform
        VAB, _, _ = batch_pharm_overlap_with_transform(
            anchors_1_k, anchors_2_k,
            vectors_1_k, vectors_2_k,
            types_1_k, types_2_k,
            q_k, t_k,
            N_real=N_k, M_real=M_k)

        # Tanimoto score
        denom = VAA_k + VBB_k - VAB
        score = torch.where(denom > 1e-8, VAB / denom, torch.zeros_like(VAB))

        # Compute numerical gradients for a few steps
        # (pharmacophore overlap doesn't have analytical SE3 gradients in this implementation)
        eps = 1e-4

        # Quaternion gradients (finite difference)
        dQ = torch.zeros_like(q_k)
        for i in range(4):
            q_plus = q_k.clone()
            q_plus[:, i] += eps
            q_plus = F.normalize(q_plus, dim=1)

            VAB_plus, _, _ = batch_pharm_overlap_with_transform(
                anchors_1_k, anchors_2_k,
                vectors_1_k, vectors_2_k,
                types_1_k, types_2_k,
                q_plus, t_k,
                N_real=N_k, M_real=M_k)
            denom_plus = VAA_k + VBB_k - VAB_plus
            score_plus = torch.where(denom_plus > 1e-8, VAB_plus / denom_plus, torch.zeros_like(VAB_plus))

            dQ[:, i] = (score_plus - score) / eps

        # Translation gradients (finite difference)
        dT = torch.zeros_like(t_k)
        for i in range(3):
            t_plus = t_k.clone()
            t_plus[:, i] += eps

            VAB_plus, _, _ = batch_pharm_overlap_with_transform(
                anchors_1_k, anchors_2_k,
                vectors_1_k, vectors_2_k,
                types_1_k, types_2_k,
                q_k, t_plus,
                N_real=N_k, M_real=M_k)
            denom_plus = VAA_k + VBB_k - VAB_plus
            score_plus = torch.where(denom_plus > 1e-8, VAB_plus / denom_plus, torch.zeros_like(VAB_plus))

            dT[:, i] = (score_plus - score) / eps

        # Track best
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        mask_q = better.unsqueeze(1)
        best_q = torch.where(mask_q, q_k, best_q)
        best_t = torch.where(mask_q, t_k, best_t)

        # Early stopping
        current_max = best_score.max().item()
        if current_max - prev_max_score < early_stop_tol:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                break
        else:
            no_improve_count = 0
            prev_max_score = current_max

        # Tangent-space projection for quaternion
        radial = (dQ * q_k).sum(dim=1, keepdim=True)
        dQ_tan = dQ - q_k * radial

        # Adam update (using fused kernel for efficiency)
        fused_adam_qt(
            q_k, t_k,
            dQ_tan,  # Ascent (maximize score)
            dT,
            m_q, v_q, m_t, v_t, lr)

    # ------------------------------------------------------------------
    # 5) Gather final results
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, topk)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * topk

    return (final_score.flatten()[sel],
            best_q.view(BATCH, topk, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, topk, 3)[torch.arange(BATCH), best])


def fast_optimize_pharm_overlay(
        ref_pharms: torch.Tensor,
        fit_pharms: torch.Tensor,
        ref_anchors: torch.Tensor,
        fit_anchors: torch.Tensor,
        ref_vectors: torch.Tensor,
        fit_vectors: torch.Tensor,
        similarity: str = 'tanimoto',
        extended_points: bool = False,
        only_extended: bool = False,
        num_repeats: int = 50,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated pharmacophore alignment.

    Drop-in replacement for optimize_pharm_overlay with GPU acceleration.
    Falls back to CPU implementation if CUDA is not available.

    Parameters
    ----------
    ref_pharms : torch.Tensor (N,)
        Reference pharmacophore type indices
    fit_pharms : torch.Tensor (M,)
        Fit pharmacophore type indices
    ref_anchors : torch.Tensor (N, 3)
        Reference anchor positions
    fit_anchors : torch.Tensor (M, 3)
        Fit anchor positions
    ref_vectors : torch.Tensor (N, 3)
        Reference direction vectors
    fit_vectors : torch.Tensor (M, 3)
        Fit direction vectors
    similarity : str
        Similarity function ('tanimoto', 'tversky', etc.)
    extended_points : bool
        Use extended points for HBA/HBD
    only_extended : bool
        Only score extended points
    num_repeats : int
        Number of seeds
    topk : int
        Number of top poses to refine
    steps_fine : int
        Fine optimization steps
    lr : float
        Learning rate

    Returns
    -------
    aligned_anchors : torch.Tensor (M, 3)
        Transformed fit anchors
    aligned_vectors : torch.Tensor (M, 3)
        Rotated fit vectors
    SE3_transform : torch.Tensor (4, 4)
        Best SE(3) transformation matrix
    score : torch.Tensor scalar
        Best similarity score
    """
    if not check_gpu_available():
        from ..alignment import optimize_pharm_overlay
        return optimize_pharm_overlay(
            ref_pharms, fit_pharms,
            ref_anchors, fit_anchors,
            ref_vectors, fit_vectors,
            similarity, extended_points, only_extended,
            num_repeats, **kwargs)

    device = torch.device('cuda')

    # Move to GPU and add batch dimension
    ref_anchors_gpu = ref_anchors.to(device, dtype=torch.float32).unsqueeze(0)
    fit_anchors_gpu = fit_anchors.to(device, dtype=torch.float32).unsqueeze(0)
    ref_vectors_gpu = ref_vectors.to(device, dtype=torch.float32).unsqueeze(0)
    fit_vectors_gpu = fit_vectors.to(device, dtype=torch.float32).unsqueeze(0)
    ref_types_gpu = ref_pharms.to(device, dtype=torch.int64).unsqueeze(0)
    fit_types_gpu = fit_pharms.to(device, dtype=torch.int64).unsqueeze(0)

    N_real = torch.tensor([ref_anchors.shape[0]], device=device, dtype=torch.int32)
    M_real = torch.tensor([fit_anchors.shape[0]], device=device, dtype=torch.int32)

    # Precompute self-overlaps
    VAA = batch_pharm_self_overlap(ref_anchors_gpu, ref_vectors_gpu, ref_types_gpu, N_real)
    VBB = batch_pharm_self_overlap(fit_anchors_gpu, fit_vectors_gpu, fit_types_gpu, M_real)

    # Run alignment
    score, q_best, t_best = coarse_fine_pharm_align_many(
        ref_anchors_gpu, fit_anchors_gpu,
        ref_vectors_gpu, fit_vectors_gpu,
        ref_types_gpu, fit_types_gpu,
        VAA, VBB,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
        N_real=N_real,
        M_real=M_real)

    # Apply transform
    aligned_anchors = apply_se3_transform(fit_anchors_gpu[0], q_best[0], t_best[0])
    aligned_vectors = apply_so3_transform(fit_vectors_gpu[0], q_best[0])

    # Build SE(3) matrix
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]

    return (aligned_anchors.cpu(),
            aligned_vectors.cpu(),
            SE3.cpu(),
            score[0].cpu())


def fast_optimize_pharm_overlay_batch(
        ref_pharms_batch: torch.Tensor,
        fit_pharms_batch: torch.Tensor,
        ref_anchors_batch: torch.Tensor,
        fit_anchors_batch: torch.Tensor,
        ref_vectors_batch: torch.Tensor,
        fit_vectors_batch: torch.Tensor,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated batch pharmacophore alignment.

    Parameters
    ----------
    ref_pharms_batch : torch.Tensor (B, N)
        Batch of reference type indices
    fit_pharms_batch : torch.Tensor (B, M)
        Batch of fit type indices
    ref_anchors_batch : torch.Tensor (B, N, 3)
        Batch of reference anchors
    fit_anchors_batch : torch.Tensor (B, M, 3)
        Batch of fit anchors
    ref_vectors_batch : torch.Tensor (B, N, 3)
        Batch of reference vectors
    fit_vectors_batch : torch.Tensor (B, M, 3)
        Batch of fit vectors
    N_real, M_real : torch.Tensor (B,)
        True feature counts
    topk : int
        Number of top poses
    steps_fine : int
        Fine optimization steps
    lr : float
        Learning rate

    Returns
    -------
    aligned_anchors : torch.Tensor (B, M, 3)
    aligned_vectors : torch.Tensor (B, M, 3)
    q_batch : torch.Tensor (B, 4)
    t_batch : torch.Tensor (B, 3)
    scores : torch.Tensor (B,)
    """
    device = ref_anchors_batch.device
    BATCH = ref_anchors_batch.shape[0]
    N_pad = ref_anchors_batch.shape[1]
    M_pad = fit_anchors_batch.shape[1]

    if N_real is None:
        N_real = ref_anchors_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = fit_anchors_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # Precompute self-overlaps
    VAA = batch_pharm_self_overlap(ref_anchors_batch, ref_vectors_batch, ref_pharms_batch, N_real)
    VBB = batch_pharm_self_overlap(fit_anchors_batch, fit_vectors_batch, fit_pharms_batch, M_real)

    # Run alignment
    scores, q_best, t_best = coarse_fine_pharm_align_many(
        ref_anchors_batch, fit_anchors_batch,
        ref_vectors_batch, fit_vectors_batch,
        ref_pharms_batch, fit_pharms_batch,
        VAA, VBB,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
        N_real=N_real,
        M_real=M_real)

    # Apply transforms
    aligned_anchors = apply_se3_transform(fit_anchors_batch, q_best, t_best)
    aligned_vectors = apply_so3_transform(fit_vectors_batch, q_best)

    return aligned_anchors, aligned_vectors, q_best, t_best, scores
