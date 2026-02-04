# shepherd_score/alignment_utils/fast_surface_se3.py
# Fast GPU-accelerated surface alignment using the existing Triton kernel.
# Surface alignment uses the same Gaussian overlap math as volumetric,
# just with surface points and surface-specific alpha values.

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..score.gaussian_overlap_triton import (
    overlap_score_grad_se3_batch,
    fused_adam_qt,
    fused_adam_qt_with_tangent_proj,
    _batch_self_overlap
)
from .fast_common import (
    check_gpu_available,
    legacy_seeds_torch,
    build_coarse_grid,
    quat_mul,
    apply_se3_transform,
    quaternion_to_rotation_matrix
)


@torch.no_grad()
def _overlap_in_chunks_surface(A, B, q, t, *, alpha: float,
                               N_real: torch.Tensor,
                               M_real: torch.Tensor,
                               NEED_GRAD: bool = True,
                               BLOCK: int = 32):
    """
    Evaluate the fused overlap kernel in chunks respecting CUDA grid limits.
    Uses smaller BLOCK size (32) for surface point clouds which are typically smaller.

    Parameters
    ----------
    A, B : torch.Tensor (K, N, 3) / (K, M, 3)
        Padded coordinate blocks
    q, t : torch.Tensor (K, 4) / (K, 3)
        Quaternions and translations
    alpha : float
        Gaussian width parameter for surface
    N_real, M_real : torch.Tensor (K,)
        True point counts
    NEED_GRAD : bool
        Whether to compute gradients
    BLOCK : int
        Tile size (smaller for surface)

    Returns
    -------
    VAB : torch.Tensor (K,)
    dQ : torch.Tensor (K, 4)
    dT : torch.Tensor (K, 3)
    """
    K = A.shape[0]
    N_real = N_real.to(torch.int32).contiguous()
    M_real = M_real.to(torch.int32).contiguous()

    out_V = torch.empty(K, device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)

    CHUNK = 65_535  # CUDA grid-z hard limit

    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)

        V, dQ, dT = overlap_score_grad_se3_batch(
            A[start:end], B[start:end],
            q[start:end], t[start:end],
            alpha=alpha,
            N_real=N_real[start:end],
            M_real=M_real[start:end],
            NEED_GRAD=NEED_GRAD,
            BLOCK=BLOCK)

        out_V[start:end] = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


def _self_overlap_surface_chunks(P_pad, N_real, alpha, BLOCK=32):
    """Compute self-overlap in chunks for surface point clouds."""
    K = P_pad.size(0)
    CHUNK = 65_535
    V_all = torch.empty(K, device=P_pad.device, dtype=P_pad.dtype)

    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(P_pad[s:e], N_real[s:e], alpha)

    return V_all


def coarse_fine_surface_align_many(
        A_batch: torch.Tensor,
        B_batch: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized padding-aware surface alignment over a batch of (A, B) pairs.

    Uses the same coarse-to-fine strategy as volumetric alignment:
    1. Build 500 pose hypotheses (250 rotations × 2 translations)
    2. Evaluate all poses (coarse scoring)
    3. Select top-k poses
    4. Fine optimization with Adam

    Parameters
    ----------
    A_batch, B_batch : torch.Tensor (B, N_pad, 3) / (B, M_pad, 3)
        Surface point coordinates
    VAA, VBB : torch.Tensor (B,)
        Pre-computed Gaussian self-overlaps
    alpha : float
        Surface-specific Gaussian width parameter
    topk : int
        Number of top poses to refine
    steps_fine : int
        Number of fine optimization steps
    lr : float
        Learning rate for Adam optimizer
    N_real, M_real : torch.Tensor (B,)
        Optional true point counts
    early_stop_patience : int
        Number of iterations without improvement before stopping
    early_stop_tol : float
        Minimum improvement threshold

    Returns
    -------
    final_score : torch.Tensor (B,)
        Best Tanimoto scores
    q_best : torch.Tensor (B, 4)
        Best quaternions
    t_best : torch.Tensor (B, 3)
        Best translations
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _, M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) Build coarse grid of 500 poses
    # ------------------------------------------------------------------
    q_grid, t_grid = build_coarse_grid(A_batch, B_batch, N_real, M_real, num_seeds=50)
    G = q_grid.size(1)  # 500 orientations

    # ------------------------------------------------------------------
    # 2) Coarse evaluation (micro-batched for memory)
    # ------------------------------------------------------------------
    ORI_CHUNK = 25_000
    PAIR_CHUNK = 65_535
    coarse_score = torch.empty(BATCH, G, device=device, dtype=A_batch.dtype)

    for o0 in range(0, G, ORI_CHUNK):
        o1 = min(o0 + ORI_CHUNK, G)
        g_len = o1 - o0

        for p0 in range(0, BATCH, PAIR_CHUNK):
            p1 = min(p0 + PAIR_CHUNK, BATCH)
            slice_len = p1 - p0

            A_rep = A_batch[p0:p1].unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad, 3)
            B_rep = B_batch[p0:p1].unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad, 3)
            q_rep = q_grid[p0:p1, o0:o1].reshape(-1, 4).contiguous()
            t_rep = t_grid[p0:p1, o0:o1].reshape(-1, 3).contiguous()

            N_rep = N_real[p0:p1].repeat_interleave(g_len)
            M_rep = M_real[p0:p1].repeat_interleave(g_len)

            VAB_slice, _, _ = _overlap_in_chunks_surface(
                A_rep, B_rep, q_rep, t_rep,
                alpha=alpha, N_real=N_rep, M_real=M_rep, NEED_GRAD=False)

            coarse_score[p0:p1, o0:o1] = VAB_slice.view(slice_len, g_len)

    # Convert to Tanimoto
    coarse_score = coarse_score / (VAA[:, None] + VBB[:, None] - coarse_score)

    # ------------------------------------------------------------------
    # 3) Select top-k orientations
    # ------------------------------------------------------------------
    best_idx = coarse_score.topk(k=topk, dim=1).indices
    q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
    t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()

    # ------------------------------------------------------------------
    # 4) Fine optimization with Adam
    # ------------------------------------------------------------------
    A_k = A_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, M_pad, 3)
    q_k = q_best.view(-1, 4)
    t_k = t_best.view(-1, 3)

    N_k = N_real.repeat_interleave(topk)
    M_k = M_real.repeat_interleave(topk)
    VAA_rep = VAA.repeat_interleave(topk)
    VBB_rep = VBB.repeat_interleave(topk)
    VAA_plus_VBB = VAA_rep + VBB_rep

    # Adam state
    m_q = torch.zeros_like(q_k)
    v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k)
    v_t = torch.zeros_like(t_k)

    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q = q_k.clone()
    best_t = t_k.clone()

    # Early stopping state
    prev_max_score = -float('inf')
    no_improve_count = 0

    for step in range(steps_fine):
        VAB, dQ, dT = _overlap_in_chunks_surface(
            A_k, B_k, q_k, t_k,
            alpha=alpha, N_real=N_k, M_real=M_k)

        denom = VAA_plus_VBB - VAB
        score = VAB / denom
        scale = VAA_plus_VBB / (denom * denom)

        # Track best
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        mask_q = better.unsqueeze(1)
        best_q = torch.where(mask_q, q_k, best_q)
        best_t = torch.where(mask_q, t_k, best_t)

        # Early stopping check every 5 iterations to reduce GPU→CPU sync overhead
        if step % 5 == 0:
            current_max = best_score.max().item()
            if current_max - prev_max_score < early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    break
            else:
                no_improve_count = 0
                prev_max_score = current_max

        # Fused Adam with tangent-space projection (avoids intermediate dQ_tan tensor)
        fused_adam_qt_with_tangent_proj(
            q_k, t_k,
            -dQ * scale.unsqueeze(1),
            -dT * scale.unsqueeze(1),
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


def fast_optimize_ROCS_overlay(
        ref_points: torch.Tensor,
        fit_points: torch.Tensor,
        alpha: float,
        num_repeats: int = 50,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated surface/ROCS alignment.

    Drop-in replacement for optimize_ROCS_overlay with GPU acceleration.
    Falls back to CPU implementation if CUDA is not available.

    Parameters
    ----------
    ref_points : torch.Tensor (N, 3)
        Reference surface/atom points
    fit_points : torch.Tensor (M, 3)
        Points to align
    alpha : float
        Gaussian width parameter
    num_repeats : int
        Number of seeds (not directly used, coarse grid is always 500)
    topk : int
        Number of top poses to refine
    steps_fine : int
        Number of fine optimization steps
    lr : float
        Learning rate

    Returns
    -------
    aligned_points : torch.Tensor (M, 3)
        Transformed fit points
    SE3_transform : torch.Tensor (4, 4)
        Best SE(3) transformation matrix
    score : torch.Tensor scalar
        Best Tanimoto score
    """
    if not check_gpu_available():
        # Fallback to CPU implementation
        from ..alignment import optimize_ROCS_overlay
        return optimize_ROCS_overlay(ref_points, fit_points, alpha, num_repeats, **kwargs)

    device = torch.device('cuda')
    ref_gpu = ref_points.to(device, dtype=torch.float32)
    fit_gpu = fit_points.to(device, dtype=torch.float32)

    # Batch dimension for single pair
    A = ref_gpu.unsqueeze(0)
    B = fit_gpu.unsqueeze(0)
    N_real = torch.tensor([ref_gpu.shape[0]], device=device, dtype=torch.int32)
    M_real = torch.tensor([fit_gpu.shape[0]], device=device, dtype=torch.int32)

    # Precompute self-overlaps
    VAA = _self_overlap_surface_chunks(A, N_real, alpha)
    VBB = _self_overlap_surface_chunks(B, M_real, alpha)

    # Run coarse-fine alignment
    score, q_best, t_best = coarse_fine_surface_align_many(
        A, B, VAA, VBB,
        alpha=alpha,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
        N_real=N_real,
        M_real=M_real)

    # Apply best transform
    aligned = apply_se3_transform(fit_gpu, q_best[0], t_best[0])

    # Build SE(3) matrix
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]

    return aligned.cpu(), SE3.cpu(), score[0].cpu()


def fast_optimize_ROCS_overlay_batch(
        ref_batch: torch.Tensor,
        fit_batch: torch.Tensor,
        alpha: float,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated batch surface alignment.

    Parameters
    ----------
    ref_batch : torch.Tensor (B, N, 3)
        Batch of reference points
    fit_batch : torch.Tensor (B, M, 3)
        Batch of fit points
    alpha : float
        Gaussian width parameter
    N_real, M_real : torch.Tensor (B,)
        True point counts (for padded batches)
    topk : int
        Number of top poses to refine
    steps_fine : int
        Fine optimization steps
    lr : float
        Learning rate

    Returns
    -------
    aligned_batch : torch.Tensor (B, M, 3)
        Transformed fit points
    q_batch : torch.Tensor (B, 4)
        Best quaternions
    t_batch : torch.Tensor (B, 3)
        Best translations
    scores : torch.Tensor (B,)
        Best Tanimoto scores
    """
    device = ref_batch.device
    BATCH = ref_batch.shape[0]
    N_pad = ref_batch.shape[1]
    M_pad = fit_batch.shape[1]

    if N_real is None:
        N_real = ref_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = fit_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # Precompute self-overlaps
    VAA = _self_overlap_surface_chunks(ref_batch, N_real, alpha)
    VBB = _self_overlap_surface_chunks(fit_batch, M_real, alpha)

    # Run coarse-fine alignment
    scores, q_best, t_best = coarse_fine_surface_align_many(
        ref_batch, fit_batch, VAA, VBB,
        alpha=alpha,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
        N_real=N_real,
        M_real=M_real)

    # Apply transforms
    aligned_batch = apply_se3_transform(fit_batch, q_best, t_best)

    return aligned_batch, q_best, t_best, scores
