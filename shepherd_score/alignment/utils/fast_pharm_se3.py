# shepherd_score/alignment/utils/fast_pharm_se3.py
# Fast GPU-accelerated pharmacophore alignment.

import torch
from typing import Tuple, Optional

from ...score.pharmacophore_overlap_triton import (
    batch_pharm_cross_overlap_with_transform,
    batch_pharm_self_overlap,
    pharm_similarity_from_overlaps,
)
try:
    from ...score.gaussian_overlap_triton import fused_adam_qt
    from ...score.pharmacophore_grad_triton import pharm_score_grad_se3_batch
except ImportError:
    # CPU-only box (no triton): numba pharm value+grad kernel + torch Adam.
    # (pharmacophore_overlap_triton is pure PyTorch, so its imports above need no guard.)
    from .cpu_overlap import fused_adam_qt, pharm_score_grad_se3_batch
from ...score.analytical_gradients._torch import build_lookup_tables
from . import fast_common as _fc
from .fast_common import (
    check_gpu_available,
    build_coarse_grid,
    batched_seeds_torch,
    apply_se3_transform,
    apply_so3_transform,
    quaternion_to_rotation_matrix
)

# Analytical (autograd-free) pharmacophore value+gradient, reused from the
# upstream analytical-gradients module. Used to drive the fine loop without
# building a per-step autograd graph (only valid for un-padded inputs; the
# padded batch path keeps the autograd fallback).
from ...score.analytical_gradients import (
    compute_overlap_and_grad_pharm,
    compute_self_overlaps_pharm,
    apply_tanimoto_chain_rule,
    apply_tversky_chain_rule,
    project_grad_R_to_quaternion,
    _rotation_matrix_from_unit_quat,
)

_PHARM_SIGMA_MAP = {'tversky': 0.95, 'tversky_ref': 1.0, 'tversky_fit': 0.05}


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
        similarity: str = "tanimoto",
        extended_points: bool = False,
        only_extended: bool = False,
        num_seeds: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
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
    # 1) pose hypotheses
    # ------------------------------------------------------------------
    if trans_centers is None:
        # Reference seed set (identity + 4 PCA + Fibonacci, COM translation);
        # fine-optimise ALL seeds and take the per-pair max -- NO coarse-grid +
        # top-k pruning. Pruning on the raw (un-optimised) pharmacophore overlap
        # repeatedly discarded the true basin for pseudo-symmetric molecules,
        # pulling the score well below the reference (worst case > 0.6). See
        # coarse_fine_align_many for the full rationale.
        quats, t_seeds = batched_seeds_torch(anchors_1, anchors_2, N_real, M_real,
                                             num_seeds=num_seeds)
        P = quats.size(1)
        q_best = quats.clone()
        t_best = t_seeds.clone()
    else:
        # Legacy translation-seeded path: coarse grid + top-k pruning (unchanged).
        q_grid, t_grid = build_coarse_grid(
            anchors_1, anchors_2, N_real, M_real, num_seeds=num_seeds,
            trans_centers_batch=trans_centers, trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans,
        )
        G = q_grid.size(1)
        ORI_CHUNK = 10_000  # Smaller chunks due to pharmacophore complexity
        coarse_score = torch.empty(BATCH, G, device=device, dtype=anchors_1.dtype)
        with torch.no_grad():
            for o0 in range(0, G, ORI_CHUNK):
                o1 = min(o0 + ORI_CHUNK, G)
                g_len = o1 - o0
                q_rep = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
                t_rep = t_grid[:, o0:o1].reshape(-1, 3).contiguous()
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
                VAB_slice = batch_pharm_cross_overlap_with_transform(
                    anchors_1_exp, anchors_2_exp, vectors_1_exp, vectors_2_exp,
                    types_1_exp, types_2_exp, q_rep, t_rep,
                    extended_points=extended_points, only_extended=only_extended,
                    N_real=N_rep, M_real=M_rep,
                )
                scores_slice = pharm_similarity_from_overlaps(
                    VAB_slice, VAA_exp, VBB_exp, similarity=similarity,
                )
                coarse_score[:, o0:o1] = scores_slice.view(BATCH, g_len)
        best_idx = coarse_score.topk(k=topk, dim=1).indices
        q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
        t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
        P = topk

    # ------------------------------------------------------------------
    # 2) Fine optimization with Adam over ALL P poses
    # ------------------------------------------------------------------
    q_param = q_best.reshape(-1, 4).contiguous()
    t_param = t_best.reshape(-1, 3).contiguous()

    # Expand data for P poses
    anchors_1_k = anchors_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad, 3)
    anchors_2_k = anchors_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad, 3)
    vectors_1_k = vectors_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad, 3)
    vectors_2_k = vectors_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad, 3)
    types_1_k = types_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, N_pad)
    types_2_k = types_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, M_pad)

    N_k = N_real.repeat_interleave(P)
    M_k = M_real.repeat_interleave(P)
    VAA_k = VAA.repeat_interleave(P)
    VBB_k = VBB.repeat_interleave(P)

    # Adam state
    m_q = torch.zeros_like(q_param)
    v_q = torch.zeros_like(q_param)
    m_t = torch.zeros_like(t_param)
    v_t = torch.zeros_like(t_param)

    best_score = torch.full((len(q_param),), -float("inf"), device=device)
    best_q = q_param.clone()
    best_t = t_param.clone()

    prev_max_score = -float('inf')
    no_improve_count = 0

    # The Triton value+grad kernel (pharm_score_grad_se3_batch) is the fast,
    # masking-aware path: it matches compute_overlap_and_grad_pharm to ~1e-7 on
    # value AND gradient (benchmarks/pharm_kernel_parity.py) and is ~20-100x
    # faster, and -- unlike the analytical torch path -- it handles N_real/M_real
    # masking, so it also covers padded buckets (which previously fell back to the
    # slow per-step autograd path). It implements the base overlap; extended_points
    # keeps the analytical/autograd fallback.
    use_kernel = (not extended_points) and anchors_1_k.is_cuda
    use_analytical = use_kernel or bool((N_k == N_pad).all() and (M_k == M_pad).all())
    _pk_tables = None
    if use_analytical:
        # Self-overlaps MUST use the same overlap as the per-step cross-overlap
        # (mixing conventions makes the Tanimoto ratio inconsistent and collapses
        # the score). Self-overlap = overlap of a molecule with itself under
        # identity; pose-invariant, computed once over the P seed-expanded rows.
        _P = anchors_1_k.shape[0]
        _I3 = torch.eye(3, device=device, dtype=anchors_1_k.dtype).expand(_P, 3, 3)
        _z = torch.zeros(_P, 3, device=device, dtype=anchors_1_k.dtype)
        if use_kernel:
            _pk_tables = build_lookup_tables(device, anchors_1_k.dtype)
            _al, _Ks, _cats = _pk_tables
            VAA_an, _, _ = pharm_score_grad_se3_batch(
                _I3, _z, types_1_k, types_1_k, anchors_1_k, anchors_1_k, vectors_1_k, vectors_1_k,
                _al, _Ks, _cats, N_real=N_k, M_real=N_k, NEED_GRAD=False)
            VBB_an, _, _ = pharm_score_grad_se3_batch(
                _I3, _z, types_2_k, types_2_k, anchors_2_k, anchors_2_k, vectors_2_k, vectors_2_k,
                _al, _Ks, _cats, N_real=M_k, M_real=M_k, NEED_GRAD=False)
        else:
            VAA_an, _, _ = compute_overlap_and_grad_pharm(
                _I3, _z, types_1_k, types_1_k, anchors_1_k, anchors_1_k, vectors_1_k, vectors_1_k,
                extended_points=extended_points, only_extended=only_extended,
            )
            VBB_an, _, _ = compute_overlap_and_grad_pharm(
                _I3, _z, types_2_k, types_2_k, anchors_2_k, anchors_2_k, vectors_2_k, vectors_2_k,
                extended_points=extended_points, only_extended=only_extended,
            )

    for step in range(steps_fine):
        if use_analytical:
            # One fused value+grad pass, no autograd graph, no per-step host sync.
            q_unit = torch.nn.functional.normalize(q_param, dim=1)
            R = _rotation_matrix_from_unit_quat(q_unit)
            if use_kernel:
                _al, _Ks, _cats = _pk_tables
                O_AB, grad_R, grad_t = pharm_score_grad_se3_batch(
                    R, t_param, types_1_k, types_2_k,
                    anchors_1_k, anchors_2_k, vectors_1_k, vectors_2_k,
                    _al, _Ks, _cats, N_real=N_k, M_real=M_k)
            else:
                O_AB, grad_R, grad_t = compute_overlap_and_grad_pharm(
                    R, t_param, types_1_k, types_2_k,
                    anchors_1_k, anchors_2_k, vectors_1_k, vectors_2_k,
                    extended_points=extended_points, only_extended=only_extended,
                )
            score = pharm_similarity_from_overlaps(O_AB, VAA_an, VBB_an, similarity=similarity)
            if similarity == 'tanimoto':
                _, sgrad_R, sgrad_t = apply_tanimoto_chain_rule(O_AB, VAA_an + VBB_an, grad_R, grad_t)
            else:
                sigma = _PHARM_SIGMA_MAP[similarity]
                D = sigma * VAA_an + (1.0 - sigma) * VBB_an
                _, sgrad_R, sgrad_t = apply_tversky_chain_rule(O_AB, D, grad_R, grad_t)
            # grad w.r.t. unit quat -> raw quat (normalisation Jacobian); this
            # already lies in the tangent space, so the projection below is a no-op.
            grad_q_unit = project_grad_R_to_quaternion(sgrad_R, q_unit)
            qn = q_param.norm(dim=1, keepdim=True).clamp(min=1e-12)
            dQ = (grad_q_unit - q_unit * (q_unit * grad_q_unit).sum(1, keepdim=True)) / qn
            dT = sgrad_t
        else:
            q_var = q_param.detach().requires_grad_(True)
            t_var = t_param.detach().requires_grad_(True)

            VAB = batch_pharm_cross_overlap_with_transform(
                anchors_1_k,
                anchors_2_k,
                vectors_1_k,
                vectors_2_k,
                types_1_k,
                types_2_k,
                q_var,
                t_var,
                extended_points=extended_points,
                only_extended=only_extended,
                N_real=N_k,
                M_real=M_k,
            )
            score = pharm_similarity_from_overlaps(VAB, VAA_k, VBB_k, similarity=similarity)
            loss = -score.sum()

            dQ, dT = torch.autograd.grad(loss, (q_var, t_var), create_graph=False)

        # Track best (use the detached score so no autograd graph is retained)
        score_det = score.detach()
        better = score_det > best_score
        best_score = torch.where(better, score_det, best_score)
        mask_q = better.unsqueeze(1)
        best_q = torch.where(mask_q, q_param, best_q)
        best_t = torch.where(mask_q, t_param, best_t)

        # Early stopping check every 5 iterations to reduce GPU→CPU sync overhead
        if step % 5 == 0:
            current_max = best_score.max().item()
            if current_max - prev_max_score < early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= (_fc.ES_PATIENCE_OVERRIDE or early_stop_patience):
                    break
            else:
                no_improve_count = 0
                prev_max_score = current_max

        # Tangent-space projection for quaternion (q_param is ~unit; for the
        # analytical branch dQ is already tangent so this is a no-op).
        radial = (dQ * q_param).sum(dim=1, keepdim=True)
        dQ_tan = dQ - q_param * radial

        # Adam update (using fused kernel for efficiency)
        fused_adam_qt(q_param, t_param, dQ_tan.detach(), dT.detach(), m_q, v_q, m_t, v_t, lr)

    # ------------------------------------------------------------------
    # 5) Gather final results
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P

    return (final_score.flatten()[sel],
            best_q[sel],
            best_t[sel])


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
        trans_centers: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
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
        from .._torch import optimize_pharm_overlay
        return optimize_pharm_overlay(
            ref_pharms, fit_pharms,
            ref_anchors, fit_anchors,
            ref_vectors, fit_vectors,
            similarity, extended_points, only_extended,
            num_repeats,
            trans_centers=trans_centers,
            lr=lr,
            max_num_steps=steps_fine,
            **kwargs)

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

    if trans_centers is not None:
        trans_centers_batch = trans_centers.to(device=device, dtype=torch.float32).unsqueeze(0)
        trans_centers_real = torch.tensor([trans_centers.shape[0]], device=device, dtype=torch.int32)
    else:
        trans_centers_batch = None
        trans_centers_real = None

    # Precompute self-overlaps
    VAA = batch_pharm_self_overlap(
        ref_anchors_gpu,
        ref_vectors_gpu,
        ref_types_gpu,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=N_real,
    )
    VBB = batch_pharm_self_overlap(
        fit_anchors_gpu,
        fit_vectors_gpu,
        fit_types_gpu,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=M_real,
    )

    # Run alignment
    score, q_best, t_best = coarse_fine_pharm_align_many(
        ref_anchors_gpu, fit_anchors_gpu,
        ref_vectors_gpu, fit_vectors_gpu,
        ref_types_gpu, fit_types_gpu,
        VAA, VBB,
        similarity=similarity,
        extended_points=extended_points,
        only_extended=only_extended,
        num_seeds=num_repeats,
        trans_centers=trans_centers_batch,
        trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
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
        *,
        similarity: str = "tanimoto",
        extended_points: bool = False,
        only_extended: bool = False,
        num_repeats: int = 50,
        trans_centers_batch: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None,
        topk: int = 30,
        steps_fine: int = 100,
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
    VAA = batch_pharm_self_overlap(
        ref_anchors_batch,
        ref_vectors_batch,
        ref_pharms_batch,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=N_real,
    )
    VBB = batch_pharm_self_overlap(
        fit_anchors_batch,
        fit_vectors_batch,
        fit_pharms_batch,
        extended_points=extended_points,
        only_extended=only_extended,
        N_real=M_real,
    )

    # Run alignment
    scores, q_best, t_best = coarse_fine_pharm_align_many(
        ref_anchors_batch, fit_anchors_batch,
        ref_vectors_batch, fit_vectors_batch,
        ref_pharms_batch, fit_pharms_batch,
        VAA, VBB,
        similarity=similarity,
        extended_points=extended_points,
        only_extended=only_extended,
        num_seeds=num_repeats,
        trans_centers=trans_centers_batch,
        trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
        N_real=N_real,
        M_real=M_real)

    # Apply transforms
    aligned_anchors = apply_se3_transform(fit_anchors_batch, q_best, t_best)
    aligned_vectors = apply_so3_transform(fit_vectors_batch, q_best)

    return aligned_anchors, aligned_vectors, q_best, t_best, scores
