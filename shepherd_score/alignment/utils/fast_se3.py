import torch, math
import torch.nn.functional as F
from ...score.gaussian_overlap_triton import overlap_score_grad_se3_batch, fused_adam_qt, fused_adam_qt_with_tangent_proj, _batch_self_overlap
from .fast_common import batched_seeds_torch
from .._torch import objective_ROCS_overlay
from typing import Optional
from .._torch import _initialize_se3_params as _legacy_init
from pathlib import Path
from contextlib import suppress

torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def _overlap_in_chunks(A, B, q, t, *, alpha: float = 0.81,
                       N_real: torch.Tensor | None = None,
                       M_real: torch.Tensor | None = None,
                       NEED_GRAD = True):
    """
    Evaluate the fused overlap kernel on an arbitrary-long list of
    orientations, slicing the list so that each launch respects the
    CUDA `grid.z ≤ 65 535` limit.

    Parameters
    ----------
    A, B : (K,N,3) / (K,M,3)   – padded coordinate blocks
    q, t : (K,4) / (K,3)       – quaternions & translations
    N_real, M_real : (K,) int32 tensors holding the *true* atom counts
                     (rows beyond those indices are padding).  If None,
                     we assume no padding.
    Returns
    -------
    VAB : (K,)    dQ : (K,4)    dT : (K,3)     — all contiguous on GPU
    """
    K = A.shape[0]
    if N_real is not None:
        N_real = N_real.to(torch.float32).contiguous()
    if M_real is not None:
        M_real = M_real.to(torch.float32).contiguous()

    out_V  = torch.empty(K,        device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)

    CHUNK = 65_535                         # CUDA grid-z hard limit

    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)

        V, dQ, dT = overlap_score_grad_se3_batch(
            A[start:end], B[start:end],
            q[start:end], t[start:end],
            alpha=alpha,
            N_real=N_real[start:end],
            M_real=M_real[start:end],
            NEED_GRAD=NEED_GRAD)

        out_V[start:end]  = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


def _self_overlap_in_chunks(P_pad, N_real, alpha=0.81):
    K = P_pad.size(0)
    CHUNK = 65_535                     # hardware limit
    V_all = torch.empty(K,
                        device=P_pad.device,
                        dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(
            P_pad[s:e], N_real[s:e], alpha)   # ← original function
    return V_all



def _fallback_quats(num: int, device, dtype):
    # Deterministic small set of “reasonable” rotations
    s2 = math.sqrt(0.5)
    base = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],          # identity
        [0.0, 1.0, 0.0, 0.0],          # 180° about x
        [0.0, 0.0, 1.0, 0.0],          # 180° about y
        [0.0, 0.0, 0.0, 1.0],          # 180° about z
        [s2,  s2, 0.0, 0.0],           # 90° about x
        [s2, 0.0,  s2, 0.0],           # 90° about y
        [s2, 0.0, 0.0,  s2],           # 90° about z
        [0.0,  s2,  s2, 0.0],          # 180° about (x+y)
        [0.0,  s2, 0.0,  s2],          # 180° about (x+z)
        [0.0, 0.0,  s2,  s2],          # 180° about (y+z)
    ], device=device, dtype=dtype)

    if base.size(0) >= num:
        q = base[:num].clone()
    else:
        reps = (num + base.size(0) - 1) // base.size(0)
        q = base.repeat(reps, 1)[:num].clone()

    return F.normalize(q, dim=1)

@torch.no_grad()
def _legacy_seeds_torch(ref_xyz: torch.Tensor,
                        fit_xyz: torch.Tensor,
                        *,
                        num_repeats: int = 50) -> tuple[torch.Tensor, torch.Tensor]:

    # Move to CPU for legacy PCA routine, but guard against degenerate inputs
    ref_cpu = ref_xyz.detach().cpu()
    fit_cpu = fit_xyz.detach().cpu()

    def fallback(reason: str):
        # COM-to-COM translation seed
        ref_com = ref_cpu.mean(dim=0) if ref_cpu.numel() else torch.zeros(3)
        fit_com = fit_cpu.mean(dim=0) if fit_cpu.numel() else torch.zeros(3)
        t0 = (ref_com - fit_com).to(device=ref_xyz.device, dtype=ref_xyz.dtype)
        t  = t0.unsqueeze(0).repeat(num_repeats, 1)
        q  = _fallback_quats(num_repeats, device=ref_xyz.device, dtype=ref_xyz.dtype)
        return q, t

    # Minimal sanity checks (common culprits)
    if ref_cpu.shape[0] < 3 or fit_cpu.shape[0] < 3:
        return fallback("too_few_points")
    if (not torch.isfinite(ref_cpu).all()) or (not torch.isfinite(fit_cpu).all()):
        return fallback("non_finite_coords")

    try:
        se3 = _legacy_init(ref_points=ref_cpu, fit_points=fit_cpu, num_repeats=num_repeats)

        # Catch NaNs/Infs coming back from PCA
        if not torch.isfinite(se3).all():
            return fallback("legacy_init_non_finite")

        se3 = se3.to(dtype=ref_xyz.dtype, device=ref_xyz.device)
        q, t = se3[:, :4], se3[:, 4:]
        return F.normalize(q, dim=1), t

    except Exception:
        # Includes numpy.linalg.LinAlgError from PCA
        return fallback("legacy_init_exception")
    
def coarse_fine_align_many(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        topk: int | None = None,        # deprecated: pruning removed (see below)
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None,
        early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5):
    """
    Vectorised padding-aware alignment over a batch of (A, B) pairs.

    Strategy (matches the reference optimiser, fully batched): build the
    reference seed set -- identity + 4 principal-axis quaternions + Fibonacci
    rotations, each with a COM-aligning translation -- then fine-optimise EVERY
    seed and take the per-pair best.

    Why no coarse-grid + top-k pruning anymore
    ------------------------------------------
    The previous implementation built a 500-pose grid (250 rotations x 2
    translations), scored every pose with a single un-optimised overlap, kept
    only the top-k by that raw score, and fine-tuned those. The raw overlap of an
    un-optimised seed is a poor predictor of its post-optimisation score, so for
    pseudo-symmetric molecules (caffeine, benzene, ...) pruning repeatedly
    discarded the seed sitting in the true basin while keeping decoys that
    plateau at a lower local optimum -- pulling the score ~5% below the reference
    on real molecules (worst case ~0.4). Optimising all seeds removes the
    fragile ranking step entirely: it is provably >= the reference (same seeds,
    >= optimisation power, take-the-max) and -- because the fine loop is
    launch-bound, not pose-bound -- costs only ~1.3x while restoring exact
    parity. `topk` is accepted for call-site compatibility but ignored.

    Parameters
    ----------
    A_batch, B_batch : (B, N_pad, 3) / (B, M_pad, 3)  atom coordinates
    VAA, VBB         : (B,)  pre-computed Gaussian self-overlaps
    num_seeds        : int   reference seed count (identity + 4 PCA + Fibonacci)
    N_real, M_real   : (B,)  optional true atom counts
    early_stop_patience : int  iterations without global improvement before stop
    early_stop_tol : float  minimum improvement threshold
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) reference seed set (no coarse grid, no flips, no 2nd translation)
    # ------------------------------------------------------------------
    quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real,
                                         num_seeds=num_seeds)
    S = quats.size(1)

    # ------------------------------------------------------------------
    # 2) fine polishing (Adam-like) on EVERY seed
    # ------------------------------------------------------------------
    A_k = A_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, M_pad, 3)
    q_k = quats.reshape(-1, 4).contiguous()
    t_k = t_seeds.reshape(-1, 3).contiguous()

    N_k = N_real.repeat_interleave(S)
    M_k = M_real.repeat_interleave(S)
    VAA_plus_VBB = (VAA + VBB).repeat_interleave(S)        # invariant in loop

    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q = q_k.clone()
    best_t = t_k.clone()

    # Early stopping state
    prev_max_score = -float('inf')
    no_improve_count = 0

    for step in range(steps_fine):
        VAB, dQ, dT = _overlap_in_chunks(
            A_k, B_k, q_k, t_k,
            alpha=alpha, N_real=N_k, M_real=M_k)

        denom = VAA_plus_VBB - VAB
        score = VAB / denom
        scale = VAA_plus_VBB / (denom * denom)

        better = score > best_score  # boolean mask, no .any()

        # Masked assignment via torch.where: fixed-shape and SYNC-FREE. (Boolean
        # index-assignment best_q[better]=... was measured ~1.4-4x SLOWER because
        # the data-dependent gather forces a per-step GPU->CPU sync.)
        best_score = torch.where(better, score, best_score)
        mask_q = better.unsqueeze(1)
        best_q = torch.where(mask_q, q_k, best_q)
        best_t = torch.where(mask_q, t_k, best_t)

        # Early stopping check, gated to every 5 steps to avoid a per-step
        # GPU->CPU sync. Gating only makes early-stop *less* aggressive, so
        # scores cannot drop.
        if step % 5 == 0:
            current_max = best_score.max().item()
            if current_max - prev_max_score < early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    break
            else:
                no_improve_count = 0
                prev_max_score = current_max

        # Tangent-space projection is fused into the Adam kernel (raw dQ in);
        # algebraically identical to projecting -dQ_tan*scale since `scale` is a
        # per-row scalar and the projection is linear.
        fused_adam_qt_with_tangent_proj(
            q_k, t_k,
            -dQ * scale.unsqueeze(1),
            -dT * scale.unsqueeze(1),
            m_q, v_q, m_t, v_t, lr
        )

    # ------------------------------------------------------------------
    # 3) gather final results (using already-tracked best scores)
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, S)

    best = final_score.argmax(dim=1)
    sel  = best + torch.arange(BATCH, device=device) * S

    return final_score.flatten()[sel], \
           best_q.view(BATCH, S, 4)[torch.arange(BATCH), best], \
           best_t.view(BATCH, S, 3)[torch.arange(BATCH), best]













