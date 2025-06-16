import torch, math
from ..score.gaussian_overlap_triton import overlap_score_grad_se3, gaussian_self_overlap, overlap_score_grad_se3_batch, fused_adam_qt
from typing import Optional

torch.backends.cuda.matmul.allow_tf32 = True 

def _get_points_fibonacci(n: int,
                          radius: float = 1.0,
                          *,
                          device: Optional[torch.device] = None,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return `n` quasi-uniform unit quaternions produced by mapping the +X axis
    onto Fibonacci-sphere points.

    Output: tensor (n, 4) on `device`, `dtype`, *normalised*.
    """
    if device is None:
        device = torch.device("cpu")

    # 1. spherical Fibonacci lattice (unit vectors)
    idx   = torch.arange(n, dtype=dtype, device=device)
    phi   = 2.0 * math.pi * idx / math.sqrt(5.0)
    cos_t = 1.0 - 2.0 * (idx + 0.5) / n
    sin_t = torch.sqrt(torch.clamp(1.0 - cos_t * cos_t, 0.0, 1.0))

    vx = sin_t * torch.cos(phi) * radius
    vy = sin_t * torch.sin(phi) * radius
    vz = cos_t * radius
    v_dst = torch.stack((vx, vy, vz), dim=1)                # (n,3)

    # 2. source vector (+X) broadcasted
    v_src = torch.tensor([1.0, 0.0, 0.0],
                         dtype=dtype, device=device).expand_as(v_dst)

    # 3. axis = v_src × v_dst
    axis = torch.linalg.cross(v_src, v_dst, dim=1)
    axis_norm = axis.norm(dim=1, keepdim=True)

    # replace zero-norm axes (parallel vectors) with [0,0,1]
    axis_safe = axis.clone()
    mask = axis_norm.squeeze() > 1e-8
    axis_safe[~mask] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    axis_safe[mask] /= axis_norm[mask]

    # 4. angle & quaternion
    dot = (v_src * v_dst).sum(1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    half = torch.acos(dot) * 0.5
    qw   = torch.cos(half)
    qxyz = axis_safe * torch.sin(half).unsqueeze(1)

    quat = torch.cat((qw.unsqueeze(1), qxyz), dim=1)        # (n,4)
    return torch.nn.functional.normalize(quat, dim=1)


@torch.no_grad()
def adam_step(p, g, m, v, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m.mul_(beta1).add_(g, alpha=1-beta1)
    v.mul_(beta2).addcmul_(g, g, value=1-beta2)
    p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)


@torch.no_grad()
def se3_align_single(A, B, *,
                     alpha: float = 0.81,
                     num_init: int = 64,
                     steps: int = 60,
                     lr: float = 0.05,
                     device: str | torch.device = "cuda",
                     VAA: float | None = None,
                     VBB: float | None = None):
    """
    Fast shape-only alignment of one pair (no Autograd graph).

    A, B : (N,3) / (M,3)   CUDA tensors *or* CPU → will be moved to device
    VAA, VBB : optional pre-computed self overlaps (Python floats);
               pass None to compute once here.

    Returns
        q_best (4,)  t_best (3,)  score_best (float)
    """
    A = A.to(device, torch.float32).contiguous()
    B = B.to(device, torch.float32).contiguous()

    # ------------------------------------------------------------
    # self overlap – ONLY ONCE
    # ------------------------------------------------------------
    if VAA is None:
        VAA = gaussian_self_overlap(A, alpha).item()
    if VBB is None:
        VBB = gaussian_self_overlap(B, alpha).item()
    Vsum = VAA + VBB                                    # scalar

    # ------------------------------------------------------------
    # initial quats / translations
    # ------------------------------------------------------------
    from .se3_np import quats_from_fibonacci
    q0  = torch.tensor(quats_from_fibonacci(num_init), device=device)
    t0  = (A.mean(0) - B.mean(0)).expand(num_init, 3)

    best_score = -1.0
    best_q = best_t = None

    for q, t in zip(q0, t0):
        q = q.clone(); t = t.clone()
        m_q = torch.zeros_like(q); v_q = torch.zeros_like(q)
        m_t = torch.zeros_like(t); v_t = torch.zeros_like(t)

        for _ in range(steps):
            VAB, dQ, dT = overlap_score_grad_se3(A, B, q, t, alpha)

            denom = Vsum - VAB                     # scalar tensor
            scale = Vsum / (denom * denom)         # ∂T/∂VAB factor
            # grad wrt score = scale * dVAB
            adam_step(q, dQ * scale, m_q, v_q, lr)
            adam_step(t, dT * scale, m_t, v_t, lr)
            q = torch.nn.functional.normalize(q, dim=0)

        # final score
        VAB, _, _ = overlap_score_grad_se3(A, B, q, t, alpha)
        score = (VAB / (Vsum - VAB)).item()

        if score > best_score:
            best_score, best_q, best_t = score, q.clone(), t.clone()

    return best_q, best_t, best_score

@torch.no_grad()
def coarse_fine_align(A: torch.Tensor,
                      B: torch.Tensor,
                      *,
                      alpha: float = 0.81,
                      VAA: float | None = None,
                      VBB: float | None = None,
                      coarse_rots: int = 512,
                      coarse_trans: int = 32,
                      topk: int = 8,
                      steps_fine: int = 15,
                      lr: float = 0.08,
                      device=None):

    if device is None:
        device = A.device
    A = A.to(device, torch.float32)
    B = B.to(device, torch.float32)

    # ---------- self overlaps ONCE -------------------------------- ▼
    if VAA is None:
        VAA = gaussian_self_overlap(A, alpha).item()
    if VBB is None:
        VBB = gaussian_self_overlap(B, alpha).item()
    Vsum = VAA + VBB

    # ---------- safe translation grid ------------------------------
    n_trans = min(coarse_trans, A.shape[0])

    quats = _get_points_fibonacci(coarse_rots, device=device)
    quats = torch.nn.functional.normalize(quats, dim=1)

    trans  = A[:n_trans] - B.mean(0)
    q_grid = quats.repeat_interleave(n_trans, 0)
    t_grid = trans.repeat(coarse_rots, 1)
    G = q_grid.size(0)

    # ---------- coarse scoring  (one launch) ----------------------- ▼
    A_rep = A.unsqueeze(0).expand(G, -1, -1)
    B_rep = B.unsqueeze(0).expand(G, -1, -1)

    VAB_grid, _, _ = overlap_score_grad_se3_batch(A_rep, B_rep, q_grid, t_grid, alpha=alpha)
    coarse_score  = VAB_grid / (Vsum - VAB_grid)

    keep = min(topk, G)
    idx  = torch.topk(coarse_score, keep).indices
    q = q_grid[idx].clone()
    t = t_grid[idx].clone()

    # ---------- fine stage -----------------------------------------
    m_q = torch.zeros_like(q); v_q = torch.zeros_like(q)
    m_t = torch.zeros_like(t); v_t = torch.zeros_like(t)

    A_k = A.unsqueeze(0).expand(keep, -1, -1).contiguous()
    B_k = B.unsqueeze(0).expand(keep, -1, -1).contiguous()

    for _ in range(steps_fine):
        # analytic score + gradients for all k orientations in one call
        VAB, dQ, dT = overlap_score_grad_se3_batch(A_k, B_k, q, t, alpha=alpha)

        # denominator  (VAA + VBB − VAB)   — broadcast to (k,1)
        denom = (Vsum - VAB).unsqueeze(1)

        # fused Adam update (handles renormalising q internally)
        fused_adam_qt(
            q, t,
            dQ / denom,                # scale grads exactly as in the batched path
            dT / denom,
            m_q, v_q, m_t, v_t,
            lr
        )

    VAB_final, _, _ = overlap_score_grad_se3_batch(A_k, B_k, q, t, alpha=alpha)
    final_score = VAB_final / (Vsum - VAB_final)
    best = torch.argmax(final_score)

    return final_score[best].item(), q[best], t[best]          # ▲ return unchanged

@torch.no_grad()
def _overlap_in_chunks(A, B, q, t, *, alpha: float = 0.81,
                       N_real: torch.Tensor | None = None,
                       M_real: torch.Tensor | None = None):
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
    if N_real is None:
        N_real = A.new_full((K,), A.shape[1], dtype=torch.int32)
    if M_real is None:
        M_real = B.new_full((K,), B.shape[1], dtype=torch.int32)

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
            M_real=M_real[start:end])

        out_V[start:end]  = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


@torch.no_grad()
def coarse_fine_align_many(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        coarse_rots: int = 512,
        coarse_trans: int = 32,
        topk: int = 8,
        steps_fine: int = 15,
        lr: float = 0.08,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None):
    """
    Padding-aware vectorised alignment for a bucket of pairs.

    A_batch, B_batch : (BATCH, N_pad, 3) / (BATCH, M_pad, 3)
    VAA, VBB         : (BATCH,)  Gaussian self-overlaps
    N_real, M_real   : (BATCH,)  true atom counts (optional when not padded)
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = B_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ---------- coarse grid (shared for the whole bucket) -----------------
    quats = _get_points_fibonacci(coarse_rots, device=device)
    quats = torch.nn.functional.normalize(quats, dim=1)        # (R,4)

    n_trans = int(torch.minimum(
        torch.tensor(coarse_trans, device=device, dtype=torch.int32),
        N_real).min())

    B_cent = (B_batch.sum(1) / M_real.float().unsqueeze(1)).unsqueeze(1)  # (B,1,3)
    trans  = A_batch[:, :n_trans] - B_cent                                # (B,n_trans,3)

    q_grid = quats.repeat_interleave(n_trans, 0)                 # (G,4)
    q_grid = q_grid.unsqueeze(0).expand(BATCH, -1, -1)           # (B,G,4)
    t_grid = trans.repeat(1, coarse_rots, 1)                     # (B,G,3)
    G = q_grid.size(1)

    # --------- double micro-batch: orientations × pairs -------------------
    ORI_CHUNK  = 25_000                 # orientations per slice
    PAIR_CHUNK = 4_096                  # pairs per slice  (≈ 400 × 4 096 × 3 × 4 ≈ 19 MB)

    coarse_score = torch.empty(BATCH, G, device=device, dtype=A_batch.dtype)

    for o0 in range(0, G, ORI_CHUNK):
        o1    = min(o0 + ORI_CHUNK, G)
        g_len = o1 - o0

        # --- loop over pair slices so the contiguous copy stays small ----
        for p0 in range(0, BATCH, PAIR_CHUNK):          ##### NEW pair-chunk #####
            p1 = min(p0 + PAIR_CHUNK, BATCH)
            slice_len = p1 - p0

            A_rep = A_batch[p0:p1].unsqueeze(1) \
                               .expand(-1, g_len, -1, -1) \
                               .reshape(-1, N_pad, 3)
            B_rep = B_batch[p0:p1].unsqueeze(1) \
                               .expand(-1, g_len, -1, -1) \
                               .reshape(-1, M_pad, 3)

            q_rep = q_grid[p0:p1, o0:o1, :].reshape(-1, 4).contiguous()
            t_rep = t_grid[p0:p1, o0:o1, :].reshape(-1, 3).contiguous()

            N_rep = N_real[p0:p1].repeat_interleave(g_len)
            M_rep = M_real[p0:p1].repeat_interleave(g_len)

            VAB_slice, _, _ = _overlap_in_chunks(
                A_rep, B_rep, q_rep, t_rep,
                alpha=alpha, N_real=N_rep, M_real=M_rep)

            coarse_score[p0:p1, o0:o1] = VAB_slice.view(slice_len, g_len)

    # ---------- pick top-k and continue exactly as before -----------------
    coarse_score = coarse_score / (VAA[:, None] + VBB[:, None] - coarse_score)
    best_idx = coarse_score.topk(k=topk, dim=1).indices           # (B,topk)

    q_best = torch.gather(
        q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
    t_best = torch.gather(
        t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()

    # ------------- fine polishing (unchanged) -----------------------------
    A_k = A_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, M_pad, 3)
    q_k = q_best.view(-1, 4)
    t_k = t_best.view(-1, 3)

    N_k = N_real.repeat_interleave(topk)
    M_k = M_real.repeat_interleave(topk)

    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

    for _ in range(steps_fine):
        VAB, dQ, dT = _overlap_in_chunks(
            A_k, B_k, q_k, t_k,
            alpha=alpha, N_real=N_k, M_real=M_k)

        denom = VAA.repeat_interleave(topk) + VBB.repeat_interleave(topk) - VAB
        fused_adam_qt(q_k, t_k, dQ / denom[:, None], dT / denom[:, None],
                      m_q, v_q, m_t, v_t, lr)

    final_VAB, _, _ = _overlap_in_chunks(
        A_k, B_k, q_k, t_k,
        alpha=alpha, N_real=N_k, M_real=M_k)

    final_score = final_VAB / (VAA.repeat_interleave(topk) +
                               VBB.repeat_interleave(topk) - final_VAB)
    final_score = final_score.view(BATCH, topk)
    best = final_score.argmax(dim=1)

    sel = best + torch.arange(BATCH, device=device) * topk
    return final_score.flatten()[sel], \
           q_k.view(BATCH, topk, 4)[torch.arange(BATCH), best], \
           t_k.view(BATCH, topk, 3)[torch.arange(BATCH), best]
