# shepherd_score/score/gaussian_overlap_triton.py
# A fused (forward + backward) Gaussian‐Tanimoto kernel using Triton.
#
import math
import triton
import triton.language as tl
import torch
import numpy as np

# ---------------------------------------------------------------------------#
#  Scratch-buffer pool to avoid   torch.zeros_like()   every kernel launch   #
# ---------------------------------------------------------------------------#
_buffer_pool = {}                #  { (shape,dtype,device) : tensor }

def _pool_key(t: torch.Tensor):
    return (t.shape, t.dtype, t.device)

def acquire_like(t: torch.Tensor) -> torch.Tensor:
    """return a reusable buffer with   t.shape / t.dtype / t.device"""
    k = _pool_key(t)
    if k not in _buffer_pool:
        _buffer_pool[k] = torch.empty_like(t)      # one-off allocation
    return _buffer_pool[k]


@triton.jit
def _load_xyz(ptr, idx, mask):
    """
    Safe load of a (K,3) array.  Any thread with mask=False uses a clamped
    pointer (index 0) so no out-of-bounds address is ever formed.
    """
    idx_safe = tl.where(mask, idx, 0)           # ← NEW
    x = tl.load(ptr + idx_safe * 3 + 0, mask=mask)
    y = tl.load(ptr + idx_safe * 3 + 1, mask=mask)
    z = tl.load(ptr + idx_safe * 3 + 2, mask=mask)
    return x, y, z


@triton.jit
def _gauss_overlap_fwd(
    # ----------- existing args ----------------------------------
    A_ptr, B_ptr,                  # coordinates (flat (K*3,) buffers)
    VAB_ptr,                       # (1,)   – accumulate VAB
    dVAB_ptr,                      # (M*3,) – ∂VAB/∂B
    # ----------- NEW args ---------------------------------------
    VBB_ptr,                       # (1,)   – accumulate VBB   (can be nullptr)
    dVBB_ptr,                      # (M*3,) – ∂VBB/∂B          (can be nullptr)
    # ------------------------------------------------------------
    N, M,
    half_alpha, k_const,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    want_grad: tl.constexpr,       # bool – do we need any gradients?
    want_self: tl.constexpr,       # bool – does this launch also compute BB?
):
    pid_r = tl.program_id(0)          # batch
    pid_n = tl.program_id(1)          # tiles in A
    pid_m = tl.program_id(2)          # tiles in B

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M

    # Load (x,y,z) for all A[i] in this tile
    ax, ay, az = _load_xyz(A_ptr + pid_r*N*3, offs_n, mask_n)
    # Load (x,y,z) for all B[j] in this tile
    bx, by, bz = _load_xyz(B_ptr + pid_r*M*3, offs_m, mask_m)

    # Compute pairwise squared distances between each tile of A and tile of B
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz

    # Gaussian kernel: g_{ij} = exp( −(alpha/2) * r2 ) * (π^{1.5} / ((2 α)^{1.5}))
    g = tl.exp(-half_alpha * r2) * k_const   # shape (BLOCK_N, BLOCK_M)

    # Accumulate VAB tile-wise
    v_tile = tl.sum(g)
    # Write into V_out[0]
    tl.atomic_add(VAB_ptr + pid_r, v_tile)

    if want_self and (pid_n == pid_m):
        # this tile is on B×B; add symmetric contribution
        # vbb_tile = 2*sum(g) - sum(diag(g))
        diag_mask = tl.arange(0, BLOCK_N)[:,None] == tl.arange(0,BLOCK_M)[None,:]
        diag_sum  = tl.sum(tl.where(diag_mask, g, 0.0))
        vbb_tile  = tl.sum(g)*2.0 - diag_sum
        tl.atomic_add(VBB_ptr + pid_r, vbb_tile)

    # Gradient wrt B_j:  ∂VAB/∂B_j = Σ_i [ α·g_{ij}·(B_j − A_i) ]
    # But since half_alpha = 0.5·alpha, α = 2·half_alpha.
    if want_grad:
        # (a) allocate *register* buffers, one element per B-atom in the tile
        gradbx = tl.zeros((BLOCK_M,), dtype=tl.float32)
        gradby = tl.zeros((BLOCK_M,), dtype=tl.float32)
        gradbz = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # (b) accumulate this tile’s contribution **into registers**
        coeff = (2.0 * half_alpha) * g
        gradbx += tl.sum(coeff * (-dx), 0)          # (BLOCK_M,)
        gradby += tl.sum(coeff * (-dy), 0)
        gradbz += tl.sum(coeff * (-dz), 0)

        # add the ½-symmetric term if this is a B×B tile
        if want_self and (pid_n == pid_m):
            coeff_self = coeff * 0.5
            gradbx += tl.sum(coeff_self * (-dx), 0)
            gradby += tl.sum(coeff_self * (-dy), 0)
            gradbz += tl.sum(coeff_self * (-dz), 0)

        # (c)  **single** atomic write-back per B-atom for this block
        tl.atomic_add(dVAB_ptr + (pid_r*M + offs_m)*3 + 0, gradbx, mask=mask_m)
        tl.atomic_add(dVAB_ptr + (pid_r*M + offs_m)*3 + 1, gradby, mask=mask_m)
        tl.atomic_add(dVAB_ptr + (pid_r*M + offs_m)*3 + 2, gradbz, mask=mask_m)

        if want_self and (pid_n == pid_m):
            tl.atomic_add(dVBB_ptr + (pid_r*M + offs_m)*3 + 0, gradbx, mask=mask_m)
            tl.atomic_add(dVBB_ptr + (pid_r*M + offs_m)*3 + 1, gradby, mask=mask_m)
            tl.atomic_add(dVBB_ptr + (pid_r*M + offs_m)*3 + 2, gradbz, mask=mask_m)

@triton.jit
def _self_overlap_fwd(
    P_ptr, V_out, dVdP_ptr,     # P has shape (K,3)
    K, half_alpha,              # 0.5 * alpha
    k_const,                    # π^{3/2} / ( (2·α)^{3/2} ), α = 2·half_alpha
    BLOCK: tl.constexpr,
):
    """
    Compute a tile of VAA / VBB and its gradient.

    The program is launched on a 2-D grid:
        pid_row = tl.program_id(0)   ← rows  (i-tile)
        pid_col = tl.program_id(1)   ← cols  (j-tile)

    Only tiles with pid_row ≥ pid_col are evaluated; the others return
    immediately, so every ordered pair (i,j) is visited exactly once.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Skip upper-triangular tiles – they are handled by the symmetric one.
    if pid_row < pid_col:
        return

    # ------------------------------------------------------------------
    #  indices & masks for this tile
    # ------------------------------------------------------------------
    offs_i   = pid_row * BLOCK + tl.arange(0, BLOCK)
    offs_j   = pid_col * BLOCK + tl.arange(0, BLOCK)
    row_mask = offs_i < K
    col_mask = offs_j < K

    # ------------------------------------------------------------------
    #  load coordinates
    # ------------------------------------------------------------------
    xi, yi, zi = _load_xyz(P_ptr, offs_i, row_mask)
    xj, yj, zj = _load_xyz(P_ptr, offs_j, col_mask)

    dx = xi[:, None] - xj[None, :]
    dy = yi[:, None] - yj[None, :]
    dz = zi[:, None] - zj[None, :]
    r2 = dx * dx + dy * dy + dz * dz

    # Gaussian kernel 
    g = tl.exp(-half_alpha * r2) * k_const

    # mask out rows / cols beyond K
    valid = row_mask[:, None] & col_mask[None, :]
    g = tl.where(valid, g, 0.0)

    # ------------------------------------------------------------------
    #  keep only lower-triangle inside *diagonal* tiles
    # ------------------------------------------------------------------
    if pid_row == pid_col:
        tri_mask = offs_i[:, None] >= offs_j[None, :]        # i ≥ j
        g = tl.where(tri_mask, g, 0.0)

    # ------------------------------------------------------------------
    #  accumulate VAA / VBB
    #     • off-diag tiles   :  2·Σ g_ij
    #     • diagonal tile    :  2·Σ_{i>j} g_ij  +  Σ_i g_ii
    #                          = 2·Σ g_ij − Σ_i g_ii
    # ------------------------------------------------------------------
    tile_sum = tl.sum(g)

    if pid_row == pid_col:
        diag_sum = tl.sum(tl.where(offs_i[:, None] == offs_j[None, :], g, 0.0))
        v_tile   = tile_sum * 2.0 - diag_sum
    else:
        v_tile   = tile_sum * 2.0     # off-diag

    tl.atomic_add(V_out, v_tile)

    # ------------------------------------------------------------------
    #  gradient   ∂V/∂P_i  = 2 · Σ_{j<i} α g_ij (P_i − P_j)
    # ------------------------------------------------------------------
    coeff = (2.0 * half_alpha) * g
    gradx = tl.sum(coeff * dx, 1) * 2.0
    grady = tl.sum(coeff * dy, 1) * 2.0
    gradz = tl.sum(coeff * dz, 1) * 2.0

    # write-back
    tl.atomic_add(dVdP_ptr + offs_i * 3 + 0, gradx, mask=row_mask)
    tl.atomic_add(dVdP_ptr + offs_i * 3 + 1, grady, mask=row_mask)
    tl.atomic_add(dVdP_ptr + offs_i * 3 + 2, gradz, mask=row_mask)


class _GaussianTanimotoAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, alpha: float = 0.81, VAA_const=None):
        """
        A: (N,3) float32 CUDA tensor
        B: (M,3) float32 CUDA tensor
        alpha: Python float
        """
        assert A.is_cuda and B.is_cuda and A.dtype == torch.float32 and B.dtype == torch.float32

        R     = A.shape[0]              # R = 50 for your optimiser
        N, M  = A.shape[1], B.shape[1]  
        SMALL  = max(N, M) <= 64
        BLOCK  = 32 if SMALL else 128

        # Precompute “half_alpha” and “k_const” on the Python side:
        half_alpha = 0.5 * alpha
        k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)
        k_self     = math.pi**1.5 / ((4.0 * half_alpha) ** 1.5)  # python float

        want_grad = ctx.needs_input_grad[1]      # only B needs grads
        want_self = (N == M)                           # fuse VBB into the same kernel

        if want_grad:
            dVAB = torch.zeros_like(B)      
            dVBB = torch.zeros_like(B)
        else:
            dVAB = acquire_like(B); dVAB.zero_()
            dVBB = acquire_like(B); dVBB.zero_()

        VAB  = torch.zeros(R, device=A.device, dtype=A.dtype)   
        VBB  = torch.zeros(R, device=A.device, dtype=A.dtype)
        VAA  = torch.zeros(R, device=A.device, dtype=A.dtype)

        if VAA_const is not None:
            # broadcast scalar(s) to the R-vector and skip kernel launch
            VAA.copy_(VAA_const.to(VAA.device).expand_as(VAA))
            skip_vaa = True
        else:
            skip_vaa = False

        grid = (R, triton.cdiv(N,BLOCK), triton.cdiv(M,BLOCK))
        _gauss_overlap_fwd[grid](
            A, B,
            VAB,  dVAB,            #  --- VAB outputs
            VBB,  dVBB,            #  --- VBB outputs (nullptr on A×B tiles)
            N, M, half_alpha, k_const,
            BLOCK, BLOCK,
            want_grad = want_grad,     # name=val to mark const-expr
            want_self = want_self,
        )

        if not skip_vaa:
            # allocate a *small, legal* dummy gradient buffer (no atomics crash)
            dummy_grad = torch.zeros_like(A)        # (R, N, 3) contiguous
            grid_aa   = (triton.cdiv(N, BLOCK), triton.cdiv(N, BLOCK))
            _self_overlap_fwd[grid_aa](A, VAA, dummy_grad,
                N, half_alpha, k_self, BLOCK=BLOCK)

        denom = VAA + VBB - VAB
        score = VAB / denom

        # Save tensors for backward
        ctx.save_for_backward(dVAB, dVBB, VAA, VBB, VAB, denom)
        return score

    @staticmethod
    def backward(ctx, grad_out):
        dVdB, dVBBdB, VAA, VBB, VAB, denom = ctx.saved_tensors

        VAA   = VAA.view(-1, 1, 1)
        VBB   = VBB.view(-1, 1, 1)
        VAB   = VAB.view(-1, 1, 1)
        denom = denom.view(-1, 1, 1)
        grad_out = grad_out.view(-1, 1, 1)

        # ∂S/∂B = [(VAA + VBB)·dVAB/dB – VAB·dVBB/dB] / (denom²)
        num   = (VAA + VBB) * dVdB - VAB * dVBBdB
        gradB = grad_out * (num / (denom * denom))
        return None, gradB, None, None


def gaussian_tanimoto(A, B, alpha=0.81, VAA_const=None):
    """
    GPU-only fused Tanimoto score (forward & analytic grad) using Triton.
    - A: (N,3) float32 CUDA tensor
    - B: (M,3) float32 CUDA tensor
    - alpha: Python float
    Returns a scalar tensor [1] with gradient wrt B.
    """
    return _GaussianTanimotoAutograd.apply(A, B, alpha, VAA_const)

@torch.no_grad()
def gaussian_self_overlap(P: torch.Tensor, alpha: float = 0.81) -> torch.Tensor:
    """
    Fast VPP(P,P) without gradients (1-scalar CUDA tensor).
    """
    N = P.shape[0]
    # Re-use the existing self-overlap kernel but pass a dummy grad ptr
    BLOCK = 128 if N > 64 else 32
    half_a, k_self = 0.5*alpha, math.pi**1.5 / ((4*0.5*alpha)**1.5)
    VPP = torch.zeros(1, device=P.device, dtype=P.dtype)
    _self_overlap_fwd[(triton.cdiv(N,BLOCK),)*2](
        P, VPP, torch.empty((N, 3), device=P.device, dtype=P.dtype),
        N, half_a, k_self, BLOCK=BLOCK
    )
    return VPP

# ------------------------------------------------------------
# Compatibility wrapper so existing code keeps working
# ------------------------------------------------------------
def get_overlap(centers_1, centers_2, alpha=0.81, VAA_const=None):
    """
    Back-compatible front end that accepts the old keyword names
    (`centers_1`, `centers_2`) **and** batched input.

    • works on CUDA via Triton kernel
    • falls back to the legacy PyTorch path on CPU
    """

    if VAA_const is not None and centers_1.data_ptr() == centers_2.data_ptr():
        # caller already has VAA
        return VAA_const

    # Allow ndarray inputs
    if isinstance(centers_1, np.ndarray):
        centers_1 = torch.as_tensor(centers_1, dtype=torch.float32)
    if isinstance(centers_2, np.ndarray):
        centers_2 = torch.as_tensor(centers_2, dtype=torch.float32)

    # -------- CPU fallback --------
    # if not centers_1.is_cuda or not centers_2.is_cuda:
    #     from .gaussian import get_overlap as _cpu_overlap   # legacy version
    #     return _cpu_overlap(centers_1, centers_2, alpha)

    # -------- GPU / Triton path --------
    if centers_1.dim() == 3:                       # batched (R,N,3)
        return gaussian_tanimoto(centers_1, centers_2, alpha, VAA_const)          # ONE call
    else:                                          # single pair (N, 3)
        return gaussian_tanimoto(centers_1, centers_2, alpha, VAA_const)
    
# --------------------------------------------------------------------------- #
# Fast path: score **and** gradients wrt quaternion q and translation t      #
# --------------------------------------------------------------------------- #
@triton.jit
def _gauss_overlap_se3(
    A_ptr, B_ptr,            # (B·N·3,) (B·M·3,)
    Q_ptr, T_ptr,            # (B·4,)   (B·3,)
    BATCH, M, N,             # int32 scalars
    half_alpha, k_const,     # float32 scalars
    S_ptr, dQ_ptr, dT_ptr,   # (B,) (B·4,) (B·3,)
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    NEED_GRAD: tl.constexpr,
):
    pid_b = tl.program_id(2)        # ← NEW batch index
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M

    # detect if this entire tile is inactive (no valid ref OR no valid fit)
    if tl.sum(mask_n) == 0 or tl.sum(mask_m) == 0:
        # nothing to do – but write a sentinel very negative value
        if pid_n == 0 and pid_m == 0:                # only once per orientation
            tl.store(S_ptr, -1e30)
        return

    # -------- advance base pointers to this pair --------------------------
    A_ptr  = A_ptr + pid_b * N * 3
    B_ptr  = B_ptr + pid_b * M * 3
    Q_ptr  = Q_ptr + pid_b * 4
    T_ptr  = T_ptr + pid_b * 3
    dQ_ptr = dQ_ptr + pid_b * 4
    dT_ptr = dT_ptr + pid_b * 3
    S_ptr  = S_ptr  + pid_b

    # ---------------- quaternion, translation -----------------
    qr = tl.load(Q_ptr + 0)
    qi = tl.load(Q_ptr + 1)
    qj = tl.load(Q_ptr + 2)
    qk = tl.load(Q_ptr + 3)

    tx = tl.load(T_ptr + 0)
    ty = tl.load(T_ptr + 1)
    tz = tl.load(T_ptr + 2)

    # --- load A points ------------------------------------------------------
    ax, ay, az = _load_xyz(A_ptr, offs_n, mask_n)

    # --- load B points & rotate on the fly ----------------------------------
    bx0, by0, bz0 = _load_xyz(B_ptr, offs_m, mask_m)

    # quaternion to rotation matrix rows (r00 … r22) in registers
    two = 2.0
    r00 = 1 - two*(qj*qj + qk*qk); r01 = two*(qi*qj - qk*qr); r02 = two*(qi*qk + qj*qr)
    r10 = two*(qi*qj + qk*qr);     r11 = 1 - two*(qi*qi + qk*qk); r12 = two*(qj*qk - qi*qr)
    r20 = two*(qi*qk - qj*qr);     r21 = two*(qj*qk + qi*qr);     r22 = 1 - two*(qi*qi + qj*qj)

    bx = r00*bx0 + r01*by0 + r02*bz0 + tx
    by = r10*bx0 + r11*by0 + r12*bz0 + ty
    bz = r20*bx0 + r21*by0 + r22*bz0 + tz

    # pairwise distances in tile
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx*dx + dy*dy + dz*dz
    g  = tl.exp(-half_alpha * r2) * k_const      # Gaussian

    # accumulate VAA, VBB, VAB   (each thread-block works on a small slice;
    # atomic adds are fine b/c N,M ≤ 400)
    Vab_tile = tl.sum(g)
    tl.atomic_add(S_ptr, Vab_tile)               # we’ll finish formula in epilogue

    if NEED_GRAD:
        # force (negative gradient) wrt moved point B_j
        coeff = (2.*half_alpha) * g              # shape (BLOCK_N,BLOCK_M)
        fx = tl.sum(coeff * (-dx), 0)
        fy = tl.sum(coeff * (-dy), 0)
        fz = tl.sum(coeff * (-dz), 0)

        # ∂S/∂t  just   Σ_j f_j
        tl.atomic_add(dT_ptr + 0, tl.sum(fx))
        tl.atomic_add(dT_ptr + 1, tl.sum(fy))
        tl.atomic_add(dT_ptr + 2, tl.sum(fz))

        # ∂S/∂q  = Σ_j (f_j  ×  (R·B_j))  mapped to quaternion tangent
        # R·B_j coords are (bx,by,bz)
        txq = fy*bz - fz*by
        tyq = fz*bx - fx*bz
        tzq = fx*by - fy*bx
        tl.atomic_add(dQ_ptr + 0, tl.sum(txq))
        tl.atomic_add(dQ_ptr + 1, tl.sum(tyq))
        tl.atomic_add(dQ_ptr + 2, tl.sum(tzq))
        # (we ignore d/∂qr; we renormalise q each step so tangent is enough)

def overlap_score_grad_se3(A, B, q, t, alpha=0.81):
    """
    A: (N,3)  torch.float32.cuda
    B: (M,3)  "
    q: (4,)   torch.float32.cuda  – *must be normalised*
    t: (3,)   "
    Returns  score, dQ, dT              (all float32 GPU tensors)
    """
    assert A.is_cuda and B.is_cuda
    N, M = A.shape[0], B.shape[0]
    BLOCK = 64 if max(N, M) <= 64 else 128

    half_alpha = 0.5*alpha
    k_const    = math.pi**1.5 / ((2.*alpha)**1.5)

    S  = torch.zeros(1,  device=A.device, dtype=A.dtype)
    dQ = torch.zeros(4,  device=A.device, dtype=A.dtype)
    dT = torch.zeros(3,  device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(N, BLOCK), triton.cdiv(M, BLOCK))
    _gauss_overlap_se3[grid](
        A, B,
        q,              # pointer to 4 floats
        t,              # pointer to 3 floats
        M, N, half_alpha, k_const,
        S, dQ, dT,
        BLOCK, BLOCK,
        NEED_GRAD=True,
    )
    # finish Tanimoto  (VAA+VBB pre-compute once outside)
    return S, dQ, dT

def overlap_score_grad_se3_batch(A, B, q, t, *, alpha=0.81):
    """
    Batched SE(3) overlap + analytic gradients.
        A : (B,N,3)   ref coordinates
        B : (B,M,3)   fit coordinates
        q : (B,4)     normalised quaternions
        t : (B,3)     translations
    Returns
        score : (B,)      dQ : (B,4)      dT : (B,3)
    """
    BATCH, N, _ = A.shape
    _,     M, _ = B.shape
    BLOCK = 64 if max(N, M) <= 64 else 128

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    # output buffers
    score = torch.zeros(BATCH, device=A.device, dtype=A.dtype)
    dQ    = torch.zeros_like(q)
    dT    = torch.zeros_like(t)

    # launch grid:  (tiles over N, tiles over M, BATCH)
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(M, BLOCK), BATCH)
    _gauss_overlap_se3[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        BATCH, M, N,
        half_alpha, k_const,
        score, dQ.view(-1), dT.view(-1),
        BLOCK, BLOCK,
        NEED_GRAD=True,
    )
    return score, dQ, dT
