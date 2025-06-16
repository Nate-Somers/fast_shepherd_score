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
    A_ptr, B_ptr,                 # (B·N_pad·3,) (B·M_pad·3,)
    Q_ptr, T_ptr,                 # (B·4,) (B·3,)
    Nreal_ptr, Mreal_ptr,         # NEW  (B,) int32 – real atom counts
    BATCH, M_pad, N_pad,          # padded sizes (int32)
    half_alpha, k_const,          # scalars
    S_ptr, dQ_ptr, dT_ptr,        # outputs
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    NEED_GRAD: tl.constexpr,
):
    pid_b = tl.program_id(2)        # ← NEW batch index
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    realN = tl.load(Nreal_ptr + pid_b)        
    realM = tl.load(Mreal_ptr + pid_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < realN                     # ← ONLY true atoms
    mask_m = offs_m < realM

    # detect if this entire tile is inactive (no valid ref OR no valid fit)
    if tl.sum(mask_n) == 0 or tl.sum(mask_m) == 0:
        # nothing to do – but write a sentinel very negative value
        if pid_n == 0 and pid_m == 0:                # only once per orientation
            tl.store(S_ptr, -1e30)
        return

    # -------- advance base pointers to this pair --------------------------
    A_ptr  = A_ptr + pid_b * N_pad * 3
    B_ptr  = B_ptr + pid_b * M_pad * 3
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

@torch.no_grad()
def overlap_score_grad_se3(A: torch.Tensor,
                           B: torch.Tensor,
                           q: torch.Tensor,
                           t: torch.Tensor,
                           *,
                           alpha: float = 0.81):
    """
    Single-pair SE(3) Gaussian-Tanimoto overlap *with analytic gradients*.

    Parameters
    ----------
    A : (N,3)  CUDA - ref coordinates  (float32)
    B : (M,3)  CUDA - fit coordinates  (float32)
    q : (4,)   CUDA - unit quaternion  (float32)
    t : (3,)   CUDA - translation      (float32)

    Returns
    -------
    VAB : (1,)      dQ : (4,)      dT : (3,)
        • VAB is the **overlap numerator** (not the final Tanimoto score);
          the caller keeps the old “add self-overlaps later” contract.
    """
    assert A.is_cuda and B.is_cuda, "inputs must already live on the GPU"
    N, M   = A.shape[0], B.shape[0]
    BLOCK  = 64 if max(N, M) <= 64 else 128
    device = A.device
    dtype  = A.dtype

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    N_real = torch.tensor([N], device=device, dtype=torch.int32)
    M_real = torch.tensor([M], device=device, dtype=torch.int32)

    VAB = torch.zeros(1,  device=device, dtype=dtype)
    dQ  = torch.zeros(4,  device=device, dtype=dtype)
    dT  = torch.zeros(3,  device=device, dtype=dtype)

    grid = (triton.cdiv(N, BLOCK),          #   blocks over ref atoms
            triton.cdiv(M, BLOCK),          #   blocks over fit atoms
            1)                              #   single orientation

    _gauss_overlap_se3[grid](
        A.contiguous().view(-1),            # A_ptr
        B.contiguous().view(-1),            # B_ptr
        q.contiguous().view(-1),            # Q_ptr
        t.contiguous().view(-1),            # T_ptr
        N_real,                             # NEW  (B,) int32
        M_real,                             # NEW  (B,) int32
        1,                                  # BATCH
        M, N,                               # padded sizes = real sizes
        half_alpha, k_const,
        VAB, dQ, dT,
        BLOCK, BLOCK,
        NEED_GRAD=True,
    )

    return VAB, dQ, dT          #  unchanged public contract

def overlap_score_grad_se3_batch(A, B, q, t, *,
                                 alpha: float = 0.81,
                                 N_real: torch.Tensor | None = None,
                                 M_real: torch.Tensor | None = None,
                                 custom_grid=None):
    """
    A,B : (K,N_pad,3)/(K,M_pad,3)   q,t : (K,4)/(K,3)
    N_real/M_real : (K,) int32 - number of *valid* atoms per orientation.
                    If None, assume all N_pad / M_pad are valid.
    """
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    BLOCK = 64 if max(N_pad, M_pad) <= 64 else 128

    if N_real is None:
        N_real = A.new_full((K,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = B.new_full((K,), M_pad, dtype=torch.int32)

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    score = torch.zeros(K,  device=A.device, dtype=A.dtype)
    dQ    = torch.zeros_like(q)
    dT    = torch.zeros_like(t)

    grid = custom_grid if custom_grid is not None else \
           (triton.cdiv(N_pad, BLOCK), triton.cdiv(M_pad, BLOCK), K)

    _gauss_overlap_se3[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),          # NEW
        M_real.contiguous(),          # NEW
        K, M_pad, N_pad,
        half_alpha, k_const,
        score, dQ.view(-1), dT.view(-1),
        BLOCK, BLOCK,
        NEED_GRAD=True,
    )
    return score, dQ, dT

# ----------------------------------------------------------------------
#  Fused Adam update for (q,t)     – 1 thread-block = 1..256 orientations
# ----------------------------------------------------------------------
@triton.jit
def _adam_qt(
    Q_ptr, T_ptr,
    dQ_ptr, dT_ptr,
    Mq_ptr, Vq_ptr,
    Mt_ptr, Vt_ptr,
    K,
    lr: tl.constexpr,
    beta1: tl.constexpr = 0.9,
    beta2: tl.constexpr = 0.999,
    eps:   tl.constexpr = 1e-8,      
    BLOCK: tl.constexpr = 256,     # threads per CTA (must divide 1024)
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < K               # lanes beyond K are masked-out NOPs

    # ---------------- flat loads (4 components) ---------------------------
    # q           – parameters
    q0 = tl.load(Q_ptr + offs*4 + 0, mask=mask)
    q1 = tl.load(Q_ptr + offs*4 + 1, mask=mask)
    q2 = tl.load(Q_ptr + offs*4 + 2, mask=mask)
    q3 = tl.load(Q_ptr + offs*4 + 3, mask=mask)

    # dq          – gradients
    dq0 = tl.load(dQ_ptr + offs*4 + 0, mask=mask)
    dq1 = tl.load(dQ_ptr + offs*4 + 1, mask=mask)
    dq2 = tl.load(dQ_ptr + offs*4 + 2, mask=mask)
    dq3 = tl.load(dQ_ptr + offs*4 + 3, mask=mask)

    # first-moment & second-moment for q
    mq0 = tl.load(Mq_ptr + offs*4 + 0, mask=mask)
    mq1 = tl.load(Mq_ptr + offs*4 + 1, mask=mask)
    mq2 = tl.load(Mq_ptr + offs*4 + 2, mask=mask)
    mq3 = tl.load(Mq_ptr + offs*4 + 3, mask=mask)

    vq0 = tl.load(Vq_ptr + offs*4 + 0, mask=mask)
    vq1 = tl.load(Vq_ptr + offs*4 + 1, mask=mask)
    vq2 = tl.load(Vq_ptr + offs*4 + 2, mask=mask)
    vq3 = tl.load(Vq_ptr + offs*4 + 3, mask=mask)

    # ---------------- flat loads (3 components) ---------------------------
    t0 = tl.load(T_ptr + offs*3 + 0, mask=mask)
    t1 = tl.load(T_ptr + offs*3 + 1, mask=mask)
    t2 = tl.load(T_ptr + offs*3 + 2, mask=mask)

    dt0 = tl.load(dT_ptr + offs*3 + 0, mask=mask)
    dt1 = tl.load(dT_ptr + offs*3 + 1, mask=mask)
    dt2 = tl.load(dT_ptr + offs*3 + 2, mask=mask)

    mt0 = tl.load(Mt_ptr + offs*3 + 0, mask=mask)
    mt1 = tl.load(Mt_ptr + offs*3 + 1, mask=mask)
    mt2 = tl.load(Mt_ptr + offs*3 + 2, mask=mask)

    vt0 = tl.load(Vt_ptr + offs*3 + 0, mask=mask)
    vt1 = tl.load(Vt_ptr + offs*3 + 1, mask=mask)
    vt2 = tl.load(Vt_ptr + offs*3 + 2, mask=mask)

    # ---------------- Adam update (each component) ------------------------
    mq0 = beta1*mq0 + (1-beta1)*dq0;  vq0 = beta2*vq0 + (1-beta2)*dq0*dq0
    mq1 = beta1*mq1 + (1-beta1)*dq1;  vq1 = beta2*vq1 + (1-beta2)*dq1*dq1
    mq2 = beta1*mq2 + (1-beta1)*dq2;  vq2 = beta2*vq2 + (1-beta2)*dq2*dq2
    mq3 = beta1*mq3 + (1-beta1)*dq3;  vq3 = beta2*vq3 + (1-beta2)*dq3*dq3

    q0 = q0 - lr * mq0 / tl.sqrt(vq0 + eps)
    q1 = q1 - lr * mq1 / tl.sqrt(vq1 + eps)
    q2 = q2 - lr * mq2 / tl.sqrt(vq2 + eps)
    q3 = q3 - lr * mq3 / tl.sqrt(vq3 + eps)

    mt0 = beta1*mt0 + (1-beta1)*dt0; vt0 = beta2*vt0 + (1-beta2)*dt0*dt0
    mt1 = beta1*mt1 + (1-beta1)*dt1; vt1 = beta2*vt1 + (1-beta2)*dt1*dt1
    mt2 = beta1*mt2 + (1-beta1)*dt2; vt2 = beta2*vt2 + (1-beta2)*dt2*dt2

    t0 = t0 - lr * mt0 / tl.sqrt(vt0 + eps)
    t1 = t1 - lr * mt1 / tl.sqrt(vt1 + eps)
    t2 = t2 - lr * mt2 / tl.sqrt(vt2 + eps)

    # ---------------- renormalise quaternion -----------------------------
    inv_norm = 1.0 / tl.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    q0 *= inv_norm; q1 *= inv_norm; q2 *= inv_norm; q3 *= inv_norm

    # ---------------- stores ---------------------------------------------
    tl.store(Q_ptr + offs*4 + 0, q0, mask=mask)
    tl.store(Q_ptr + offs*4 + 1, q1, mask=mask)
    tl.store(Q_ptr + offs*4 + 2, q2, mask=mask)
    tl.store(Q_ptr + offs*4 + 3, q3, mask=mask)

    tl.store(Mq_ptr + offs*4 + 0, mq0, mask=mask)
    tl.store(Mq_ptr + offs*4 + 1, mq1, mask=mask)
    tl.store(Mq_ptr + offs*4 + 2, mq2, mask=mask)
    tl.store(Mq_ptr + offs*4 + 3, mq3, mask=mask)

    tl.store(Vq_ptr + offs*4 + 0, vq0, mask=mask)
    tl.store(Vq_ptr + offs*4 + 1, vq1, mask=mask)
    tl.store(Vq_ptr + offs*4 + 2, vq2, mask=mask)
    tl.store(Vq_ptr + offs*4 + 3, vq3, mask=mask)

    tl.store(T_ptr + offs*3 + 0, t0, mask=mask)
    tl.store(T_ptr + offs*3 + 1, t1, mask=mask)
    tl.store(T_ptr + offs*3 + 2, t2, mask=mask)

    tl.store(Mt_ptr + offs*3 + 0, mt0, mask=mask)
    tl.store(Mt_ptr + offs*3 + 1, mt1, mask=mask)
    tl.store(Mt_ptr + offs*3 + 2, mt2, mask=mask)

    tl.store(Vt_ptr + offs*3 + 0, vt0, mask=mask)
    tl.store(Vt_ptr + offs*3 + 1, vt1, mask=mask)
    tl.store(Vt_ptr + offs*3 + 2, vt2, mask=mask)


def fused_adam_qt(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr):
    K = q.shape[0]
    grid = (triton.cdiv(K, 256),)

    _adam_qt[grid](
        q.contiguous().view(-1),  t.contiguous().view(-1),
        dQ.contiguous().view(-1), dT.contiguous().view(-1),
        m_q.view(-1), v_q.view(-1), m_t.view(-1), v_t.view(-1),
        K, lr=lr,            #  ← constexpr keyword

        # beta1 / beta2 / eps / BLOCK keep their defaults
    )

# ------------------------------------------------------------------ #
#  One-time warm-up so the first real call doesn't pay for PTX build #
# ------------------------------------------------------------------ #
if torch.cuda.is_available():
    dummy = torch.zeros(512, 4, device="cuda", dtype=torch.float32)
    _adam_qt[(2,)](                         # 512 // 256 = 2 blocks
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        512, lr=0.001                      # K=512, any lr
    )
    torch.cuda.synchronize()