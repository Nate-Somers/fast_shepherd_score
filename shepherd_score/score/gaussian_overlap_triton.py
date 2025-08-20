# shepherd_score/score/gaussian_overlap_triton.py
# A fused (forward + backward) Gaussian‐Tanimoto kernel using Triton.
#
import math, sys
import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def _load_xyz(ptr, idx, mask):
    """
    Safe load of a (K,3) array.  Any thread with mask=False uses a clamped
    pointer (index 0) so no out-of-bounds address is ever formed.
    """
    idx_safe = tl.where(mask, idx, 0)           
    x = tl.load(ptr + idx_safe * 3 + 0, mask=mask, cache_modifier=".ca")
    y = tl.load(ptr + idx_safe * 3 + 1, mask=mask, cache_modifier=".ca")
    z = tl.load(ptr + idx_safe * 3 + 2, mask=mask, cache_modifier=".ca")
    return x, y, z

# ----------------- score and gradients wrt quaternion q and translation t -----------------     
@triton.jit
def _gauss_overlap_se3_tiled(
    A_ptr, B_ptr,                 # flat (B * N_pad * 3), (B * M_pad * 3)
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B,)
    BATCH, M_pad, N_pad,          # ints
    half_alpha, k_const,          # scalars
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,), dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # single tile edge (e.g. 64)
    NEED_GRAD: tl.constexpr
):
    # -------- which alignment (one CTA per pair) --------
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)
    realM = tl.load(Mreal_ptr + pid)

    # -------- base pointers for this pair ---------------
    A_ptr  = A_ptr  + pid * N_pad * 3
    B_ptr  = B_ptr  + pid * M_pad * 3
    Q_ptr  = Q_ptr  + pid * 4
    T_ptr  = T_ptr  + pid * 3
    dQ_ptr = dQ_ptr + pid * 4
    dT_ptr = dT_ptr + pid * 3
    S_ptr  = S_ptr  + pid

    # -------- quaternion / translation ------------------
    qr = tl.load(Q_ptr + 0); qi = tl.load(Q_ptr + 1)
    qj = tl.load(Q_ptr + 2); qk = tl.load(Q_ptr + 3)
    tx = tl.load(T_ptr + 0); ty = tl.load(T_ptr + 1); tz = tl.load(T_ptr + 2)

    # rotation matrix (registers)
    two = 2.0
    r00 = 1 - two*(qj*qj + qk*qk); r01 = two*(qi*qj - qk*qr); r02 = two*(qi*qk + qj*qr)
    r10 = two*(qi*qj + qk*qr);     r11 = 1 - two*(qi*qi + qk*qk); r12 = two*(qj*qk - qi*qr)
    r20 = two*(qi*qk - qj*qr);     r21 = two*(qj*qk + qi*qr);     r22 = 1 - two*(qi*qi + qj*qj)

    # -------- accumulators (register) -------------------
    Vab_acc = 0.0
    dTx = 0.0; dTy = 0.0; dTz = 0.0
    dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0

    inv_ln2 = 1.4426950408889634

    # NOTE: outer loop over A tiles, inner loop over B tiles
    # Each tile load is once per loop -> reuse inside nested loops.
    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN

        # load A tile (x,y,z) into registers
        a_idx = tl.where(mask_n, offs_n, 0)
        ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
        ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
        az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM

            b_idx = tl.where(mask_m, offs_m, 0)
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            # rotate + translate B tile
            bx = r00*bx0 + r01*by0 + r02*bz0 + tx
            by = r10*bx0 + r11*by0 + r12*bz0 + ty
            bz = r20*bx0 + r21*by0 + r22*bz0 + tz

            # broadcast differences (BLOCK x BLOCK)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            pair_mask = mask_n[:, None] & mask_m[None, :]
            g = tl.where(pair_mask, g, 0.0)

            # overlap accumulation
            Vab_acc += tl.sum(g)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                # forces sum over i for each j (axis 0)
                fx = tl.sum(coeff * dx, 0)
                fy = tl.sum(coeff * dy, 0)
                fz = tl.sum(coeff * dz, 0)

                # translation grads (sum over valid j)
                dTx += tl.sum(fx)
                dTy += tl.sum(fy)
                dTz += tl.sum(fz)

                # quaternion grads (reuse original body-frame coords bx0,by0,bz0)
                # mask_m already applied via fx,fy,fz sums above (masked zeros)
                wq = qr; xq = qi; yq = qj; zq = qk
                four = 4.0

                # contributions per j
                dw = (
                    fx * (-two*zq*by0 + two*yq*bz0) +
                    fy * ( two*zq*bx0 - two*xq*bz0) +
                    fz * (-two*yq*bx0 + two*xq*by0)
                )
                dxq = (
                    fx * ( two*yq*by0 + two*zq*bz0) +
                    fy * ( two*yq*bx0 - four*xq*by0 - two*wq*bz0) +
                    fz * ( two*zq*bx0 + two*wq*by0 - four*xq*bz0)
                )
                dyq = (
                    fx * (-four*yq*bx0 + two*xq*by0 + two*wq*bz0) +
                    fy * (  two*xq*bx0                 + two*zq*bz0) +
                    fz * (-two*wq*bx0 + two*zq*by0 - four*yq*bz0)
                )
                dzq = (
                    fx * (-four*zq*bx0 - two*wq*by0 + two*xq*bz0) +
                    fy * ( two*wq*bx0 - four*zq*by0 + two*yq*bz0) +
                    fz * ( two*xq*bx0 + two*yq*by0                )
                )

                # mask again (safer if any fx,fy,fz lanes picked noise)
                dw  = tl.where(mask_m, dw,  0.0)
                dxq = tl.where(mask_m, dxq, 0.0)
                dyq = tl.where(mask_m, dyq, 0.0)
                dzq = tl.where(mask_m, dzq, 0.0)

                dQw += tl.sum(dw)
                dQx += tl.sum(dxq)
                dQy += tl.sum(dyq)
                dQz += tl.sum(dzq)

    # -------- single final write (no atomics needed) -------
    tl.store(S_ptr, Vab_acc)

    if NEED_GRAD:
        tl.store(dT_ptr + 0, dTx)
        tl.store(dT_ptr + 1, dTy)
        tl.store(dT_ptr + 2, dTz)
        tl.store(dQ_ptr + 0, dQw)
        tl.store(dQ_ptr + 1, dQx)
        tl.store(dQ_ptr + 2, dQy)
        tl.store(dQ_ptr + 3, dQz)


def overlap_score_grad_se3_batch(
    A, B, q, t, *,
    alpha: float = 0.81,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    BLOCK: int = 64
):
    """
    One CTA per alignment (pair). Internal tile loops over A,B.
    Shapes:
      A : (K, N_pad, 3)
      B : (K, M_pad, 3)
      q : (K, 4)
      t : (K, 3)
    """
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    device = A.device
    dtype  = A.dtype

    if N_real is None:
        N_real = torch.full((K,), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((K,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    out_S  = torch.zeros(K, device=device, dtype=dtype)
    out_dQ = torch.zeros_like(q)
    out_dT = torch.zeros_like(t)

    grid = (K,)    # 1-D launch: one CTA per alignment

    _gauss_overlap_se3_tiled[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad,
        half_alpha, k_const,
        out_S, out_dQ.view(-1), out_dT.view(-1),
        BLOCK,
        NEED_GRAD=NEED_GRAD,
    )
    return out_S, out_dQ, out_dT


#  Fused Adam update for (q,t)     – 1 thread-block = 1..256 orientations
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
        K, lr=lr       
    )

#  One-time warm-up so the first real call doesn't pay for PTX build #
if torch.cuda.is_available():
    dummy = torch.zeros(512, 4, device="cuda", dtype=torch.float32)
    _adam_qt[(2,)](                         # 512 // 256 = 2 blocks
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        dummy.view(-1), dummy.view(-1),
        512, lr=0.001,                    # K=512, any lr
    )
    torch.cuda.synchronize()

# ---------------------------------------------------------------------
# helper: batched self-overlap   VPP(P,P)  for a padded tensor
# ---------------------------------------------------------------------
@torch.no_grad()
def _batch_self_overlap(P_pad: torch.Tensor,
                              N_real: torch.Tensor,
                              alpha: float = 0.81) -> torch.Tensor:
    K, N_pad, _ = P_pad.shape
    q_id = torch.tensor([1.,0.,0.,0.], device=P_pad.device, dtype=P_pad.dtype).expand(K,4)
    t_0  = torch.zeros(K,3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_se3_batch(
        P_pad, P_pad, q_id, t_0,
        alpha=alpha,
        N_real=N_real, M_real=N_real,
        NEED_GRAD=False)
    return V






















# ---------------------------------------------------------------------------
#  QUATERNION-TORQUE CHAIN –  COMPLETE TRACE
#      run with:  python gaussian_overlap_triton.py --trace-quat-grad
# ---------------------------------------------------------------------------
if __name__ == "__main__" and "--trace-quat-grad" in sys.argv:
    import torch, math

    torch.manual_seed(0)
    dev, dt = "cuda", torch.float32
    α         = 0.81
    half_α    = 0.5 * α
    k_const   = math.pi**1.5 / ((2.0 * α) ** 1.5)

    # ---------- tiny system so the kernel is a single tile -----------------
    N, M = 32, 17
    A   = torch.randn(N, 3, device=dev, dtype=dt) 
    B0  = torch.randn(N, 3, device=dev, dtype=dt) 
    q    = torch.randn(4,  device=dev, dtype=dt);  q /= q.norm()
    t    = torch.randn(3,  device=dev, dtype=dt)

    # ----------------------------------------------------------------------
    #  1)  AUTOGRAD REFERENCE  (CPU, double-checked ground truth)
    # ----------------------------------------------------------------------
    A_cpu, B_cpu = A.cpu(), B0.cpu()
    q_unit = q.cpu().clone().detach()          # copy from CUDA → CPU
    q_unit /= q_unit.norm()                    # ensure unit length (out-of-graph)
    q_unit.requires_grad_(True)
    t_ref = t.cpu().clone().detach().requires_grad_(True)

    def quat_to_mat(q):
        w, x, y, z = q.unbind(-1)
        two = 2.0
        return torch.stack([
            torch.stack([1-two*(y*y+z*z), two*(x*y-z*w),   two*(x*z+y*w)]),
            torch.stack([two*(x*y+z*w),   1-two*(x*x+z*z), two*(y*z-x*w)]),
            torch.stack([two*(x*z-y*w),   two*(y*z+x*w),   1-two*(x*x+y*y)])
        ])

    R_ref  = quat_to_mat(q_unit)                        #   ⇒ gradient is tangent

    Bp    = (R_ref @ B_cpu.T).T + t_ref                 # (M,3)
    Bp.retain_grad()                   #  ← keep ∂V/∂Bp after backward()

    d     = A_cpu[:, None, :] - Bp[None, :, :]  
    d_0     = A_cpu[:, None, :] - B_cpu[None, :, :]        # (N,M,3)
    g_ref_0 = torch.exp(-half_α * (d_0*d_0).sum(-1)) * k_const
    g_ref = torch.exp(-half_α * (d*d).sum(-1)) * k_const
    VAB   = g_ref.sum()
    VAB.backward()

    dQ_autograd = q_unit.grad.detach()          # gradient in the same space as the kernel
    dT_autograd = t_ref.grad.detach()

    F_autograd = Bp.grad.detach()
    R_B_autograd      = Bp.detach() - t_ref.detach()
    tau_autograd   = torch.cross(R_B_autograd, F_autograd).sum(0)
    Sigma_d_autograd = (F_autograd * R_B_autograd).sum() + (dT_autograd * t_ref).sum()

    # ----------------------------------------------------------------------
    #  2)  BUILD EVERY “ATOMISTIC” PIECE ON THE CPU (no autograd)
    # ----------------------------------------------------------------------
    coeff  = (2*half_α) * g_ref
    F     = torch.sum(coeff.unsqueeze(-1) * d, 0)  # (M,3)
    dT_handwritten = F.sum(0)

    # (2) rotated B-coordinates   r_j = R(q) · B_j
    R_B  = (R_ref @ B_cpu.T).T                          # (M, 3)

    # torque   τ = Σ r_j × F_j
    tau_B = torch.linalg.cross(R_B, F).sum(0)          # (3,)  

    #  **TANGENT-SPACE** gradient  

    def dR_dq(q: torch.Tensor):
        """
        Return ∂R/∂w, ∂R/∂x, ∂R/∂y, ∂R/∂z  (each 3 × 3) for a unit quaternion q.
        Shapes: input  (4,)  →  tuple of 4 tensors, each (3,3)
        """
        w, x, y, z = q           # all are 0-D tensors
        two   = q.new_tensor(2.0)
        zero  = q.new_tensor(0.0)  # 0-D, same dtype/device as q

        dR_dw = torch.stack([
            torch.stack([  zero,   -two*z,   +two*y ]),
            torch.stack([ +two*z,   zero,    -two*x ]),
            torch.stack([ -two*y,  +two*x,    zero  ])
        ])

        dR_dx = torch.stack([
            torch.stack([  zero,     two*y,     two*z ]),
            torch.stack([  two*y,  -4*x,     -two*w ]),
            torch.stack([  two*z,   two*w,   -4*x  ])
        ])

        dR_dy = torch.stack([
            torch.stack([ -4*y,     two*x,     two*w ]),
            torch.stack([  two*x,    zero,     two*z ]),
            torch.stack([ -two*w,   two*z,   -4*y  ])
        ])

        dR_dz = torch.stack([
            torch.stack([ -4*z,    -two*w,    two*x ]),
            torch.stack([  two*w,   -4*z,     two*y ]),
            torch.stack([  two*x,    two*y,    zero  ])
        ])

        return dR_dw, dR_dx, dR_dy, dR_dz

    dR_dw, dR_dx, dR_dy, dR_dz = dR_dq(q_unit)          # matrices from the formulas
    B_body = B_cpu                                       # (M,3) in body frame

    def dVdq_component(dR):
        return torch.sum(F * (torch.matmul(dR, B_body.T).T))  # scalar

    dQ_handwritten = torch.tensor([
        dVdq_component(dR_dw),
        dVdq_component(dR_dx),
        dVdq_component(dR_dy),
        dVdq_component(dR_dz)
    ], dtype=torch.float32)

    # ----------------------------------------------------------------------
    #  4)  GRAB REAL KERNEL OUTPUT
    # ----------------------------------------------------------------------
    V_buf, dQ_buf, dT_buf = overlap_score_grad_se3_batch(A, B0, q, t, alpha=α)
    torch.cuda.synchronize()
    dQ_kernel = dQ_buf.cpu()
    dT_kernel = dT_buf.cpu()

    # ----------------------------------------------------------------------
    #  5)  REPORT
    # ----------------------------------------------------------------------
    def show(name, ref, test):
        Δ = (test - ref).abs()
        print(f"{name:<18s}  max|Δ| = {Δ.max():+.3e}   "
              f"{'✅' if torch.allclose(ref, test, rtol=1e-5, atol=1e-6) else '❌'}")

    print("\n================  TORQUE / QUATERNION CHAIN TRACE  ================")
    print("-------------------------------------------------------------------")
    print("dQ_autograd          :", dQ_autograd.numpy())
    print("dQ_handwritten       :", dQ_handwritten.detach().numpy())
    print("dQ_kernel            :", dQ_kernel.numpy())
    print("dT_autograd          :", dT_autograd.numpy())
    print("dT_handwritten       :", dT_handwritten.detach().numpy())
    print("dT_kernel            :", dT_kernel.detach().numpy())
    print("===================================================================")
    print("F_autograd           :", F_autograd.detach().numpy())
    print("F_handwritten        :", F.detach().numpy())
    print("RB_autograd          :", R_B_autograd.detach().numpy())
    print("RB_handwritten       :", R_B.detach().numpy())
    print("tau_autograd         :", tau_autograd.detach().numpy())
    print("tau_handwritten      :", tau_B.detach().numpy())
    print("Sigma_d_autograd     :", Sigma_d_autograd.detach().numpy())
    print("-------------------------------------------------------------------")
    print("g_ref_0              :", g_ref_0.numpy())
    print("g_ref                :", g_ref.detach().numpy())
    print("q_unit               :", q_unit.detach().numpy())
    print("R_ref                :", R_ref.detach().numpy())
    print("B                    :", B_cpu.numpy())
    print("d                    :", d.detach().numpy())
    print("===================================================================")



