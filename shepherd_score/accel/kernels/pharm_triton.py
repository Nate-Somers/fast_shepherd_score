# shepherd_score/accel/kernels/pharm_triton.py
# Triton value + SE(3) gradient for the typed/directional pharmacophore overlap.
# Mirrors score.analytical_gradients._torch.compute_overlap_and_grad_pharm
# (extended_points=False) so its outputs (O_AB, grad_R, grad_t) are a drop-in
# replacement for the analytical torch path in the pharm fine loop.
#
# One CTA per pose (grid=(P,)); a single BLOCK x BLOCK tile covers the (small)
# feature counts. Indexing follows the analytical code: FIT (B, rotated by R,t)
# is index i; REF (A, fixed) is index j; the type/alpha/K/cat come from FIT.
import torch
import triton
import triton.language as tl


# Self-tunes num_warps per (N_pad, M_pad) on the actual device. BLOCK is derived
# from the feature count (next_pow2), not a GPU-specific constant; only the warp
# count is hardware-dependent, so autotune picks it -- nothing hardcoded per GPU.
@triton.autotune(configs=[triton.Config({}, num_warps=_w, num_stages=_s)
                          for _w in (1, 2, 4, 8) for _s in (1, 2, 3, 4)],
                 key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _pharm_score_grad_kernel(
    Aanc_ptr, Banc_ptr,            # ref/fit anchors  (P*N*3) / (P*M*3)
    Avec_ptr, Bvec_ptr,            # ref/fit vectors  (raw; normalised inside)
    Atyp_ptr, Btyp_ptr,            # ref/fit type idx (P*N) / (P*M)  int32
    Nreal_ptr, Mreal_ptr,          # (P,) ref / fit real counts
    R_ptr, T_ptr,                  # (P*9) row-major R, (P*3) t
    alphas_ptr, Ks_ptr, cats_ptr,  # (T,) type lookup
    P, N_pad, M_pad,
    O_ptr, gR_ptr, gt_ptr,         # outputs: (P,), (P*9) row-major, (P*3)
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)   # ref count
    realM = tl.load(Mreal_ptr + pid)   # fit count

    Aanc = Aanc_ptr + pid * N_pad * 3
    Banc = Banc_ptr + pid * M_pad * 3
    Avec = Avec_ptr + pid * N_pad * 3
    Bvec = Bvec_ptr + pid * M_pad * 3
    Atyp = Atyp_ptr + pid * N_pad
    Btyp = Btyp_ptr + pid * M_pad
    Rb = R_ptr + pid * 9
    Tb = T_ptr + pid * 3

    r00 = tl.load(Rb + 0); r01 = tl.load(Rb + 1); r02 = tl.load(Rb + 2)
    r10 = tl.load(Rb + 3); r11 = tl.load(Rb + 4); r12 = tl.load(Rb + 5)
    r20 = tl.load(Rb + 6); r21 = tl.load(Rb + 7); r22 = tl.load(Rb + 8)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)

    # ---- FIT (i): rotate anchors + normalised vectors by (R, t) ----
    oi = tl.arange(0, BLOCK)
    mi = oi < realM
    fi = tl.where(mi, oi, 0)
    fax0 = tl.load(Banc + fi * 3 + 0, mask=mi, other=0.0)
    fay0 = tl.load(Banc + fi * 3 + 1, mask=mi, other=0.0)
    faz0 = tl.load(Banc + fi * 3 + 2, mask=mi, other=0.0)
    fvx = tl.load(Bvec + fi * 3 + 0, mask=mi, other=0.0)
    fvy = tl.load(Bvec + fi * 3 + 1, mask=mi, other=0.0)
    fvz = tl.load(Bvec + fi * 3 + 2, mask=mi, other=0.0)
    fnorm = tl.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
    finv = 1.0 / tl.where(fnorm > 1e-12, fnorm, 1e-12)
    fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv   # normalised orig fit vector
    ityp = tl.load(Btyp + fi, mask=mi, other=0)
    ialpha = tl.load(alphas_ptr + ityp)
    iK = tl.load(Ks_ptr + ityp)
    icat = tl.load(cats_ptr + ityp)
    # rotated fit anchor (a_t) and rotated normalised fit vector (v_t)
    fatx = r00 * fax0 + r01 * fay0 + r02 * faz0 + tx
    faty = r10 * fax0 + r11 * fay0 + r12 * faz0 + ty
    fatz = r20 * fax0 + r21 * fay0 + r22 * faz0 + tz
    fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
    fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
    fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn

    # ---- REF (j): fixed anchors + normalised vectors ----
    oj = tl.arange(0, BLOCK)
    mj = oj < realN
    rj = tl.where(mj, oj, 0)
    rax = tl.load(Aanc + rj * 3 + 0, mask=mj, other=0.0)
    ray = tl.load(Aanc + rj * 3 + 1, mask=mj, other=0.0)
    raz = tl.load(Aanc + rj * 3 + 2, mask=mj, other=0.0)
    rvx = tl.load(Avec + rj * 3 + 0, mask=mj, other=0.0)
    rvy = tl.load(Avec + rj * 3 + 1, mask=mj, other=0.0)
    rvz = tl.load(Avec + rj * 3 + 2, mask=mj, other=0.0)
    rnorm = tl.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    rinv = 1.0 / tl.where(rnorm > 1e-12, rnorm, 1e-12)
    rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
    jtyp = tl.load(Atyp + rj, mask=mj, other=0)

    # ---- (i,j) interaction tile ----
    dx = fatx[:, None] - rax[None, :]
    dy = faty[:, None] - ray[None, :]
    dz = fatz[:, None] - raz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    alpha_i = ialpha[:, None]
    E = tl.exp2((-alpha_i * 0.5 * r2) * 1.4426950408889634)   # exp2(x*log2 e)=e^x; hw ex2.approx
    pair = (ityp[:, None] == jtyp[None, :]) & (icat[:, None] != 3) & mi[:, None] & mj[None, :]
    E = tl.where(pair, E, 0.0)

    # direction weighting from D = v_t_i . v_ref_j
    D = fvtx[:, None] * rvxn[None, :] + fvty[:, None] * rvyn[None, :] + fvtz[:, None] * rvzn[None, :]
    Dcl = tl.minimum(tl.maximum(D, 0.0), 1.0)
    w_dir = (Dcl + 2.0) / 3.0
    w_arom = (tl.abs(D) + 2.0) / 3.0
    cat_i = icat[:, None]
    w = tl.where(cat_i == 1, w_dir, tl.where(cat_i == 2, w_arom, 1.0))

    Ki = iK[:, None]
    KwE = Ki * w * E
    O = tl.sum(KwE)

    if NEED_GRAD:
        aKwE = -alpha_i * KwE                       # -alpha K w E
        # grad_t = sum_ij aKwE * (a_t_i - a_ref_j)
        gtx = tl.sum(aKwE * dx); gty = tl.sum(aKwE * dy); gtz = tl.sum(aKwE * dz)
        # grad_R_spatial[a,b] = sum aKwE * delta_a * a_orig_fit_i[b]
        ax0 = fax0[:, None]; ay0 = fay0[:, None]; az0 = faz0[:, None]
        gR00 = tl.sum(aKwE * dx * ax0); gR01 = tl.sum(aKwE * dx * ay0); gR02 = tl.sum(aKwE * dx * az0)
        gR10 = tl.sum(aKwE * dy * ax0); gR11 = tl.sum(aKwE * dy * ay0); gR12 = tl.sum(aKwE * dy * az0)
        gR20 = tl.sum(aKwE * dz * ax0); gR21 = tl.sum(aKwE * dz * ay0); gR22 = tl.sum(aKwE * dz * az0)
        # grad_R_weight[a,b] = sum coeff * ref_vn_j[a] * fit_vn_orig_i[b]
        c = tl.where(cat_i == 1, tl.where((D > 0.0) & (D < 1.0), 1.0, 0.0),
                     tl.where(cat_i == 2, tl.where(D > 0.0, 1.0, tl.where(D < 0.0, -1.0, 0.0)), 0.0))
        coeff = (1.0 / 3.0) * Ki * E * c
        rvxnb = rvxn[None, :]; rvynb = rvyn[None, :]; rvznb = rvzn[None, :]
        fvxnb = fvxn[:, None]; fvynb = fvyn[:, None]; fvznb = fvzn[:, None]
        gR00 += tl.sum(coeff * rvxnb * fvxnb); gR01 += tl.sum(coeff * rvxnb * fvynb); gR02 += tl.sum(coeff * rvxnb * fvznb)
        gR10 += tl.sum(coeff * rvynb * fvxnb); gR11 += tl.sum(coeff * rvynb * fvynb); gR12 += tl.sum(coeff * rvynb * fvznb)
        gR20 += tl.sum(coeff * rvznb * fvxnb); gR21 += tl.sum(coeff * rvznb * fvynb); gR22 += tl.sum(coeff * rvznb * fvznb)

    tl.store(O_ptr + pid, O)
    if NEED_GRAD:
        gtb = gt_ptr + pid * 3
        tl.store(gtb + 0, gtx); tl.store(gtb + 1, gty); tl.store(gtb + 2, gtz)
        gRbo = gR_ptr + pid * 9
        tl.store(gRbo + 0, gR00); tl.store(gRbo + 1, gR01); tl.store(gRbo + 2, gR02)
        tl.store(gRbo + 3, gR10); tl.store(gRbo + 4, gR11); tl.store(gRbo + 5, gR12)
        tl.store(gRbo + 6, gR20); tl.store(gRbo + 7, gR21); tl.store(gRbo + 8, gR22)


# ===========================================================================
#  DIRECTIONLESS "color" kernel for vol_color: same same-type-only typed Gaussian
#  but isotropic (w=1, no vectors, no weight-gradient), takes the QUATERNION q
#  (not R) and emits dV/dq DIRECTLY in-register -- byte-identical dV/dq tail to the
#  shape kernel (_gauss_overlap_se3_tiled), so the driver drops the
#  rotation->quaternion projection / normalization-Jacobian tail entirely.
#  q is assumed unit (the adam renormalizes each step), as in the shape kernel.
#  A = ref anchors (axis 0), B = fit anchors (axis 1, rotated). dx = A - rot(B):
#  the same sign convention as the shape kernel, so its dV/dq tail is reused verbatim.
# ===========================================================================
@triton.autotune(configs=[triton.Config({}, num_warps=_w, num_stages=_s)
                          for _w in (1, 2, 4, 8) for _s in (1, 2, 3, 4)],
                 key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _pharm_color_grad_kernel(
    Aanc_ptr, Banc_ptr,            # ref/fit anchors  (P*N*3) / (P*M*3)
    Atyp_ptr, Btyp_ptr,            # ref/fit type idx (P*N) / (P*M)  int32
    Nreal_ptr, Mreal_ptr,          # (P,) ref / fit real counts
    Q_ptr, T_ptr,                  # (P*4) quaternion (w,x,y,z), (P*3) t
    alphas_ptr, Ks_ptr, cats_ptr,  # (T,) type lookup
    P, N_pad, M_pad,
    O_ptr, dQ_ptr, dT_ptr,         # outputs: (P,), (P*4), (P*3)
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)   # ref count
    realM = tl.load(Mreal_ptr + pid)   # fit count

    Aanc = Aanc_ptr + pid * N_pad * 3
    Banc = Banc_ptr + pid * M_pad * 3
    Atyp = Atyp_ptr + pid * N_pad
    Btyp = Btyp_ptr + pid * M_pad
    Qb = Q_ptr + pid * 4
    Tb = T_ptr + pid * 3

    qr = tl.load(Qb + 0); qi = tl.load(Qb + 1); qj = tl.load(Qb + 2); qk = tl.load(Qb + 3)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)
    two = 2.0; four = 4.0
    r00 = 1 - two * (qj * qj + qk * qk); r01 = two * (qi * qj - qk * qr); r02 = two * (qi * qk + qj * qr)
    r10 = two * (qi * qj + qk * qr); r11 = 1 - two * (qi * qi + qk * qk); r12 = two * (qj * qk - qi * qr)
    r20 = two * (qi * qk - qj * qr); r21 = two * (qj * qk + qi * qr); r22 = 1 - two * (qi * qi + qj * qj)
    inv_ln2 = 1.4426950408889634

    # ---- REF (axis 0, n): fixed anchors + types ----
    on = tl.arange(0, BLOCK)
    mn = on < realN
    ni = tl.where(mn, on, 0)
    ax = tl.load(Aanc + ni * 3 + 0, mask=mn, other=0.0)
    ay = tl.load(Aanc + ni * 3 + 1, mask=mn, other=0.0)
    az = tl.load(Aanc + ni * 3 + 2, mask=mn, other=0.0)
    ntyp = tl.load(Atyp + ni, mask=mn, other=0)

    # ---- FIT (axis 1, m): anchors rotated by (R(q), t); per-type alpha/K/cat ----
    om = tl.arange(0, BLOCK)
    mm = om < realM
    mIdx = tl.where(mm, om, 0)
    bx0 = tl.load(Banc + mIdx * 3 + 0, mask=mm, other=0.0)
    by0 = tl.load(Banc + mIdx * 3 + 1, mask=mm, other=0.0)
    bz0 = tl.load(Banc + mIdx * 3 + 2, mask=mm, other=0.0)
    mtyp = tl.load(Btyp + mIdx, mask=mm, other=0)
    malpha = tl.load(alphas_ptr + mtyp)
    mK = tl.load(Ks_ptr + mtyp)
    mcat = tl.load(cats_ptr + mtyp)
    bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
    by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
    bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz

    # ---- (n,m) tile: same-type isotropic Gaussian ----
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    half_a = malpha[None, :] * 0.5
    g = tl.exp2((-half_a * r2) * inv_ln2) * mK[None, :]
    pair = mn[:, None] & mm[None, :] & (ntyp[:, None] == mtyp[None, :]) & (mcat[None, :] != 3)
    g = tl.where(pair, g, 0.0)
    O = tl.sum(g)

    if NEED_GRAD:
        coeff = malpha[None, :] * g                 # alpha_m * g  (N x M)
        fx = tl.sum(coeff * dx, 0)                  # per-fit force (M,)
        fy = tl.sum(coeff * dy, 0)
        fz = tl.sum(coeff * dz, 0)
        dTx = tl.sum(fx); dTy = tl.sum(fy); dTz = tl.sum(fz)
        # dV/dq via the body-frame fit coords (bx0,by0,bz0) -- verbatim shape-kernel tail
        wq = qr; xq = qi; yq = qj; zq = qk
        dw = (fx * (-two * zq * by0 + two * yq * bz0) +
              fy * (two * zq * bx0 - two * xq * bz0) +
              fz * (-two * yq * bx0 + two * xq * by0))
        dxq = (fx * (two * yq * by0 + two * zq * bz0) +
               fy * (two * yq * bx0 - four * xq * by0 - two * wq * bz0) +
               fz * (two * zq * bx0 + two * wq * by0 - four * xq * bz0))
        dyq = (fx * (-four * yq * bx0 + two * xq * by0 + two * wq * bz0) +
               fy * (two * xq * bx0 + two * zq * bz0) +
               fz * (-two * wq * bx0 + two * zq * by0 - four * yq * bz0))
        dzq = (fx * (-four * zq * bx0 - two * wq * by0 + two * xq * bz0) +
               fy * (two * wq * bx0 - four * zq * by0 + two * yq * bz0) +
               fz * (two * xq * bx0 + two * yq * by0))
        dw = tl.where(mm, dw, 0.0); dxq = tl.where(mm, dxq, 0.0)
        dyq = tl.where(mm, dyq, 0.0); dzq = tl.where(mm, dzq, 0.0)
        dQw = tl.sum(dw); dQx = tl.sum(dxq); dQy = tl.sum(dyq); dQz = tl.sum(dzq)

    tl.store(O_ptr + pid, O)
    if NEED_GRAD:
        dQb = dQ_ptr + pid * 4
        tl.store(dQb + 0, dQw); tl.store(dQb + 1, dQx); tl.store(dQb + 2, dQy); tl.store(dQb + 3, dQz)
        dTb = dT_ptr + pid * 3
        tl.store(dTb + 0, dTx); tl.store(dTb + 1, dTy); tl.store(dTb + 2, dTz)


def pharm_color_score_grad_se3_batch(
    A, B, q, t, ref_types, fit_types, alphas, Ks, cats, *,
    N_real=None, M_real=None, NEED_GRAD=True, BLOCK=None, num_warps=None, num_stages=None,
):
    """Triton directionless-color value+QUATERNION-grad kernel for vol_color.
    A = ref anchors (P,N,3), B = fit anchors (P,M,3), q=(P,4), t=(P,3);
    ref/fit_types (P,N)/(P,M) int. Returns (O, dQ, dT) with dQ = dO/dq (like the shape
    kernel) -- no rotation->quaternion projection needed downstream."""
    P, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev = A.device
    if BLOCK is None:
        BLOCK = triton.next_power_of_2(max(N_pad, M_pad, 16))
    if N_real is None:
        N_real = torch.full((P,), N_pad, device=dev, dtype=torch.int32)
    if M_real is None:
        M_real = torch.full((P,), M_pad, device=dev, dtype=torch.int32)

    O = torch.zeros(P, device=dev, dtype=A.dtype)
    dQ = torch.zeros(P, 4, device=dev, dtype=A.dtype)
    dT = torch.zeros(P, 3, device=dev, dtype=A.dtype)

    _pharm_color_grad_kernel[(P,)](
        A.contiguous().view(-1), B.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        N_real.to(torch.int32).contiguous(), M_real.to(torch.int32).contiguous(),
        q.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, N_pad, M_pad,
        O, dQ.view(-1), dT.view(-1),
        BLOCK, NEED_GRAD,
    )
    return O, dQ, dT


# ===========================================================================
#  DIRECTIONAL pharm value + QUATERNION gradient (pharm mode in-register dQ).
#  Same typed/directional Gaussian + weight as _pharm_score_grad_kernel, but takes
#  q (builds R; assumes |q|=1) and emits dV/dq directly via the shape dR/dq tail
#  applied to (positional force, fit anchor) + (weight force, fit vector).
#  Layout: REF=axis 0 (n), FIT=axis 1 (m); dx = ref - rot(fit) (shape convention).
# ===========================================================================
@triton.autotune(configs=[triton.Config({}, num_warps=_w, num_stages=_s)
                          for _w in (1, 2, 4, 8) for _s in (1, 2, 3, 4)],
                 key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _pharm_grad_dq_kernel(
    Aanc_ptr, Banc_ptr, Avec_ptr, Bvec_ptr,
    Atyp_ptr, Btyp_ptr, Nreal_ptr, Mreal_ptr,
    Q_ptr, T_ptr, alphas_ptr, Ks_ptr, cats_ptr,
    P, N_pad, M_pad,
    O_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)   # ref
    realM = tl.load(Mreal_ptr + pid)   # fit
    Aanc = Aanc_ptr + pid * N_pad * 3; Banc = Banc_ptr + pid * M_pad * 3
    Avec = Avec_ptr + pid * N_pad * 3; Bvec = Bvec_ptr + pid * M_pad * 3
    Atyp = Atyp_ptr + pid * N_pad; Btyp = Btyp_ptr + pid * M_pad
    Qb = Q_ptr + pid * 4; Tb = T_ptr + pid * 3

    qr = tl.load(Qb + 0); qi = tl.load(Qb + 1); qj = tl.load(Qb + 2); qk = tl.load(Qb + 3)
    tx = tl.load(Tb + 0); ty = tl.load(Tb + 1); tz = tl.load(Tb + 2)
    two = 2.0; four = 4.0
    r00 = 1 - two * (qj * qj + qk * qk); r01 = two * (qi * qj - qk * qr); r02 = two * (qi * qk + qj * qr)
    r10 = two * (qi * qj + qk * qr); r11 = 1 - two * (qi * qi + qk * qk); r12 = two * (qj * qk - qi * qr)
    r20 = two * (qi * qk - qj * qr); r21 = two * (qj * qk + qi * qr); r22 = 1 - two * (qi * qi + qj * qj)
    inv_ln2 = 1.4426950408889634

    # REF (axis 0, n): anchors + normalised vectors + types
    on = tl.arange(0, BLOCK); mn = on < realN; ni = tl.where(mn, on, 0)
    ax = tl.load(Aanc + ni * 3 + 0, mask=mn, other=0.0)
    ay = tl.load(Aanc + ni * 3 + 1, mask=mn, other=0.0)
    az = tl.load(Aanc + ni * 3 + 2, mask=mn, other=0.0)
    rvx = tl.load(Avec + ni * 3 + 0, mask=mn, other=0.0)
    rvy = tl.load(Avec + ni * 3 + 1, mask=mn, other=0.0)
    rvz = tl.load(Avec + ni * 3 + 2, mask=mn, other=0.0)
    rnorm = tl.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    rinv = 1.0 / tl.where(rnorm > 1e-12, rnorm, 1e-12)
    rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
    ntyp = tl.load(Atyp + ni, mask=mn, other=0)

    # FIT (axis 1, m): anchors + normalised vectors + per-type alpha/K/cat
    om = tl.arange(0, BLOCK); mm = om < realM; mIdx = tl.where(mm, om, 0)
    bx0 = tl.load(Banc + mIdx * 3 + 0, mask=mm, other=0.0)
    by0 = tl.load(Banc + mIdx * 3 + 1, mask=mm, other=0.0)
    bz0 = tl.load(Banc + mIdx * 3 + 2, mask=mm, other=0.0)
    fvx = tl.load(Bvec + mIdx * 3 + 0, mask=mm, other=0.0)
    fvy = tl.load(Bvec + mIdx * 3 + 1, mask=mm, other=0.0)
    fvz = tl.load(Bvec + mIdx * 3 + 2, mask=mm, other=0.0)
    fnorm = tl.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
    finv = 1.0 / tl.where(fnorm > 1e-12, fnorm, 1e-12)
    fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv
    mtyp = tl.load(Btyp + mIdx, mask=mm, other=0)
    malpha = tl.load(alphas_ptr + mtyp); mK = tl.load(Ks_ptr + mtyp); mcat = tl.load(cats_ptr + mtyp)
    bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
    by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
    bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
    fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
    fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
    fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn

    # (n,m) tile: dx = ref - rot(fit)
    dx = ax[:, None] - bx[None, :]
    dy = ay[:, None] - by[None, :]
    dz = az[:, None] - bz[None, :]
    r2 = dx * dx + dy * dy + dz * dz
    al_m = malpha[None, :]
    E = tl.exp2((-al_m * 0.5 * r2) * inv_ln2)
    pair = mn[:, None] & mm[None, :] & (ntyp[:, None] == mtyp[None, :]) & (mcat[None, :] != 3)
    E = tl.where(pair, E, 0.0)
    D = fvtx[None, :] * rvxn[:, None] + fvty[None, :] * rvyn[:, None] + fvtz[None, :] * rvzn[:, None]
    Dcl = tl.minimum(tl.maximum(D, 0.0), 1.0)
    cat_m = mcat[None, :]
    w = tl.where(cat_m == 1, (Dcl + 2.0) / 3.0, tl.where(cat_m == 2, (tl.abs(D) + 2.0) / 3.0, 1.0))
    KwE = mK[None, :] * w * E
    O = tl.sum(KwE)

    if NEED_GRAD:
        # positional force per fit m (sum over ref n): coeff = alpha_m * KwE
        cpos = al_m * KwE
        fx = tl.sum(cpos * dx, 0); fy = tl.sum(cpos * dy, 0); fz = tl.sum(cpos * dz, 0)
        dTx = tl.sum(fx); dTy = tl.sum(fy); dTz = tl.sum(fz)
        # weight force per fit m: coeff = (1/3) K_m E c ; wforce = sum_n coeff * ref_vn_n
        c = tl.where(cat_m == 1, tl.where((D > 0.0) & (D < 1.0), 1.0, 0.0),
                     tl.where(cat_m == 2, tl.where(D > 0.0, 1.0, tl.where(D < 0.0, -1.0, 0.0)), 0.0))
        cwt = (1.0 / 3.0) * mK[None, :] * E * c
        wfx = tl.sum(cwt * rvxn[:, None], 0); wfy = tl.sum(cwt * rvyn[:, None], 0); wfz = tl.sum(cwt * rvzn[:, None], 0)
        # dV/dq = shape dR/dq tail with (positional force, fit anchor) + (weight force, fit vector)
        dw = (fx * (-two * qk * by0 + two * qj * bz0) + fy * (two * qk * bx0 - two * qi * bz0) + fz * (-two * qj * bx0 + two * qi * by0)
              + wfx * (-two * qk * fvyn + two * qj * fvzn) + wfy * (two * qk * fvxn - two * qi * fvzn) + wfz * (-two * qj * fvxn + two * qi * fvyn))
        dxq = (fx * (two * qj * by0 + two * qk * bz0) + fy * (two * qj * bx0 - four * qi * by0 - two * qr * bz0) + fz * (two * qk * bx0 + two * qr * by0 - four * qi * bz0)
               + wfx * (two * qj * fvyn + two * qk * fvzn) + wfy * (two * qj * fvxn - four * qi * fvyn - two * qr * fvzn) + wfz * (two * qk * fvxn + two * qr * fvyn - four * qi * fvzn))
        dyq = (fx * (-four * qj * bx0 + two * qi * by0 + two * qr * bz0) + fy * (two * qi * bx0 + two * qk * bz0) + fz * (-two * qr * bx0 + two * qk * by0 - four * qj * bz0)
               + wfx * (-four * qj * fvxn + two * qi * fvyn + two * qr * fvzn) + wfy * (two * qi * fvxn + two * qk * fvzn) + wfz * (-two * qr * fvxn + two * qk * fvyn - four * qj * fvzn))
        dzq = (fx * (-four * qk * bx0 - two * qr * by0 + two * qi * bz0) + fy * (two * qr * bx0 - four * qk * by0 + two * qj * bz0) + fz * (two * qi * bx0 + two * qj * by0)
               + wfx * (-four * qk * fvxn - two * qr * fvyn + two * qi * fvzn) + wfy * (two * qr * fvxn - four * qk * fvyn + two * qj * fvzn) + wfz * (two * qi * fvxn + two * qj * fvyn))
        dw = tl.where(mm, dw, 0.0); dxq = tl.where(mm, dxq, 0.0); dyq = tl.where(mm, dyq, 0.0); dzq = tl.where(mm, dzq, 0.0)
        dQw = tl.sum(dw); dQx = tl.sum(dxq); dQy = tl.sum(dyq); dQz = tl.sum(dzq)

    tl.store(O_ptr + pid, O)
    if NEED_GRAD:
        dQb = dQ_ptr + pid * 4
        tl.store(dQb + 0, dQw); tl.store(dQb + 1, dQx); tl.store(dQb + 2, dQy); tl.store(dQb + 3, dQz)
        dTb = dT_ptr + pid * 3
        tl.store(dTb + 0, dTx); tl.store(dTb + 1, dTy); tl.store(dTb + 2, dTz)


def pharm_grad_dq_se3_batch(
    q, t, ref_types, fit_types, ref_anchors, fit_anchors, ref_vectors, fit_vectors,
    alphas, Ks, cats, *, N_real=None, M_real=None, NEED_GRAD=True, BLOCK=None, num_warps=None, num_stages=None,
):
    """Triton directional pharm value+QUATERNION-grad kernel (in-register dQ). Takes q (not R);
    returns (O, dQ, dT) with dQ = dO/dq -- no R->q projection needed downstream."""
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev = ref_anchors.device
    if BLOCK is None:
        BLOCK = triton.next_power_of_2(max(N_pad, M_pad, 16))
    if N_real is None:
        N_real = torch.full((P,), N_pad, device=dev, dtype=torch.int32)
    if M_real is None:
        M_real = torch.full((P,), M_pad, device=dev, dtype=torch.int32)
    O = torch.zeros(P, device=dev, dtype=ref_anchors.dtype)
    dQ = torch.zeros(P, 4, device=dev, dtype=ref_anchors.dtype)
    dT = torch.zeros(P, 3, device=dev, dtype=ref_anchors.dtype)
    _pharm_grad_dq_kernel[(P,)](
        ref_anchors.contiguous().view(-1), fit_anchors.contiguous().view(-1),
        ref_vectors.contiguous().view(-1), fit_vectors.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        N_real.to(torch.int32).contiguous(), M_real.to(torch.int32).contiguous(),
        q.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, N_pad, M_pad, O, dQ.view(-1), dT.view(-1), BLOCK, NEED_GRAD,
    )
    return O, dQ, dT


def pharm_score_grad_se3_batch(
    R, t, ref_types, fit_types, ref_anchors, fit_anchors, ref_vectors, fit_vectors,
    alphas, Ks, cats, *, N_real=None, M_real=None, NEED_GRAD=True,
    BLOCK=None, num_warps=1,
):
    """Triton value+grad for the pharm overlap; outputs match
    compute_overlap_and_grad_pharm(extended_points=False): (O_AB, grad_R, grad_t).
    R: (P,3,3), t: (P,3); *_anchors/*_vectors: (P,N/M,3); *_types: (P,N/M) int.
    """
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev = ref_anchors.device
    if BLOCK is None:
        BLOCK = triton.next_power_of_2(max(N_pad, M_pad, 16))
    if N_real is None:
        N_real = torch.full((P,), N_pad, device=dev, dtype=torch.int32)
    if M_real is None:
        M_real = torch.full((P,), M_pad, device=dev, dtype=torch.int32)

    O = torch.zeros(P, device=dev, dtype=ref_anchors.dtype)
    gR = torch.zeros(P, 3, 3, device=dev, dtype=ref_anchors.dtype)
    gt = torch.zeros(P, 3, device=dev, dtype=ref_anchors.dtype)

    _pharm_score_grad_kernel[(P,)](
        ref_anchors.contiguous().view(-1), fit_anchors.contiguous().view(-1),
        ref_vectors.contiguous().view(-1), fit_vectors.contiguous().view(-1),
        ref_types.to(torch.int32).contiguous().view(-1), fit_types.to(torch.int32).contiguous().view(-1),
        N_real.to(torch.int32).contiguous(), M_real.to(torch.int32).contiguous(),
        R.contiguous().view(-1), t.contiguous().view(-1),
        alphas.contiguous(), Ks.contiguous(), cats.to(torch.int32).contiguous(),
        P, N_pad, M_pad,
        O, gR.view(-1), gt.view(-1),
        BLOCK, NEED_GRAD,
    )
    return O, gR, gt
