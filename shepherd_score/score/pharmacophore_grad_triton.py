# shepherd_score/score/pharmacophore_grad_triton.py
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
@triton.autotune(configs=[triton.Config({}, num_warps=_w) for _w in (1, 2, 4, 8)],
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
    E = tl.exp(-alpha_i * 0.5 * r2)
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
