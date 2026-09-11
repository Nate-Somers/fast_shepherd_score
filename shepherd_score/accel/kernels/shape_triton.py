"""Fused forward+backward Gaussian-Tanimoto overlap kernels in Triton.

Provides the value + SE(3) gradient (dV/dq, dV/dt) kernel used by the batched
coarse-to-fine aligners on CUDA tensors, plus the fused Adam quaternion/translation
update. The numba CPU twins live in :mod:`~shepherd_score.accel.kernels.cpu`.
"""
from __future__ import annotations

import math
import triton
import triton.language as tl
import torch


# ----------------- score and gradients wrt quaternion q and translation t -----------------
# BLOCK/num_warps/num_stages are chosen by triton.autotune per (N_pad, M_pad) on the actual
# device and cached, so the kernel self-tunes to any GPU / Triton version. Candidates are
# deliberately small tiles (BLOCK <= 64, <= 4 warps): the kernel runs ONE CTA per pair, so a
# batch already launches thousands of CTAs and occupancy comes from the pair count -- larger
# tiles are never selected and only lengthen the cold-start autotune sweep.
_OVERLAP_CONFIGS = [
    triton.Config({'BLOCK': _b}, num_warps=_w, num_stages=_s)
    for _b in (16, 32, 64) for _w in (1, 2, 4) for _s in (1, 2, 3, 4)
]

# TESTED AND REVERTED: the 4-warp ceiling above is a single-pose observation, so a wider sweep
# (num_warps up to 8) was tried for the multi-pose kernel, which does POSES times the work per
# CTA. It changed NOTHING -- surf reproduced at 1.661/1.661/1.701 and vol was flat within run
# variance -- while making the autotune sweep 33% larger (gate 5 went 6s -> 48.6s). The
# autotuner selects the same config either way, so the ceiling was never the constraint.


# --- shared SE(3) device functions (inlined at zero cost by @triton.jit) ------------------
# The quaternion->rotation-matrix build and the overlap-force->quaternion-gradient tail were
# copy-pasted byte-for-byte into every overlap kernel (shape + ESP here, and imported by
# esp_triton). Factoring them into these @triton.jit helpers makes a correctness fix a SINGLE
# edit instead of N identical ones and removes the silent-divergence risk. Triton inlines a
# @triton.jit callee into its caller, so this is bit-identical to the inline blocks (validated:
# kernel V/dQ/dT match a torch-autograd reference).
@triton.jit
def _quat_to_rotmat(qr, qi, qj, qk):
    """Rotation matrix (row-major r00..r22) from a quaternion (w,x,y,z). q need NOT be unit
    -- the value kernels apply this to the raw optimiser state and the norm cancels in the
    Tanimoto ratio (renormalisation happens in the Adam step)."""
    two = 2.0
    r00 = 1 - two*(qj*qj + qk*qk); r01 = two*(qi*qj - qk*qr); r02 = two*(qi*qk + qj*qr)
    r10 = two*(qi*qj + qk*qr);     r11 = 1 - two*(qi*qi + qk*qk); r12 = two*(qj*qk - qi*qr)
    r20 = two*(qi*qk - qj*qr);     r21 = two*(qj*qk + qi*qr);     r22 = 1 - two*(qi*qi + qj*qj)
    return r00, r01, r02, r10, r11, r12, r20, r21, r22


@triton.jit
def _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk):
    """Per-point quaternion-gradient contributions (dw,dx,dy,dz) from the translation-force
    components (fx,fy,fz) and the body-frame fit coords (bx0,by0,bz0). Analytic d(R b)/dq
    contracted with the force; caller masks padding and reduces. Works elementwise, so fx..bz0
    may be scalars or per-lane vectors."""
    two = 2.0; four = 4.0
    dw = (fx * (-two*qk*by0 + two*qj*bz0)
          + fy * ( two*qk*bx0 - two*qi*bz0)
          + fz * (-two*qj*bx0 + two*qi*by0))
    dxq = (fx * ( two*qj*by0 + two*qk*bz0)
           + fy * ( two*qj*bx0 - four*qi*by0 - two*qr*bz0)
           + fz * ( two*qk*bx0 + two*qr*by0 - four*qi*bz0))
    dyq = (fx * (-four*qj*bx0 + two*qi*by0 + two*qr*bz0)
           + fy * ( two*qi*bx0                 + two*qk*bz0)
           + fz * (-two*qr*bx0 + two*qk*by0 - four*qj*bz0))
    dzq = (fx * (-four*qk*bx0 - two*qr*by0 + two*qi*bz0)
           + fy * ( two*qr*bx0 - four*qk*by0 + two*qj*bz0)
           + fz * ( two*qi*bx0 + two*qj*by0))
    return dw, dxq, dyq, dzq


# cache_results=True persists the chosen (BLOCK, num_warps) to the Triton cache dir
# keyed by (N_pad, M_pad), so the per-process autotune sweep (~4 s/shape) is paid
# once per machine, not once per process -- big win for fresh-process workloads.
@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _gauss_overlap_se3_tiled(
    A_ptr, B_ptr,                 # flat (B * N_pad * 3), (B * M_pad * 3)
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B,)
    BATCH, M_pad, N_pad,          # ints
    half_alpha, k_const,          # scalars
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,), dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # tile edge (chosen by autotune)
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr,          # poses per molecule; 1 == one molecule per CTA (legacy)
    SOA: tl.constexpr = False,    # coordinate layout: (K,3,P_pad) instead of (K,P_pad,3)
):
    # -------- which alignment (one CTA per pair) --------
    pid = tl.program_id(0)
    # One CTA per POSE, but coordinates belong to a MOLECULE. With SEEDS > 1 the caller passes
    # the molecule blocks UNREPLICATED and the SEEDS consecutive CTAs of a molecule all read its
    # one copy; with SEEDS == 1 this is pid, i.e. the original one-molecule-per-CTA layout.
    mol = pid // SEEDS
    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)

    # -------- base pointers: coords by molecule, pose state by CTA ---------
    A_ptr  = A_ptr  + mol * N_pad * 3
    B_ptr  = B_ptr  + mol * M_pad * 3
    Q_ptr  = Q_ptr  + pid * 4
    T_ptr  = T_ptr  + pid * 3
    dQ_ptr = dQ_ptr + pid * 4
    dT_ptr = dT_ptr + pid * 3
    S_ptr  = S_ptr  + pid

    # -------- quaternion / translation ------------------
    qr = tl.load(Q_ptr + 0); qi = tl.load(Q_ptr + 1)
    qj = tl.load(Q_ptr + 2); qk = tl.load(Q_ptr + 3)
    tx = tl.load(T_ptr + 0); ty = tl.load(T_ptr + 1); tz = tl.load(T_ptr + 2)

    # rotation matrix (registers) -- shared device fn (inlined, bit-identical)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

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

        # load A tile (x,y,z) into registers.
        # SOA: the three coordinate planes are contiguous, so a tile load is BLOCK adjacent
        # floats. In the AoS layout each of these is a stride-3 gather, which Nsight Compute
        # measured at 8.9 of 32 bytes used per sector (L40S, job 22595748) -- ~3x the sectors
        # for the same data, on a kernel whose L1/TEX pipe is already 91% busy while DRAM sits
        # at 5%. Values, arithmetic and reduction order are untouched, so the two layouts are
        # bit-identical; only the address changes.
        a_idx = tl.where(mask_n, offs_n, 0)
        if SOA:
            ax = tl.load(A_ptr + 0 * N_pad + a_idx, mask=mask_n, other=0.0)
            ay = tl.load(A_ptr + 1 * N_pad + a_idx, mask=mask_n, other=0.0)
            az = tl.load(A_ptr + 2 * N_pad + a_idx, mask=mask_n, other=0.0)
        else:
            ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
            ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
            az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM

            b_idx = tl.where(mask_m, offs_m, 0)
            if SOA:
                bx0 = tl.load(B_ptr + 0 * M_pad + b_idx, mask=mask_m, other=0.0)
                by0 = tl.load(B_ptr + 1 * M_pad + b_idx, mask=mask_m, other=0.0)
                bz0 = tl.load(B_ptr + 2 * M_pad + b_idx, mask=mask_m, other=0.0)
            else:
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

                # quaternion grads (shared device fn; reuses body-frame coords bx0,by0,bz0).
                # mask_m already applied via fx,fy,fz sums above (masked zeros).
                dw, dxq, dyq, dzq = _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk)

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


# ---------------------------------------------------------------------------------------
# Multi-pose variant: POSES poses of ONE molecule per CTA.
#
# The single-pose kernel above is latency/occupancy-bound, not bandwidth- or compute-bound:
# measured at ~5% of an L40S's fp32 peak and ~17% of its SFU throughput, doing only ~N_pad*M_pad
# (~1024) pair-evals per CTA with <= 4 warps. Removing the coordinate replication was worth 1.01x,
# which ruled out memory traffic; what is left is simply too little work per CTA to hide latency.
#
# This does the same arithmetic with POSES times more of it per CTA. The A and B tiles are loaded
# ONCE per (n-tile, m-tile) and reused across POSES poses, so the quaternion->rotmat build, the
# tile loads and the loop overhead amortise over POSES instead of being paid per pose.
#


@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'])
@triton.jit
def _gauss_overlap_se3_multipose(
    A_ptr, B_ptr,
    Q_ptr, T_ptr,
    Nreal_ptr, Mreal_ptr,
    BATCH, M_pad, N_pad,
    half_alpha, k_const,
    S_ptr, dQ_ptr, dT_ptr,
    BLOCK: tl.constexpr,
    NEED_GRAD: tl.constexpr,
    SEEDS: tl.constexpr,
    POSES: tl.constexpr,
    POSES_PAD: tl.constexpr,      # next power of two >= POSES (tl.arange extent)
):
    """POSES poses of ONE molecule per CTA, vectorised over the pose axis.

    The single-pose kernel is latency/occupancy-bound, not bandwidth- or compute-bound: measured
    at ~5% of an L40S fp32 peak and ~17% of its SFU, doing only ~N_pad*M_pad (~1024) pair-evals
    per CTA with <= 4 warps. Deduplicating its coordinate traffic was worth 1.008x, which ruled
    memory out; what remains is too little work per CTA to hide latency.

    Here the A and B tiles load ONCE and the pose axis broadcasts over them, so every tile load,
    the quaternion->rotmat build and the loop overhead amortise over POSES poses, and each CTA
    carries POSES times the arithmetic.

    Poses are a TENSOR dimension rather than an unrolled Python loop -- Triton has no
    ``__setitem__``, so list accumulators are not expressible. That makes POSES a ``tl.arange``
    extent, hence power-of-two, and it must divide SEEDS so a CTA never straddles two molecules.

    Identical arithmetic and identical per-pose accumulation order to the single-pose kernel,
    so results are bit-identical; only which CTA performs them changes.
    """
    pid = tl.program_id(0)
    base = pid * POSES
    mol = base // SEEDS                    # SEEDS % POSES == 0 => all POSES share this molecule

    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)
    A_ptr = A_ptr + mol * N_pad * 3
    B_ptr = B_ptr + mol * M_pad * 3

    # -------- pose state: (POSES,) vectors, built once --------
    # tl.arange demands a power-of-two extent, so run POSES_PAD lanes and mask the tail. That
    # is what lets POSES be 5 or 10 -- vol has 10 seeds, so without this the only testable value
    # was 2, which is too little extra work per CTA to decide anything.
    p_off = tl.arange(0, POSES_PAD)
    mask_p = p_off < POSES
    qo = (base + p_off) * 4
    to = (base + p_off) * 3
    qr = tl.load(Q_ptr + qo + 0, mask=mask_p, other=0.0)
    qi = tl.load(Q_ptr + qo + 1, mask=mask_p, other=0.0)
    qj = tl.load(Q_ptr + qo + 2, mask=mask_p, other=0.0)
    qk = tl.load(Q_ptr + qo + 3, mask=mask_p, other=0.0)
    tx = tl.load(T_ptr + to + 0, mask=mask_p, other=0.0)
    ty = tl.load(T_ptr + to + 1, mask=mask_p, other=0.0)
    tz = tl.load(T_ptr + to + 2, mask=mask_p, other=0.0)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

    Vab_acc = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTx = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTy = tl.zeros([POSES_PAD], dtype=tl.float32)
    dTz = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQw = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQx = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQy = tl.zeros([POSES_PAD], dtype=tl.float32)
    dQz = tl.zeros([POSES_PAD], dtype=tl.float32)

    inv_ln2 = 1.4426950408889634

    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN
        a_idx = tl.where(mask_n, offs_n, 0)
        ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
        ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
        az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM
            b_idx = tl.where(mask_m, offs_m, 0)
            # ONE load, reused by every pose via the broadcast below
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            # (POSES, BLOCK): each pose's rotation applied to the shared body-frame tile
            bx = r00[:, None]*bx0[None, :] + r01[:, None]*by0[None, :] + r02[:, None]*bz0[None, :] + tx[:, None]
            by = r10[:, None]*bx0[None, :] + r11[:, None]*by0[None, :] + r12[:, None]*bz0[None, :] + ty[:, None]
            bz = r20[:, None]*bx0[None, :] + r21[:, None]*by0[None, :] + r22[:, None]*bz0[None, :] + tz[:, None]

            # (POSES, BLOCK_n, BLOCK_m)
            dx = ax[None, :, None] - bx[:, None, :]
            dy = ay[None, :, None] - by[:, None, :]
            dz = az[None, :, None] - bz[:, None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            pair_mask = mask_n[None, :, None] & mask_m[None, None, :]
            g = tl.where(pair_mask, g, 0.0)

            # Reduce the (n, m) plane in ONE pass over a flattened row, matching the
            # single-pose kernel's tl.sum(g) over a flat (BLOCK, BLOCK). The two-stage
            # tl.sum(tl.sum(g,2),1) is a DIFFERENT summation tree, and float addition is not
            # associative -- that, not the arithmetic, is why multi-pose was not bit-identical.
            Vab_acc += tl.sum(tl.reshape(g, [POSES_PAD, BLOCK * BLOCK]), 1)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                fx = tl.sum(coeff * dx, 1)            # (POSES, BLOCK_m): sum over i
                fy = tl.sum(coeff * dy, 1)
                fz = tl.sum(coeff * dz, 1)

                dTx += tl.sum(fx, 1)
                dTy += tl.sum(fy, 1)
                dTz += tl.sum(fz, 1)

                dw, dxq, dyq, dzq = _quat_grad_tail(
                    fx, fy, fz,
                    bx0[None, :], by0[None, :], bz0[None, :],
                    qr[:, None], qi[:, None], qj[:, None], qk[:, None])
                mm = mask_m[None, :]
                dQw += tl.sum(tl.where(mm, dw, 0.0), 1)
                dQx += tl.sum(tl.where(mm, dxq, 0.0), 1)
                dQy += tl.sum(tl.where(mm, dyq, 0.0), 1)
                dQz += tl.sum(tl.where(mm, dzq, 0.0), 1)

    tl.store(S_ptr + base + p_off, Vab_acc, mask=mask_p)
    if NEED_GRAD:
        tl.store(dT_ptr + to + 0, dTx, mask=mask_p)
        tl.store(dT_ptr + to + 1, dTy, mask=mask_p)
        tl.store(dT_ptr + to + 2, dTz, mask=mask_p)
        tl.store(dQ_ptr + qo + 0, dQw, mask=mask_p)
        tl.store(dQ_ptr + qo + 1, dQx, mask=mask_p)
        tl.store(dQ_ptr + qo + 2, dQy, mask=mask_p)
        tl.store(dQ_ptr + qo + 3, dQz, mask=mask_p)


def overlap_score_grad_se3_batch(
    A, B, q, t, *,
    alpha: float = 0.81,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    BLOCK: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    seeds_per_mol: int = 1,
    poses_per_cta: int = 1,
    soa: bool = False,
):
    """
    One CTA per POSE. Internal tile loops over A,B.
    Shapes (``seeds_per_mol == 1``, the default and the legacy layout):
      A : (K, N_pad, 3)
      B : (K, M_pad, 3)
      q : (K, 4)
      t : (K, 3)

    With ``seeds_per_mol = S > 1`` the coordinate blocks are UNREPLICATED and the pose tensors
    carry every pose:
      A : (K // S, N_pad, 3)      N_real, M_real : (K // S,)
      q : (K, 4)                  t : (K, 3)
    CTA ``i`` then reads molecule ``i // S``. Identical arithmetic on identical values -- only
    the address changes -- so results are bit-identical to the replicated layout.

    If BLOCK is None, an optimal block size is auto-selected based on N_pad and M_pad.
    """
    K = q.shape[0]                       # POSES == CTAs
    S = int(seeds_per_mol)
    # SoA hands the kernel (K, 3, P_pad) -- the three coordinate planes contiguous -- so the
    # point count is the LAST axis there and the middle one in the default AoS layout.
    if soa:
        n_mol, _, N_pad = A.shape
        _, _, M_pad = B.shape
    else:
        n_mol, N_pad, _ = A.shape
        _, M_pad, _ = B.shape
    if S < 1 or K % S != 0 or n_mol != K // S:
        raise ValueError(
            f"seeds_per_mol={S} inconsistent: q has {K} poses, A has {n_mol} molecules "
            f"(expected {K // S if S else 0})")
    device = A.device
    dtype  = A.dtype

    if N_real is None:
        N_real = torch.full((n_mol,), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((n_mol,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    half_alpha = 0.5 * alpha
    k_const    = math.pi**1.5 / ((2.0 * alpha) ** 1.5)

    # Both kernels below STORE every element of out_S, and every element of out_dQ/out_dT when
    # NEED_GRAD (the multi-pose variant's mask only hides the tl.arange padding lanes, which
    # address no real pose), so pre-zeroing is three memset kernels per fine step writing
    # buffers that are about to be overwritten -- 0.0159 us/mol of device time on a vol screen
    # at N=100,000 (job 22593930). Without NEED_GRAD the gradient buffers ARE left unwritten,
    # so those keep their zeros rather than handing a caller uninitialised memory.
    out_S  = torch.empty(K, device=device, dtype=dtype)
    if NEED_GRAD:
        out_dQ = torch.empty_like(q)
        out_dT = torch.empty_like(t)
    else:
        out_dQ = torch.zeros_like(q)
        out_dT = torch.zeros_like(t)

    POSES = int(poses_per_cta)
    if POSES > 1:
        # POSES poses of ONE molecule per CTA. Every pose in a CTA must share a molecule, so
        # SEEDS must divide evenly by POSES; K % POSES follows from that.
        if S <= 1 or S % POSES != 0 or K % POSES != 0:
            raise ValueError(
                f"poses_per_cta={POSES} needs the deduped layout and SEEDS % POSES == 0 "
                f"(got seeds_per_mol={S}, K={K})")
        POSES_PAD = 1 << (POSES - 1).bit_length()      # tl.arange extent; tail lanes masked
        _gauss_overlap_se3_multipose[(K // POSES,)](
            A.contiguous().view(-1), B.contiguous().view(-1),
            q.contiguous().view(-1), t.contiguous().view(-1),
            N_real.contiguous(), M_real.contiguous(),
            K, M_pad, N_pad, half_alpha, k_const,
            out_S, out_dQ.view(-1), out_dT.view(-1),
            NEED_GRAD=NEED_GRAD, SEEDS=S, POSES=POSES, POSES_PAD=POSES_PAD,
        )
        return out_S, out_dQ, out_dT

    grid = (K,)    # 1-D launch: one CTA per alignment

    # BLOCK + num_warps are chosen by triton.autotune per (N_pad, M_pad) on the
    # ACTUAL device (see _OVERLAP_CONFIGS) -- nothing GPU-specific. The legacy
    # BLOCK/num_warps/num_stages kwargs are accepted for back-compat but ignored.
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
        NEED_GRAD=NEED_GRAD,
        SEEDS=S,
        SOA=bool(soa),
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
    PROJECT: tl.constexpr = False, # if True, tangent-project dQ: dQ -= q*(dQ.q)
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

    if PROJECT:                      # tangent-space projection: dQ -= q*(dQ.q)
        radial = dq0*q0 + dq1*q1 + dq2*q2 + dq3*q3
        dq0 = dq0 - q0 * radial
        dq1 = dq1 - q1 * radial
        dq2 = dq2 - q2 * radial
        dq3 = dq3 - q3 * radial

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
    _warmup_adam_qt()                       # lazy one-time PTX build (must not run at import)
    K = q.shape[0]
    grid = (triton.cdiv(K, 256),)

    _adam_qt[grid](
        q.contiguous().view(-1),  t.contiguous().view(-1),
        dQ.contiguous().view(-1), dT.contiguous().view(-1),
        m_q.view(-1), v_q.view(-1), m_t.view(-1), v_t.view(-1),
        K, lr=lr
    )


def fused_adam_qt_with_tangent_proj(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr):
    """
    Fused Adam update with tangent-space projection for quaternion gradients.

    Unlike fused_adam_qt, this function accepts the raw gradient dQ (before
    tangent projection) and performs the projection internally in the kernel,
    saving one memory read/write cycle.

    Args:
        q, t: quaternion (K,4) and translation (K,3) parameters (updated in-place)
        dQ: raw quaternion gradients (K,4) - NOT tangent-projected
        dT: translation gradients (K,3)
        m_q, v_q, m_t, v_t: Adam moment tensors (updated in-place)
        lr: learning rate
    """
    K = q.shape[0]
    grid = (triton.cdiv(K, 256),)

    _adam_qt[grid](
        q.contiguous().view(-1),  t.contiguous().view(-1),
        dQ.contiguous().view(-1), dT.contiguous().view(-1),
        m_q.view(-1), v_q.view(-1), m_t.view(-1), v_t.view(-1),
        K, lr=lr, PROJECT=True
    )


#  One-time _adam_qt warm-up so the first real call doesn't pay the PTX build.
#  Must stay lazy -- never call this at import. Importing this module must NOT
#  initialize CUDA: it would allocate GPU memory just by importing, and (crucially)
#  it would poison the fork-based multi-GPU pool (shepherd_score.accel.multi_gpu),
#  which can only fork its workers while the parent has not initialized CUDA.
#  Triton JIT-compiles on first launch regardless; this only front-loads it.
_ADAM_QT_WARMED = False


def _warmup_adam_qt():
    global _ADAM_QT_WARMED
    if _ADAM_QT_WARMED or not torch.cuda.is_available():
        return
    _ADAM_QT_WARMED = True
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


# =======================================================================================
# FUSED FINE STEP: overlap + Tanimoto + best-pose tracking + Adam, in ONE kernel
# =======================================================================================
# MEASURED MOTIVATION (job 22593930, L40S, vol screen N=100,000, CUPTI per-kernel):
# a fine step was 17 CUDA kernels -- the overlap kernel, three output memsets (the
# ``torch.zeros`` for S/dQ/dT), the twelve elementwise ops of
# ``_GraphedFineSurf._tanimoto_adam_tail``, and ``_adam_qt``. The overlap kernel is
# 0.4977 us/mol of device time; everything else in the step adds 0.143 us/mol (elementwise
# 0.0901 + Adam 0.0366 + memset 0.0159), i.e. 22% of the step's GPU time spent on work that
# touches seven scalars per pose. dQ/dT make a full HBM round trip purely to be multiplied
# by a scalar and negated -- the overlap kernel already holds them in registers.
#
# One CTA owns one POSE, so the whole tail is a per-CTA SCALAR epilogue: load this pose's
# norm / best / best-pose / Adam moments, finish the step, store back. Nothing leaves
# registers between the gradient and the parameter update, and a step becomes ONE launch.
#
# The arithmetic is copied operation-for-operation from ``_tanimoto_adam_tail`` plus
# ``_adam_qt`` (PROJECT=True), in the same order, on the same values. The one place the two
# could legitimately differ is float division: torch divides with IEEE rounding, Triton's
# ``/`` does not promise to. So the two Tanimoto divides below pin ``ieee_rounding=True``
# while the Adam divide keeps the plain ``/`` that ``_adam_qt`` already ships. Bit-identity
# is therefore a MEASURED claim, not a structural one -- see ``benchmarks/fused_step_parity.py``.


@triton.jit
def _overlap_accum_tile(A_ptr, B_ptr, realN, realM,
                        qr, qi, qj, qk, tx, ty, tz,
                        half_alpha, k_const, N_pad, M_pad,
                        BLOCK: tl.constexpr, NEED_GRAD: tl.constexpr):
    """Tiled Gaussian-overlap value + SE(3) gradient for ONE pose, as a device function.

    Lifted verbatim out of :func:`_gauss_overlap_se3_tiled` so the fused-step kernel and the
    standalone kernel share one copy of the arithmetic. ``@triton.jit`` callees are inlined,
    which is what makes that sharing free -- and bit-identical to the inline block it
    replaces (the same argument the file already relies on for ``_quat_to_rotmat``).
    """
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat(qr, qi, qj, qk)

    Vab_acc = 0.0
    dTx = 0.0; dTy = 0.0; dTz = 0.0
    dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0

    inv_ln2 = 1.4426950408889634

    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN

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

            bx = r00*bx0 + r01*by0 + r02*bz0 + tx
            by = r10*bx0 + r11*by0 + r12*bz0 + ty
            bz = r20*bx0 + r21*by0 + r22*bz0 + tz

            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            r2 = dx*dx + dy*dy + dz*dz

            g = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const
            pair_mask = mask_n[:, None] & mask_m[None, :]
            g = tl.where(pair_mask, g, 0.0)

            Vab_acc += tl.sum(g)

            if NEED_GRAD:
                coeff = (2.0 * half_alpha) * g
                fx = tl.sum(coeff * dx, 0)
                fy = tl.sum(coeff * dy, 0)
                fz = tl.sum(coeff * dz, 0)

                dTx += tl.sum(fx)
                dTy += tl.sum(fy)
                dTz += tl.sum(fz)

                dw, dxq, dyq, dzq = _quat_grad_tail(fx, fy, fz, bx0, by0, bz0, qr, qi, qj, qk)

                dw  = tl.where(mask_m, dw,  0.0)
                dxq = tl.where(mask_m, dxq, 0.0)
                dyq = tl.where(mask_m, dyq, 0.0)
                dzq = tl.where(mask_m, dzq, 0.0)

                dQw += tl.sum(dw)
                dQx += tl.sum(dxq)
                dQy += tl.sum(dyq)
                dQz += tl.sum(dzq)

    return Vab_acc, dTx, dTy, dTz, dQw, dQx, dQy, dQz


# This kernel MUTATES q / t / best / best-pose / Adam moments in place, and the autotuner runs
# each candidate config several times on the real buffers -- without ``restore_value`` a tuning
# sweep would silently advance the optimiser by ~100 steps before the first real one. The list
# names the ARGUMENTS to snapshot and roll back between trials.
@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True,
                 restore_value=['Q_ptr', 'T_ptr', 'Best_ptr', 'Bq_ptr', 'Bt_ptr',
                                'Mq_ptr', 'Vq_ptr', 'Mt_ptr', 'Vt_ptr'])
@triton.jit
def _gauss_overlap_se3_fused_step(
    A_ptr, B_ptr,                       # (nmol*N_pad*3), (nmol*M_pad*3)
    Q_ptr, T_ptr,                       # (P*4), (P*3)  -- read AND written
    Nreal_ptr, Mreal_ptr,               # (nmol,)
    Norm_ptr,                           # (P,)  VAA+VBB per pose
    Best_ptr, Bq_ptr, Bt_ptr,           # (P,), (P*4), (P*3)  best score + pose so far
    Mq_ptr, Vq_ptr, Mt_ptr, Vt_ptr,     # Adam moments
    BATCH, M_pad, N_pad,
    half_alpha, k_const, lr,
    BLOCK: tl.constexpr,
    SEEDS: tl.constexpr,
    beta1: tl.constexpr = 0.9,
    beta2: tl.constexpr = 0.999,
    eps:   tl.constexpr = 1e-8,
):
    pid = tl.program_id(0)
    mol = pid // SEEDS                  # SEEDS>1: coordinates are per MOLECULE, poses per CTA
    realN = tl.load(Nreal_ptr + mol)
    realM = tl.load(Mreal_ptr + mol)

    A_ptr = A_ptr + mol * N_pad * 3
    B_ptr = B_ptr + mol * M_pad * 3
    qo = pid * 4
    to = pid * 3

    qr = tl.load(Q_ptr + qo + 0); qi = tl.load(Q_ptr + qo + 1)
    qj = tl.load(Q_ptr + qo + 2); qk = tl.load(Q_ptr + qo + 3)
    tx = tl.load(T_ptr + to + 0); ty = tl.load(T_ptr + to + 1); tz = tl.load(T_ptr + to + 2)

    Vab, dTx, dTy, dTz, dQw, dQx, dQy, dQz = _overlap_accum_tile(
        A_ptr, B_ptr, realN, realM, qr, qi, qj, qk, tx, ty, tz,
        half_alpha, k_const, N_pad, M_pad, BLOCK, True)

    # ---- Tanimoto score + d(score)/d(VAB) scale   (== _tanimoto_adam_tail ops 1-4) ------
    norm  = tl.load(Norm_ptr + pid)
    denom = norm - Vab
    score = tl.fdiv(Vab, denom, ieee_rounding=True)
    d2    = denom * denom
    scale = tl.fdiv(norm, d2, ieee_rounding=True)

    # ---- best-pose tracking   (== the gt + three where's) -------------------------------
    bst = tl.load(Best_ptr + pid)
    better = score > bst
    tl.store(Best_ptr + pid, tl.where(better, score, bst))
    tl.store(Bq_ptr + qo + 0, tl.where(better, qr, tl.load(Bq_ptr + qo + 0)))
    tl.store(Bq_ptr + qo + 1, tl.where(better, qi, tl.load(Bq_ptr + qo + 1)))
    tl.store(Bq_ptr + qo + 2, tl.where(better, qj, tl.load(Bq_ptr + qo + 2)))
    tl.store(Bq_ptr + qo + 3, tl.where(better, qk, tl.load(Bq_ptr + qo + 3)))
    tl.store(Bt_ptr + to + 0, tl.where(better, tx, tl.load(Bt_ptr + to + 0)))
    tl.store(Bt_ptr + to + 1, tl.where(better, ty, tl.load(Bt_ptr + to + 1)))
    tl.store(Bt_ptr + to + 2, tl.where(better, tz, tl.load(Bt_ptr + to + 2)))

    # ---- ascent direction: -(grad * scale)   (== the two mul + neg_ pairs) --------------
    dq0 = -(dQw * scale); dq1 = -(dQx * scale)
    dq2 = -(dQy * scale); dq3 = -(dQz * scale)
    dt0 = -(dTx * scale); dt1 = -(dTy * scale); dt2 = -(dTz * scale)

    # ---- tangent projection + Adam + renormalise   (== _adam_qt, PROJECT=True) ----------
    radial = dq0*qr + dq1*qi + dq2*qj + dq3*qk
    dq0 = dq0 - qr * radial
    dq1 = dq1 - qi * radial
    dq2 = dq2 - qj * radial
    dq3 = dq3 - qk * radial

    mq0 = tl.load(Mq_ptr + qo + 0); mq1 = tl.load(Mq_ptr + qo + 1)
    mq2 = tl.load(Mq_ptr + qo + 2); mq3 = tl.load(Mq_ptr + qo + 3)
    vq0 = tl.load(Vq_ptr + qo + 0); vq1 = tl.load(Vq_ptr + qo + 1)
    vq2 = tl.load(Vq_ptr + qo + 2); vq3 = tl.load(Vq_ptr + qo + 3)
    mt0 = tl.load(Mt_ptr + to + 0); mt1 = tl.load(Mt_ptr + to + 1)
    mt2 = tl.load(Mt_ptr + to + 2)
    vt0 = tl.load(Vt_ptr + to + 0); vt1 = tl.load(Vt_ptr + to + 1)
    vt2 = tl.load(Vt_ptr + to + 2)

    mq0 = beta1*mq0 + (1-beta1)*dq0;  vq0 = beta2*vq0 + (1-beta2)*dq0*dq0
    mq1 = beta1*mq1 + (1-beta1)*dq1;  vq1 = beta2*vq1 + (1-beta2)*dq1*dq1
    mq2 = beta1*mq2 + (1-beta1)*dq2;  vq2 = beta2*vq2 + (1-beta2)*dq2*dq2
    mq3 = beta1*mq3 + (1-beta1)*dq3;  vq3 = beta2*vq3 + (1-beta2)*dq3*dq3

    qr = qr - lr * mq0 / tl.sqrt(vq0 + eps)
    qi = qi - lr * mq1 / tl.sqrt(vq1 + eps)
    qj = qj - lr * mq2 / tl.sqrt(vq2 + eps)
    qk = qk - lr * mq3 / tl.sqrt(vq3 + eps)

    mt0 = beta1*mt0 + (1-beta1)*dt0; vt0 = beta2*vt0 + (1-beta2)*dt0*dt0
    mt1 = beta1*mt1 + (1-beta1)*dt1; vt1 = beta2*vt1 + (1-beta2)*dt1*dt1
    mt2 = beta1*mt2 + (1-beta1)*dt2; vt2 = beta2*vt2 + (1-beta2)*dt2*dt2

    tx = tx - lr * mt0 / tl.sqrt(vt0 + eps)
    ty = ty - lr * mt1 / tl.sqrt(vt1 + eps)
    tz = tz - lr * mt2 / tl.sqrt(vt2 + eps)

    inv_norm = 1.0 / tl.sqrt(qr*qr + qi*qi + qj*qj + qk*qk)
    qr *= inv_norm; qi *= inv_norm; qj *= inv_norm; qk *= inv_norm

    tl.store(Q_ptr + qo + 0, qr); tl.store(Q_ptr + qo + 1, qi)
    tl.store(Q_ptr + qo + 2, qj); tl.store(Q_ptr + qo + 3, qk)
    tl.store(Mq_ptr + qo + 0, mq0); tl.store(Mq_ptr + qo + 1, mq1)
    tl.store(Mq_ptr + qo + 2, mq2); tl.store(Mq_ptr + qo + 3, mq3)
    tl.store(Vq_ptr + qo + 0, vq0); tl.store(Vq_ptr + qo + 1, vq1)
    tl.store(Vq_ptr + qo + 2, vq2); tl.store(Vq_ptr + qo + 3, vq3)
    tl.store(T_ptr + to + 0, tx); tl.store(T_ptr + to + 1, ty)
    tl.store(T_ptr + to + 2, tz)
    tl.store(Mt_ptr + to + 0, mt0); tl.store(Mt_ptr + to + 1, mt1)
    tl.store(Mt_ptr + to + 2, mt2)
    tl.store(Vt_ptr + to + 0, vt0); tl.store(Vt_ptr + to + 1, vt1)
    tl.store(Vt_ptr + to + 2, vt2)


def fused_fine_step(A, B, q, t, norm, best, bq, bt, mq, vq, mt, vt, *,
                    alpha: float = 0.81, N_real, M_real, lr: float,
                    seeds_per_mol: int = 1):
    """One complete fine-optimiser step for every pose, in a single kernel launch.

    Replaces ``overlap_score_grad_se3_batch`` + the twelve-op Tanimoto/best/gradient tail +
    ``fused_adam_qt_with_tangent_proj``. ``q, t, best, bq, bt, mq, vq, mt, vt`` are all
    updated IN PLACE; nothing is returned and nothing is allocated, which is also what makes
    it capture cleanly into a CUDA graph.

    ``A``/``B`` follow the same convention as :func:`overlap_score_grad_se3_batch`: with
    ``seeds_per_mol = S > 1`` they hold ``P // S`` UNREPLICATED molecules and CTA ``i`` reads
    molecule ``i // S``.
    """
    K = q.shape[0]
    S = int(seeds_per_mol)
    n_mol, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    if S < 1 or K % S != 0 or n_mol != K // S:
        raise ValueError(
            f"seeds_per_mol={S} inconsistent: q has {K} poses, A has {n_mol} molecules "
            f"(expected {K // S if S else 0})")

    half_alpha = 0.5 * alpha
    k_const = math.pi**1.5 / ((2.0 * alpha) ** 1.5)
    # Same int32 coercion the standalone wrapper does -- the kernel compares the tile offsets
    # against these directly, so an int64 count would read the wrong stride.
    N_real = N_real.to(device=A.device, dtype=torch.int32, copy=False)
    M_real = M_real.to(device=A.device, dtype=torch.int32, copy=False)

    _gauss_overlap_se3_fused_step[(K,)](
        A.contiguous().view(-1), B.contiguous().view(-1),
        q.view(-1), t.view(-1),
        N_real.contiguous(), M_real.contiguous(),
        norm,
        best, bq.view(-1), bt.view(-1),
        mq.view(-1), vq.view(-1), mt.view(-1), vt.view(-1),
        K, M_pad, N_pad, half_alpha, k_const, float(lr),
        SEEDS=S,
    )


# =======================================================================================
# FUSED TANIMOTO + BEST-POSE + ADAM TAIL: thirteen elementwise launches -> one
# =======================================================================================
# The fine step's tail is, per pose, seven scalars of arithmetic. It shipped as twelve torch
# elementwise ops plus ``_adam_qt``, each of which reads its inputs from HBM and writes its
# output back: ``denom``, ``score``, ``d2``, ``scale``, ``better``, then three ``where``s and
# two mul+neg pairs, and only then the Adam kernel. Measured on an L40S vol screen at
# N=100,000 (job 22593930, CUPTI): elementwise 0.0901 + Adam 0.0366 + the three output memsets
# 0.0159 = 0.143 us/mol of device time, against 0.4977 for the overlap kernel itself.
#
# This does the same arithmetic with ONE thread per pose and one launch, so every intermediate
# stays in registers. It is the pose-parallel counterpart of ``_gauss_overlap_se3_fused_step``:
# that one folded the same work into the overlap CTA and measured 1.004x (the tail was never
# launch-bound -- 128 threads redundantly doing one pose's scalars costs what it saves). Here
# the parallelism matches the work.
#
# BIT-IDENTITY: the operations, their order, and their inputs are unchanged, and every one is
# elementwise, so there is no reduction whose order could shift. The one place torch and Triton
# can legitimately disagree is float division, so the two Tanimoto divides pin
# ``ieee_rounding=True`` and the Adam divide keeps the plain ``/`` that ``_adam_qt`` ships.


@triton.jit
def _tanimoto_best_adam_kernel(
    V_ptr, dQ_ptr, dT_ptr,              # (P,), (P*4), (P*3)   this step's value + gradient
    Norm_ptr,                           # (P,)                 VAA+VBB per pose
    Q_ptr, T_ptr,                       # (P*4), (P*3)         parameters, updated in place
    Best_ptr, Bq_ptr, Bt_ptr,           # (P,), (P*4), (P*3)   best score + pose so far
    Mq_ptr, Vq_ptr, Mt_ptr, Vt_ptr,     # Adam moments
    P, lr,
    BLOCK: tl.constexpr,
    beta1: tl.constexpr = 0.9,
    beta2: tl.constexpr = 0.999,
    eps:   tl.constexpr = 1e-8,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    qo = offs * 4
    to = offs * 3

    Vab  = tl.load(V_ptr + offs, mask=m)
    norm = tl.load(Norm_ptr + offs, mask=m)

    # ---- torch.sub / div / mul / div  ---------------------------------------------------
    denom = norm - Vab
    score = tl.fdiv(Vab, denom, ieee_rounding=True)
    d2    = denom * denom
    scale = tl.fdiv(norm, d2, ieee_rounding=True)

    # ---- torch.gt + three where's (best score, best q, best t) --------------------------
    qr = tl.load(Q_ptr + qo + 0, mask=m); qi = tl.load(Q_ptr + qo + 1, mask=m)
    qj = tl.load(Q_ptr + qo + 2, mask=m); qk = tl.load(Q_ptr + qo + 3, mask=m)
    tx = tl.load(T_ptr + to + 0, mask=m); ty = tl.load(T_ptr + to + 1, mask=m)
    tz = tl.load(T_ptr + to + 2, mask=m)

    bst = tl.load(Best_ptr + offs, mask=m)
    better = score > bst
    tl.store(Best_ptr + offs, tl.where(better, score, bst), mask=m)
    tl.store(Bq_ptr + qo + 0, tl.where(better, qr, tl.load(Bq_ptr + qo + 0, mask=m)), mask=m)
    tl.store(Bq_ptr + qo + 1, tl.where(better, qi, tl.load(Bq_ptr + qo + 1, mask=m)), mask=m)
    tl.store(Bq_ptr + qo + 2, tl.where(better, qj, tl.load(Bq_ptr + qo + 2, mask=m)), mask=m)
    tl.store(Bq_ptr + qo + 3, tl.where(better, qk, tl.load(Bq_ptr + qo + 3, mask=m)), mask=m)
    tl.store(Bt_ptr + to + 0, tl.where(better, tx, tl.load(Bt_ptr + to + 0, mask=m)), mask=m)
    tl.store(Bt_ptr + to + 1, tl.where(better, ty, tl.load(Bt_ptr + to + 1, mask=m)), mask=m)
    tl.store(Bt_ptr + to + 2, tl.where(better, tz, tl.load(Bt_ptr + to + 2, mask=m)), mask=m)

    # ---- the two mul + neg_ pairs: gq = -(dQ*scale), gt = -(dT*scale) --------------------
    dq0 = -(tl.load(dQ_ptr + qo + 0, mask=m) * scale)
    dq1 = -(tl.load(dQ_ptr + qo + 1, mask=m) * scale)
    dq2 = -(tl.load(dQ_ptr + qo + 2, mask=m) * scale)
    dq3 = -(tl.load(dQ_ptr + qo + 3, mask=m) * scale)
    dt0 = -(tl.load(dT_ptr + to + 0, mask=m) * scale)
    dt1 = -(tl.load(dT_ptr + to + 1, mask=m) * scale)
    dt2 = -(tl.load(dT_ptr + to + 2, mask=m) * scale)

    # ---- _adam_qt with PROJECT=True, verbatim -------------------------------------------
    radial = dq0*qr + dq1*qi + dq2*qj + dq3*qk
    dq0 = dq0 - qr * radial
    dq1 = dq1 - qi * radial
    dq2 = dq2 - qj * radial
    dq3 = dq3 - qk * radial

    mq0 = tl.load(Mq_ptr + qo + 0, mask=m); mq1 = tl.load(Mq_ptr + qo + 1, mask=m)
    mq2 = tl.load(Mq_ptr + qo + 2, mask=m); mq3 = tl.load(Mq_ptr + qo + 3, mask=m)
    vq0 = tl.load(Vq_ptr + qo + 0, mask=m); vq1 = tl.load(Vq_ptr + qo + 1, mask=m)
    vq2 = tl.load(Vq_ptr + qo + 2, mask=m); vq3 = tl.load(Vq_ptr + qo + 3, mask=m)
    mt0 = tl.load(Mt_ptr + to + 0, mask=m); mt1 = tl.load(Mt_ptr + to + 1, mask=m)
    mt2 = tl.load(Mt_ptr + to + 2, mask=m)
    vt0 = tl.load(Vt_ptr + to + 0, mask=m); vt1 = tl.load(Vt_ptr + to + 1, mask=m)
    vt2 = tl.load(Vt_ptr + to + 2, mask=m)

    mq0 = beta1*mq0 + (1-beta1)*dq0;  vq0 = beta2*vq0 + (1-beta2)*dq0*dq0
    mq1 = beta1*mq1 + (1-beta1)*dq1;  vq1 = beta2*vq1 + (1-beta2)*dq1*dq1
    mq2 = beta1*mq2 + (1-beta1)*dq2;  vq2 = beta2*vq2 + (1-beta2)*dq2*dq2
    mq3 = beta1*mq3 + (1-beta1)*dq3;  vq3 = beta2*vq3 + (1-beta2)*dq3*dq3

    qr = qr - lr * mq0 / tl.sqrt(vq0 + eps)
    qi = qi - lr * mq1 / tl.sqrt(vq1 + eps)
    qj = qj - lr * mq2 / tl.sqrt(vq2 + eps)
    qk = qk - lr * mq3 / tl.sqrt(vq3 + eps)

    mt0 = beta1*mt0 + (1-beta1)*dt0; vt0 = beta2*vt0 + (1-beta2)*dt0*dt0
    mt1 = beta1*mt1 + (1-beta1)*dt1; vt1 = beta2*vt1 + (1-beta2)*dt1*dt1
    mt2 = beta1*mt2 + (1-beta1)*dt2; vt2 = beta2*vt2 + (1-beta2)*dt2*dt2

    tx = tx - lr * mt0 / tl.sqrt(vt0 + eps)
    ty = ty - lr * mt1 / tl.sqrt(vt1 + eps)
    tz = tz - lr * mt2 / tl.sqrt(vt2 + eps)

    inv_norm = 1.0 / tl.sqrt(qr*qr + qi*qi + qj*qj + qk*qk)
    qr *= inv_norm; qi *= inv_norm; qj *= inv_norm; qk *= inv_norm

    tl.store(Q_ptr + qo + 0, qr, mask=m); tl.store(Q_ptr + qo + 1, qi, mask=m)
    tl.store(Q_ptr + qo + 2, qj, mask=m); tl.store(Q_ptr + qo + 3, qk, mask=m)
    tl.store(Mq_ptr + qo + 0, mq0, mask=m); tl.store(Mq_ptr + qo + 1, mq1, mask=m)
    tl.store(Mq_ptr + qo + 2, mq2, mask=m); tl.store(Mq_ptr + qo + 3, mq3, mask=m)
    tl.store(Vq_ptr + qo + 0, vq0, mask=m); tl.store(Vq_ptr + qo + 1, vq1, mask=m)
    tl.store(Vq_ptr + qo + 2, vq2, mask=m); tl.store(Vq_ptr + qo + 3, vq3, mask=m)
    tl.store(T_ptr + to + 0, tx, mask=m); tl.store(T_ptr + to + 1, ty, mask=m)
    tl.store(T_ptr + to + 2, tz, mask=m)
    tl.store(Mt_ptr + to + 0, mt0, mask=m); tl.store(Mt_ptr + to + 1, mt1, mask=m)
    tl.store(Mt_ptr + to + 2, mt2, mask=m)
    tl.store(Vt_ptr + to + 0, vt0, mask=m); tl.store(Vt_ptr + to + 1, vt1, mask=m)
    tl.store(Vt_ptr + to + 2, vt2, mask=m)


def tanimoto_best_adam(VAB, dQ, dT, norm, q, t, best, bq, bt, mq, vq, mt, vt, lr,
                       BLOCK: int = 256):
    """Single-channel Tanimoto + best-pose tracking + tangent-projected Adam, in one launch.

    Drop-in for ``_GraphedFineSurf._tanimoto_adam_tail``'s twelve torch ops followed by
    ``fused_adam_qt_with_tangent_proj``. Everything is updated in place and nothing is
    allocated, so it captures into a CUDA graph on the same terms the ops it replaces did.
    """
    P = q.shape[0]
    _tanimoto_best_adam_kernel[(triton.cdiv(P, BLOCK),)](
        VAB, dQ.view(-1), dT.view(-1), norm,
        q.view(-1), t.view(-1),
        best, bq.view(-1), bt.view(-1),
        mq.view(-1), vq.view(-1), mt.view(-1), vt.view(-1),
        P, float(lr), BLOCK=BLOCK,
    )
