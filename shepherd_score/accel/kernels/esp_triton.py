# shepherd_score/accel/kernels/esp_triton.py
# A fused (forward + backward) ESP-weighted Gaussian-Tanimoto kernel using Triton.
# Extends the base volumetric kernel to include electrostatic potential weighting.
#
from __future__ import annotations

import math
import triton
import triton.language as tl
import torch

from .shape_triton import _OVERLAP_CONFIGS
from ...score.constants import COULOMB_SCALING, LAM_SCALING


# Self-tunes per (N_pad, M_pad) on the actual device -- no GPU-specific hardcoding.
# cache_results=True persists the choice to disk so the sweep is once-per-machine.
@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _gauss_overlap_esp_se3_tiled(
    A_ptr, B_ptr,                 # coordinates: flat (B * N_pad * 3), (B * M_pad * 3)
    CA_ptr, CB_ptr,               # charges: flat (B * N_pad), (B * M_pad)
    Q_ptr, T_ptr,                 # (B * 4), (B * 3)
    Nreal_ptr, Mreal_ptr,         # (B,)
    BATCH, M_pad, N_pad,          # ints
    half_alpha, k_const,          # Gaussian parameters
    inv_lam,                      # 1/lam for ESP weighting
    S_ptr, dQ_ptr, dT_ptr,        # outputs (S: (B,), dQ: (B*4), dT: (B*3))
    BLOCK: tl.constexpr,          # single tile edge (e.g. 64)
    NEED_GRAD: tl.constexpr
):
    """
    ESP-weighted Gaussian overlap kernel with SE(3) gradients.

    The overlap formula is:
    V = sum_ij( k_const * exp(-alpha/2 * R_ij^2) * exp(-(C_i - C_j)^2 / lam) )

    Where R_ij is spatial distance, C_i/C_j are ESP values at points i/j.

    The charge weighting exp(-(C_i-C_j)^2/lam) is independent of SE(3) params,
    so gradients have the same structure as the base kernel, just scaled by
    the charge weight factor.
    """
    # -------- which alignment (one CTA per pair) --------
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)
    realM = tl.load(Mreal_ptr + pid)

    # -------- base pointers for this pair ---------------
    A_ptr  = A_ptr  + pid * N_pad * 3
    B_ptr  = B_ptr  + pid * M_pad * 3
    CA_ptr = CA_ptr + pid * N_pad
    CB_ptr = CB_ptr + pid * M_pad
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
    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN

        # load A tile coordinates (x,y,z) into registers
        a_idx = tl.where(mask_n, offs_n, 0)
        ax = tl.load(A_ptr + a_idx * 3 + 0, mask=mask_n, other=0.0)
        ay = tl.load(A_ptr + a_idx * 3 + 1, mask=mask_n, other=0.0)
        az = tl.load(A_ptr + a_idx * 3 + 2, mask=mask_n, other=0.0)

        # load A tile charges
        ca = tl.load(CA_ptr + a_idx, mask=mask_n, other=0.0)

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM

            b_idx = tl.where(mask_m, offs_m, 0)
            bx0 = tl.load(B_ptr + b_idx * 3 + 0, mask=mask_m, other=0.0)
            by0 = tl.load(B_ptr + b_idx * 3 + 1, mask=mask_m, other=0.0)
            bz0 = tl.load(B_ptr + b_idx * 3 + 2, mask=mask_m, other=0.0)

            # load B tile charges
            cb = tl.load(CB_ptr + b_idx, mask=mask_m, other=0.0)

            # rotate + translate B tile
            bx = r00*bx0 + r01*by0 + r02*bz0 + tx
            by = r10*bx0 + r11*by0 + r12*bz0 + ty
            bz = r20*bx0 + r21*by0 + r22*bz0 + tz

            # broadcast differences (BLOCK x BLOCK)
            dx = ax[:, None] - bx[None, :]
            dy = ay[:, None] - by[None, :]
            dz = az[:, None] - bz[None, :]
            r2 = dx*dx + dy*dy + dz*dz

            # charge difference squared (BLOCK x BLOCK)
            dc = ca[:, None] - cb[None, :]
            c2 = dc * dc

            # Gaussian spatial term
            g_spatial = tl.exp2((-half_alpha * r2) * inv_ln2) * k_const

            # ESP charge weighting term: exp(-c2 / lam) = exp2(-c2 * inv_lam / ln2)
            g_charge = tl.exp2((-c2 * inv_lam) * inv_ln2)

            # Combined overlap
            g = g_spatial * g_charge

            pair_mask = mask_n[:, None] & mask_m[None, :]
            g = tl.where(pair_mask, g, 0.0)

            # overlap accumulation
            Vab_acc += tl.sum(g)

            if NEED_GRAD:
                # Gradient coefficient includes both spatial and charge terms
                # d/dR(g) = d/dR(g_spatial * g_charge) = g_charge * d/dR(g_spatial)
                # Since g_charge doesn't depend on R
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

                # mask again
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


def overlap_score_grad_esp_se3_batch(
    A, B,
    charges_A, charges_B,
    q, t, *,
    alpha: float = 0.81,
    lam: float = 0.3,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    NEED_GRAD: bool = True,
    BLOCK: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """
    ESP-weighted overlap with SE(3) gradients.

    One CTA per alignment (pair). Internal tile loops over A, B.

    Shapes:
      A : (K, N_pad, 3) - coordinates of molecule A (reference)
      B : (K, M_pad, 3) - coordinates of molecule B (fit)
      charges_A : (K, N_pad) - ESP values at A points
      charges_B : (K, M_pad) - ESP values at B points
      q : (K, 4) - quaternions
      t : (K, 3) - translations

    Returns:
      VAB : (K,) - ESP-weighted overlap scores
      dQ : (K, 4) - quaternion gradients
      dT : (K, 3) - translation gradients
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
    inv_lam    = 1.0 / lam

    out_S  = torch.zeros(K, device=device, dtype=dtype)
    out_dQ = torch.zeros_like(q)
    out_dT = torch.zeros_like(t)

    grid = (K,)    # 1-D launch: one CTA per alignment

    # BLOCK + num_warps chosen by triton.autotune per (N_pad, M_pad) on the actual
    # device; legacy BLOCK/num_warps/num_stages kwargs accepted but ignored.
    _gauss_overlap_esp_se3_tiled[grid](
        A.contiguous().view(-1),
        B.contiguous().view(-1),
        charges_A.contiguous().view(-1),
        charges_B.contiguous().view(-1),
        q.contiguous().view(-1),
        t.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad,
        half_alpha, k_const, inv_lam,
        out_S, out_dQ.view(-1), out_dT.view(-1),
        NEED_GRAD=NEED_GRAD,
    )
    return out_S, out_dQ, out_dT


# ============================================================================
#  ShaEP ESP surface-comparison kernel (esp_combo mode), VALUE-ONLY.
#
#  For each real field point i of the "observer" molecule, compute the Coulomb
#  ESP induced there by the OTHER molecule's atoms, mask the point out if it
#  falls inside that molecule's vdW+probe volume, and accumulate a Gaussian of
#  the ESP difference:
#      esp = sum_i  keep_i * exp( -(point_esp_i - sum_m q_m/d_im)^2 / lam )
#  This is the fused replacement for the (B, N_surf, M_atoms) torch.cdist that
#  the eager `_batch_esp_comparison` materialized twice per fine step. It is
#  value-only: esp_combo steers the pose with the SHAPE gradient (the ESP term
#  is scored / used for seed selection), so no dO/dq tail is needed.
#
#  The caller passes points + atoms already in the world frame (the driver
#  applies the SE(3) transform to whichever cloud is moving), so the kernel
#  needs no quaternion -- it is a pure pairwise reduction.
# ============================================================================
@triton.autotune(configs=_OVERLAP_CONFIGS, key=['N_pad', 'M_pad'], cache_results=True)
@triton.jit
def _esp_comparison_tiled(
    P_ptr, A_ptr,             # field points (B*N_pad*3), source atoms (B*M_pad*3)
    Q_ptr, R_ptr,             # atom charges (B*M_pad), atom vdW radii (B*M_pad)
    PE_ptr,                   # precomputed ESP at field points (B*N_pad)
    Nreal_ptr, Mreal_ptr,     # (B,) real field-point / atom counts
    BATCH, M_pad, N_pad,      # ints
    inv_lam, coulomb, probe,  # scalars
    S_ptr,                    # output ESP comparison (B,)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    realN = tl.load(Nreal_ptr + pid)
    realM = tl.load(Mreal_ptr + pid)

    P_ptr  = P_ptr  + pid * N_pad * 3
    A_ptr  = A_ptr  + pid * M_pad * 3
    Q_ptr  = Q_ptr  + pid * M_pad
    R_ptr  = R_ptr  + pid * M_pad
    PE_ptr = PE_ptr + pid * N_pad
    S_ptr  = S_ptr  + pid

    inv_ln2 = 1.4426950408889634
    total = 0.0

    # outer loop over field-point tiles; inner loop over atom tiles
    for n0 in range(0, N_pad, BLOCK):
        offs_n = n0 + tl.arange(0, BLOCK)
        mask_n = offs_n < realN
        n_idx = tl.where(mask_n, offs_n, 0)
        px = tl.load(P_ptr + n_idx * 3 + 0, mask=mask_n, other=0.0)
        py = tl.load(P_ptr + n_idx * 3 + 1, mask=mask_n, other=0.0)
        pz = tl.load(P_ptr + n_idx * 3 + 2, mask=mask_n, other=0.0)
        pe = tl.load(PE_ptr + n_idx, mask=mask_n, other=0.0)

        esp_acc = tl.zeros([BLOCK], dtype=tl.float32)   # ESP at each field point
        block_cnt = tl.zeros([BLOCK], dtype=tl.float32)  # # of blocking atoms

        for m0 in range(0, M_pad, BLOCK):
            offs_m = m0 + tl.arange(0, BLOCK)
            mask_m = offs_m < realM
            m_idx = tl.where(mask_m, offs_m, 0)
            ax = tl.load(A_ptr + m_idx * 3 + 0, mask=mask_m, other=0.0)
            ay = tl.load(A_ptr + m_idx * 3 + 1, mask=mask_m, other=0.0)
            az = tl.load(A_ptr + m_idx * 3 + 2, mask=mask_m, other=0.0)
            qc = tl.load(Q_ptr + m_idx, mask=mask_m, other=0.0)
            rad = tl.load(R_ptr + m_idx, mask=mask_m, other=0.0)

            dx = px[:, None] - ax[None, :]
            dy = py[:, None] - ay[None, :]
            dz = pz[:, None] - az[None, :]
            d = tl.sqrt(dx * dx + dy * dy + dz * dz)
            d = tl.where(d < 1e-6, 1e-6, d)

            pair_m = mask_m[None, :]
            # ESP contribution sum_m q_m / d  (padded atoms masked out -> 0)
            esp_acc += tl.sum(tl.where(pair_m, qc[None, :] / d, 0.0), axis=1)
            # a real atom within (radius + probe) blocks this point
            blocked = (d < (rad[None, :] + probe)) & pair_m
            block_cnt += tl.sum(tl.where(blocked, 1.0, 0.0), axis=1)

        esp_acc = esp_acc * coulomb
        diff = pe - esp_acc
        keep = mask_n & (block_cnt == 0.0)
        val = tl.where(keep, tl.exp2((-(diff * diff) * inv_lam) * inv_ln2), 0.0)
        total += tl.sum(val)

    tl.store(S_ptr, total)


def esp_comparison_batch(
    points, atoms, charges, point_esp, radii, *,
    N_real: torch.Tensor | None = None,
    M_real: torch.Tensor | None = None,
    probe_radius: float = 1.0,
    lam: float = 0.001,
    BLOCK: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """Fused ShaEP ESP surface comparison (value-only). One CTA per pair.

    Shapes (all world-frame, padded):
      points    : (K, N_pad, 3) observer field points (where ESP is compared)
      atoms     : (K, M_pad, 3) source-molecule atom coordinates (with H)
      charges   : (K, M_pad)    source-molecule partial charges
      point_esp : (K, N_pad)    precomputed ESP at ``points``
      radii     : (K, M_pad)    source-molecule vdW radii (for the volume mask)
      N_real/M_real : (K,) int   true field-point / atom counts (padding masks)

    Returns ``esp`` (K,): the masked Gaussian-of-ESP-difference sum, matching the
    eager ``_batch_esp_comparison`` / ``_esp_comparison`` torch reference. ``lam``
    is the raw weighting parameter; ``LAM_SCALING`` is applied internally.
    """
    K, N_pad, _ = points.shape
    _, M_pad, _ = atoms.shape
    device = points.device
    dtype = points.dtype

    if N_real is None:
        N_real = torch.full((K,), N_pad, device=device, dtype=torch.int32)
    else:
        N_real = N_real.to(device=device, dtype=torch.int32, copy=False)
    if M_real is None:
        M_real = torch.full((K,), M_pad, device=device, dtype=torch.int32)
    else:
        M_real = M_real.to(device=device, dtype=torch.int32, copy=False)

    inv_lam = 1.0 / (LAM_SCALING * lam)
    out_S = torch.zeros(K, device=device, dtype=dtype)

    grid = (K,)
    _esp_comparison_tiled[grid](
        points.contiguous().view(-1),
        atoms.contiguous().view(-1),
        charges.contiguous().view(-1),
        radii.contiguous().view(-1),
        point_esp.contiguous().view(-1),
        N_real.contiguous(),
        M_real.contiguous(),
        K, M_pad, N_pad,
        inv_lam, float(COULOMB_SCALING), float(probe_radius),
        out_S,
    )
    return out_S


@torch.no_grad()
def _batch_self_overlap_esp(
    P_pad: torch.Tensor,
    charges_pad: torch.Tensor,
    N_real: torch.Tensor,
    alpha: float = 0.81,
    lam: float = 0.3
) -> torch.Tensor:
    """
    Batched self-overlap VPP(P,P) for ESP-weighted Gaussian overlap.

    Parameters
    ----------
    P_pad : (K, N_pad, 3) - padded coordinates
    charges_pad : (K, N_pad) - padded ESP values
    N_real : (K,) int32 - true point counts

    Returns
    -------
    V : (K,) - self-overlap values
    """
    K, N_pad, _ = P_pad.shape
    q_id = torch.tensor([1., 0., 0., 0.], device=P_pad.device, dtype=P_pad.dtype).expand(K, 4)
    t_0  = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_esp_se3_batch(
        P_pad, P_pad,
        charges_pad, charges_pad,
        q_id, t_0,
        alpha=alpha,
        lam=lam,
        N_real=N_real, M_real=N_real,
        NEED_GRAD=False)
    return V
