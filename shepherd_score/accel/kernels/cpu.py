"""CPU (numba) fallbacks for the Triton overlap+grad and Adam kernels.

These let the batched coarse-to-fine driver (``fast_se3.coarse_fine_align_many`` and
``_align_batch_vol``) run on a CPU-only box with **no Triton/CUDA**. ``fast_se3`` imports
these when ``shepherd_score.accel.kernels.shape_triton`` (which hard-requires triton)
fails to import.

Design / correctness
--------------------
``overlap_score_grad_se3_batch`` replicates ``_gauss_overlap_se3_tiled`` (the Triton
kernel) **operation-for-operation**: identical quaternion convention ``q=(w,x,y,z)``,
the standard unit-quaternion rotation matrix, ``B`` (fit) rotated against ``A`` (ref),
``Vab = K·Σ exp(-α/2·r²)`` with ``K = π^1.5/(2α)^1.5``, and the exact analytical
``dVab/dq`` (4) and ``dVab/dt`` (3). It is therefore numerically **exact** (it computes
the true overlap+gradient), though not bit-identical to the GPU (``math.exp`` ≠ Triton's
``exp2``; see SPEED_EXPERIMENTS_CPU.md "Byte-identical exploration"). The heavy O(K·N·M)
work is a fused single-pass ``@njit(parallel=True)`` kernel (one row per pose, prange over
poses); the cheap O(K) quaternion/Adam bookkeeping stays in torch.

``fused_adam_qt_with_tangent_proj`` replicates ``_adam_qt_with_tangent_proj`` exactly
(tangent projection ``dQ-q(dQ·q)``, Adam β1=0.9 β2=0.999 eps=1e-8 *inside* the sqrt, no
bias correction, quaternion renormalisation), in-place, pure torch.

Single-core throughput is set by ``NUMBA_NUM_THREADS`` (=1 pins one core); the default
uses all cores for aggregate throughput.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from numba import njit, prange

from ...score.constants import COULOMB_SCALING, LAM_SCALING

_K_PI = math.pi ** 1.5


@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_kernel(A, B, q, t, Nr, Mr, alpha, need_grad):
    """Fused overlap value + SE(3) gradient, one pose per prange iteration.

    A (K,N,3) ref, B (K,M,3) fit, q (K,4)=(w,x,y,z), t (K,3). Nr/Mr (K,) int real counts.
    Returns V (K,), dQ (K,4)=dVab/dq, dT (K,3)=dVab/dt. fp64 accumulation.
    Mirrors _gauss_overlap_se3_tiled in gaussian_overlap_triton.py.
    """
    K = A.shape[0]
    Kc = _K_PI / (2.0 * alpha) ** 1.5
    a2 = alpha / 2.0
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        # standard unit-quaternion -> rotation matrix (assumes |q|=1, as Triton does)
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, m, 0]; by0 = B[k, m, 1]; bz0 = B[k, m, 2]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):
                dx = A[k, n, 0] - bx; dy = A[k, n, 1] - by; dz = A[k, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                g = Kc * math.exp(-a2 * r2)
                Vacc += g
                if need_grad:
                    c = alpha * g            # coeff = 2*half_alpha*g = alpha*g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                # dVab/dq, body-frame coords (bx0,by0,bz0); identical to the Triton tail
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        V[k] = Vacc
        dT[k, 0] = dTx; dT[k, 1] = dTy; dT[k, 2] = dTz
        dQ[k, 0] = dQw; dQ[k, 1] = dQx; dQ[k, 2] = dQy; dQ[k, 3] = dQz
    return V, dQ, dT


def overlap_score_grad_se3_batch(A, B, q, t, *, alpha: float = 0.81,
                                 N_real=None, M_real=None, NEED_GRAD: bool = True,
                                 BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``overlap_score_grad_se3_batch``. Returns (V, dQ, dT)
    as torch tensors on A.device with A.dtype. Extra kwargs ignored (GPU-only knobs)."""
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    V, dQ, dT = _overlap_grad_kernel(An, Bn, qn, tn, Nr, Mr, float(alpha), bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


@torch.no_grad()
def fused_adam_qt_with_tangent_proj(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr,
                                    beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam with quaternion tangent projection + renorm. Exact replica of the
    Triton ``_adam_qt_with_tangent_proj`` (no bias correction; eps inside the sqrt)."""
    radial = (dQ * q).sum(dim=1, keepdim=True)
    dq = dQ - q * radial
    m_q.mul_(beta1).add_(dq, alpha=1.0 - beta1)
    v_q.mul_(beta2).addcmul_(dq, dq, value=1.0 - beta2)
    q.add_(-lr * m_q / (v_q + eps).sqrt())
    m_t.mul_(beta1).add_(dT, alpha=1.0 - beta1)
    v_t.mul_(beta2).addcmul_(dT, dT, value=1.0 - beta2)
    t.add_(-lr * m_t / (v_t + eps).sqrt())
    q.div_(q.norm(dim=1, keepdim=True))


@torch.no_grad()
def fused_adam_qt(q, t, dQ, dT, m_q, v_q, m_t, v_t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam (no tangent projection); CPU replica of Triton ``_adam_qt``."""
    m_q.mul_(beta1).add_(dQ, alpha=1.0 - beta1)
    v_q.mul_(beta2).addcmul_(dQ, dQ, value=1.0 - beta2)
    q.add_(-lr * m_q / (v_q + eps).sqrt())
    m_t.mul_(beta1).add_(dT, alpha=1.0 - beta1)
    v_t.mul_(beta2).addcmul_(dT, dT, value=1.0 - beta2)
    t.add_(-lr * m_t / (v_t + eps).sqrt())
    q.div_(q.norm(dim=1, keepdim=True))


@torch.no_grad()
def _batch_self_overlap(P_pad: torch.Tensor, N_real: torch.Tensor, alpha: float = 0.81):
    """Self-overlap VAA/VBB via the identity pose; CPU replica."""
    K = P_pad.shape[0]
    q_id = torch.zeros(K, 4, device=P_pad.device, dtype=P_pad.dtype); q_id[:, 0] = 1.0
    t_0 = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_se3_batch(P_pad, P_pad, q_id, t_0, alpha=alpha,
                                           N_real=N_real, M_real=N_real, NEED_GRAD=False)
    return V


def fused_surf_step_batch(*args, **kwargs):
    """The Triton fused surf step has no CPU analogue; the CPU driver uses the eager
    fine loop (guarded by torch.cuda.is_available()), so this should never be called."""
    raise NotImplementedError(
        "fused_surf_step_batch is CUDA-only; the CPU path uses the eager fine loop.")


# ===========================================================================
#  ESP-weighted overlap (esp / vol_esp): shape kernel x charge weight.
#  Replicates gaussian_overlap_esp_triton._gauss_overlap_esp_se3_tiled:
#  V = K * sum exp(-a/2 r^2) * exp(-(Ci-Cj)^2/lam), gradient = shape grad scaled
#  by the (SE(3)-invariant) charge weight (folded into g). The two exps are fused
#  into one exp(-a/2 r^2 - c2/lam) (algebraically identical; the L5 lever).
# ===========================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _overlap_grad_esp_kernel(A, B, CA, CB, q, t, Nr, Mr, alpha, inv_lam, need_grad):
    K = A.shape[0]
    Kc = _K_PI / (2.0 * alpha) ** 1.5
    a2 = alpha / 2.0
    V = np.zeros(K, dtype=np.float64)
    dQ = np.zeros((K, 4), dtype=np.float64)
    dT = np.zeros((K, 3), dtype=np.float64)
    for k in prange(K):
        qr = q[k, 0]; qi = q[k, 1]; qj = q[k, 2]; qk_ = q[k, 3]
        tx = t[k, 0]; ty = t[k, 1]; tz = t[k, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[k]; m_real = Mr[k]
        Vacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):
            bx0 = B[k, m, 0]; by0 = B[k, m, 1]; bz0 = B[k, m, 2]; cbm = CB[k, m]
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):
                dx = A[k, n, 0] - bx; dy = A[k, n, 1] - by; dz = A[k, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                dc = CA[k, n] - cbm
                g = Kc * math.exp(-a2 * r2 - dc * dc * inv_lam)
                Vacc += g
                if need_grad:
                    c = alpha * g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        V[k] = Vacc
        dT[k, 0] = dTx; dT[k, 1] = dTy; dT[k, 2] = dTz
        dQ[k, 0] = dQw; dQ[k, 1] = dQx; dQ[k, 2] = dQy; dQ[k, 3] = dQz
    return V, dQ, dT


def overlap_score_grad_esp_se3_batch(A, B, charges_A, charges_B, q, t, *,
                                     alpha: float = 0.81, lam: float = 0.3,
                                     N_real=None, M_real=None, NEED_GRAD: bool = True,
                                     BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``overlap_score_grad_esp_se3_batch``."""
    K, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    CAn = np.ascontiguousarray(charges_A.detach().cpu().numpy())
    CBn = np.ascontiguousarray(charges_B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    V, dQ, dT = _overlap_grad_esp_kernel(An, Bn, CAn, CBn, qn, tn, Nr, Mr,
                                         float(alpha), 1.0 / float(lam), bool(NEED_GRAD))
    return (torch.as_tensor(V, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


# ===========================================================================
#  ShaEP ESP surface comparison (esp_combo), VALUE-ONLY. CPU twin of the Triton
#  esp_triton._esp_comparison_tiled, op-for-op: for each real field point i,
#  Coulomb ESP from the other molecule's atoms, vdW+probe volume mask, Gaussian
#  of the ESP difference, summed over points. Replaces the (K,N_surf,M_atoms)
#  torch.cdist the eager `_batch_esp_comparison` materialized. fp64 accumulation;
#  math.exp vs the Triton exp2 is the only intended divergence. No gradient
#  (esp_combo steers the pose with the shape gradient).
# ===========================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _esp_comparison_kernel(P, A, Q, R, PE, Nr, Mr, inv_lam, coulomb, probe):
    K = P.shape[0]
    S = np.zeros(K, dtype=np.float64)
    for k in prange(K):
        n_real = Nr[k]; m_real = Mr[k]
        total = 0.0
        for i in range(n_real):
            px = P[k, i, 0]; py = P[k, i, 1]; pz = P[k, i, 2]
            pe = PE[k, i]
            esp = 0.0
            blocked = False
            for m in range(m_real):
                dx = px - A[k, m, 0]; dy = py - A[k, m, 1]; dz = pz - A[k, m, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d < 1e-6:
                    d = 1e-6
                esp += Q[k, m] / d
                if d < R[k, m] + probe:
                    blocked = True
            esp *= coulomb
            if not blocked:
                diff = pe - esp
                total += math.exp(-(diff * diff) * inv_lam)
        S[k] = total
    return S


def esp_comparison_batch(points, atoms, charges, point_esp, radii, *,
                         N_real=None, M_real=None, probe_radius: float = 1.0,
                         lam: float = 0.001, BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton ``esp_comparison_batch`` (value-only ShaEP ESP
    term). Returns ``esp`` (K,) as a torch tensor on ``points.device``. ``lam`` is
    the raw weighting parameter; ``LAM_SCALING`` is applied internally."""
    K, N_pad, _ = points.shape
    _, M_pad, _ = atoms.shape
    dev, dt = points.device, points.dtype
    Pn = np.ascontiguousarray(points.detach().cpu().numpy())
    An = np.ascontiguousarray(atoms.detach().cpu().numpy())
    Qn = np.ascontiguousarray(charges.detach().cpu().numpy())
    Rn = np.ascontiguousarray(radii.detach().cpu().numpy())
    PEn = np.ascontiguousarray(point_esp.detach().cpu().numpy())
    Nr = (np.full(K, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(K, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    inv_lam = 1.0 / (LAM_SCALING * float(lam))
    S = _esp_comparison_kernel(Pn, An, Qn, Rn, PEn, Nr, Mr,
                               inv_lam, float(COULOMB_SCALING), float(probe_radius))
    return torch.as_tensor(S, device=dev, dtype=dt)


# ===========================================================================
#  Pharmacophore overlap value+grad (pharm). Replicates pharmacophore_grad_triton
#  ._pharm_score_grad_kernel op-for-op: typed Gaussians (per-type alpha/K/cat),
#  directional weighting w (cat 1 = (clamp(D,0,1)+2)/3, cat 2 = (|D|+2)/3, else 1),
#  type-match + non-dummy(cat!=3) masking; returns O, grad_R (3x3), grad_t. FIT=i
#  (rotated by R,t), REF=j (fixed); type/alpha/K/cat from FIT.
# ===========================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _pharm_grad_kernel(RaA, FaB, RvA, FvB, RtA, FtB, R, t, alphas, Ks, cats, Nr, Mr, need_grad):
    P = RaA.shape[0]
    O = np.zeros(P, dtype=np.float64)
    gR = np.zeros((P, 3, 3), dtype=np.float64)
    gt = np.zeros((P, 3), dtype=np.float64)
    for p in prange(P):
        r00 = R[p, 0, 0]; r01 = R[p, 0, 1]; r02 = R[p, 0, 2]
        r10 = R[p, 1, 0]; r11 = R[p, 1, 1]; r12 = R[p, 1, 2]
        r20 = R[p, 2, 0]; r21 = R[p, 2, 1]; r22 = R[p, 2, 2]
        tx = t[p, 0]; ty = t[p, 1]; tz = t[p, 2]
        n_real = Nr[p]; m_real = Mr[p]
        Oacc = 0.0
        gtx = 0.0; gty = 0.0; gtz = 0.0
        g00 = 0.0; g01 = 0.0; g02 = 0.0; g10 = 0.0; g11 = 0.0; g12 = 0.0; g20 = 0.0; g21 = 0.0; g22 = 0.0
        for i in range(m_real):  # FIT
            fax0 = FaB[p, i, 0]; fay0 = FaB[p, i, 1]; faz0 = FaB[p, i, 2]
            fvx = FvB[p, i, 0]; fvy = FvB[p, i, 1]; fvz = FvB[p, i, 2]
            fnorm = math.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
            finv = 1.0 / (fnorm if fnorm > 1e-12 else 1e-12)
            fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv
            ityp = FtB[p, i]
            ialpha = alphas[ityp]; iK = Ks[ityp]; icat = cats[ityp]
            if icat == 3:
                continue
            fatx = r00 * fax0 + r01 * fay0 + r02 * faz0 + tx
            faty = r10 * fax0 + r11 * fay0 + r12 * faz0 + ty
            fatz = r20 * fax0 + r21 * fay0 + r22 * faz0 + tz
            fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
            fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
            fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn
            for j in range(n_real):  # REF
                if RtA[p, j] != ityp:
                    continue
                rax = RaA[p, j, 0]; ray = RaA[p, j, 1]; raz = RaA[p, j, 2]
                rvx = RvA[p, j, 0]; rvy = RvA[p, j, 1]; rvz = RvA[p, j, 2]
                rnorm = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
                rinv = 1.0 / (rnorm if rnorm > 1e-12 else 1e-12)
                rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
                dx = fatx - rax; dy = faty - ray; dz = fatz - raz
                r2 = dx * dx + dy * dy + dz * dz
                E = math.exp(-ialpha * 0.5 * r2)
                D = fvtx * rvxn + fvty * rvyn + fvtz * rvzn
                if icat == 1:
                    Dcl = 0.0 if D < 0.0 else (1.0 if D > 1.0 else D)
                    w = (Dcl + 2.0) / 3.0
                elif icat == 2:
                    w = (abs(D) + 2.0) / 3.0
                else:
                    w = 1.0
                KwE = iK * w * E
                Oacc += KwE
                if need_grad:
                    aKwE = -ialpha * KwE
                    gtx += aKwE * dx; gty += aKwE * dy; gtz += aKwE * dz
                    g00 += aKwE * dx * fax0; g01 += aKwE * dx * fay0; g02 += aKwE * dx * faz0
                    g10 += aKwE * dy * fax0; g11 += aKwE * dy * fay0; g12 += aKwE * dy * faz0
                    g20 += aKwE * dz * fax0; g21 += aKwE * dz * fay0; g22 += aKwE * dz * faz0
                    if icat == 1:
                        c = 1.0 if (D > 0.0 and D < 1.0) else 0.0
                    elif icat == 2:
                        c = 1.0 if D > 0.0 else (-1.0 if D < 0.0 else 0.0)
                    else:
                        c = 0.0
                    coeff = (1.0 / 3.0) * iK * E * c
                    g00 += coeff * rvxn * fvxn; g01 += coeff * rvxn * fvyn; g02 += coeff * rvxn * fvzn
                    g10 += coeff * rvyn * fvxn; g11 += coeff * rvyn * fvyn; g12 += coeff * rvyn * fvzn
                    g20 += coeff * rvzn * fvxn; g21 += coeff * rvzn * fvyn; g22 += coeff * rvzn * fvzn
        O[p] = Oacc
        gt[p, 0] = gtx; gt[p, 1] = gty; gt[p, 2] = gtz
        gR[p, 0, 0] = g00; gR[p, 0, 1] = g01; gR[p, 0, 2] = g02
        gR[p, 1, 0] = g10; gR[p, 1, 1] = g11; gR[p, 1, 2] = g12
        gR[p, 2, 0] = g20; gR[p, 2, 1] = g21; gR[p, 2, 2] = g22
    return O, gR, gt


def pharm_score_grad_se3_batch(R, t, ref_types, fit_types, ref_anchors, fit_anchors,
                               ref_vectors, fit_vectors, alphas, Ks, cats, *,
                               N_real=None, M_real=None, NEED_GRAD: bool = True,
                               BLOCK=None, num_warps=1):
    """CPU drop-in for the Triton ``pharm_score_grad_se3_batch``. Returns (O, grad_R, grad_t)."""
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev, dt = ref_anchors.device, ref_anchors.dtype

    def _np(x):
        return np.ascontiguousarray(x.detach().cpu().numpy())
    RaA = _np(ref_anchors); FaB = _np(fit_anchors)
    RvA = _np(ref_vectors); FvB = _np(fit_vectors)
    RtA = ref_types.detach().cpu().numpy().astype(np.int64)
    FtB = fit_types.detach().cpu().numpy().astype(np.int64)
    Rn = _np(R); tn = _np(t)
    al = alphas.detach().cpu().numpy().astype(np.float64)
    Ksn = Ks.detach().cpu().numpy().astype(np.float64)
    cn = cats.detach().cpu().numpy().astype(np.int64)
    Nr = (np.full(P, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(P, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    O, gR, gt = _pharm_grad_kernel(RaA, FaB, RvA, FvB, RtA, FtB, Rn, tn, al, Ksn, cn, Nr, Mr, bool(NEED_GRAD))
    return (torch.as_tensor(O, device=dev, dtype=dt),
            torch.as_tensor(gR, device=dev, dtype=dt),
            torch.as_tensor(gt, device=dev, dtype=dt))


# ===========================================================================
#  DIRECTIONLESS pharmacophore "color" overlap value + QUATERNION gradient
#  (vol_color mode). Same same-type-only typed Gaussian as the pharm kernel, but
#  (a) DIRECTIONLESS (every real type is an isotropic point Gaussian: w=1, no
#  vector machinery, no weight gradient), and (b) emits dV/dq DIRECTLY in-register
#  -- exactly like the shape kernel _overlap_grad_kernel -- so the driver needs no
#  rotation-matrix -> quaternion projection / normalization-Jacobian tail.
#  q is assumed unit (the adam renormalizes it each step), matching the shape kernel.
#  A = ref anchors, B = fit anchors (rotated by R(q),t); At/Bt = ref/fit type idx.
#  dx = A - rot(B): identical sign convention to _overlap_grad_kernel, so the dV/dq
#  tail below is byte-identical to the (validated) shape-kernel tail.
# ===========================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _pharm_color_grad_kernel(A, B, q, t, At, Bt, alphas, Ks, cats, Nr, Mr, need_grad):
    P = A.shape[0]
    O = np.zeros(P, dtype=np.float64)
    dQ = np.zeros((P, 4), dtype=np.float64)
    dT = np.zeros((P, 3), dtype=np.float64)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty = t[p, 1]; tz = t[p, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[p]; m_real = Mr[p]
        Oacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for m in range(m_real):  # FIT
            bx0 = B[p, m, 0]; by0 = B[p, m, 1]; bz0 = B[p, m, 2]
            ityp = Bt[p, m]
            if cats[ityp] == 3:   # dummy / pad -> skip
                continue
            ialpha = alphas[ityp]; iK = Ks[ityp]
            a2 = ialpha * 0.5
            bx = r00 * bx0 + r01 * by0 + r02 * bz0 + tx
            by = r10 * bx0 + r11 * by0 + r12 * bz0 + ty
            bz = r20 * bx0 + r21 * by0 + r22 * bz0 + tz
            fxj = 0.0; fyj = 0.0; fzj = 0.0
            for n in range(n_real):  # REF (same-type only)
                if At[p, n] != ityp:
                    continue
                dx = A[p, n, 0] - bx; dy = A[p, n, 1] - by; dz = A[p, n, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                g = iK * math.exp(-a2 * r2)
                Oacc += g
                if need_grad:
                    c = ialpha * g
                    fxj += c * dx; fyj += c * dy; fzj += c * dz
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                dQw += fxj * (-2.0 * qk_ * by0 + 2.0 * qj * bz0) + fyj * (2.0 * qk_ * bx0 - 2.0 * qi * bz0) + fzj * (-2.0 * qj * bx0 + 2.0 * qi * by0)
                dQx += fxj * (2.0 * qj * by0 + 2.0 * qk_ * bz0) + fyj * (2.0 * qj * bx0 - 4.0 * qi * by0 - 2.0 * qr * bz0) + fzj * (2.0 * qk_ * bx0 + 2.0 * qr * by0 - 4.0 * qi * bz0)
                dQy += fxj * (-4.0 * qj * bx0 + 2.0 * qi * by0 + 2.0 * qr * bz0) + fyj * (2.0 * qi * bx0 + 2.0 * qk_ * bz0) + fzj * (-2.0 * qr * bx0 + 2.0 * qk_ * by0 - 4.0 * qj * bz0)
                dQz += fxj * (-4.0 * qk_ * bx0 - 2.0 * qr * by0 + 2.0 * qi * bz0) + fyj * (2.0 * qr * bx0 - 4.0 * qk_ * by0 + 2.0 * qj * bz0) + fzj * (2.0 * qi * bx0 + 2.0 * qj * by0)
        O[p] = Oacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return O, dQ, dT


def pharm_color_score_grad_se3_batch(A, B, q, t, ref_types, fit_types, alphas, Ks, cats, *,
                                     N_real=None, M_real=None, NEED_GRAD: bool = True,
                                     BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton directionless-color value+quaternion-grad kernel.
    A = ref anchors (P,N,3), B = fit anchors (P,M,3), q=(P,4), t=(P,3); ref/fit_types
    (P,N)/(P,M). Returns (O, dQ, dT) with dQ = dO/dq (quaternion), like the shape kernel."""
    P, N_pad, _ = A.shape
    _, M_pad, _ = B.shape
    dev, dt = A.device, A.dtype
    An = np.ascontiguousarray(A.detach().cpu().numpy())
    Bn = np.ascontiguousarray(B.detach().cpu().numpy())
    qn = np.ascontiguousarray(q.detach().cpu().numpy())
    tn = np.ascontiguousarray(t.detach().cpu().numpy())
    At = ref_types.detach().cpu().numpy().astype(np.int64)
    Bt = fit_types.detach().cpu().numpy().astype(np.int64)
    al = alphas.detach().cpu().numpy().astype(np.float64)
    Ksn = Ks.detach().cpu().numpy().astype(np.float64)
    cn = cats.detach().cpu().numpy().astype(np.int64)
    Nr = (np.full(P, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(P, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    O, dQ, dT = _pharm_color_grad_kernel(An, Bn, qn, tn, At, Bt, al, Ksn, cn, Nr, Mr, bool(NEED_GRAD))
    return (torch.as_tensor(O, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


# ===========================================================================
#  DIRECTIONAL pharmacophore overlap value + QUATERNION gradient (pharm mode).
#  Same typed/directional Gaussian + weight (cat 1/2) as _pharm_grad_kernel, but
#  takes the quaternion q (assumes |q|=1, as the adam renormalizes it) and emits
#  dV/dq DIRECTLY in-register, so the pharm driver drops the
#  rotation->quaternion projection + normalization-Jacobian tail. dV/dq is the
#  projection of grad_R = grad_R_positional + grad_R_weight onto q, computed by
#  reusing the validated shape-kernel dR/dq tail TWICE: once with the positional
#  "force" (sum_j aKwE*(rotfit-ref)) and the body-frame fit ANCHOR, and once with
#  the weight "force" (sum_j coeff*ref_vn) and the body-frame fit VECTOR.
# ===========================================================================
@njit(parallel=True, fastmath=True, cache=True)
def _pharm_grad_dq_kernel(A, B, q, t, At, Bt, RvA, FvB, alphas, Ks, cats, Nr, Mr, need_grad):
    P = A.shape[0]
    O = np.zeros(P, dtype=np.float64)
    dQ = np.zeros((P, 4), dtype=np.float64)
    dT = np.zeros((P, 3), dtype=np.float64)
    for p in prange(P):
        qr = q[p, 0]; qi = q[p, 1]; qj = q[p, 2]; qk_ = q[p, 3]
        tx = t[p, 0]; ty = t[p, 1]; tz = t[p, 2]
        r00 = 1.0 - 2.0 * (qj * qj + qk_ * qk_); r01 = 2.0 * (qi * qj - qk_ * qr); r02 = 2.0 * (qi * qk_ + qj * qr)
        r10 = 2.0 * (qi * qj + qk_ * qr); r11 = 1.0 - 2.0 * (qi * qi + qk_ * qk_); r12 = 2.0 * (qj * qk_ - qi * qr)
        r20 = 2.0 * (qi * qk_ - qj * qr); r21 = 2.0 * (qj * qk_ + qi * qr); r22 = 1.0 - 2.0 * (qi * qi + qj * qj)
        n_real = Nr[p]; m_real = Mr[p]
        Oacc = 0.0
        dTx = 0.0; dTy = 0.0; dTz = 0.0
        dQw = 0.0; dQx = 0.0; dQy = 0.0; dQz = 0.0
        for i in range(m_real):  # FIT
            fax0 = B[p, i, 0]; fay0 = B[p, i, 1]; faz0 = B[p, i, 2]
            ityp = Bt[p, i]
            icat = cats[ityp]
            if icat == 3:
                continue
            ialpha = alphas[ityp]; iK = Ks[ityp]
            fvx = FvB[p, i, 0]; fvy = FvB[p, i, 1]; fvz = FvB[p, i, 2]
            fnorm = math.sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
            finv = 1.0 / (fnorm if fnorm > 1e-12 else 1e-12)
            fvxn = fvx * finv; fvyn = fvy * finv; fvzn = fvz * finv
            fatx = r00 * fax0 + r01 * fay0 + r02 * faz0 + tx
            faty = r10 * fax0 + r11 * fay0 + r12 * faz0 + ty
            fatz = r20 * fax0 + r21 * fay0 + r22 * faz0 + tz
            fvtx = r00 * fvxn + r01 * fvyn + r02 * fvzn
            fvty = r10 * fvxn + r11 * fvyn + r12 * fvzn
            fvtz = r20 * fvxn + r21 * fvyn + r22 * fvzn
            fxj = 0.0; fyj = 0.0; fzj = 0.0   # positional force (sum_j aKwE*(rotfit-ref))
            wfx = 0.0; wfy = 0.0; wfz = 0.0   # weight force (sum_j coeff*ref_vn)
            for j in range(n_real):  # REF (same-type only)
                if At[p, j] != ityp:
                    continue
                rax = A[p, j, 0]; ray = A[p, j, 1]; raz = A[p, j, 2]
                rvx = RvA[p, j, 0]; rvy = RvA[p, j, 1]; rvz = RvA[p, j, 2]
                rnorm = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
                rinv = 1.0 / (rnorm if rnorm > 1e-12 else 1e-12)
                rvxn = rvx * rinv; rvyn = rvy * rinv; rvzn = rvz * rinv
                dx = fatx - rax; dy = faty - ray; dz = fatz - raz   # rotfit - ref
                r2 = dx * dx + dy * dy + dz * dz
                E = math.exp(-ialpha * 0.5 * r2)
                D = fvtx * rvxn + fvty * rvyn + fvtz * rvzn
                if icat == 1:
                    Dcl = 0.0 if D < 0.0 else (1.0 if D > 1.0 else D)
                    w = (Dcl + 2.0) / 3.0
                elif icat == 2:
                    w = (abs(D) + 2.0) / 3.0
                else:
                    w = 1.0
                KwE = iK * w * E
                Oacc += KwE
                if need_grad:
                    aKwE = -ialpha * KwE
                    fxj += aKwE * dx; fyj += aKwE * dy; fzj += aKwE * dz
                    if icat == 1:
                        c = 1.0 if (D > 0.0 and D < 1.0) else 0.0
                    elif icat == 2:
                        c = 1.0 if D > 0.0 else (-1.0 if D < 0.0 else 0.0)
                    else:
                        c = 0.0
                    coeff = (1.0 / 3.0) * iK * E * c
                    wfx += coeff * rvxn; wfy += coeff * rvyn; wfz += coeff * rvzn
            if need_grad:
                dTx += fxj; dTy += fyj; dTz += fzj
                # positional dV/dq: shape tail with force (fxj,..) and body fit ANCHOR (fa*0)
                dQw += fxj * (-2.0 * qk_ * fay0 + 2.0 * qj * faz0) + fyj * (2.0 * qk_ * fax0 - 2.0 * qi * faz0) + fzj * (-2.0 * qj * fax0 + 2.0 * qi * fay0)
                dQx += fxj * (2.0 * qj * fay0 + 2.0 * qk_ * faz0) + fyj * (2.0 * qj * fax0 - 4.0 * qi * fay0 - 2.0 * qr * faz0) + fzj * (2.0 * qk_ * fax0 + 2.0 * qr * fay0 - 4.0 * qi * faz0)
                dQy += fxj * (-4.0 * qj * fax0 + 2.0 * qi * fay0 + 2.0 * qr * faz0) + fyj * (2.0 * qi * fax0 + 2.0 * qk_ * faz0) + fzj * (-2.0 * qr * fax0 + 2.0 * qk_ * fay0 - 4.0 * qj * faz0)
                dQz += fxj * (-4.0 * qk_ * fax0 - 2.0 * qr * fay0 + 2.0 * qi * faz0) + fyj * (2.0 * qr * fax0 - 4.0 * qk_ * fay0 + 2.0 * qj * faz0) + fzj * (2.0 * qi * fax0 + 2.0 * qj * fay0)
                # weight dV/dq: shape tail with force (wf*) and body fit VECTOR (fv*n)
                dQw += wfx * (-2.0 * qk_ * fvyn + 2.0 * qj * fvzn) + wfy * (2.0 * qk_ * fvxn - 2.0 * qi * fvzn) + wfz * (-2.0 * qj * fvxn + 2.0 * qi * fvyn)
                dQx += wfx * (2.0 * qj * fvyn + 2.0 * qk_ * fvzn) + wfy * (2.0 * qj * fvxn - 4.0 * qi * fvyn - 2.0 * qr * fvzn) + wfz * (2.0 * qk_ * fvxn + 2.0 * qr * fvyn - 4.0 * qi * fvzn)
                dQy += wfx * (-4.0 * qj * fvxn + 2.0 * qi * fvyn + 2.0 * qr * fvzn) + wfy * (2.0 * qi * fvxn + 2.0 * qk_ * fvzn) + wfz * (-2.0 * qr * fvxn + 2.0 * qk_ * fvyn - 4.0 * qj * fvzn)
                dQz += wfx * (-4.0 * qk_ * fvxn - 2.0 * qr * fvyn + 2.0 * qi * fvzn) + wfy * (2.0 * qr * fvxn - 4.0 * qk_ * fvyn + 2.0 * qj * fvzn) + wfz * (2.0 * qi * fvxn + 2.0 * qj * fvyn)
        O[p] = Oacc
        dT[p, 0] = dTx; dT[p, 1] = dTy; dT[p, 2] = dTz
        dQ[p, 0] = dQw; dQ[p, 1] = dQx; dQ[p, 2] = dQy; dQ[p, 3] = dQz
    return O, dQ, dT


def pharm_grad_dq_se3_batch(q, t, ref_types, fit_types, ref_anchors, fit_anchors,
                            ref_vectors, fit_vectors, alphas, Ks, cats, *,
                            N_real=None, M_real=None, NEED_GRAD: bool = True,
                            BLOCK=None, num_warps=None, num_stages=None):
    """CPU drop-in for the Triton directional pharm value+QUATERNION-grad kernel. Takes q
    (not R) and returns (O, dQ, dT) with dQ = dO/dq -- no R->q projection needed downstream."""
    P, N_pad, _ = ref_anchors.shape
    _, M_pad, _ = fit_anchors.shape
    dev, dt = ref_anchors.device, ref_anchors.dtype

    def _np(x):
        return np.ascontiguousarray(x.detach().cpu().numpy())
    RaA = _np(ref_anchors); FaB = _np(fit_anchors)
    RvA = _np(ref_vectors); FvB = _np(fit_vectors)
    RtA = ref_types.detach().cpu().numpy().astype(np.int64)
    FtB = fit_types.detach().cpu().numpy().astype(np.int64)
    qn = _np(q); tn = _np(t)
    al = alphas.detach().cpu().numpy().astype(np.float64)
    Ksn = Ks.detach().cpu().numpy().astype(np.float64)
    cn = cats.detach().cpu().numpy().astype(np.int64)
    Nr = (np.full(P, N_pad, np.int64) if N_real is None
          else N_real.detach().cpu().numpy().astype(np.int64))
    Mr = (np.full(P, M_pad, np.int64) if M_real is None
          else M_real.detach().cpu().numpy().astype(np.int64))
    O, dQ, dT = _pharm_grad_dq_kernel(RaA, FaB, qn, tn, RtA, FtB, RvA, FvB, al, Ksn, cn, Nr, Mr, bool(NEED_GRAD))
    return (torch.as_tensor(O, device=dev, dtype=dt),
            torch.as_tensor(dQ, device=dev, dtype=dt),
            torch.as_tensor(dT, device=dev, dtype=dt))


@torch.no_grad()
def _batch_self_overlap_esp(P_pad, charges, N_real, alpha: float = 0.81, lam: float = 0.3):
    """ESP self-overlap via the identity pose; CPU replica of _batch_self_overlap_esp."""
    K = P_pad.shape[0]
    q_id = torch.zeros(K, 4, device=P_pad.device, dtype=P_pad.dtype); q_id[:, 0] = 1.0
    t_0 = torch.zeros(K, 3, device=P_pad.device, dtype=P_pad.dtype)
    V, _, _ = overlap_score_grad_esp_se3_batch(P_pad, P_pad, charges, charges, q_id, t_0,
                                               alpha=alpha, lam=lam, N_real=N_real,
                                               M_real=N_real, NEED_GRAD=False)
    return V
