"""CPU (numba) fallbacks for the Triton overlap+grad and Adam kernels.

These let the batched coarse-to-fine driver (``fast_se3.coarse_fine_align_many`` and
``_align_batch_vol``) run on a CPU-only box with **no Triton/CUDA**. ``fast_se3`` imports
these when ``shepherd_score.score.gaussian_overlap_triton`` (which hard-requires triton)
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

_K_PI = math.pi ** 1.5


@njit(parallel=True, fastmath=False, cache=True)
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
