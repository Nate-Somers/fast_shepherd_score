"""Opt-in CPU (numba) fine-loop driver — a first-class CPU path, NOT the GPU fallback.

The GPU drivers' eager loop runs the numba overlap+grad kernel for the heavy O(K·N·M) work
but does the per-step score/best/Adam tail in **torch** and re-marshals the constant coordinate
blocks ``A``/``B`` torch->numpy every step. On CPU that tail is (a) serial (only the kernel
``prange``s) so multi-core gain is Amdahl-capped, and (b) catastrophic when torch's own thread
pool (``OMP_NUM_THREADS``) contends with numba's ``prange`` — measured ~5-8x slower at 8 threads.

This module removes torch from the hot loop entirely. The constant inputs are converted to
numpy ONCE; each step calls the existing (validated) overlap+grad njit kernel, then a small
``@njit(parallel=True)`` TAIL that does Tanimoto score + best-pose tracking + tangent-projected
Adam + renorm, all in ``prange`` over poses, in place. The Python loop only chains the two njit
calls and does the every-5-steps early-stop check (a numpy ``.max()`` — no torch).

PARITY: the overlap+grad kernels are reused unchanged (same dV/dq math), and the tail replicates
``fused_adam_qt_with_tangent_proj`` exactly (β1=0.9, β2=0.999, eps=1e-8 inside the sqrt, no bias
correction, unit-quaternion renorm) and ``_update_best`` (track the pre-Adam pose by score). The
seed set and step count are the caller's (identical to the Triton path — same ``_MODE_SEEDS`` /
``_MODE_STEPS``). The only intended numeric difference vs the eager path is fastmath reassociation.
"""
from __future__ import annotations

import os
import numpy as np
from numba import njit, prange

# Adam constants — must match fused_adam_qt_with_tangent_proj in cpu.py / the Triton tail.
_B1 = 0.9
_B2 = 0.999
_EPS = 1e-8

# Lever 2+3: use the SoA fp32 + SVML-vectorized overlap kernels (cpu_soa.py) when this numba
# build has SVML (numba<=0.59 + icc_rt; config.USING_SVML). They vectorize the exp-bound inner
# loop for ~2x (atoms) to ~4.4x (200-pt surface), at fp32 gradient accuracy (~1e-4 rel). Without
# SVML they would be no faster (scalar exp is the barrier), so fall back to the fp64 AoS kernels
# in cpu.py. Disable with FSS_CPU_SOA=0.
try:
    import numba.core.config as _nbcfg
    _SVML = bool(_nbcfg.USING_SVML)
except Exception:
    _SVML = False
_USE_SOA = (os.environ.get("FSS_CPU_SOA", "1") != "0") and _SVML


@njit(parallel=True, fastmath=False, cache=True)
def _tail_tanimoto(O, dQ, dT, q, t, mq, vq, mt, vt, best, bq, bt, norm, lr):
    """One fine-loop tail step for a single-channel Tanimoto objective (vol/surf/vol_esp/
    surf_esp/pharm). In place. O,(dQ,dT) = overlap value + dO/dq, dO/dt from the kernel at the
    CURRENT (q,t). Tracks the pre-Adam pose as best (== _update_best), then tangent-projected
    Adam on q + plain Adam on t + renorm (== fused_adam_qt_with_tangent_proj)."""
    P = O.shape[0]
    for p in prange(P):
        Op = O[p]
        denom = norm[p] - Op
        if denom == 0.0:
            denom = 1e-12
        score = Op / denom
        scale = norm[p] / (denom * denom)            # d(Tanimoto)/dO  (>=0)
        # best-pose tracking (pre-Adam pose), identical to _update_best
        if score > best[p]:
            best[p] = score
            bq[p, 0] = q[p, 0]; bq[p, 1] = q[p, 1]; bq[p, 2] = q[p, 2]; bq[p, 3] = q[p, 3]
            bt[p, 0] = t[p, 0]; bt[p, 1] = t[p, 1]; bt[p, 2] = t[p, 2]
        # descent gradient g = -scale * dO/d(.)   (ascend Tanimoto == descend -Tanimoto)
        gq0 = -scale * dQ[p, 0]; gq1 = -scale * dQ[p, 1]
        gq2 = -scale * dQ[p, 2]; gq3 = -scale * dQ[p, 3]
        gt0 = -scale * dT[p, 0]; gt1 = -scale * dT[p, 1]; gt2 = -scale * dT[p, 2]
        # tangent projection of the quaternion grad: dq = g - q (g·q)
        radial = gq0 * q[p, 0] + gq1 * q[p, 1] + gq2 * q[p, 2] + gq3 * q[p, 3]
        dq0 = gq0 - q[p, 0] * radial; dq1 = gq1 - q[p, 1] * radial
        dq2 = gq2 - q[p, 2] * radial; dq3 = gq3 - q[p, 3] * radial
        # Adam on q (no bias correction; eps inside the sqrt)
        mq[p, 0] = _B1 * mq[p, 0] + (1.0 - _B1) * dq0
        mq[p, 1] = _B1 * mq[p, 1] + (1.0 - _B1) * dq1
        mq[p, 2] = _B1 * mq[p, 2] + (1.0 - _B1) * dq2
        mq[p, 3] = _B1 * mq[p, 3] + (1.0 - _B1) * dq3
        vq[p, 0] = _B2 * vq[p, 0] + (1.0 - _B2) * dq0 * dq0
        vq[p, 1] = _B2 * vq[p, 1] + (1.0 - _B2) * dq1 * dq1
        vq[p, 2] = _B2 * vq[p, 2] + (1.0 - _B2) * dq2 * dq2
        vq[p, 3] = _B2 * vq[p, 3] + (1.0 - _B2) * dq3 * dq3
        q[p, 0] -= lr * mq[p, 0] / np.sqrt(vq[p, 0] + _EPS)
        q[p, 1] -= lr * mq[p, 1] / np.sqrt(vq[p, 1] + _EPS)
        q[p, 2] -= lr * mq[p, 2] / np.sqrt(vq[p, 2] + _EPS)
        q[p, 3] -= lr * mq[p, 3] / np.sqrt(vq[p, 3] + _EPS)
        # Adam on t (no tangent projection)
        mt[p, 0] = _B1 * mt[p, 0] + (1.0 - _B1) * gt0
        mt[p, 1] = _B1 * mt[p, 1] + (1.0 - _B1) * gt1
        mt[p, 2] = _B1 * mt[p, 2] + (1.0 - _B1) * gt2
        vt[p, 0] = _B2 * vt[p, 0] + (1.0 - _B2) * gt0 * gt0
        vt[p, 1] = _B2 * vt[p, 1] + (1.0 - _B2) * gt1 * gt1
        vt[p, 2] = _B2 * vt[p, 2] + (1.0 - _B2) * gt2 * gt2
        t[p, 0] -= lr * mt[p, 0] / np.sqrt(vt[p, 0] + _EPS)
        t[p, 1] -= lr * mt[p, 1] / np.sqrt(vt[p, 1] + _EPS)
        t[p, 2] -= lr * mt[p, 2] / np.sqrt(vt[p, 2] + _EPS)
        # renorm q to the unit sphere
        qn = np.sqrt(q[p, 0] * q[p, 0] + q[p, 1] * q[p, 1] + q[p, 2] * q[p, 2] + q[p, 3] * q[p, 3])
        if qn < 1e-12:
            qn = 1e-12
        q[p, 0] /= qn; q[p, 1] /= qn; q[p, 2] /= qn; q[p, 3] /= qn


@njit(parallel=True, fastmath=False, cache=True)
def _tail_vol_color(Vs, dQs, dTs, Oc, dQc, dTc, q, t, mq, vq, mt, vt, best, bq, bt,
                    norm_s, norm_c, w, lr):
    """vol_color tail: score = (1-w)*shape_Tc + w*color_Tc; combined descent grad
    g = (1-w)*(-scale_s*dO_s) + w*(-scale_c*dO_c); tangent-projected Adam (matches the eager
    vol_color loop / _GraphedFineVolColor)."""
    P = Vs.shape[0]
    for p in prange(P):
        ds = norm_s[p] - Vs[p]
        if ds == 0.0:
            ds = 1e-12
        ss = Vs[p] / ds
        scale_s = norm_s[p] / (ds * ds)
        dc = norm_c[p] - Oc[p]
        if dc == 0.0:
            dc = 1e-12
        sc = Oc[p] / dc
        scale_c = norm_c[p] / (dc * dc)
        score = (1.0 - w) * ss + w * sc
        if score > best[p]:
            best[p] = score
            bq[p, 0] = q[p, 0]; bq[p, 1] = q[p, 1]; bq[p, 2] = q[p, 2]; bq[p, 3] = q[p, 3]
            bt[p, 0] = t[p, 0]; bt[p, 1] = t[p, 1]; bt[p, 2] = t[p, 2]
        cs = -(1.0 - w) * scale_s
        cc = -w * scale_c
        gq0 = cs * dQs[p, 0] + cc * dQc[p, 0]; gq1 = cs * dQs[p, 1] + cc * dQc[p, 1]
        gq2 = cs * dQs[p, 2] + cc * dQc[p, 2]; gq3 = cs * dQs[p, 3] + cc * dQc[p, 3]
        gt0 = cs * dTs[p, 0] + cc * dTc[p, 0]; gt1 = cs * dTs[p, 1] + cc * dTc[p, 1]
        gt2 = cs * dTs[p, 2] + cc * dTc[p, 2]
        radial = gq0 * q[p, 0] + gq1 * q[p, 1] + gq2 * q[p, 2] + gq3 * q[p, 3]
        dq0 = gq0 - q[p, 0] * radial; dq1 = gq1 - q[p, 1] * radial
        dq2 = gq2 - q[p, 2] * radial; dq3 = gq3 - q[p, 3] * radial
        mq[p, 0] = _B1 * mq[p, 0] + (1.0 - _B1) * dq0
        mq[p, 1] = _B1 * mq[p, 1] + (1.0 - _B1) * dq1
        mq[p, 2] = _B1 * mq[p, 2] + (1.0 - _B1) * dq2
        mq[p, 3] = _B1 * mq[p, 3] + (1.0 - _B1) * dq3
        vq[p, 0] = _B2 * vq[p, 0] + (1.0 - _B2) * dq0 * dq0
        vq[p, 1] = _B2 * vq[p, 1] + (1.0 - _B2) * dq1 * dq1
        vq[p, 2] = _B2 * vq[p, 2] + (1.0 - _B2) * dq2 * dq2
        vq[p, 3] = _B2 * vq[p, 3] + (1.0 - _B2) * dq3 * dq3
        q[p, 0] -= lr * mq[p, 0] / np.sqrt(vq[p, 0] + _EPS)
        q[p, 1] -= lr * mq[p, 1] / np.sqrt(vq[p, 1] + _EPS)
        q[p, 2] -= lr * mq[p, 2] / np.sqrt(vq[p, 2] + _EPS)
        q[p, 3] -= lr * mq[p, 3] / np.sqrt(vq[p, 3] + _EPS)
        mt[p, 0] = _B1 * mt[p, 0] + (1.0 - _B1) * gt0
        mt[p, 1] = _B1 * mt[p, 1] + (1.0 - _B1) * gt1
        mt[p, 2] = _B1 * mt[p, 2] + (1.0 - _B1) * gt2
        vt[p, 0] = _B2 * vt[p, 0] + (1.0 - _B2) * gt0 * gt0
        vt[p, 1] = _B2 * vt[p, 1] + (1.0 - _B2) * gt1 * gt1
        vt[p, 2] = _B2 * vt[p, 2] + (1.0 - _B2) * gt2 * gt2
        t[p, 0] -= lr * mt[p, 0] / np.sqrt(vt[p, 0] + _EPS)
        t[p, 1] -= lr * mt[p, 1] / np.sqrt(vt[p, 1] + _EPS)
        t[p, 2] -= lr * mt[p, 2] / np.sqrt(vt[p, 2] + _EPS)
        qn = np.sqrt(q[p, 0] * q[p, 0] + q[p, 1] * q[p, 1] + q[p, 2] * q[p, 2] + q[p, 3] * q[p, 3])
        if qn < 1e-12:
            qn = 1e-12
        q[p, 0] /= qn; q[p, 1] /= qn; q[p, 2] /= qn; q[p, 3] /= qn


def _f32c(x):
    """torch CPU tensor -> contiguous float32 numpy (no copy when already so)."""
    return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)


def _i64(x):
    return np.ascontiguousarray(x.detach().cpu().numpy()).astype(np.int64)


def fine_loop_cpu(overlap_fn, q_seed, t_seed, norm, *, lr, steps,
                  es_patience, es_tol, tail="tanimoto", tail_args=()):
    """Run the whole fine loop on CPU with NO torch in the hot path.

    overlap_fn(q_np, t_np) -> the kernel outputs the tail needs at the current pose:
        tanimoto:  (O, dQ, dT)
        vol_color: (Vs, dQs, dTs, Oc, dQc, dTc)
    q_seed/t_seed/norm: float32 numpy (P,4)/(P,3)/(P,). Returns (best, bq, bt) float32 numpy.
    Early-stop semantics match the eager loop (global best, checked every 5 steps).
    """
    P = q_seed.shape[0]
    q = q_seed.copy(); t = t_seed.copy()
    mq = np.zeros((P, 4), np.float32); vq = np.zeros((P, 4), np.float32)
    mt = np.zeros((P, 3), np.float32); vt = np.zeros((P, 3), np.float32)
    best = np.full(P, -np.inf, np.float32)
    bq = q_seed.copy(); bt = t_seed.copy()
    prev = -np.inf; no_improve = 0
    lr = np.float32(lr)
    for step in range(steps):
        out = overlap_fn(q, t)
        if tail == "tanimoto":
            O, dQ, dT = out
            _tail_tanimoto(O, dQ, dT, q, t, mq, vq, mt, vt, best, bq, bt, norm, lr)
        else:  # vol_color
            Vs, dQs, dTs, Oc, dQc, dTc = out
            norm_c, w = tail_args
            _tail_vol_color(Vs, dQs, dTs, Oc, dQc, dTc, q, t, mq, vq, mt, vt, best, bq, bt,
                            norm, norm_c, np.float32(w), lr)
        if step % 5 == 0:
            cur = float(best.max())
            if cur - prev < es_tol:
                no_improve += 1
                if no_improve >= es_patience:
                    break
            else:
                no_improve = 0
                prev = cur
    return best, bq, bt


# ===========================================================================
# Per-mode glue: marshal the torch inputs to numpy ONCE, build the overlap closure,
# run the fused CPU loop, return torch tensors on the original device. These are what
# the GPU drivers call on their CPU branch.
# ===========================================================================
def cpu_fused_shape(A_k, B_k, q_seed, t_seed, N_k, M_k, norm, alpha, lr, steps,
                    es_patience, es_tol):
    """vol / surf (and the shape channel): Gaussian overlap Tanimoto."""
    import torch
    Nr = _i64(N_k); Mr = _i64(M_k); a_f = float(alpha)
    if _USE_SOA:
        from .cpu_soa import _overlap_grad_kernel_soa, to_soa
        A_np = to_soa(_f32c(A_k)); B_np = to_soa(_f32c(B_k))     # (K,3,N) fp32, contiguous in n

        def _ov(qn, tn):
            return _overlap_grad_kernel_soa(A_np, B_np, qn, tn, Nr, Mr, a_f, True)   # fp32 out
    else:
        from .cpu import _overlap_grad_kernel
        A_np = _f32c(A_k); B_np = _f32c(B_k)

        def _ov(qn, tn):
            V, dQ, dT = _overlap_grad_kernel(A_np, B_np, qn, tn, Nr, Mr, a_f, True)
            return V.astype(np.float32), dQ.astype(np.float32), dT.astype(np.float32)

    bs, bq, bt = fine_loop_cpu(_ov, _f32c(q_seed), _f32c(t_seed), _f32c(norm),
                               lr=lr, steps=steps, es_patience=es_patience, es_tol=es_tol)
    dev = A_k.device
    return (torch.from_numpy(bs).to(dev), torch.from_numpy(bq).to(dev),
            torch.from_numpy(bt).to(dev))


def cpu_fused_esp(A_k, B_k, CA_k, CB_k, q_seed, t_seed, N_k, M_k, norm, alpha, lam, lr, steps,
                  es_patience, es_tol):
    """vol_esp / surf_esp: ESP-weighted Gaussian overlap Tanimoto (shape kernel × charge)."""
    import torch
    CA = _f32c(CA_k); CB = _f32c(CB_k); Nr = _i64(N_k); Mr = _i64(M_k)
    a_f = float(alpha); inv_lam = 1.0 / float(lam)
    if _USE_SOA:
        from .cpu_soa import _overlap_grad_esp_kernel_soa, to_soa
        A_np = to_soa(_f32c(A_k)); B_np = to_soa(_f32c(B_k))     # (K,3,N) fp32; charges stay (K,N)

        def _ov(qn, tn):
            return _overlap_grad_esp_kernel_soa(A_np, B_np, CA, CB, qn, tn, Nr, Mr, a_f, inv_lam, True)
    else:
        from .cpu import _overlap_grad_esp_kernel
        A_np = _f32c(A_k); B_np = _f32c(B_k)

        def _ov(qn, tn):
            V, dQ, dT = _overlap_grad_esp_kernel(A_np, B_np, CA, CB, qn, tn, Nr, Mr, a_f, inv_lam, True)
            return V.astype(np.float32), dQ.astype(np.float32), dT.astype(np.float32)

    bs, bq, bt = fine_loop_cpu(_ov, _f32c(q_seed), _f32c(t_seed), _f32c(norm),
                               lr=lr, steps=steps, es_patience=es_patience, es_tol=es_tol)
    dev = A_k.device
    return (torch.from_numpy(bs).to(dev), torch.from_numpy(bq).to(dev),
            torch.from_numpy(bt).to(dev))


def cpu_fused_pharm(anc1_k, anc2_k, vec1_k, vec2_k, t1_k, t2_k, q_seed, t_seed,
                    N_k, M_k, norm, al, Ks, cats, lr, steps, es_patience, es_tol):
    """pharm: directional pharmacophore overlap Tanimoto (in-register dO/dq kernel)."""
    import torch
    from .cpu import _pharm_grad_dq_kernel
    Ra = _f32c(anc1_k); Fa = _f32c(anc2_k); Rv = _f32c(vec1_k); Fv = _f32c(vec2_k)
    Rt = _i64(t1_k); Ft = _i64(t2_k); Nr = _i64(N_k); Mr = _i64(M_k)
    aln = np.ascontiguousarray(al.detach().cpu().numpy(), dtype=np.float64)
    Ksn = np.ascontiguousarray(Ks.detach().cpu().numpy(), dtype=np.float64)
    cn = np.ascontiguousarray(cats.detach().cpu().numpy()).astype(np.int64)

    def _ov(qn, tn):
        O, dQ, dT = _pharm_grad_dq_kernel(Ra, Fa, qn, tn, Rt, Ft, Rv, Fv, aln, Ksn, cn, Nr, Mr, True)
        return O.astype(np.float32), dQ.astype(np.float32), dT.astype(np.float32)

    bs, bq, bt = fine_loop_cpu(_ov, _f32c(q_seed), _f32c(t_seed), _f32c(norm),
                               lr=lr, steps=steps, es_patience=es_patience, es_tol=es_tol)
    dev = anc1_k.device
    return (torch.from_numpy(bs).to(dev), torch.from_numpy(bq).to(dev),
            torch.from_numpy(bt).to(dev))


def cpu_fused_vol_color(A_k, B_k, anc1_k, anc2_k, pt1_k, pt2_k, q_seed, t_seed,
                        Nc_k, Mc_k, Na_k, Ma_k, norm_s, norm_c, al, Ks, cats,
                        alpha, color_weight, lr, steps, es_patience, es_tol):
    """vol_color: (1-w)*shape_Tc + w*directionless-color_Tc, combined-objective descent."""
    import torch
    from .cpu import _pharm_color_grad_kernel                  # color channel: typed, AoS
    An1 = _f32c(anc1_k); An2 = _f32c(anc2_k)
    Pt1 = _i64(pt1_k); Pt2 = _i64(pt2_k)
    Nc = _i64(Nc_k); Mc = _i64(Mc_k); Na = _i64(Na_k); Ma = _i64(Ma_k)
    aln = np.ascontiguousarray(al.detach().cpu().numpy(), dtype=np.float64)
    Ksn = np.ascontiguousarray(Ks.detach().cpu().numpy(), dtype=np.float64)
    cn = np.ascontiguousarray(cats.detach().cpu().numpy()).astype(np.int64)
    a_f = float(alpha)
    if _USE_SOA:                                               # shape channel SoA fp32 + SVML
        from .cpu_soa import _overlap_grad_kernel_soa, to_soa
        A_np = to_soa(_f32c(A_k)); B_np = to_soa(_f32c(B_k))

        def _shape(qn, tn):
            return _overlap_grad_kernel_soa(A_np, B_np, qn, tn, Nc, Mc, a_f, True)
    else:
        from .cpu import _overlap_grad_kernel
        A_np = _f32c(A_k); B_np = _f32c(B_k)

        def _shape(qn, tn):
            Vs, dQs, dTs = _overlap_grad_kernel(A_np, B_np, qn, tn, Nc, Mc, a_f, True)
            return Vs.astype(np.float32), dQs.astype(np.float32), dTs.astype(np.float32)

    def _ov(qn, tn):
        Vs, dQs, dTs = _shape(qn, tn)
        Oc, dQc, dTc = _pharm_color_grad_kernel(An1, An2, qn, tn, Pt1, Pt2, aln, Ksn, cn, Na, Ma, True)
        return (Vs, dQs, dTs,
                Oc.astype(np.float32), dQc.astype(np.float32), dTc.astype(np.float32))

    bs, bq, bt = fine_loop_cpu(_ov, _f32c(q_seed), _f32c(t_seed), _f32c(norm_s),
                               lr=lr, steps=steps, es_patience=es_patience, es_tol=es_tol,
                               tail="vol_color", tail_args=(_f32c(norm_c), float(color_weight)))
    dev = A_k.device
    return (torch.from_numpy(bs).to(dev), torch.from_numpy(bq).to(dev),
            torch.from_numpy(bt).to(dev))
