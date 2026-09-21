"""``vol_esp`` / ``surf_esp`` driver entry points (charge-weighted Gaussian overlap, Tanimoto)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..kernels.dispatch import overlap_score_grad_esp_se3_batch, _batch_self_overlap_esp
from ._common import (check_gpu_available, apply_se3_transform,  # noqa: F401 (re-export)
                      quaternion_to_rotation_matrix)
from ._shim import batch, run, se3_of


@torch.no_grad()
def _overlap_in_chunks_esp(A, B, CA, CB, q, t, *, alpha: float, lam: float, N_real, M_real,
                           NEED_GRAD: bool = True, seeds_per_mol: int = 1, poses_per_cta: int = 1):
    """Evaluate the fused ESP overlap kernel in grid-safe chunks."""
    K = q.shape[0]
    N_real = N_real.to(torch.int32).contiguous()
    M_real = M_real.to(torch.int32).contiguous()
    out_V = torch.empty(K, device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)
    S = int(seeds_per_mol)
    CHUNK = 65_535 if S == 1 else max(S, (65_535 // S) * S)
    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)
        ms, me = start // S, end // S
        extra = {} if S == 1 else {"seeds_per_mol": S}
        if int(poses_per_cta) > 1:
            extra["poses_per_cta"] = int(poses_per_cta)
        V, dQ, dT = overlap_score_grad_esp_se3_batch(
            A[ms:me], B[ms:me], CA[ms:me], CB[ms:me], q[start:end], t[start:end],
            alpha=alpha, lam=lam, N_real=N_real[ms:me], M_real=M_real[ms:me],
            NEED_GRAD=NEED_GRAD, **extra)
        out_V[start:end] = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT
    return out_V, out_dQ, out_dT


def _self_overlap_esp_chunks(P_pad, C_pad, N_real, alpha, lam):
    K = P_pad.size(0)
    CHUNK = 65_535
    V_all = torch.empty(K, device=P_pad.device, dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap_esp(P_pad[s:e], C_pad[s:e], N_real[s:e], alpha, lam)
    return V_all


def coarse_fine_esp_align_many(A_batch, B_batch, CA_batch, CB_batch, VAA=None, VBB=None, *,
                               alpha: float = 0.81, lam: float = 0.3, num_seeds: int = 50,
                               trans_centers=None, trans_centers_real=None,
                               num_repeats_per_trans: int = 10, topk: int = 30,
                               steps_fine: int = 100, lr: float = 0.075, N_real=None,
                               M_real=None, seeds=None, early_stop_patience: int = 5,
                               early_stop_tol: float = 1e-5):
    """Batched ESP-weighted alignment; ``lam`` is passed to the kernel as given."""
    b = batch(A_batch, B_batch, N_real, M_real)
    chans = {"heavy": b, "charges": batch(CA_batch, CB_batch, b.n_real, b.m_real)}
    return run("vol_esp", chans, alpha=alpha, lam=lam, num_seeds=num_seeds, steps_fine=steps_fine,
               lr=lr, early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol,
               seeds=seeds, trans_centers=trans_centers, trans_centers_real=trans_centers_real,
               num_repeats_per_trans=num_repeats_per_trans, topk=topk)


def fast_optimize_ROCS_esp_overlay_batch(
        ref_batch, fit_batch, ref_charges_batch, fit_charges_batch, alpha: float, lam: float,
        N_real=None, M_real=None, trans_centers_batch=None, trans_centers_real=None,
        num_repeats_per_trans: int = 10, num_seeds: int = 50, topk: int = 30,
        steps_fine: int = 100, lr: float = 0.075, seeds=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ESP alignment: ``(aligned_fit, q, t, scores)``."""
    scores, q_best, t_best = coarse_fine_esp_align_many(
        ref_batch, fit_batch, ref_charges_batch, fit_charges_batch, alpha=alpha, lam=lam,
        num_seeds=num_seeds, trans_centers=trans_centers_batch,
        trans_centers_real=trans_centers_real, num_repeats_per_trans=num_repeats_per_trans,
        topk=topk, steps_fine=steps_fine, lr=lr, N_real=N_real, M_real=M_real, seeds=seeds)
    return apply_se3_transform(fit_batch, q_best, t_best), q_best, t_best, scores


def fast_optimize_ROCS_esp_overlay(ref_points, fit_points, ref_charges, fit_charges, alpha: float,
                                   lam: float, num_repeats: int = 50, trans_centers=None,
                                   num_repeats_per_trans: int = 10, topk: int = 30,
                                   steps_fine: int = 100, lr: float = 0.075, **kwargs):
    """Single-pair GPU ESP alignment: ``(aligned_points, SE3 (4,4), score)`` on CPU. Falls back
    to the eager reference optimizer when CUDA is unavailable.

    ``num_repeats`` is ACCEPTED AND IGNORED on the accelerated path, as it always has been: this
    function has never forwarded it, so the seed count is the batched default (50) and a caller
    passing 1 still searches 50 orientations. It IS honoured by the CPU fallback below, which is
    the reference optimizer. Left as is deliberately -- wiring it through is a behaviour change
    (``tests/test_fast_batch_alignment.py`` compares this against the batched entry point at
    ``num_repeats=1`` and expects them to agree to 1e-4, which only holds while both run 50)."""
    if not check_gpu_available():
        from ...alignment._torch import optimize_ROCS_esp_overlay
        return optimize_ROCS_esp_overlay(ref_points, fit_points, ref_charges, fit_charges,
                                         alpha, lam, num_repeats, **kwargs)
    device = torch.device("cuda")
    f = lambda x: x.to(device, dtype=torch.float32)
    A, B = f(ref_points).unsqueeze(0), f(fit_points).unsqueeze(0)
    CA, CB = f(ref_charges).unsqueeze(0), f(fit_charges).unsqueeze(0)
    tcb = tcr = None
    if trans_centers is not None:
        tc = f(trans_centers)
        tcb = tc.unsqueeze(0)
        tcr = torch.tensor([tc.shape[0]], device=device, dtype=torch.int32)
    aligned, q, t, score = fast_optimize_ROCS_esp_overlay_batch(
        A, B, CA, CB, alpha, lam, trans_centers_batch=tcb, trans_centers_real=tcr,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk,
        steps_fine=steps_fine, lr=lr)
    return aligned[0].cpu(), se3_of(q[0], t[0]).cpu(), score[0].cpu()
