"""``vol`` / ``surf`` driver entry points (Gaussian volume overlap, Tanimoto).

The fine loop lives in :mod:`engine`; this module keeps the chunked kernel wrappers and the
historical ``coarse_fine_align_many`` signature.
"""
from __future__ import annotations

import torch

from ..kernels.dispatch import overlap_score_grad_se3_batch, _batch_self_overlap
from ._common import apply_se3_transform, quaternion_to_rotation_matrix  # noqa: F401 (re-export)
from ._shim import batch, run

torch.backends.cuda.matmul.allow_tf32 = True

#: Poses of ONE molecule per CTA for the deduplicated multi-pose shape kernel; measured to pay
#: for surf only (contention-insensitive ~36k aligns/s; vol 1.04-1.07x for a larger divergence;
#: the ESP kernel negative). Mirrors ``ModeSpec.multipose``.
_MODE_POSES = {"surf": 8}


@torch.no_grad()
def _overlap_in_chunks(A, B, q, t, *, alpha: float = 0.81, N_real=None, M_real=None,
                       NEED_GRAD=True, seeds_per_mol: int = 1, poses_per_cta: int = 1):
    """Evaluate the fused overlap kernel on an arbitrary-long pose list, slicing at the CUDA
    ``grid.z <= 65535`` limit and keeping every molecule's seed group whole."""
    K = q.shape[0]
    if N_real is None:
        N_real = A.new_full((A.shape[0],), A.shape[1], dtype=torch.int32)
    if M_real is None:
        M_real = B.new_full((B.shape[0],), B.shape[1], dtype=torch.int32)
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
        V, dQ, dT = overlap_score_grad_se3_batch(
            A[ms:me], B[ms:me], q[start:end], t[start:end], alpha=alpha,
            N_real=N_real[ms:me], M_real=M_real[ms:me], NEED_GRAD=NEED_GRAD, **extra)
        out_V[start:end] = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT
    return out_V, out_dQ, out_dT


def _self_overlap_in_chunks(P_pad, N_real, alpha=0.81):
    K = P_pad.size(0)
    CHUNK = 65_535
    V_all = torch.empty(K, device=P_pad.device, dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(P_pad[s:e], N_real[s:e], alpha)
    return V_all


def coarse_fine_align_many(A_batch, B_batch, VAA=None, VBB=None, *, alpha: float = 0.81,
                           num_seeds: int = 50, steps_fine: int = 100, lr: float = 0.075,
                           N_real=None, M_real=None, early_stop_patience: int = 2,
                           early_stop_tol: float = 1e-5, seeds=None, prune_after: int = 0,
                           prune_keep: int = 0, mode: str | None = None):
    """Batched shape alignment of ``(A_batch, B_batch)`` pairs; every seed is fine-optimised
    and the per-pair maximum taken. ``VAA``/``VBB`` are accepted for signature compatibility;
    the engine recomputes the self-overlaps from the same kernel. ``mode="surf"`` selects the
    surface mode (its deduplicated multi-pose layout); anything else is ``vol``."""
    chans = {"atoms" if mode != "surf" else "surf": batch(A_batch, B_batch, N_real, M_real)}
    return run("surf" if mode == "surf" else "vol", chans, alpha=alpha, num_seeds=num_seeds,
               steps_fine=steps_fine, lr=lr, early_stop_patience=early_stop_patience,
               early_stop_tol=early_stop_tol, seeds=seeds)
