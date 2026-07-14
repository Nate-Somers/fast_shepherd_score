from __future__ import annotations

import torch
# Kernels are dispatched per-call by tensor device (Triton on CUDA, numba on CPU), so one
# process can run both -- e.g. backend="numba" runs CPU tensors through the numba kernels
# even on a GPU box.
from ..kernels.dispatch import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj,
    _batch_self_overlap,
)
from ._common import batched_seeds_torch, _update_best
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from typing import Optional

torch.backends.cuda.matmul.allow_tf32 = True


@torch.no_grad()
def _overlap_in_chunks(A, B, q, t, *, alpha: float = 0.81,
                       N_real: torch.Tensor | None = None,
                       M_real: torch.Tensor | None = None,
                       NEED_GRAD = True):
    """
    Evaluate the fused overlap kernel on an arbitrary-long list of
    orientations, slicing the list so that each launch respects the
    CUDA `grid.z ≤ 65 535` limit.

    Parameters
    ----------
    A, B : (K,N,3) / (K,M,3)   – padded coordinate blocks
    q, t : (K,4) / (K,3)       – quaternions & translations
    N_real, M_real : (K,) int32 tensors holding the *true* atom counts
                     (rows beyond those indices are padding).  If None,
                     we assume no padding.
    Returns
    -------
    VAB : (K,)    dQ : (K,4)    dT : (K,3)     — all contiguous on GPU
    """
    K = A.shape[0]
    if N_real is not None:
        N_real = N_real.to(torch.float32).contiguous()
    if M_real is not None:
        M_real = M_real.to(torch.float32).contiguous()

    out_V  = torch.empty(K,        device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)

    CHUNK = 65_535                         # CUDA grid-z hard limit

    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)

        V, dQ, dT = overlap_score_grad_se3_batch(
            A[start:end], B[start:end],
            q[start:end], t[start:end],
            alpha=alpha,
            N_real=N_real[start:end],
            M_real=M_real[start:end],
            NEED_GRAD=NEED_GRAD)

        out_V[start:end]  = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


def _self_overlap_in_chunks(P_pad, N_real, alpha=0.81):
    K = P_pad.size(0)
    CHUNK = 65_535                     # hardware limit
    V_all = torch.empty(K,
                        device=P_pad.device,
                        dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(
            P_pad[s:e], N_real[s:e], alpha)   # ← original function
    return V_all


class _GraphedFineSurf(_GraphedFineBase):
    """Capture one in-place surf/vol fine step into a CUDA graph; replay = N steps.

    All loop-carried state (q/t, Adam moments, best_*) lives in persistent buffers
    updated in place; per-step temporaries use out= so there are no host syncs and
    no reallocation -- the requirements for graph capture. Inputs (A/B/seeds/...)
    are copied into the buffers before each replay, so one captured graph serves
    every bucket of the same (N_pad, M_pad, P) shape. capture()/run() are inherited
    from _GraphedFineBase.
    """

    def __init__(self, N_pad, M_pad, P, steps, alpha, lr, device):
        self.alpha = float(alpha); self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.Nr = torch.empty(P, device=device, dtype=torch.int32)
        self.Mr = torch.empty(P, device=device, dtype=torch.int32)
        self.norm = f(P)
        self.qs = f(P, 4); self.ts = f(P, 3)            # seeds (replay start state)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        self.denom = f(P); self.d2 = f(P); self.score = f(P); self.scale = f(P)
        self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        VAB, dQ, dT = overlap_score_grad_se3_batch(
            self.A, self.B, self.q, self.t, alpha=self.alpha, N_real=self.Nr, M_real=self.Mr)
        self._tanimoto_adam_tail(VAB, dQ, dT)

    def _tanimoto_adam_tail(self, VAB, dQ, dT):
        """Single-channel Tanimoto score + best-pose tracking + tangent-projected Adam, all
        in-place into persistent buffers. Shared by surf/vol and the ESP subclass (whose
        only difference is the fused shape+ESP overlap kernel that produces VAB/dQ/dT)."""
        torch.sub(self.norm, VAB, out=self.denom)
        torch.div(VAB, self.denom, out=self.score)
        torch.mul(self.denom, self.denom, out=self.d2)
        torch.div(self.norm, self.d2, out=self.scale)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        torch.mul(dQ, self.scale.unsqueeze(1), out=self.gq); self.gq.neg_()
        torch.mul(dT, self.scale.unsqueeze(1), out=self.gt); self.gt.neg_()
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, Nr, Mr, norm, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.norm.copy_(norm); self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_fine(A_k, B_k, q_seed, t_seed, N_k, M_k, norm, alpha, lr, steps, N_pad, M_pad, P,
                      es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "surf", N_pad, M_pad, P, steps, round(float(alpha), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineSurf(N_pad, M_pad, P, steps, alpha, lr, A_k.device),
        key, (A_k, B_k, N_k, M_k, norm, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_align_many(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None,
        early_stop_patience: int = 2,   # vol/surf converge fast; esp/pharm use 5
        early_stop_tol: float = 1e-5,
        seeds: tuple | None = None,
        prune_after: int = 0, prune_keep: int = 0):
    """
    Vectorised padding-aware alignment over a batch of (A, B) pairs.

    Strategy (matches the reference optimiser, fully batched): build the
    reference seed set -- identity + 4 principal-axis quaternions + Fibonacci
    rotations, each with a COM-aligning translation -- then fine-optimise EVERY
    seed and take the per-pair best.

    No coarse-grid top-k prune
    --------------------------
    Seeds are never ranked by their raw, un-optimised overlap: that score does
    not predict a seed's post-optimisation result, so ranking on it discards the
    seed sitting in the true basin for pseudo-symmetric molecules and keeps
    decoys that plateau at a lower local optimum. Every seed therefore enters the
    fine loop, and the per-pair max is taken.

    Parameters
    ----------
    A_batch, B_batch : (B, N_pad, 3) / (B, M_pad, 3)  atom coordinates
    VAA, VBB         : (B,)  pre-computed Gaussian self-overlaps
    num_seeds        : int   reference seed count (identity + 4 PCA + Fibonacci)
    N_real, M_real   : (B,)  optional true atom counts
    early_stop_patience : int  iterations without global improvement before stop
    early_stop_tol : float  minimum improvement threshold
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) reference seed set (no coarse grid, no flips, no 2nd translation)
    # ------------------------------------------------------------------
    # Seeds may be precomputed once per band and passed in, so the sub-batcher
    # can chunk ONLY the (memory-heavy) fine loop without re-paying the
    # launch-bound seed-gen per chunk. Slicing precomputed per-pair seeds is
    # exactly equivalent to recomputing per chunk (seeds are per-pair independent).
    if seeds is None:
        quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real,
                                             num_seeds=num_seeds)
    else:
        quats, t_seeds = seeds
    S = quats.size(1)

    # ------------------------------------------------------------------
    # 2) fine polishing (Adam-like) on EVERY seed
    # ------------------------------------------------------------------
    A_k = A_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, M_pad, 3)
    q_seed = quats.reshape(-1, 4).contiguous()
    t_seed = t_seeds.reshape(-1, 3).contiguous()

    N_k = N_real.repeat_interleave(S)
    M_k = M_real.repeat_interleave(S)
    VAA_plus_VBB = (VAA + VBB).repeat_interleave(S)        # invariant in loop
    P = q_seed.shape[0]

    best_score = best_q = best_t = None

    # --- CUDA-graph fast path for the launch-bound regime --------------------
    # Compute-aware cap: one formula covers vol and surf, which share this kernel -- see
    # graph_cap.
    if (A_batch.is_cuda and P <= graph_cap(N_pad * M_pad)
            and A_batch.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_fine(
                A_k.contiguous(), B_k.contiguous(), q_seed, t_seed, N_k, M_k,
                VAA_plus_VBB, alpha, lr, steps_fine, N_pad, M_pad, P,
                es_patience=early_stop_patience, es_tol=early_stop_tol)
        except Exception:
            best_score = None                              # capture failed -> eager

    # --- opt-in CPU (numba) fast path: fully-fused fine loop, NO torch in the hot loop ------
    # Skipped for fp64; any failure falls through to the eager loop. Same seeds/steps as
    # every other path.
    if (best_score is None and not A_batch.is_cuda
            and A_batch.dtype == torch.float32):
        try:
            from ..kernels.cpu_fused import cpu_fused_shape
            best_score, best_q, best_t = cpu_fused_shape(
                A_k, B_k, q_seed, t_seed, N_k, M_k, VAA_plus_VBB, alpha, lr, steps_fine,
                early_stop_patience, early_stop_tol)
        except Exception:
            best_score = None                              # fused failed -> eager

    # --- eager fall-back (large P, non-CUDA, fp64, or capture failure) -------
    if best_score is None:
        q_k = q_seed.clone()
        t_k = t_seed.clone()
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

        best_score = torch.full((len(q_k),), -float('inf'), device=device)
        best_q = q_k.clone()
        best_t = t_k.clone()

        # Early stopping state
        es_patience = early_stop_patience
        es_tol = early_stop_tol
        prev_max_score = -float('inf')
        no_improve_count = 0

        for step in range(steps_fine):
            VAB, dQ, dT = _overlap_in_chunks(
                A_k, B_k, q_k, t_k,
                alpha=alpha, N_real=N_k, M_real=M_k)

            denom = VAA_plus_VBB - VAB
            score = VAB / denom
            scale = VAA_plus_VBB / (denom * denom)

            best_score, best_q, best_t = _update_best(score, q_k, t_k, best_score, best_q, best_t)

            # Early stopping check, gated to every 5 steps to avoid a per-step
            # GPU->CPU sync. Gating only makes early-stop *less* aggressive.
            if step % 5 == 0:
                current_max = best_score.max().item()
                if current_max - prev_max_score < es_tol:
                    no_improve_count += 1
                    if no_improve_count >= es_patience:
                        break
                else:
                    no_improve_count = 0
                    prev_max_score = current_max

            fused_adam_qt_with_tangent_proj(
                q_k, t_k,
                -dQ * scale.unsqueeze(1),
                -dT * scale.unsqueeze(1),
                m_q, v_q, m_t, v_t, lr
            )

    # ------------------------------------------------------------------
    # 3) gather final results (using already-tracked best scores)
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, S)

    best = final_score.argmax(dim=1)
    sel  = best + torch.arange(BATCH, device=device) * S

    return final_score.flatten()[sel], \
           best_q.view(BATCH, S, 4)[torch.arange(BATCH), best], \
           best_t.view(BATCH, S, 3)[torch.arange(BATCH), best]
