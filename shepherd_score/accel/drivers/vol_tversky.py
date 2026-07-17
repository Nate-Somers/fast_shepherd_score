"""Batched coarse-to-fine driver for the asymmetric ``vol_tversky`` shape overlay.

``vol_tversky`` is NOT a new channel: it is the *same* atom-centred Gaussian SHAPE overlap
as ``vol`` (the ``overlap_score_grad_se3_batch`` kernel), scored with an asymmetric **Tversky**
reduction instead of Tanimoto::

    T = AB / (AB + ta*(AA - AB) + tb*(BB - AB))
      = AB / (k*AB + C),   k = 1 - ta - tb,   C = ta*AA + tb*BB

``AB`` is the cross overlap of the reference with the SE(3)-transformed fit; ``AA``/``BB`` are
the reference/fit self-overlaps. ``AA`` and ``BB`` are SE(3)-INVARIANT (a rigid transform
preserves a self-overlap), so ``C`` and ``k`` are precomputed once per pair and only ``AB``
(and its gradient) flow through the fine loop. The chain rule collapses to a single per-pair
scalar::

    dT/dAB = C / (k*AB + C)^2

so the pose gradient is ``scale * dAB/dq`` with ``scale = C / denom^2`` -- structurally identical
to the Tanimoto tail in ``shape.py`` (which uses ``scale = (AA+BB) / denom^2``), differing ONLY
in the reduction constants. We therefore REUSE the shape value+gradient kernel (Triton on CUDA,
numba on CPU) and the shared CUDA-graph fine loop, and only swap the host-side reduction.

Range note: unlike Tanimoto, the Tversky score is NOT bounded to [0, 1] -- a small dense query
inside a larger molecule can exceed 1.0 (the Gaussian cross-overlap is non-idempotent). Nothing
here clamps the score or assumes a [0, 1] range; best-pose tracking is a plain running max.
"""
from __future__ import annotations

import torch

# Kernels are dispatched per-call by tensor device (Triton on CUDA, numba on CPU), so one
# process can run both -- e.g. backend="numba" runs CPU tensors through the numba kernels.
from ..kernels.dispatch import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj,
)
from ._common import batched_seeds_torch, _update_best
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
# Reuse the shape driver's padding-safe chunked kernel wrappers verbatim (no new kernel).
from .shape import _overlap_in_chunks, _self_overlap_in_chunks  # noqa: F401  (re-export for callers)

torch.backends.cuda.matmul.allow_tf32 = True


class _GraphedFineTversky(_GraphedFineBase):
    """Capture one in-place vol_tversky fine step into a CUDA graph; replay = N steps.

    Identical loop-carried state and Adam tail as ``shape._GraphedFineSurf``; the ONLY
    difference is the reduction constants -- Tversky uses ``denom = k*VAB + C`` and
    ``scale = C/denom^2`` where ``C = ta*VAA + tb*VBB`` (loaded per pair) and ``k = 1-ta-tb``
    (baked into this graph). All temporaries use ``out=`` so there is no host sync and the
    step is capturable.
    """

    def __init__(self, N_pad, M_pad, P, steps, alpha, lr, k, device):
        self.alpha = float(alpha); self.lr = float(lr); self.k = float(k)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        self.A = f(P, N_pad, 3); self.B = f(P, M_pad, 3)
        self.Nr = torch.empty(P, device=device, dtype=torch.int32)
        self.Mr = torch.empty(P, device=device, dtype=torch.int32)
        self.C = f(P)                                    # ta*VAA + tb*VBB (pose-invariant)
        self.qs = f(P, 4); self.ts = f(P, 3)             # seeds (replay start state)
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
        # Tversky: denom = k*VAB + C ; T = VAB/denom ; dT/dVAB = C/denom^2. NOT clamped to [0,1].
        torch.add(self.C, VAB, alpha=self.k, out=self.denom)
        torch.div(VAB, self.denom, out=self.score)
        torch.mul(self.denom, self.denom, out=self.d2)
        torch.div(self.C, self.d2, out=self.scale)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        torch.mul(dQ, self.scale.unsqueeze(1), out=self.gq); self.gq.neg_()
        torch.mul(dT, self.scale.unsqueeze(1), out=self.gt); self.gt.neg_()
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, Nr, Mr, C, qs, ts):
        self.A.copy_(A); self.B.copy_(B)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.C.copy_(C); self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_fine_tversky(A_k, B_k, q_seed, t_seed, N_k, M_k, C, alpha, lr, k, steps,
                              N_pad, M_pad, P, es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_tversky", N_pad, M_pad, P, steps,
           round(float(alpha), 4), round(float(lr), 5), round(float(k), 6))
    return run_graphed(
        lambda: _GraphedFineTversky(N_pad, M_pad, P, steps, alpha, lr, k, A_k.device),
        key, (A_k, B_k, N_k, M_k, C, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_align_many_tversky(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        tversky_alpha: float = 0.95,
        tversky_beta: float = 0.05,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None,
        early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5,
        seeds: tuple | None = None):
    """Batched vol_tversky alignment over a batch of (A, B) pairs.

    Mirrors ``shape.coarse_fine_align_many`` exactly (same seed set, same fine schedule, same
    reused shape kernel + CUDA-graph loop); only the similarity reduction is Tversky instead of
    Tanimoto. ``VAA``/``VBB`` are the pose-invariant self-overlaps; ``tversky_alpha``/
    ``tversky_beta`` weight the missing-reference / extra-fit volume. Because ``AA``/``BB`` are
    pose-invariant, maximizing ``AB`` maximizes the Tversky score, so the optimal pose matches
    ``vol``; we still descend on the Tversky gradient (``scale = C/denom^2``) for exact parity
    with the eager reference.
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # Tversky reduction constants (pose-invariant). k can be negative-ish but with the
    # defaults (ta=0.95, tb=0.05) k == 0, i.e. denom == C is a pure per-pair constant; the
    # general form is kept so arbitrary (ta, tb) work.
    k = 1.0 - float(tversky_alpha) - float(tversky_beta)
    C = tversky_alpha * VAA + tversky_beta * VBB                       # (BATCH,)

    # --- reference seed set (identity + 4 PCA + structured/Fibonacci), same as shape ---
    if seeds is None:
        quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real,
                                             num_seeds=num_seeds)
    else:
        quats, t_seeds = seeds
    S = quats.size(1)

    # --- fine polishing on EVERY seed ---
    A_k = A_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, M_pad, 3)
    q_seed = quats.reshape(-1, 4).contiguous()
    t_seed = t_seeds.reshape(-1, 3).contiguous()

    N_k = N_real.repeat_interleave(S)
    M_k = M_real.repeat_interleave(S)
    C_k = C.repeat_interleave(S)                                       # per-pose constant
    P = q_seed.shape[0]

    best_score = best_q = best_t = None

    # --- CUDA-graph fast path (launch-bound regime); shares graph_cap with vol/surf ---
    if (A_batch.is_cuda and P <= graph_cap(N_pad * M_pad)
            and A_batch.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_fine_tversky(
                A_k.contiguous(), B_k.contiguous(), q_seed, t_seed, N_k, M_k,
                C_k, alpha, lr, k, steps_fine, N_pad, M_pad, P,
                es_patience=early_stop_patience, es_tol=early_stop_tol)
        except Exception:
            best_score = None                                          # capture failed -> eager

    # --- eager fall-back (CPU/numba, large P, fp64, or capture failure) ---
    # NOTE: no cpu_fused fast path here -- cpu_fused_shape hardcodes the Tanimoto reduction.
    # The eager loop dispatches the SAME shape value+grad kernel to numba on CPU tensors, so
    # this IS the numba backend for vol_tversky.
    if best_score is None:
        q_k = q_seed.clone()
        t_k = t_seed.clone()
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

        best_score = torch.full((len(q_k),), -float('inf'), device=device)
        best_q = q_k.clone()
        best_t = t_k.clone()

        es_patience = early_stop_patience
        es_tol = early_stop_tol
        prev_max_score = -float('inf')
        no_improve_count = 0

        for step in range(steps_fine):
            VAB, dQ, dT = _overlap_in_chunks(
                A_k, B_k, q_k, t_k,
                alpha=alpha, N_real=N_k, M_real=M_k)

            denom = k * VAB + C_k                                      # Tversky denominator
            score = VAB / denom                                       # NOT clamped to [0,1]
            scale = C_k / (denom * denom)                             # dT/dVAB

            best_score, best_q, best_t = _update_best(score, q_k, t_k, best_score, best_q, best_t)

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

    # --- gather per-pair best over seeds ---
    final_score = best_score.view(BATCH, S)

    best = final_score.argmax(dim=1)
    sel  = best + torch.arange(BATCH, device=device) * S

    return final_score.flatten()[sel], \
           best_q.view(BATCH, S, 4)[torch.arange(BATCH), best], \
           best_t.view(BATCH, S, 3)[torch.arange(BATCH), best]
