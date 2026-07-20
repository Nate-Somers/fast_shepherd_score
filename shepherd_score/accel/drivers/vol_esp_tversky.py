"""Batched coarse-to-fine driver for the asymmetric ``vol_esp_tversky`` overlay.

``vol_esp_tversky`` is NOT a new channel: it is the SAME electrostatic-weighted atom-centred
Gaussian overlap as ``vol_esp`` (the fused shape+ESP value+grad kernel
``overlap_score_grad_esp_se3_batch``, which vol_esp/surf_esp both dispatch to), scored
with the asymmetric **Tversky** reduction instead of Tanimoto -- exactly what ``vol_tversky`` is
to ``vol``::

    T = AB / (AB + ta*(AA - AB) + tb*(BB - AB))
      = AB / (k*AB + C),   k = 1 - ta - tb,   C = ta*AA + tb*BB

``AB`` is the cross ESP overlap of the reference with the SE(3)-transformed fit; ``AA``/``BB`` are
the reference/fit ESP self-overlaps. ``AA``/``BB`` are SE(3)-INVARIANT (a rigid transform
preserves a self-overlap and the charges are unchanged), so ``C`` and ``k`` are precomputed once
per pair and only ``AB`` (and its gradient) flow through the fine loop; the chain rule collapses
to a single per-pair scalar ``dT/dAB = C/denom^2`` (``scale = C/denom^2``). We therefore REUSE the
ESP value+gradient kernel (Triton on CUDA, numba on CPU) and the ``vol_tversky`` host-side
reduction, and write NO new kernel.

Because ``AA``/``BB`` are pose-invariant, maximizing ``AB`` maximizes ``T``, so the optimal pose
== ``vol_esp``'s; we still descend on the Tversky gradient for exact parity with the eager
reference. Like Tanimoto-vs-``vol``, the Tversky score is NOT bounded to [0, 1] and is never
clamped; best-pose tracking is a plain running max.
"""
from __future__ import annotations

import torch

# Kernels dispatch per-call by tensor device (Triton on CUDA, numba on CPU), so one process can
# run both -- e.g. backend="numba" runs CPU tensors through the numba ESP kernel.
from ..kernels.dispatch import (
    overlap_score_grad_esp_se3_batch, fused_adam_qt_with_tangent_proj,
)
from ._common import batched_seeds_torch, _update_best
from ._graphed import run_graphed, graph_cap
# Reuse the vol_tversky graphed reduction tail (denom=k*VAB+C, scale=C/denom^2) verbatim.
from .vol_tversky import _GraphedFineTversky
# Reuse vol_esp's padding-safe chunked ESP kernel wrappers + ESP self-overlap verbatim (no new kernel).
from .esp import _overlap_in_chunks_esp, _self_overlap_esp_chunks  # noqa: F401  (re-export for callers)

torch.backends.cuda.matmul.allow_tf32 = True


class _GraphedFineEspTversky(_GraphedFineTversky):
    """Capture one in-place vol_esp_tversky fine step into a CUDA graph; replay = N steps.

    The ``vol_tversky`` reduction (``denom = k*VAB + C``, ``scale = C/denom^2``) with ``VAB``
    produced by the FUSED shape+ESP kernel instead of the shape kernel. Adds persistent charge
    buffers (``CA``/``CB``) and the ``lam`` scalar -- exactly as ``_GraphedFineEsp`` adds them to
    the surf/vol Tanimoto loop -- and reuses ``_GraphedFineTversky._tversky_adam_tail`` verbatim.
    """

    def __init__(self, N_pad, M_pad, P, steps, alpha, lam, lr, k, device):
        self.lam = float(lam)
        self.CA = torch.empty(P, N_pad, device=device, dtype=torch.float32)
        self.CB = torch.empty(P, M_pad, device=device, dtype=torch.float32)
        super().__init__(N_pad, M_pad, P, steps, alpha, lr, k, device)

    def _step(self):
        VAB, dQ, dT = overlap_score_grad_esp_se3_batch(
            self.A, self.B, self.CA, self.CB, self.q, self.t,
            alpha=self.alpha, lam=self.lam, N_real=self.Nr, M_real=self.Mr)
        self._tversky_adam_tail(VAB, dQ, dT)

    def _load(self, A, B, CA, CB, Nr, Mr, C, qs, ts):
        super()._load(A, B, Nr, Mr, C, qs, ts)
        self.CA.copy_(CA); self.CB.copy_(CB)


def _run_graphed_fine_esp_tversky(A_k, B_k, CA_k, CB_k, q_seed, t_seed, N_k, M_k, C,
                                  alpha, lam, lr, k, steps, N_pad, M_pad, P,
                                  es_patience=0, es_tol=1e-5):
    key = (A_k.device.index, "vol_esp_tversky", N_pad, M_pad, P, steps,
           round(float(alpha), 4), round(float(lam), 6), round(float(lr), 5), round(float(k), 6))
    return run_graphed(
        lambda: _GraphedFineEspTversky(N_pad, M_pad, P, steps, alpha, lam, lr, k, A_k.device),
        key, (A_k, B_k, CA_k, CB_k, N_k, M_k, C, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_esp_tversky_align_many(
        A_batch, B_batch, CA_batch, CB_batch, VAA, VBB, *,
        alpha: float = 0.81,
        lam: float = 0.1,
        tversky_alpha: float = 0.95,
        tversky_beta: float = 0.05,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None,
        early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5,
        seeds: tuple | None = None):
    """Batched vol_esp_tversky alignment over a batch of (A, B) pairs.

    Mirrors ``vol_tversky.coarse_fine_align_many_tversky`` exactly (same seed set, same fine
    schedule, same Tversky reduction + CUDA-graph loop) but with the SHAPE kernel swapped for the
    ESP kernel and the shape self-overlaps swapped for the ESP self-overlaps ``VAA``/``VBB``.
    ``CA_batch``/``CB_batch`` are the per-atom partial charges; ``lam`` is RAW (atom-centred,
    matches ``vol_esp``). Because ``AA``/``BB`` are pose-invariant, maximizing ``AB`` maximizes the
    Tversky score, so the optimal pose matches ``vol_esp``; we still descend on the Tversky
    gradient (``scale = C/denom^2``) for exact parity with the eager reference.
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # Tversky reduction constants (pose-invariant); identical form to vol_tversky. With the
    # defaults (ta=0.95, tb=0.05) k == 0 so denom == C is a pure per-pair constant; the general
    # form is kept so arbitrary (ta, tb) work.
    k = 1.0 - float(tversky_alpha) - float(tversky_beta)
    C = tversky_alpha * VAA + tversky_beta * VBB                       # (BATCH,)

    # --- reference seed set (identity + 4 PCA + structured/Fibonacci), same as shape/esp ---
    if seeds is None:
        quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real,
                                             num_seeds=num_seeds)
    else:
        quats, t_seeds = seeds
    S = quats.size(1)

    # --- fine polishing on EVERY seed ---
    A_k = A_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, M_pad, 3)
    CA_k = CA_batch.unsqueeze(1).expand(-1, S, -1).reshape(-1, N_pad)
    CB_k = CB_batch.unsqueeze(1).expand(-1, S, -1).reshape(-1, M_pad)
    q_seed = quats.reshape(-1, 4).contiguous()
    t_seed = t_seeds.reshape(-1, 3).contiguous()

    N_k = N_real.repeat_interleave(S)
    M_k = M_real.repeat_interleave(S)
    C_k = C.repeat_interleave(S)                                       # per-pose constant
    P = q_seed.shape[0]

    best_score = best_q = best_t = None

    # --- CUDA-graph fast path (launch-bound regime); ESP-kernel work cap via graph_cap ---
    if (A_batch.is_cuda and P <= graph_cap(N_pad * M_pad)
            and A_batch.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_fine_esp_tversky(
                A_k.contiguous(), B_k.contiguous(), CA_k.contiguous(), CB_k.contiguous(),
                q_seed, t_seed, N_k, M_k, C_k, alpha, lam, lr, k, steps_fine, N_pad, M_pad, P,
                es_patience=early_stop_patience, es_tol=early_stop_tol)
        except Exception:
            best_score = None                                          # capture failed -> eager

    # --- eager fall-back (CPU/numba, large P, fp64, or capture failure) ---
    # NOTE: no cpu_fused fast path here -- cpu_fused_esp hardcodes the Tanimoto reduction. The
    # eager loop dispatches the SAME ESP value+grad kernel to numba on CPU tensors, so this IS
    # the numba backend for vol_esp_tversky (mirrors vol_tversky, which likewise avoids
    # cpu_fused_shape's hardcoded Tanimoto).
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
            VAB, dQ, dT = _overlap_in_chunks_esp(
                A_k, B_k, CA_k, CB_k, q_k, t_k,
                alpha=alpha, lam=lam, N_real=N_k, M_real=M_k)

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
