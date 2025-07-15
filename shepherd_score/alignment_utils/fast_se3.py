import torch, math
from ..score.gaussian_overlap_triton import overlap_score_grad_se3_batch, fused_adam_qt
from ..alignment import objective_ROCS_overlay
from typing import Optional
from ..alignment import _initialize_se3_params as _legacy_init
from pathlib import Path
import torch.nn.functional as F
from contextlib import suppress

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


def _self_overlap_in_chunks(P_pad, N_real, *, alpha=0.81):
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


@torch.no_grad()
def _legacy_seeds_torch(ref_xyz: torch.Tensor,
                        fit_xyz: torch.Tensor,
                        *,
                        num_repeats: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the exact legacy seeds as (q,t) on the *same device* / dtype as
    `ref_xyz`.  Shapes = (num_repeats,4) and (num_repeats,3).
    """
    # legacy helper wants CPU tensors (it builds PCA etc. in Python)
    ref_cpu = ref_xyz.detach().cpu()
    fit_cpu = fit_xyz.detach().cpu()

    se3 = _legacy_init(ref_points=ref_cpu,          # <-- Torch tensors!
                       fit_points=fit_cpu,
                       num_repeats=num_repeats)     # (R,7) torch.float32 CPU

    # move back to caller’s device / dtype
    se3 = se3.to(dtype=ref_xyz.dtype, device=ref_xyz.device)
    q, t = se3[:, :4], se3[:, 4:]
    return torch.nn.functional.normalize(q, dim=1), t


def coarse_fine_align_many(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        topk: int = 30,
        steps_fine: int = 75,
        lr: float = 0.075,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None):
    """
    Vectorised padding-aware alignment over a batch of (A, B) pairs.

    Parameters
    ----------
    A_batch, B_batch : (B, N_pad, 3) / (B, M_pad, 3)  atom coordinates
    VAA, VBB         : (B,)  pre-computed Gaussian self-overlaps
    N_real, M_real   : (B,)  optional true atom counts
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) build a coarse grid of 250 rotations × 2 translations = 500 poses
    # ------------------------------------------------------------------
    qs, ts = [], []
    for i in range(BATCH):
        q_i, t_i = _legacy_seeds_torch(
            A_batch[i, :N_real[i]], B_batch[i, :M_real[i]])
        qs.append(q_i)
        ts.append(t_i)

    quats   = torch.stack(qs, dim=0)             # (B, 50, 4)
    t_seeds = torch.stack(ts, dim=0)             # (B, 50, 3)

    # π-axis flips
    def quat_mul(q, r):
        w1,x1,y1,z1 = q.unbind(-1); w2,x2,y2,z2 = r.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)

    qx = torch.tensor([0., 1., 0., 0.], device=device)
    qy = torch.tensor([0., 0., 1., 0.], device=device)
    qz = torch.tensor([0., 0., 0., 1.], device=device)
    flips  = torch.stack([qx, qy, qz, quat_mul(qx, qy)], 0)  # (4,4)

    q_base = quats.reshape(-1, 4)
    q_base = torch.cat(
        [q_base,
         quat_mul(flips[:, None], q_base[None]).reshape(-1, 4)],
        dim=0).view(BATCH, -1, 4)                            # (B, 250, 4)

    # two translations per pair: COM→COM and tip→COM
    com_trans = t_seeds[:, :1, :]                            # (B,1,3)
    tips      = A_batch[torch.arange(BATCH),
                        A_batch.norm(dim=2).argmax(dim=1)]   # (B,3)
    extra_t   = (tips - B_batch.mean(1)).unsqueeze(1)        # (B,1,3)
    t_base    = torch.cat([com_trans, extra_t], dim=1)       # (B,2,3)

    # Cartesian product
    q_grid = q_base[:, :, None, :].expand(-1, -1, 2, -1).reshape(BATCH, -1, 4)
    t_grid = t_base[:, None, :, :].expand(-1, 250, -1, -1).reshape(BATCH, -1, 3)
    G = q_grid.size(1)                                       # 500 orientations

    # ------------------------------------------------------------------
    # 2) coarse evaluation (micro-batched for memory)
    # ------------------------------------------------------------------
    ORI_CHUNK  = 25_000
    PAIR_CHUNK = 4_096
    coarse_score = torch.empty(BATCH, G, device=device, dtype=A_batch.dtype)

    for o0 in range(0, G, ORI_CHUNK):
        o1 = min(o0 + ORI_CHUNK, G); g_len = o1 - o0
        for p0 in range(0, BATCH, PAIR_CHUNK):
            p1 = min(p0 + PAIR_CHUNK, BATCH); slice_len = p1 - p0

            A_rep = A_batch[p0:p1].unsqueeze(1).expand(-1, g_len, -1, -1)\
                                   .reshape(-1, N_pad, 3)
            B_rep = B_batch[p0:p1].unsqueeze(1).expand(-1, g_len, -1, -1)\
                                   .reshape(-1, M_pad, 3)
            q_rep = q_grid[p0:p1, o0:o1].reshape(-1, 4).contiguous()
            t_rep = t_grid[p0:p1, o0:o1].reshape(-1, 3).contiguous()

            N_rep = N_real[p0:p1].repeat_interleave(g_len)
            M_rep = M_real[p0:p1].repeat_interleave(g_len)

            VAB_slice, _, _ = _overlap_in_chunks(
                A_rep, B_rep, q_rep, t_rep,
                alpha=alpha, N_real=N_rep, M_real=M_rep, NEED_GRAD=False)

            coarse_score[p0:p1, o0:o1] = VAB_slice.view(slice_len, g_len)

    coarse_score = coarse_score / (VAA[:, None] + VBB[:, None] - coarse_score)

    # ------------------------------------------------------------------
    # 3) select top-k orientations
    # ------------------------------------------------------------------
    best_idx = coarse_score.topk(k=topk, dim=1).indices      # (B, topk)
    q_best   = torch.gather(q_grid, 1,
                            best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
    t_best   = torch.gather(t_grid, 1,
                            best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()

    # ------------------------------------------------------------------
    # 4) fine polishing (Adam-like) on the top-k
    # ------------------------------------------------------------------
    A_k = A_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, N_pad, 3)
    B_k = B_batch.unsqueeze(1).expand(-1, topk, -1, -1).reshape(-1, M_pad, 3)
    q_k = q_best.view(-1, 4)
    t_k = t_best.view(-1, 3)

    N_k = N_real.repeat_interleave(topk)
    M_k = M_real.repeat_interleave(topk)
    VAA_rep = VAA.repeat_interleave(topk)
    VBB_rep = VBB.repeat_interleave(topk)

    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q     = torch.empty_like(q_k)
    best_t     = torch.empty_like(t_k)

    for _ in range(steps_fine):
        VAB, dQ, dT = _overlap_in_chunks(
            A_k, B_k, q_k, t_k,
            alpha=alpha, N_real=N_k, M_real=M_k)
        
        denom  = VAA_rep + VBB_rep - VAB
        score  = VAB / denom
        scale  = (VAA_rep + VBB_rep) / (denom * denom)

        # ----- build tangent-space gradient once -----
        radial = (dQ * q_k).sum(dim=1, keepdim=True)
        dQ_tan = dQ - q_k * radial          # (K,4)

        better = score > best_score
        if better.any():
            best_score[better] = score[better]
            best_q[better]     = q_k[better]
            best_t[better]     = t_k[better]

        fused_adam_qt(q_k, t_k,
                      -dQ_tan * scale[:, None],
                      -dT * scale[:, None],
                      m_q, v_q, m_t, v_t, lr)

    # ------------------------------------------------------------------
    # 5) gather final results
    # ------------------------------------------------------------------
    final_VAB, _, _ = _overlap_in_chunks(
        A_k, B_k, q_k, t_k, alpha=alpha, N_real=N_k, M_real=M_k)

    final_score = final_VAB / (
        VAA.repeat_interleave(topk) +
        VBB.repeat_interleave(topk) - final_VAB)
    final_score = final_score.view(BATCH, topk)

    best = final_score.argmax(dim=1)
    sel  = best + torch.arange(BATCH, device=device) * topk

    return final_score.flatten()[sel], \
           q_k.view(BATCH, topk, 4)[torch.arange(BATCH), best], \
           t_k.view(BATCH, topk, 3)[torch.arange(BATCH), best]


















# ######################################################################
# # [DEBUG] 1-shot gradient parity check (analytical vs. autograd)
# # ------------------------------------------------------------------
# if _ == 0:
#     # pick the very first orientation in this micro-batch
#     test_idx = 0

#     # detach copies so we can re-enable gradient tracking
#     realN = int(N_k[test_idx])                 # true atom counts (kernel)
#     realM = int(M_k[test_idx])

#     A_real = A_k[test_idx, :realN].detach()    # (realN,3)
#     B_real = B_k[test_idx, :realM].detach()    # (realM,3)

#     q_aut = q_k[test_idx:test_idx+1].detach().clone()
#     t_aut = t_k[test_idx:test_idx+1].detach().clone()

#     # autograd needs a single (7,) tensor:  [q (4) | t (3)]
#     se3_aut = torch.cat([q_aut[0], t_aut[0]]).requires_grad_(True)


#     # legacy objective returns (1-Tanimoto) –>  score = 1-loss
#     loss = objective_ROCS_overlay(
#         se3_params = se3_aut,
#         ref_points = A_real,         # (N,3)
#         fit_points = B_real,         # (M,3)
#         alpha      = alpha)
#     score = 1.0 - loss
#     score.backward()                              # ↯ autograd grads
#     grads    = se3_aut.grad           # (7,)
#     dQ_auto  = grads[:4]              # (4,)
#     dT_auto  = grads[4:]              # (3,)


#     # analytical score-gradients  (kernel gives dVAB)
#     scale_aut   = scale[test_idx]                 # scalar
#     dQ_kernel   =  dQ_tan[test_idx] * scale_aut
#     dT_kernel   =  dT[test_idx] * scale_aut

#     # compare
#     Δq = (dQ_kernel - dQ_auto).abs().max()
#     Δt = (dT_kernel - dT_auto).abs().max()

#     q_unit   = q_k[test_idx] / q_k[test_idx].norm()     # ensure exactly unit
#     P        = torch.eye(4, device=q_unit.device) - q_unit[:,None] @ q_unit[None,:]
#     dQ_proj  = P @ dQ_kernel    # project kernel grad into tangential plane

#     # -------------------------------------------------------------- NEW
#     #   ➊ How many atoms are real vs. padding?
#     realN = int(N_k[test_idx].item())    # true atom counts you passed to the kernel
#     realM = int(M_k[test_idx].item())
#     print(f"[padding]  real N={realN}/{N_pad}   real M={realM}/{M_pad}")

#     #   ➋ Does either gradient carry a radial-(‖q‖) component?
#     print(f"[param]    ‖q‖={q_unit.norm():.6f}   "
#           f"q·kern={torch.dot(q_unit, dQ_kernel):+.3e}   "
#           f"q·auto={torch.dot(q_unit, dQ_auto):+.3e}")
#     # -------------------------------------------------------------- END

#     print("‖raw kernel grad‖ :", dQ_kernel.norm().item())
#     print("‖projected grad‖ :", dQ_proj.norm().item())
#     print("‖autograd grad‖  :", dQ_auto.norm().item())
#     print("dot(q, autograd) :", torch.dot(q_unit, dQ_auto).item())   # ~0

#     # ---- print ----------------------------------------------
#     names = ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]
#     auto_full   = torch.cat([dQ_auto,  dT_auto])
#     kern_full   = torch.cat([dQ_kernel, dT_kernel])
#     delta_full  = kern_full - auto_full

#     print("\n[grad-check] component   kernel_grad         autograd_grad        Δ")
#     for n, k, a, d in zip(names, kern_full, auto_full, delta_full):
#         print(f"                 {n:>2} :    {k:+.6e}       {a:+.6e}    {d:+.1e}")
#     print()  # blank line for clarity

#     thresh = 1e-4          # adjust if you need tighter tolerance
#     if Δq > thresh or Δt > thresh:
#         raise AssertionError(
#             f"[grad-check] mismatch —  "
#             f"dQ Δ={Δq.item():.3e}, dT Δ={Δt.item():.3e}")
#     else:
#         print(f"[grad-check] analytical == autograd  ✓ "
#               f"(max‖Δ‖: dQ {Δq:.2e}, dT {Δt:.2e})")
#     ####################################################################