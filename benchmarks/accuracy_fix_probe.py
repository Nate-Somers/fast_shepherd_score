"""
Compare candidate fixes for the coarse-pruning accuracy bug, batched per
molecule over many random SE(3) transforms (known optimum = 1.0), reporting both
score recovery (mean / worst drop) and relative wall-time.

Variants (all share the SAME coarse grid: num_seeds=50 -> 500 poses):
  base      : topk=30, steps=75                       (current production)
  topk64    : topk=64, steps=75                       (brute-force: keep more)
  topk120   : topk=120, steps=75
  warmrank  : coarse->top-64, warm 12 steps, rerank by warm score->top-30,
              then 63 deep steps                       (rank by potential)

Drop = 1.0 - fork_score (self-copy optimum is exactly 1.0).
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.real_workloads import DRUGS, _build_molecule, _transform
from benchmarks.alignment_bench.workloads import _random_rotation
from shepherd_score.alignment.utils.fast_common import build_coarse_grid, batched_seeds_torch
from shepherd_score.alignment.utils.fast_se3 import _overlap_in_chunks, _self_overlap_in_chunks
from shepherd_score.score.gaussian_overlap_triton import fused_adam_qt_with_tangent_proj


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def seeds_all_batch(A, B, N, M, VAA, VBB, *, num_seeds, total_steps, lr=0.075, alpha=0.81):
    """No coarse grid, no pruning: build the raw seed set (identity + PCA +
    Fibonacci, COM translation) exactly like the original reference, then
    fine-optimize ALL of them and take the per-pair max. Memory ~ num_seeds
    poses (not 500), so it stays safe at large batch."""
    device = A.device
    K = A.shape[0]
    q_s, t_s = batched_seeds_torch(A, B, N, M, num_seeds=num_seeds)   # (K,S,4),(K,S,3)
    S = q_s.shape[1]
    q_k = q_s.reshape(-1, 4).clone(); t_k = t_s.reshape(-1, 3).clone()
    A_k = A.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, A.shape[1], 3)
    B_k = B.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, B.shape[1], 3)
    Nk = N.repeat_interleave(S); Mk = M.repeat_interleave(S)
    norm = (VAA + VBB).repeat_interleave(S)
    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
    best = torch.full((q_k.shape[0],), -float('inf'), device=device)
    for _ in range(total_steps):
        VAB, dQ, dT = _overlap_in_chunks(A_k, B_k, q_k, t_k, alpha=alpha, N_real=Nk, M_real=Mk)
        denom = norm - VAB
        best = torch.maximum(best, VAB / denom)
        scale = norm / (denom * denom)
        fused_adam_qt_with_tangent_proj(q_k, t_k, -dQ * scale.unsqueeze(1),
                                        -dT * scale.unsqueeze(1), m_q, v_q, m_t, v_t, lr)
    return best.view(K, S).max(dim=1).values


@torch.no_grad()
def cf_batch(A, B, N, M, VAA, VBB, *, topk_coarse, warm_steps, topk_fine,
             total_steps, lr=0.075, alpha=0.81):
    """Batched coarse->(warm rerank)->deep fine over a single-bucket cohort.

    A:(K,N,3) B:(K,M,3) N,M:(K,) VAA,VBB:(K,). Returns best score per pair (K,).
    warm_steps==0 disables reranking (plain top-k = production behaviour).
    """
    device = A.device
    K = A.shape[0]
    q_grid, t_grid = build_coarse_grid(A, B, N, M, num_seeds=50)   # (K,G,4),(K,G,3)
    G = q_grid.shape[1]

    # ---- coarse eval (no grad) ----
    A_rep = A.unsqueeze(1).expand(-1, G, -1, -1).reshape(-1, A.shape[1], 3)
    B_rep = B.unsqueeze(1).expand(-1, G, -1, -1).reshape(-1, B.shape[1], 3)
    Nr = N.repeat_interleave(G); Mr = M.repeat_interleave(G)
    V, _, _ = _overlap_in_chunks(A_rep, B_rep, q_grid.reshape(-1, 4).contiguous(),
                                 t_grid.reshape(-1, 3).contiguous(),
                                 alpha=alpha, N_real=Nr, M_real=Mr, NEED_GRAD=False)
    coarse = (V / (VAA.repeat_interleave(G) + VBB.repeat_interleave(G) - V)).view(K, G)

    tc = topk_coarse
    idx = coarse.topk(k=min(tc, G), dim=1).indices                  # (K,tc)
    q_k = torch.gather(q_grid, 1, idx.unsqueeze(-1).expand(-1, -1, 4)).reshape(-1, 4).clone()
    t_k = torch.gather(t_grid, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).reshape(-1, 3).clone()

    def fine(q_k, t_k, Kc, nsteps, A, B, N, M, VAA, VBB):
        A_k = A.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(-1, A.shape[1], 3)
        B_k = B.unsqueeze(1).expand(-1, Kc, -1, -1).reshape(-1, B.shape[1], 3)
        Nk = N.repeat_interleave(Kc); Mk = M.repeat_interleave(Kc)
        norm = (VAA + VBB).repeat_interleave(Kc)
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
        best = torch.full((q_k.shape[0],), -float('inf'), device=device)
        cur = best
        for _ in range(nsteps):
            VAB, dQ, dT = _overlap_in_chunks(A_k, B_k, q_k, t_k, alpha=alpha, N_real=Nk, M_real=Mk)
            denom = norm - VAB
            cur = VAB / denom
            scale = norm / (denom * denom)
            best = torch.maximum(best, cur)
            fused_adam_qt_with_tangent_proj(q_k, t_k, -dQ * scale.unsqueeze(1),
                                            -dT * scale.unsqueeze(1), m_q, v_q, m_t, v_t, lr)
        return q_k, t_k, best, cur

    if warm_steps <= 0:
        _, _, best, _ = fine(q_k, t_k, tc, total_steps, A, B, N, M, VAA, VBB)
        return best.view(K, tc).max(dim=1).values

    # warm pass over tc candidates
    q_k, t_k, best_w, cur_w = fine(q_k, t_k, tc, warm_steps, A, B, N, M, VAA, VBB)
    # rerank by best-so-far within each pair, keep topk_fine
    best_w2 = best_w.view(K, tc)
    sel = best_w2.topk(k=min(topk_fine, tc), dim=1).indices           # (K,tf)
    flat = (sel + torch.arange(K, device=device).unsqueeze(1) * tc).reshape(-1)
    q_d = q_k[flat].clone(); t_d = t_k[flat].clone()
    keep_best = best_w.view(K, tc).gather(1, sel).reshape(-1)
    _, _, best_d, _ = fine(q_d, t_d, sel.shape[1], total_steps - warm_steps,
                           A, B, N, M, VAA, VBB)
    best_d = torch.maximum(best_d, keep_best)
    return best_d.view(K, sel.shape[1]).max(dim=1).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", type=int, default=12)
    ap.add_argument("--rot-max-deg", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    # ("grid"|"seeds", kwargs). 'grid' = 500-pose coarse grid + topk fine.
    # 'seeds' = raw seed set, no coarse, all fine-optimized (memory-light).
    variants = {
        "base":      ("grid",  dict(topk_coarse=30,  warm_steps=0, topk_fine=30,  total_steps=75,  lr=0.075)),
        "full500":   ("grid",  dict(topk_coarse=500, warm_steps=0, topk_fine=500, total_steps=100, lr=0.075)),
        "seeds50":   ("seeds", dict(num_seeds=50,  total_steps=100, lr=0.075)),
        "seeds100":  ("seeds", dict(num_seeds=100, total_steps=100, lr=0.075)),
        "seeds250":  ("seeds", dict(num_seeds=250, total_steps=100, lr=0.075)),
    }
    drops = {k: [] for k in variants}
    times = {k: 0.0 for k in variants}

    print(f"FIX PROBE: {args.transforms} transforms/molecule, rot<={args.rot_max_deg:.0f}deg, "
          f"drop=1-score, time=min over {args.reps} reps")
    hdr = f'{"molecule":16s} {"n":>4s} ' + " ".join(f'{k:>9s}' for k in variants)
    print(hdr); print("-" * len(hdr))

    for name, smi, heavy in DRUGS:
        mol = _build_molecule(smi)
        As, Bs = [], []
        for _ in range(args.transforms):
            for _ in range(32):
                R = _random_rotation(rng)
                if (np.trace(R) - 1.0) / 2.0 >= np.cos(np.deg2rad(args.rot_max_deg)):
                    break
            t = rng.standard_normal(3); t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * 3.0)
            fit = _transform(mol, R, t)
            As.append(mol.surf_pos); Bs.append(fit.surf_pos)
        A = torch.as_tensor(np.stack(As), dtype=torch.float32, device=device)
        B = torch.as_tensor(np.stack(Bs), dtype=torch.float32, device=device)
        K = A.shape[0]
        N = torch.full((K,), A.shape[1], dtype=torch.int32, device=device)
        M = torch.full((K,), B.shape[1], dtype=torch.int32, device=device)
        VAA = _self_overlap_in_chunks(A, N); VBB = _self_overlap_in_chunks(B, M)

        row = []
        for k, (kind, kw) in variants.items():
            fn = seeds_all_batch if kind == "seeds" else cf_batch
            # warmup
            s = fn(A, B, N, M, VAA, VBB, **kw); _sync()
            best_t = float("inf")
            for _ in range(args.reps):
                _sync(); t0 = time.perf_counter()
                s = fn(A, B, N, M, VAA, VBB, **kw); _sync()
                best_t = min(best_t, time.perf_counter() - t0)
            d = (1.0 - s).clamp(min=0).cpu().numpy()
            drops[k].extend(d.tolist())
            times[k] += best_t
            row.append(f"{float(d.mean()):9.4f}")
        print(f'{name:16s} {A.shape[1]:4d} ' + " ".join(row))

    print("-" * len(hdr))
    print("OVERALL  mean_drop / worst_drop / total_time(s) / rel_time:")
    base_t = times["base"]
    for k in variants:
        dd = np.array(drops[k])
        print(f"  {k:10s} mean={dd.mean():.4f}  worst={dd.max():.4f}  "
              f"time={times[k]:.3f}s  rel={times[k]/base_t:.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
