"""
Accuracy-bug localization for the fork's batched surf alignment.

For each real molecule we build ONE self-SE(3)-copy pair (known optimum = 1.0)
and ask: where does the fork lose the optimum the original recovers?

Because coarse_fine_align_many tracks best_score from the seed poses onward,
   fork_final >= best_coarse_tanimoto(grid)
so if fork_final is low the *coarse grid itself* missed the basin. We therefore
report, per molecule:
  n      : surface-point count (size)
  orig   : original reference score (cpu_single_torch)  -> expected ~1.0
  coarse : best Tanimoto over the baseline 500-pose grid (num_seeds=50)
  base   : fork final, baseline (ns=50, topk=30, steps=75, lr=0.075)
  +grid  : fork final, denser grid (ns=200 -> 2000 poses)
  +topk  : fork final, topk=120
  +steps : fork final, steps=300
  +all   : fork final, ns=200 + topk=120 + steps=300

This tells us whether the fix is grid density, top-k, fine steps, or all three.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.real_workloads import DRUGS, _build_molecule, _transform
from benchmarks.alignment_bench.workloads import _random_rotation
from shepherd_score.alignment.utils.fast_common import build_coarse_grid
from shepherd_score.alignment.utils.fast_se3 import (
    coarse_fine_align_many, _overlap_in_chunks, _self_overlap_in_chunks,
)
from shepherd_score.score.gaussian_overlap_triton import fused_adam_qt_with_tangent_proj


def _pair_tensors(mol, fit, device, alpha=0.81):
    A = torch.as_tensor(mol.surf_pos, dtype=torch.float32, device=device)[None]   # (1,N,3)
    B = torch.as_tensor(fit.surf_pos, dtype=torch.float32, device=device)[None]   # (1,M,3)
    N = torch.tensor([A.shape[1]], dtype=torch.int32, device=device)
    M = torch.tensor([B.shape[1]], dtype=torch.int32, device=device)
    VAA = _self_overlap_in_chunks(A, N, alpha)
    VBB = _self_overlap_in_chunks(B, M, alpha)
    return A, B, N, M, VAA, VBB


@torch.no_grad()
def best_coarse(A, B, N, M, VAA, VBB, num_seeds, alpha=0.81):
    """Best Tanimoto over the full coarse grid built with `num_seeds`."""
    q_grid, t_grid = build_coarse_grid(A, B, N, M, num_seeds=num_seeds)  # (1,G,4),(1,G,3)
    G = q_grid.shape[1]
    A_rep = A.expand(G, -1, -1).contiguous()
    B_rep = B.expand(G, -1, -1).contiguous()
    q = q_grid[0].contiguous(); t = t_grid[0].contiguous()
    Nr = N.repeat(G); Mr = M.repeat(G)
    V, _, _ = _overlap_in_chunks(A_rep, B_rep, q, t, alpha=alpha,
                                 N_real=Nr, M_real=Mr, NEED_GRAD=False)
    tani = V / (VAA + VBB - V)
    return float(tani.max())


@torch.no_grad()
def cf(A, B, N, M, VAA, VBB, *, num_seeds, topk, steps, lr=0.075, alpha=0.81):
    """Parametrised clone of coarse_fine_align_many for ONE pair, exposing
    num_seeds (grid density). Mirrors the production fine loop exactly."""
    device = A.device
    q_grid, t_grid = build_coarse_grid(A, B, N, M, num_seeds=num_seeds)
    G = q_grid.shape[1]
    A_rep = A.expand(G, -1, -1).contiguous()
    B_rep = B.expand(G, -1, -1).contiguous()
    Nr = N.repeat(G); Mr = M.repeat(G)
    V, _, _ = _overlap_in_chunks(A_rep, B_rep, q_grid[0].contiguous(), t_grid[0].contiguous(),
                                 alpha=alpha, N_real=Nr, M_real=Mr, NEED_GRAD=False)
    coarse = V / (VAA + VBB - V)                       # (G,)
    idx = coarse.topk(k=min(topk, G)).indices
    q_k = q_grid[0, idx].clone(); t_k = t_grid[0, idx].clone()
    K = q_k.shape[0]
    A_k = A.expand(K, -1, -1).contiguous(); B_k = B.expand(K, -1, -1).contiguous()
    Nk = N.repeat(K); Mk = M.repeat(K)
    VAB_norm = (VAA + VBB).repeat(K)
    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
    best = torch.full((K,), -float('inf'), device=device)
    for _ in range(steps):
        VAB, dQ, dT = _overlap_in_chunks(A_k, B_k, q_k, t_k, alpha=alpha, N_real=Nk, M_real=Mk)
        denom = VAB_norm - VAB
        score = VAB / denom
        scale = VAB_norm / (denom * denom)
        best = torch.maximum(best, score)
        fused_adam_qt_with_tangent_proj(q_k, t_k, -dQ * scale.unsqueeze(1),
                                        -dT * scale.unsqueeze(1), m_q, v_q, m_t, v_t, lr)
    return float(best.max())


def orig_score(mol, fit, device, num_repeats, max_steps, alpha=0.81):
    from shepherd_score import alignment as A
    ref = torch.as_tensor(mol.surf_pos, dtype=torch.float32, device=device)
    fitt = torch.as_tensor(fit.surf_pos, dtype=torch.float32, device=device)
    out = A.optimize_ROCS_overlay(ref, fitt, alpha, num_repeats=num_repeats,
                                  lr=0.1, max_num_steps=max_steps)
    return float(np.asarray(out[2]).reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-max-deg", type=float, default=60.0)
    ap.add_argument("--orig-repeats", type=int, default=16)
    ap.add_argument("--orig-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-orig", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    print("ACCURACY DIAGNOSTIC (surf, self-SE(3)-copy, known optimum = 1.0)")
    print(f"rot<= {args.rot_max_deg:.0f} deg | orig=cpu_single_torch "
          f"(repeats={args.orig_repeats}, steps={args.orig_steps})")
    hdr = (f'{"molecule":16s} {"n":>4s} {"orig":>7s} {"coarse":>7s} {"base":>7s} '
           f'{"+grid":>7s} {"+topk":>7s} {"+steps":>7s} {"+all":>7s}')
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, smi, heavy in DRUGS:
        mol = _build_molecule(smi)
        # one fixed random SE(3) within the rotation cap
        for _ in range(32):
            R = _random_rotation(rng)
            if (np.trace(R) - 1.0) / 2.0 >= np.cos(np.deg2rad(args.rot_max_deg)):
                break
        t = rng.standard_normal(3); t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * 3.0)
        fit = _transform(mol, R, t)

        A, B, N, M, VAA, VBB = _pair_tensors(mol, fit, device)
        n = int(N.item())
        orig = float('nan') if args.skip_orig else orig_score(
            mol, fit, device, args.orig_repeats, args.orig_steps)
        coarse = best_coarse(A, B, N, M, VAA, VBB, num_seeds=50)
        base = cf(A, B, N, M, VAA, VBB, num_seeds=50, topk=30, steps=75)
        g = cf(A, B, N, M, VAA, VBB, num_seeds=200, topk=30, steps=75)
        k = cf(A, B, N, M, VAA, VBB, num_seeds=50, topk=120, steps=75)
        s = cf(A, B, N, M, VAA, VBB, num_seeds=50, topk=30, steps=300)
        allk = cf(A, B, N, M, VAA, VBB, num_seeds=200, topk=120, steps=300)
        rows.append((name, n, orig, coarse, base, g, k, s, allk))
        print(f'{name:16s} {n:4d} {orig:7.3f} {coarse:7.3f} {base:7.3f} '
              f'{g:7.3f} {k:7.3f} {s:7.3f} {allk:7.3f}')

    arr = np.array([r[2:] for r in rows], dtype=float)
    labels = ["orig", "coarse", "base", "+grid", "+topk", "+steps", "+all"]
    print("-" * len(hdr))
    means = np.nanmean(arr, axis=0)
    mins = np.nanmin(arr, axis=0)
    print(f'{"MEAN":16s} {"":4s} ' + " ".join(f"{m:7.3f}" for m in means))
    print(f'{"MIN":16s} {"":4s} ' + " ".join(f"{m:7.3f}" for m in mins))


if __name__ == "__main__":
    raise SystemExit(main())
