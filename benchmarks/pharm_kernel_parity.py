"""
STRICT gradient-parity gate for the Triton pharm value+grad kernel.

Compares pharm_score_grad_se3_batch (Triton) against the analytical torch
compute_overlap_and_grad_pharm (oracle) on (O_AB, grad_R, grad_t) across sizes,
seeds, type distributions, and edge cases. Relative tolerance 1e-3 (float32).
The kernel must pass ALL before it is allowed near the fine loop.
"""
import torch

from shepherd_score.score.analytical_gradients._torch import (
    compute_overlap_and_grad_pharm, build_lookup_tables, _rotation_matrix_from_unit_quat)
from shepherd_score.score.pharmacophore_grad_triton import pharm_score_grad_se3_batch

TOL = 1e-3


def _rel(a, b):
    d = (a - b).abs().max().item()
    s = max(a.abs().max().item(), 1.0)
    return d, d / s


def check(name, P, N, M, seed, T, vec_scale=1.0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(P, 4, device=dev, generator=g); q = q / q.norm(dim=1, keepdim=True)
    R = _rotation_matrix_from_unit_quat(q)
    t = torch.randn(P, 3, device=dev, generator=g)
    ra = torch.randn(P, N, 3, device=dev, generator=g)
    fa = torch.randn(P, M, 3, device=dev, generator=g)
    rv = vec_scale * torch.randn(P, N, 3, device=dev, generator=g)
    fv = vec_scale * torch.randn(P, M, 3, device=dev, generator=g)
    rt = torch.randint(0, T, (P, N), device=dev, generator=g)
    ft = torch.randint(0, T, (P, M), device=dev, generator=g)
    alphas, Ks, cats = build_lookup_tables(dev, torch.float32)

    O0, gR0, gt0 = compute_overlap_and_grad_pharm(R, t, rt, ft, ra, fa, rv, fv, extended_points=False)
    O1, gR1, gt1 = pharm_score_grad_se3_batch(R, t, rt, ft, ra, fa, rv, fv, alphas, Ks, cats)

    dO, rO = _rel(O0, O1)
    dgR, rgR = _rel(gR0, gR1)
    dgt, rgt = _rel(gt0, gt1)
    ok = rO < TOL and rgR < TOL and rgt < TOL
    print(f"{name:18s} P={P:5d} N={N:2d} M={M:2d} | "
          f"O {rO:.1e} gR {rgR:.1e} gt {rgt:.1e} | {'PASS' if ok else 'FAIL <<<'}")
    return ok


def check_pad(name, P, realN, padN, realM, padM, seed, T):
    """Kernel on PADDED tensors (random garbage in padding, N_real/M_real set) must
    equal the analytical on the UNPADDED real slice -- validates the mi/mj masking."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(P, 4, device=dev, generator=g); q = q / q.norm(dim=1, keepdim=True)
    R = _rotation_matrix_from_unit_quat(q)
    t = torch.randn(P, 3, device=dev, generator=g)
    ra = torch.randn(P, padN, 3, device=dev, generator=g)
    fa = torch.randn(P, padM, 3, device=dev, generator=g)
    rv = torch.randn(P, padN, 3, device=dev, generator=g)
    fv = torch.randn(P, padM, 3, device=dev, generator=g)
    rt = torch.randint(0, T, (P, padN), device=dev, generator=g)
    ft = torch.randint(0, T, (P, padM), device=dev, generator=g)
    alphas, Ks, cats = build_lookup_tables(dev, torch.float32)
    Nr = torch.full((P,), realN, device=dev, dtype=torch.int32)
    Mr = torch.full((P,), realM, device=dev, dtype=torch.int32)
    Ok, gRk, gtk = pharm_score_grad_se3_batch(R, t, rt, ft, ra, fa, rv, fv, alphas, Ks, cats,
                                              N_real=Nr, M_real=Mr)
    Oa, gRa, gta = compute_overlap_and_grad_pharm(
        R, t, rt[:, :realN], ft[:, :realM], ra[:, :realN], fa[:, :realM],
        rv[:, :realN], fv[:, :realM], extended_points=False)
    _, rO = _rel(Oa, Ok); _, rgR = _rel(gRa, gRk); _, rgt = _rel(gta, gtk)
    ok = rO < TOL and rgR < TOL and rgt < TOL
    print(f"{name:18s} P={P:5d} pad {realN}/{padN},{realM}/{padM} | "
          f"O {rO:.1e} gR {rgR:.1e} gt {rgt:.1e} | {'PASS' if ok else 'FAIL <<<'}")
    return ok


def main():
    dev = torch.device("cuda")
    alphas, _, cats = build_lookup_tables(dev, torch.float32)
    T = alphas.shape[0]
    print(f"#types={T}, cats unique={sorted(set(cats.tolist()))}  (3=dummy)")
    ok = True
    ok &= check("base",          2000, 12, 12, 0, T)
    ok &= check("base-seed1",    2000, 12, 12, 1, T)
    ok &= check("base-seed2",    2000, 12, 12, 2, T)
    ok &= check("diff-N-M",      1500,  8, 14, 3, T)
    ok &= check("small",          800,  3,  5, 4, T)
    ok &= check("band16",        3000, 16, 16, 5, T)
    ok &= check("band32",        1000, 28, 28, 6, T)
    ok &= check("degenerate-vec",1000, 10, 10, 7, T, vec_scale=1e-9)
    ok &= check("large-vec",     1000, 10, 10, 8, T, vec_scale=50.0)
    # padding masking (real < pad), both ref and fit
    ok &= check_pad("pad-ref",   1500, 9, 16, 16, 16, 10, T)
    ok &= check_pad("pad-fit",   1500, 16, 16, 7, 16, 11, T)
    ok &= check_pad("pad-both",  1500, 5, 16, 11, 16, 12, T)
    print("\n=== " + ("ALL PASS" if ok else "SOME FAILED") + " ===")


if __name__ == "__main__":
    raise SystemExit(main())
