"""
Can pharm's fine-loop value+grad go on Triton via torch.compile (inductor)?

The analytical pharm value+grad (compute_overlap_and_grad_pharm) is many small
torch ops per step -> launch-bound, like everything else. torch.compile fuses it
into generated Triton kernels with the SAME math (so parity is guaranteed). This
A/Bs eager vs compiled on representative pose/feature counts and checks outputs.
"""
import time
import torch

from shepherd_score.score.analytical_gradients._torch import (
    compute_overlap_and_grad_pharm, _rotation_matrix_from_unit_quat)


def main():
    device = torch.device("cuda")
    P, N = 10000, 16     # poses (batch 200 x 50 seeds), pharm features padded to a band
    torch.manual_seed(0)
    q = torch.randn(P, 4, device=device); q = q / q.norm(dim=1, keepdim=True)
    R = _rotation_matrix_from_unit_quat(q)
    t = torch.randn(P, 3, device=device)
    ty1 = torch.randint(0, 7, (P, N), device=device)
    ty2 = torch.randint(0, 7, (P, N), device=device)
    a1 = torch.randn(P, N, 3, device=device); a2 = torch.randn(P, N, 3, device=device)
    v1 = torch.randn(P, N, 3, device=device); v2 = torch.randn(P, N, 3, device=device)

    def run(fn):
        return fn(R, t, ty1, ty2, a1, a2, v1, v2, extended_points=False)

    eager = compute_overlap_and_grad_pharm
    comp = torch.compile(compute_overlap_and_grad_pharm)

    oe, gRe, gte = run(eager); torch.cuda.synchronize()
    oc, gRc, gtc = run(comp); torch.cuda.synchronize()   # triggers compile
    print(f"parity: |dO|={ (oe-oc).abs().max().item():.2e} "
          f"|dgR|={ (gRe-gRc).abs().max().item():.2e} "
          f"|dgt|={ (gte-gtc).abs().max().item():.2e}")

    def timeit(fn, reps=30):
        for _ in range(8):
            run(fn)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run(fn); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    te = timeit(eager)
    tc = timeit(comp)
    print(f"P={P} N={N} | eager {te*1e3:7.3f}ms | compiled {tc*1e3:7.3f}ms | speedup {te/tc:.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
