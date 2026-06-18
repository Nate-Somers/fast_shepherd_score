"""
Find the best (BLOCK, num_warps) per molecule size for the surf overlap kernel,
so we can replace the heuristic with a validated occupancy-aware config. The
kernel is one CTA per pose and runs latency/occupancy-bound at large P; occupancy
is set by per-CTA resources (num_warps, BLOCK). Sweeps real band sizes.
"""
import time
import torch

from shepherd_score.score.gaussian_overlap_triton import overlap_score_grad_se3_batch


def bench(A, B, q, t, Nr, Mr, BLOCK, nw):
    for _ in range(3):
        overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Mr,
                                     BLOCK=BLOCK, num_warps=nw)
    torch.cuda.synchronize()
    ts = []
    for _ in range(8):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Mr,
                                     BLOCK=BLOCK, num_warps=nw)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    device = torch.device("cuda")
    P = 200000
    for Npad in [32, 48, 64, 80, 112]:
        N = M = Npad
        A = torch.randn(P, N, 3, device=device, dtype=torch.float32).contiguous()
        B = torch.randn(P, M, 3, device=device, dtype=torch.float32).contiguous()
        q = torch.randn(P, 4, device=device, dtype=torch.float32)
        q = (q / q.norm(dim=1, keepdim=True)).contiguous()
        t = torch.randn(P, 3, device=device, dtype=torch.float32).contiguous()
        Nr = torch.full((P,), N, dtype=torch.int32, device=device)
        Mr = torch.full((P,), M, dtype=torch.int32, device=device)
        # current default (no explicit warps) as baseline
        def base():
            for _ in range(3):
                overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Mr)
            torch.cuda.synchronize()
            ts = []
            for _ in range(8):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                overlap_score_grad_se3_batch(A, B, q, t, alpha=0.81, N_real=Nr, M_real=Mr)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            return min(ts)
        b = base()
        best = (1e9, None)
        results = []
        for BLOCK in [16, 32, 64]:
            for nw in [1, 2, 4]:
                tmin = bench(A, B, q, t, Nr, Mr, BLOCK, nw)
                results.append((tmin, BLOCK, nw))
                if tmin < best[0]:
                    best = (tmin, (BLOCK, nw))
        results.sort()
        top = results[0]
        print(f"N={Npad:3d} | default {b*1e3:7.2f}ms | best BLOCK={best[1][0]:3d} warps={best[1][1]} "
              f"{top[0]*1e3:7.2f}ms | speedup {b/top[0]:4.2f}x", flush=True)
        del A, B, q, t
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
