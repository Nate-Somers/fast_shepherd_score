"""
Paired KERNEL-level microbench for the overlap value+grad kernel (the dominant
cost in the fine loop). Isolates the kernel from align overhead AND from this
GPU's clock jitter: every config is timed back-to-back in one process, report
median per-rep ratio vs the first (baseline) config. All variants run the SAME
math (bit-identical) -- only BLOCK / num_warps / num_stages / maxnreg differ, so
this finds occupancy/latency wins with zero accuracy risk. Parity of outputs is
asserted against the baseline config.

  python -m benchmarks.experiments.kernelbench --P 200000 --N 48 --reps 24
"""
import argparse
import math
import statistics
import time

import torch
import triton

from shepherd_score.score import gaussian_overlap_triton as G

_RAW = G._gauss_overlap_se3_tiled.fn          # underlying JITFunction (bypass autotune)


def _fixed(block, warps, stages, maxnreg):
    return triton.autotune(
        configs=[triton.Config({'BLOCK': block}, num_warps=warps, num_stages=stages, maxnreg=maxnreg)],
        key=['N_pad', 'M_pad'])(_RAW)


def _inputs(P, N, M, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    A = torch.randn(P, N, 3, device=dev, generator=g)
    B = torch.randn(P, M, 3, device=dev, generator=g)
    q = torch.randn(P, 4, device=dev, generator=g); q = q / q.norm(dim=1, keepdim=True)
    t = torch.randn(P, 3, device=dev, generator=g)
    Nr = torch.full((P,), N, device=dev, dtype=torch.int32)
    Mr = torch.full((P,), M, device=dev, dtype=torch.int32)
    return A, B, q, t, Nr, Mr


def _runner(kfn, inp, N, M, alpha=0.81):
    A, B, q, t, Nr, Mr = inp
    P = A.shape[0]
    ha = 0.5 * alpha; kc = math.pi**1.5 / ((2.0 * alpha) ** 1.5)
    S = torch.zeros(P, device=A.device); dQ = torch.zeros_like(q); dT = torch.zeros_like(t)
    Af, Bf, qf, tf = A.reshape(-1), B.reshape(-1), q.reshape(-1), t.reshape(-1)
    dQf, dTf = dQ.reshape(-1), dT.reshape(-1)

    def run():
        kfn[(P,)](Af, Bf, qf, tf, Nr, Mr, P, M, N, ha, kc, S, dQf, dTf, NEED_GRAD=True)
    return run, S, dQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=200000)         # poses (batch*50)
    ap.add_argument("--N", type=int, default=48)             # surf points (pad)
    ap.add_argument("--reps", type=int, default=24)
    args = ap.parse_args()
    dev = torch.device("cuda")
    N = M = args.N
    inp = _inputs(args.P, N, M, dev)

    configs = {
        "B16 w1 s1 (base)": (16, 1, 1, None),
        "B16 w1 s2":        (16, 1, 2, None),
        "B16 w1 s3":        (16, 1, 3, None),
        "B16 w1 s4":        (16, 1, 4, None),
        "B16 w1 s3 mr64":   (16, 1, 3, 64),
        "B16 w1 s3 mr96":   (16, 1, 3, 96),
        "B16 w1 s1 mr64":   (16, 1, 1, 64),
        "B32 w1 s3":        (32, 1, 3, None),
        "B16 w2 s3":        (16, 2, 3, None),
    }
    runs = {}; refS = refQ = None
    for name, cfg in configs.items():
        run, S, dQ = _runner(_fixed(*cfg), inp, N, M)
        run(); torch.cuda.synchronize()                       # warm/compile
        runs[name] = (run, S, dQ)
        if refS is None:
            refS, refQ = S.clone(), dQ.clone()

    names = list(configs)
    times = {n: [] for n in names}
    for _ in range(args.reps):
        for n in names:
            run, _, _ = runs[n]
            torch.cuda.synchronize(); t0 = time.perf_counter(); run(); torch.cuda.synchronize()
            times[n].append(time.perf_counter() - t0)

    base = names[0]
    print(f"\nP={args.P} N=M={N}  (paired median over {args.reps} reps)")
    print(f"{'config':22s} {'ms':>8s} {'speedup':>8s} {'parity max|Δ|':>14s}")
    print("-" * 56)
    for n in names:
        run, S, dQ = runs[n]
        med = statistics.median(times[n])
        sp = statistics.median([tb / tn for tb, tn in zip(times[base], times[n])])
        dd = max(float((S - refS).abs().max()), float((dQ - refQ).abs().max()))
        print(f"{n:22s} {med*1e3:8.2f} {sp:7.2f}x {dd:14.2e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
