"""
Controlled in-process A/B of the surf fine-loop: OLD (chunk-wrapper +
torch.where best-tracking) vs NEW (direct kernel + in-place best). Same tensors,
back-to-back, interleaved, median over many reps -> isolates the Stage-1
launch/alloc reduction from cross-run contention. Pure timing (random coords;
values don't affect kernel cost).
"""
import argparse
import time
import numpy as np
import torch

from shepherd_score.score.gaussian_overlap_triton import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj,
)
from shepherd_score.alignment.utils.fast_se3 import _overlap_in_chunks


def _mk(P, N, M, dev):
    A = torch.randn(P, N, 3, device=dev)
    B = torch.randn(P, M, 3, device=dev)
    q = torch.randn(P, 4, device=dev); q /= q.norm(dim=1, keepdim=True)
    t = torch.randn(P, 3, device=dev)
    Nr = torch.full((P,), N, dtype=torch.int32, device=dev)
    Mr = torch.full((P,), M, dtype=torch.int32, device=dev)
    norm = torch.rand(P, device=dev) + 1.0
    return A, B, q, t, Nr, Mr, norm


def old_loop(A, B, q, t, Nr, Mr, norm, steps):
    qk = q.clone(); tk = t.clone()
    mq = torch.zeros_like(qk); vq = torch.zeros_like(qk)
    mt = torch.zeros_like(tk); vt = torch.zeros_like(tk)
    best = torch.full((len(qk),), -1e30, device=qk.device)
    bq = qk.clone(); bt = tk.clone()
    for _ in range(steps):
        VAB, dQ, dT = _overlap_in_chunks(A, B, qk, tk, alpha=0.81, N_real=Nr, M_real=Mr)
        denom = norm - VAB; score = VAB / denom; scale = norm / (denom * denom)
        better = score > best
        best = torch.where(better, score, best)
        m = better.unsqueeze(1)
        bq = torch.where(m, qk, bq); bt = torch.where(m, tk, bt)
        fused_adam_qt_with_tangent_proj(qk, tk, -dQ * scale.unsqueeze(1),
                                        -dT * scale.unsqueeze(1), mq, vq, mt, vt, 0.075)
    return best


def new_loop(A, B, q, t, Nr, Mr, norm, steps):
    qk = q.clone(); tk = t.clone()
    mq = torch.zeros_like(qk); vq = torch.zeros_like(qk)
    mt = torch.zeros_like(tk); vt = torch.zeros_like(tk)
    best = torch.full((len(qk),), -1e30, device=qk.device)
    bq = qk.clone(); bt = tk.clone()
    one = qk.shape[0] <= 65_535
    for _ in range(steps):
        if one:
            VAB, dQ, dT = overlap_score_grad_se3_batch(A, B, qk, tk, alpha=0.81, N_real=Nr, M_real=Mr)
        else:
            VAB, dQ, dT = _overlap_in_chunks(A, B, qk, tk, alpha=0.81, N_real=Nr, M_real=Mr)
        denom = norm - VAB; score = VAB / denom; scale = norm / (denom * denom)
        better = score > best
        bq[better] = qk[better]; bt[better] = tk[better]
        torch.maximum(best, score, out=best)
        fused_adam_qt_with_tangent_proj(qk, tk, -dQ * scale.unsqueeze(1),
                                        -dT * scale.unsqueeze(1), mq, vq, mt, vt, 0.075)
    return best


def _time(fn, args, steps, reps):
    torch.cuda.synchronize()
    for _ in range(2):
        fn(*args, steps); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(*args, steps); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--npad", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")

    print(f"FINE-LOOP A/B  (surf, N=M={args.npad}, seeds={args.seeds}, steps={args.steps}, "
          f"median/min over {args.reps} interleaved reps)")
    hdr = f'{"batch":>6s} {"P rows":>7s} | {"OLD med (ms)":>12s} {"NEW med (ms)":>12s} {"speedup(med)":>12s} {"speedup(min)":>12s}'
    print(hdr); print("-" * len(hdr))
    for batch in [16, 64, 256, 1024]:
        P = args.seeds * batch
        A, B, q, t, Nr, Mr, norm = _mk(P, args.npad, args.npad, dev)
        argt = (A, B, q, t, Nr, Mr, norm)
        # interleave to share contention
        o_med, o_min = _time(old_loop, argt, args.steps, args.reps)
        n_med, n_min = _time(new_loop, argt, args.steps, args.reps)
        print(f'{batch:6d} {P:7d} | {o_med*1e3:12.2f} {n_med*1e3:12.2f} '
              f'{o_med/n_med:11.2f}x {o_min/n_min:11.2f}x')


if __name__ == "__main__":
    raise SystemExit(main())
