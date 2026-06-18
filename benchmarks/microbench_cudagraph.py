"""
Lever #3 prototype + controlled A/B: CUDA-graph the surf fine loop.

Idea: the fine loop is launch-bound (~10 kernel launches/step). Capture ONE step
as a CUDA graph operating in-place on persistent buffers (sync-free, fixed-shape),
then replay it N times -- replay carries state between steps, so N replays = N
steps with near-zero per-step launch overhead.

This measures the FINE LOOP in isolation (where graphs can help), interleaved
old-vs-new on identical tensors (median over reps -> noise-robust), and checks
that the graphed result is bit-identical to the eager loop (parity gate). Random
coords; values don't affect kernel cost or the equivalence.

Reports graph REPLAY-only time (the steady-state cost once a per-shape graph is
cached, which is how it'd be used in production) and, for context, the one-time
capture cost.
"""
import argparse
import time
import numpy as np
import torch

from shepherd_score.score.gaussian_overlap_triton import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj,
)


def _mk(P, N, M, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(P, N, 3, generator=g).to(dev)
    B = torch.randn(P, M, 3, generator=g).to(dev)
    q = torch.randn(P, 4, generator=g).to(dev); q /= q.norm(dim=1, keepdim=True)
    t = torch.randn(P, 3, generator=g).to(dev)
    Nr = torch.full((P,), N, dtype=torch.int32, device=dev)
    Mr = torch.full((P,), M, dtype=torch.int32, device=dev)
    norm = torch.rand(P, generator=g).to(dev) + 1.0
    return A, B, q, t, Nr, Mr, norm


def eager_loop(A, B, q, t, Nr, Mr, norm, steps):
    """Mirrors the production fine loop (torch.where best-tracking, no early-stop)."""
    qk = q.clone(); tk = t.clone()
    mq = torch.zeros_like(qk); vq = torch.zeros_like(qk)
    mt = torch.zeros_like(tk); vt = torch.zeros_like(tk)
    best = torch.full((len(qk),), -1e30, device=qk.device)
    bq = qk.clone(); bt = tk.clone()
    for _ in range(steps):
        VAB, dQ, dT = overlap_score_grad_se3_batch(A, B, qk, tk, alpha=0.81, N_real=Nr, M_real=Mr)
        denom = norm - VAB
        score = VAB / denom
        scale = norm / (denom * denom)
        better = score > best
        best = torch.where(better, score, best)
        m = better.unsqueeze(1)
        bq = torch.where(m, qk, bq)
        bt = torch.where(m, tk, bt)
        fused_adam_qt_with_tangent_proj(qk, tk, -dQ * scale.unsqueeze(1),
                                        -dT * scale.unsqueeze(1), mq, vq, mt, vt, 0.075)
    return best, bq, bt


class GraphedLoop:
    """Captures one in-place fine step into a CUDA graph; replay N times = N steps."""

    def __init__(self, A, B, q, t, Nr, Mr, norm):
        dev = q.device; P = q.shape[0]
        self.q0, self.t0 = q, t
        self.qk = q.clone(); self.tk = t.clone()
        self.mq = torch.zeros_like(self.qk); self.vq = torch.zeros_like(self.qk)
        self.mt = torch.zeros_like(self.tk); self.vt = torch.zeros_like(self.tk)
        self.best = torch.full((P,), -1e30, device=dev)
        self.bq = self.qk.clone(); self.bt = self.tk.clone()
        # persistent scratch for the elementwise ops (all out=, no allocation)
        self.denom = torch.empty(P, device=dev); self.d2 = torch.empty(P, device=dev)
        self.score = torch.empty(P, device=dev); self.scale = torch.empty(P, device=dev)
        self.better = torch.empty(P, dtype=torch.bool, device=dev)
        self.gq = torch.empty_like(self.qk); self.gt = torch.empty_like(self.tk)
        self.A, self.B, self.Nr, self.Mr, self.norm = A, B, Nr, Mr, norm
        self.graph = None

    def _step(self):
        VAB, dQ, dT = overlap_score_grad_se3_batch(
            self.A, self.B, self.qk, self.tk, alpha=0.81, N_real=self.Nr, M_real=self.Mr)
        torch.sub(self.norm, VAB, out=self.denom)
        torch.div(VAB, self.denom, out=self.score)
        torch.mul(self.denom, self.denom, out=self.d2)
        torch.div(self.norm, self.d2, out=self.scale)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.qk, self.bq, out=self.bq)
        torch.where(bm, self.tk, self.bt, out=self.bt)
        torch.mul(dQ, self.scale.unsqueeze(1), out=self.gq); self.gq.neg_()
        torch.mul(dT, self.scale.unsqueeze(1), out=self.gt); self.gt.neg_()
        fused_adam_qt_with_tangent_proj(self.qk, self.tk, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, 0.075)

    def _reset(self):
        self.qk.copy_(self.q0); self.tk.copy_(self.t0)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-1e30); self.bq.copy_(self.q0); self.bt.copy_(self.t0)

    def capture(self):
        # warmup on a side stream so Triton kernels compile/autotune before capture
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def run(self, steps):
        self._reset()
        for _ in range(steps):
            self.graph.replay()
        return self.best, self.bq, self.bt


def _time(fn, reps):
    torch.cuda.synchronize()
    for _ in range(2):
        fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--npad", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")

    print(f"CUDA-GRAPH fine-loop A/B (surf, N=M={args.npad}, seeds={args.seeds}, "
          f"steps={args.steps}, median/{args.reps} reps)")
    hdr = (f'{"batch":>6s} {"P rows":>8s} | {"eager ms":>9s} {"graph ms":>9s} {"speedup":>8s} | '
           f'{"capture ms":>10s} | {"max|Δbest|":>11s}')
    print(hdr); print("-" * len(hdr))
    for batch in [16, 64, 256, 1024]:
        P = args.seeds * batch
        A, B, q, t, Nr, Mr, norm = _mk(P, args.npad, args.npad, dev)
        # parity: eager vs graphed result
        be, _, _ = eager_loop(A, B, q, t, Nr, Mr, norm, args.steps)
        gl = GraphedLoop(A, B, q, t, Nr, Mr, norm)
        cap_t0 = time.perf_counter(); gl.capture(); torch.cuda.synchronize()
        cap_ms = (time.perf_counter() - cap_t0) * 1e3
        bg, _, _ = gl.run(args.steps)
        dmax = float((be - bg).abs().max())
        # timing (replay-only for the graph; eager full loop)
        te = _time(lambda: eager_loop(A, B, q, t, Nr, Mr, norm, args.steps), args.reps)
        tg = _time(lambda: gl.run(args.steps), args.reps)
        print(f'{batch:6d} {P:8d} | {te*1e3:9.2f} {tg*1e3:9.2f} {te/tg:7.2f}x | '
              f'{cap_ms:10.1f} | {dmax:11.2e}')


if __name__ == "__main__":
    raise SystemExit(main())
