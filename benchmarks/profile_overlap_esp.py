"""
Best (BLOCK, num_warps) per molecule size for the ESP overlap kernel (charges +
Gaussian, one CTA per pose). Same occupancy story as the surf kernel; esp does
more per-pose work so the optimum may differ -- sweep to confirm before applying.
"""
import time
import torch

from shepherd_score.score.gaussian_overlap_esp_triton import overlap_score_grad_esp_se3_batch


def _t(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(8):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    device = torch.device("cuda")
    P = 200000
    for Npad in [32, 48, 64, 80, 112]:
        N = M = Npad
        A = torch.randn(P, N, 3, device=device, dtype=torch.float32).contiguous()
        B = torch.randn(P, M, 3, device=device, dtype=torch.float32).contiguous()
        CA = torch.randn(P, N, device=device, dtype=torch.float32).contiguous()
        CB = torch.randn(P, M, device=device, dtype=torch.float32).contiguous()
        q = torch.randn(P, 4, device=device, dtype=torch.float32)
        q = (q / q.norm(dim=1, keepdim=True)).contiguous()
        t = torch.randn(P, 3, device=device, dtype=torch.float32).contiguous()
        Nr = torch.full((P,), N, dtype=torch.int32, device=device)
        Mr = torch.full((P,), M, dtype=torch.int32, device=device)

        def call(BLOCK=None, nw=None):
            kw = {} if BLOCK is None else {"BLOCK": BLOCK}
            if nw is not None:
                kw["num_warps"] = nw
            overlap_score_grad_esp_se3_batch(A, B, CA, CB, q, t, alpha=0.81, lam=0.3,
                                             N_real=Nr, M_real=Mr, **kw)

        b = _t(lambda: call())                          # current default (BLOCK=64, default warps)
        best = (1e9, None)
        for BLOCK in [16, 32, 64]:
            for nw in [1, 2, 4]:
                tmin = _t(lambda B_=BLOCK, w_=nw: call(B_, w_))
                if tmin < best[0]:
                    best = (tmin, (BLOCK, nw))
        print(f"N={Npad:3d} | default {b*1e3:7.2f}ms | best BLOCK={best[1][0]:3d} warps={best[1][1]} "
              f"{best[0]*1e3:7.2f}ms | speedup {b/best[0]:4.2f}x", flush=True)
        del A, B, CA, CB, q, t
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
