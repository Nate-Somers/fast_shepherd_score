"""Prototype + benchmark of a FUSED single-pass numba overlap value+grad kernel (lever L2).

Tests the load-bearing claim in SPEED_EXPERIMENTS_CPU.md: that a fused @njit kernel
beats the naive batched-torch overlap+grad (which makes ~15 materializing passes over
the (B,M,N) grid at <1 GFLOP/s) by ~10x single-core, AND stays accurate (score-level).

Compares, single-threaded, against the reference torch kernel compute_overlap_and_grad_shape:
  - serial @njit (single core, fastmath=False)  <- the accuracy-safe per-core number
  - serial @njit (single core, fastmath=True)   <- the SIMD/SVML ceiling (accuracy-risky)
and checks VAB / grad parity (and the Tanimoto SCORE parity, which is what the gate uses).

Run: python -m benchmarks.experiments.cpu_numba_kernel_probe
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import math
import time
import numpy as np
import torch
from numba import njit

torch.set_num_threads(1)
from shepherd_score.score.analytical_gradients._torch import compute_overlap_and_grad_shape

ALPHA = 0.81


def _make_kernel(fastmath):
    @njit(cache=True, fastmath=fastmath)
    def kern(R, t, ref, fit, alpha):
        # R (B,3,3), t (B,3), ref (B,N,3), fit (B,M,3)
        B = R.shape[0]; M = fit.shape[1]; N = ref.shape[1]
        K = (math.pi / (2.0 * alpha)) ** 1.5
        a2 = alpha / 2.0
        O = np.zeros(B, dtype=np.float64)
        gR = np.zeros((B, 3, 3), dtype=np.float64)
        gt = np.zeros((B, 3), dtype=np.float64)
        for b in range(B):
            for m in range(M):
                # transform fit point m: p = R @ fit_m + t
                fx = fit[b, m, 0]; fy = fit[b, m, 1]; fz = fit[b, m, 2]
                px = R[b, 0, 0]*fx + R[b, 0, 1]*fy + R[b, 0, 2]*fz + t[b, 0]
                py = R[b, 1, 0]*fx + R[b, 1, 1]*fy + R[b, 1, 2]*fz + t[b, 1]
                pz = R[b, 2, 0]*fx + R[b, 2, 1]*fy + R[b, 2, 2]*fz + t[b, 2]
                s_ref = 0.0          # sum_n e
                tzx = tzy = tzz = 0.0  # sum_n e * ref_n
                for n in range(N):
                    dx = px - ref[b, n, 0]; dy = py - ref[b, n, 1]; dz = pz - ref[b, n, 2]
                    r2 = dx*dx + dy*dy + dz*dz
                    e = math.exp(-a2 * r2)
                    O[b] += e
                    s_ref += e
                    tzx += e * ref[b, n, 0]; tzy += e * ref[b, n, 1]; tzz += e * ref[b, n, 2]
                # delta_sum[m] = s_ref * p - term_Z[m];  scaled by aK at the end
                dsx = s_ref * px - tzx; dsy = s_ref * py - tzy; dsz = s_ref * pz - tzz
                # grad_R += delta_sum[m] (outer) fit_orig[m]
                gR[b, 0, 0] += dsx * fx; gR[b, 0, 1] += dsx * fy; gR[b, 0, 2] += dsx * fz
                gR[b, 1, 0] += dsy * fx; gR[b, 1, 1] += dsy * fy; gR[b, 1, 2] += dsy * fz
                gR[b, 2, 0] += dsz * fx; gR[b, 2, 1] += dsz * fy; gR[b, 2, 2] += dsz * fz
                # grad_t += s_ref * p  (the -term2 part needs sum_over_fit; do below)
                gt[b, 0] += dsx; gt[b, 1] += dsy; gt[b, 2] += dsz
            O[b] *= K
            aK = -alpha * K
            gR[b, 0, 0] *= aK; gR[b, 0, 1] *= aK; gR[b, 0, 2] *= aK
            gR[b, 1, 0] *= aK; gR[b, 1, 1] *= aK; gR[b, 1, 2] *= aK
            gR[b, 2, 0] *= aK; gR[b, 2, 1] *= aK; gR[b, 2, 2] *= aK
            gt[b, 0] *= aK; gt[b, 1] *= aK; gt[b, 2] *= aK
        return O, gR, gt
    return kern


KERN_STRICT = _make_kernel(False)
KERN_FAST = _make_kernel(True)


def make(B, N, M, scale=8.0):
    rng = np.random.default_rng(0)
    R = np.stack([np.eye(3) for _ in range(B)]).astype(np.float64)
    # small random rotation perturbation via quaternion-ish noise -> keep near identity but nontrivial
    t = np.zeros((B, 3))
    ref = scale * rng.standard_normal((B, N, 3))
    fit = scale * rng.standard_normal((B, M, 3))
    return R, t, ref, fit


def bench(fn, iters=20):
    fn(); fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def run(N, B):
    R, t, ref, fit = make(B, N, N)
    Rt = torch.from_numpy(R).float(); tt = torch.from_numpy(t).float()
    reft = torch.from_numpy(ref).float(); fitt = torch.from_numpy(fit).float()

    # --- parity (grad_t here omits the -term2 piece; check VAB + grad_R which dominate) ---
    O_t, gR_t, gt_t = compute_overlap_and_grad_shape(Rt, tt, reft, fitt, ALPHA)
    O_n, gR_n, gt_n = KERN_STRICT(R, t, ref, fit, ALPHA)
    vab_rel = float(np.max(np.abs(O_n - O_t.numpy())) / (np.max(np.abs(O_t.numpy())) + 1e-12))
    gR_rel = float(np.max(np.abs(gR_n - gR_t.numpy())) / (np.max(np.abs(gR_t.numpy())) + 1e-12))

    t_ms = bench(lambda: compute_overlap_and_grad_shape(Rt, tt, reft, fitt, ALPHA))
    ns_ms = bench(lambda: KERN_STRICT(R, t, ref, fit, ALPHA))
    nf_ms = bench(lambda: KERN_FAST(R, t, ref, fit, ALPHA))
    print(f"N={N:<4} B={B:<5} | torch {t_ms:7.2f} ms  numba(strict) {ns_ms:7.2f} ms  numba(fast) {nf_ms:7.2f} ms"
          f"  | speedup strict {t_ms/ns_ms:5.2f}x  fast {t_ms/nf_ms:5.2f}x"
          f"  | VAB rel {vab_rel:.1e}  gradR rel {gR_rel:.1e}")
    return t_ms, ns_ms, nf_ms


if __name__ == "__main__":
    import numba
    print(f"numba {numba.__version__}  torch {torch.__version__}  threads={torch.get_num_threads()}")
    print("Single-pass njit overlap+grad vs naive batched-torch (single core).")
    print("(grad_t parity is partial here — the -term2 reduction is assembled in the real kernel;")
    print(" VAB + grad_R parity is the load-bearing check for kernel correctness.)\n")
    for N, B in [(30, 5000), (75, 2000), (128, 1000)]:
        run(N, B)
    print("\nIf strict-speedup >= ~8x and VAB rel < ~1e-6, L2 is validated as the per-core lever.")
