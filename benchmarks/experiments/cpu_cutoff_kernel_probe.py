"""Hypothesis (lever L11): per-element distance cutoff in the fused numba kernel.

The Gaussian overlap exp(-alpha/2 r^2) is < 1e-8 beyond ~6.7 A (alpha=0.81). The GPU
log (SPEED_EXPERIMENTS.md exp #8) could skip only ~2% because SIMT can't skip individual
lanes -- but a SCALAR CPU loop skips per-ELEMENT for free. Measured pair-sparsity was 0.40,
so ~60% of exp() calls are droppable. Dropped terms are < 1e-8 each, so this is plausibly
score-safe (< 1e-5). This attacks exactly the scalar-exp bottleneck that capped the numba
kernel at ~3.3x.

Measures, single-core: skip fraction, speedup of cutoff vs no-cutoff fused njit, and
VAB + Tanimoto-SCORE parity (the gated quantity). Uses ellipsoid-shell point clouds that
mimic real drug surfaces (~12-14 A extent) since open3d isn't on this box.

Run: python -m benchmarks.experiments.cpu_cutoff_kernel_probe
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import math
import time
import numpy as np
from numba import njit

ALPHA = 0.81
# cutoff: exp(-alpha/2 * rcut2) = 1e-9  ->  rcut2 = -2*ln(1e-9)/alpha
RCUT2 = -2.0 * math.log(1e-9) / ALPHA   # ~ 51.2  (r ~ 7.15 A)


@njit(cache=True, fastmath=False)
def fused_full(ref, fit, alpha):
    B = ref.shape[0]; N = ref.shape[1]; M = fit.shape[1]
    K = (math.pi / (2.0 * alpha)) ** 1.5
    a2 = alpha / 2.0
    O = np.zeros(B)
    for b in range(B):
        for m in range(M):
            fx = fit[b, m, 0]; fy = fit[b, m, 1]; fz = fit[b, m, 2]
            for n in range(N):
                dx = fx - ref[b, n, 0]; dy = fy - ref[b, n, 1]; dz = fz - ref[b, n, 2]
                r2 = dx*dx + dy*dy + dz*dz
                O[b] += math.exp(-a2 * r2)
        O[b] *= K
    return O


@njit(cache=True, fastmath=False)
def fused_cutoff(ref, fit, alpha, rcut2):
    B = ref.shape[0]; N = ref.shape[1]; M = fit.shape[1]
    K = (math.pi / (2.0 * alpha)) ** 1.5
    a2 = alpha / 2.0
    O = np.zeros(B)
    skipped = 0
    total = 0
    for b in range(B):
        for m in range(M):
            fx = fit[b, m, 0]; fy = fit[b, m, 1]; fz = fit[b, m, 2]
            for n in range(N):
                dx = fx - ref[b, n, 0]; dy = fy - ref[b, n, 1]; dz = fz - ref[b, n, 2]
                r2 = dx*dx + dy*dy + dz*dz
                total += 1
                if r2 > rcut2:
                    skipped += 1
                    continue
                O[b] += math.exp(-a2 * r2)
        O[b] *= K
    return O, skipped, total


def ellipsoid_shell(B, n, axes, rng, jitter=0.4):
    # points near the surface of an ellipsoid (mimics a molecular surface)
    v = rng.standard_normal((B, n, 3))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    pts = v * np.array(axes)[None, None, :]
    pts += jitter * rng.standard_normal((B, n, 3))
    return pts


def bench(fn, iters=20):
    fn(); fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def run(n, B, offset):
    rng = np.random.default_rng(1)
    axes = (6.0, 4.5, 3.5)   # ~12-13 A long axis, drug-sized
    ref = ellipsoid_shell(B, n, axes, rng)
    fit = ellipsoid_shell(B, n, axes, rng) + offset   # offset mimics a distinct/imperfect pose
    # self-overlap-ish for the SCORE: VAA/VBB via the full kernel (same cloud)
    O_full = fused_full(ref, fit, ALPHA)
    O_cut, sk, tot = fused_cutoff(ref, fit, ALPHA, RCUT2)
    VAA = fused_full(ref, ref, ALPHA); VBB = fused_full(fit, fit, ALPHA)
    VAA_c = fused_cutoff(ref, ref, ALPHA, RCUT2)[0]; VBB_c = fused_cutoff(fit, fit, ALPHA, RCUT2)[0]
    score_full = O_full / (VAA + VBB - O_full)
    score_cut = O_cut / (VAA_c + VBB_c - O_cut)

    vab_rel = float(np.max(np.abs(O_cut - O_full)) / (np.max(np.abs(O_full)) + 1e-12))
    sc_abs = float(np.max(np.abs(score_cut - score_full)))
    skipfrac = sk / tot
    t_full = bench(lambda: fused_full(ref, fit, ALPHA))
    t_cut = bench(lambda: fused_cutoff(ref, fit, ALPHA, RCUT2))
    print(f"n={n:<4} B={B:<5} offset={offset:>3} | skip {skipfrac*100:5.1f}%  "
          f"speedup {t_full/t_cut:5.2f}x  | VAB rel {vab_rel:.1e}  SCORE max|d| {sc_abs:.1e}")
    return skipfrac, t_full / t_cut, sc_abs


if __name__ == "__main__":
    import numba
    print(f"numba {numba.__version__}  RCUT = {math.sqrt(RCUT2):.2f} A (exp tail 1e-9)")
    print("Per-element cutoff in the fused njit kernel (CPU can skip per-lane; GPU SIMT cannot).\n")
    print("Self-overlap geometry (offset=0, clouds coincide -> worst case, fewest skips):")
    run(75, 2000, 0); run(128, 1000, 0)
    print("\nDistinct-ish geometry (offset between clouds -> more skips, realistic for distinct pairs):")
    run(75, 2000, 3); run(128, 1000, 3); run(128, 1000, 6)
    print("\nGate: SCORE max|d| must be < 1e-5 (and winner-flips=0 in the full pipeline).")
