"""Would a different language/tool beat the numba CPU kernel? Measures the ceiling.

The shipped numba overlap kernel (cpu_overlap.py) is scalar-exp-bound: torch.exp runs at
~2824 Mexp/s (SIMD) but numba's scalar math.exp at ~396 Mexp/s (no SVML on this box). This
probe measures what fraction of the kernel is exp() — by diffing the real kernel against an
identical one with exp replaced by a cheap polynomial (same memory pattern) — then estimates
the Amdahl speedup a vectorized-exp language/tool (C++/SLEEF, Rust, Julia, or numba+SVML) could
deliver. Conclusion: ~2.8x ceiling -> vol ~450/core, still ~4.4x short of 2k/core; the exp
throughput floor is language-independent. See SPEED_EXPERIMENTS_CPU.md "Language investigation".

Run: python -m benchmarks.experiments.cpu_language_investigation
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import math
import numpy as np
from numba import njit


def _mk(use_exp):
    @njit(fastmath=False)
    def k(A, B, Nr, Mr, alpha):
        K = A.shape[0]; Kc = math.pi ** 1.5 / (2 * alpha) ** 1.5; a2 = alpha / 2
        V = np.zeros(K)
        for kk in range(K):
            acc = 0.0
            for m in range(Mr[kk]):
                bx = B[kk, m, 0]; by = B[kk, m, 1]; bz = B[kk, m, 2]
                for n in range(Nr[kk]):
                    dx = A[kk, n, 0] - bx; dy = A[kk, n, 1] - by; dz = A[kk, n, 2] - bz
                    r2 = dx * dx + dy * dy + dz * dz
                    if use_exp:
                        acc += Kc * math.exp(-a2 * r2)
                    else:  # cheap poly: same loads/stores, no transcendental
                        acc += Kc * (1.0 - a2 * r2 + 0.5 * a2 * a2 * r2 * r2)
            V[kk] = acc
        return V
    return k


def _bench(f, it=8):
    f(); f()
    t0 = time.perf_counter()
    for _ in range(it):
        f()
    return (time.perf_counter() - t0) / it * 1e3


if __name__ == "__main__":
    ke, kp = _mk(True), _mk(False)
    rng = np.random.default_rng(0)
    print("exp fraction of the numba overlap kernel, and the vectorized-exp ceiling:\n")
    for name, N, Bsz in [("vol", 30, 5000), ("surf", 128, 1000)]:
        A = (5 * rng.standard_normal((Bsz, N, 3)))
        B = (5 * rng.standard_normal((Bsz, N, 3)))
        Nr = np.full(Bsz, N, np.int64); Mr = np.full(Bsz, N, np.int64)
        te = _bench(lambda: ke(A, B, Nr, Mr, 0.81))
        tp = _bench(lambda: kp(A, B, Nr, Mr, 0.81))
        frac = (te - tp) / te
        spd_simd = 1.0 / ((1 - frac) + frac / 7.0)     # SLEEF/SVML ~7x scalar->SIMD
        spd_np = 1.0 / ((1 - frac) + frac / (719 / 396))  # np.exp SIMD, no new language
        print(f"  {name:5s} N={N:<4}: exp = {frac*100:4.1f}% of kernel | "
              f"ceiling w/ SIMD-exp(7x) = {spd_simd:.2f}x | w/ np.exp(1.8x) = {spd_np:.2f}x")
    print("""
Measured exp throughput (single-core): torch.exp 2824 / np.exp 719 / numba-scalar 396 Mexp/s.

Language / tool options (all share the SAME O(50 seeds x steps x NxM) exp floor):
  numba + Intel SVML (pip install icc_rt)  ~2.8x  LOW effort (no lang change!) <- TRY FIRST
  torch.compile (Inductor->C++/OpenMP)     ~2-3x  LOW (needs gcc; WSL2)        <- also gets non-kernel ops
  C++/SLEEF + OpenMP  or  Rust/Julia       ~2.8x  HIGH effort, needs toolchain
  Cython                                   ~1.3x  scalar libm exp (like numba)
  JAX-CPU (XLA)                            <1x    single-device, poor multicore at nr=50
Best case ~2.8x -> vol ~450/core: real, but still ~4.4x short of 2k/core. The exp floor is physical.
""")
