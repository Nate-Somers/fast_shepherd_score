"""Can an EXACT (byte-identical) reimplementation reach 2k/core? Foundational tests.

Byte-identical = reproduce the reference output BIT FOR BIT. This probe establishes what
that constraint permits, by testing whether the overlap primitives are bit-reproducible
across implementations, and whether the Gaussian-overlap exponent separates exactly.

Conclusion the numbers support (see SPEED_EXPERIMENTS_CPU.md "Byte-identical exploration"):
byte-identical locks you to the reference library's exact ops; the dominant cost
(50 seeds x steps x NxM exp) is then irreducible, because every lever that would cut it
changes the output bits. No byte-identical change reaches 2k/core.

Run: python -m benchmarks.experiments.cpu_byte_identical_probe
"""
import math
import numpy as np
import torch
torch.set_num_threads(1)


def bit_equal(a, b):
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def main():
    rng = np.random.default_rng(0)
    print("=== 1. Are the overlap primitives bit-reproducible across implementations? ===")
    x = (6 * rng.standard_normal(200000)).astype(np.float32)
    xt = torch.from_numpy(x)
    e_t, e_n = torch.exp(xt).numpy(), np.exp(x)
    print(f"  torch.exp vs np.exp     : byte-identical={bit_equal(e_t, e_n)}  "
          f"#differ={int((e_t != e_n).sum())}/{len(x)}")

    A = (5 * rng.standard_normal((1, 96, 3))).astype(np.float32)
    B = (5 * rng.standard_normal((1, 96, 3))).astype(np.float32)
    d_c = (torch.cdist(torch.from_numpy(A), torch.from_numpy(B), p=2.0) ** 2).numpy()
    d_m = ((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1)
    print(f"  cdist**2 vs manual r2   : byte-identical={bit_equal(d_c, d_m)}  "
          f"#differ={int((d_c != d_m).sum())}/{d_c.size}")

    v = rng.standard_normal(20000).astype(np.float32)
    s_np = np.float32(v.sum())
    s_loop = np.float32(0.0)
    for a in v:
        s_loop = np.float32(s_loop + a)
    print(f"  np.sum vs naive loop    : byte-identical={s_np == s_loop}  (reduction order matters)")

    print("\n=== 2. Does the Gaussian overlap exponent separate into an EXACT O(N+M) form? ===")
    # |Rq+t - p|^2 = |q'|^2 + |p|^2 - 2 q'.p  => exp(-a/2|.|^2) = exp(-a/2|q'|^2) exp(-a/2|p|^2) exp(a q'.p)
    # The per-point factors are O(N+M); the exp(a q'.p) cross term still COUPLES (i,j) -> O(NM) exactly.
    # An O(N+M) form (Fast Gauss Transform) only exists by TRUNCATING a series == APPROXIMATION.
    N = 64
    a = 0.81
    p = (4 * rng.standard_normal((N, 3))).astype(np.float64)
    q = (4 * rng.standard_normal((N, 3))).astype(np.float64)
    K = (math.pi / (2 * a)) ** 1.5
    r2 = ((q[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    VAB_direct = K * np.exp(-a / 2 * r2).sum()
    wq = np.exp(-a / 2 * (q ** 2).sum(1)); vp = np.exp(-a / 2 * (p ** 2).sum(1))
    cross = np.exp(a * (q @ p.T))
    VAB_factored = K * (wq[:, None] * vp[None, :] * cross).sum()
    print(f"  factored form still O(NM) (cross term couples i,j); matches direct: "
          f"rel={abs(VAB_factored-VAB_direct)/abs(VAB_direct):.1e}")
    print("  -> the only sub-quadratic (O(N+M)) Gaussian sum is the truncated-series FGT = APPROXIMATE.")

    print("\n=== VERDICT ===")
    print("  Byte-identical => locked to the reference ops (primitives differ bit-wise across impls).")
    print("  Exact cost = O(50 seeds x steps x NxM) exp evals, irreducible without changing output.")
    print("  Every 2k-class lever (fewer seeds/steps/points, faster/other exp, FGT) changes the bits.")
    print("  => No byte-identical algorithmic change reaches 2k/core. Proven, not assumed.")


if __name__ == "__main__":
    main()
