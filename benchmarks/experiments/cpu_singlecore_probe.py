"""Single-core throughput probe for a CPU batched overlap+grad optimizer.

Measures the REAL single-thread ceiling of the L1-style batched torch fine loop
(the device-agnostic eager path of coarse_fine_align_many), using the already-
CPU-capable analytical kernel `compute_analytical_grad_se3_shape`. No jax / triton /
open3d needed — synthetic point clouds (timing is independent of their distribution
since exp() is evaluated for every NxM pair regardless).

Goal: pin how far single-core throughput is from the 2,000 pairs/s/core bar for
vol-sized (~30 pts) vs surf-sized (~128 pts) clouds. See SPEED_EXPERIMENTS_CPU.md.

Run: python -m benchmarks.experiments.cpu_singlecore_probe
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import math
import time
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # already set once this process

from shepherd_score.score.analytical_gradients._torch import (
    compute_analytical_grad_se3_shape,
)

ALPHA = 0.81
K = (math.pi / (2.0 * ALPHA)) ** 1.5


def self_overlap(P):  # (B,K,3) -> (B,)
    d2 = torch.cdist(P, P) ** 2
    return K * torch.exp(-ALPHA / 2.0 * d2).sum(dim=(1, 2))


def make_batch(n_pairs, n_seeds, N, M, scale=8.0):
    B = n_pairs * n_seeds
    ref = scale * torch.randn(B, N, 3, dtype=torch.float32)
    fit = scale * torch.randn(B, M, 3, dtype=torch.float32)
    q = torch.zeros(B, 4, dtype=torch.float32)
    q[:, 0] = 1.0
    q = q + 0.1 * torch.randn(B, 4)
    t = torch.zeros(B, 3, dtype=torch.float32)
    se3 = torch.cat([q, t], dim=1)
    return ref, fit, se3


def adam_loop(ref, fit, se3, steps, VAA, VBB):
    m = torch.zeros_like(se3); v = torch.zeros_like(se3)
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    for _ in range(steps):
        _loss, g = compute_analytical_grad_se3_shape(se3, ref, fit, ALPHA, VAA, VBB)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        se3 = se3 - lr * m / (v.sqrt() + eps)
    return se3


def probe(name, N, M, n_pairs, n_seeds=50, steps=100, warmup=3):
    ref, fit, se3 = make_batch(n_pairs, n_seeds, N, M)
    VAA = self_overlap(ref)   # pose-invariant, computed once
    VBB = self_overlap(fit)

    # warmup (allocation, first-call paths)
    adam_loop(ref, fit, se3.clone(), warmup, VAA, VBB)

    t0 = time.perf_counter()
    adam_loop(ref, fit, se3.clone(), steps, VAA, VBB)
    dt = time.perf_counter() - t0

    pairs_s = n_pairs / dt
    per_step_ms = dt / steps * 1e3
    per_pair_us = dt / n_pairs * 1e6
    print(f"{name:>10}  N=M={N:<4} pairs={n_pairs:<5} seeds={n_seeds} steps={steps} "
          f"B={n_pairs*n_seeds:<6} | {dt:6.3f}s  {per_step_ms:7.2f} ms/step  "
          f"{per_pair_us:9.1f} us/pair  -> {pairs_s:8.1f} pairs/s/core")
    return pairs_s


if __name__ == "__main__":
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"interop={torch.get_num_interop_threads()}")
    print("Fixed-step (no early-stop) single-core throughput; early-stop would lift ~1.5-2x.\n")
    # vol-sized: ~30 heavy atoms
    probe("vol",  30,  30,  n_pairs=200, steps=100)
    # surf-sized: num_surf_points = max(24, 3*heavy) ~ 75-150; probe two points
    probe("surf-75", 75, 75, n_pairs=40, steps=100)
    probe("surf-128", 128, 128, n_pairs=20, steps=100)
    print("\n2,000 pairs/s/core bar: vol within reach? surf gap = bar / measured.")
