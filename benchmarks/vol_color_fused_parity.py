"""Paired-parity gate for the fused vol_color kernel vs the two separate kernels.

In-process (NOT cross-job -- ESP/color triton kernels are cross-job non-deterministic) it
runs random poses through (a) overlap_score_grad_se3_batch + pharm_color_score_grad_se3_batch
and (b) the single fused vol_color_score_grad_se3_batch, and asserts all six outputs match.
Masking (real < pad) and dummy types are exercised. Also times the two paths.

    python benchmarks/vol_color_fused_parity.py
"""
import sys, os, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from shepherd_score.accel.kernels.shape_triton import overlap_score_grad_se3_batch
from shepherd_score.accel.kernels.pharm_triton import pharm_color_score_grad_se3_batch
from shepherd_score.accel.kernels.vol_color_triton import vol_color_score_grad_se3_batch
from shepherd_score.score.analytical_gradients._torch import build_lookup_tables


def _mae(a, b):
    return (a - b).abs().max().item()


def run(P=4096, Ns=24, Ms=28, Na=9, Ma=11, seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    c1 = torch.randn(P, Ns, 3, device=dev)
    c2 = torch.randn(P, Ms, 3, device=dev)
    a1 = torch.randn(P, Na, 3, device=dev)
    a2 = torch.randn(P, Ma, 3, device=dev)
    # types 0..7 real + some dummy (8); the directionless tables map real->cat0, dummy->cat3
    rt = torch.randint(0, 9, (P, Na), device=dev)
    ft = torch.randint(0, 9, (P, Ma), device=dev)
    q = torch.randn(P, 4, device=dev); q = q / q.norm(dim=1, keepdim=True)
    t = torch.randn(P, 3, device=dev)
    # real counts strictly inside the pads to exercise masking
    Nc = torch.randint(Ns - 4, Ns + 1, (P,), device=dev, dtype=torch.int32)
    Mc = torch.randint(Ms - 4, Ms + 1, (P,), device=dev, dtype=torch.int32)
    Naa = torch.randint(Na - 3, Na + 1, (P,), device=dev, dtype=torch.int32)
    Maa = torch.randint(Ma - 3, Ma + 1, (P,), device=dev, dtype=torch.int32)
    al, Ks, cats = build_lookup_tables(dev, torch.float32, directionless=True)

    # --- separate (the reference) ---
    VAB, dQs, dTs = overlap_score_grad_se3_batch(c1, c2, q, t, alpha=0.81, N_real=Nc, M_real=Mc)
    Oc, dQc, dTc = pharm_color_score_grad_se3_batch(a1, a2, q, t, rt, ft, al, Ks, cats, N_real=Naa, M_real=Maa)
    # --- fused ---
    fV, fdQs, fdTs, fOc, fdQc, fdTc = vol_color_score_grad_se3_batch(
        c1, c2, a1, a2, q, t, rt, ft, al, Ks, cats, alpha=0.81,
        N_real_cent=Nc, M_real_cent=Mc, N_real_anc=Naa, M_real_anc=Maa)
    torch.cuda.synchronize()

    print(f"P={P} Ns={Ns} Ms={Ms} Na={Na} Ma={Ma}")
    print(f"  shape : V {_mae(VAB,fV):.2e}  dQ {_mae(dQs,fdQs):.2e}  dT {_mae(dTs,fdTs):.2e}")
    print(f"  color : O {_mae(Oc,fOc):.2e}  dQ {_mae(dQc,fdQc):.2e}  dT {_mae(dTc,fdTc):.2e}")
    worst = max(_mae(VAB, fV), _mae(dQs, fdQs), _mae(dTs, fdTs),
               _mae(Oc, fOc), _mae(dQc, fdQc), _mae(dTc, fdTc))

    # --- timing (best of 5 after warmup) ---
    def _sep():
        overlap_score_grad_se3_batch(c1, c2, q, t, alpha=0.81, N_real=Nc, M_real=Mc)
        pharm_color_score_grad_se3_batch(a1, a2, q, t, rt, ft, al, Ks, cats, N_real=Naa, M_real=Maa)
    def _fus():
        vol_color_score_grad_se3_batch(c1, c2, a1, a2, q, t, rt, ft, al, Ks, cats, alpha=0.81,
                                       N_real_cent=Nc, M_real_cent=Mc, N_real_anc=Naa, M_real_anc=Maa)
    for fn, name in ((_sep, "separate"), (_fus, "fused")):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        best = min(_t(fn) for _ in range(5))
        print(f"  {name:>9}: {best*1e3:.3f} ms")
    return worst


def _t(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    return time.perf_counter() - t0


if __name__ == "__main__":
    worst = 0.0
    for sz in [(4096, 24, 28, 9, 11), (16384, 32, 36, 12, 8), (1024, 16, 16, 5, 5)]:
        worst = max(worst, run(*sz))
    print(f"\nWORST |delta| across all outputs/sizes: {worst:.2e}")
    print("PASS" if worst < 5e-3 else "FAIL", "(threshold 5e-3; float summation-order diffs expected ~1e-5)")
    raise SystemExit(0 if worst < 5e-3 else 1)
