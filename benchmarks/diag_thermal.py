"""
Directly test whether the per-cell throughput swing is GPU clock throttling.

Runs the SAME fixed workload (surf same 10000, forced full 100 steps, identical
cohort) back-to-back N times and logs throughput alongside nvidia-smi SM clock /
mem clock / temperature / power each rep. If it's throttling, mol/s tracks the SM
clock; if the clock barely moves while mol/s halves, the explanation is wrong.
"""
import subprocess
import time
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _self_overlap_in_chunks
from shepherd_score.alignment.utils.fast_common import batched_seeds_torch
from shepherd_score.container._core import _band_key


def gpu_stat():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        return out.split("\n")[0]
    except Exception as e:
        return f"nvsmi-err:{type(e).__name__}"


def main():
    device = torch.device("cuda")
    nb = 10000
    co = make_real_cohort("surf", n_pairs=nb, bucket_kind="same", seed=3)
    pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=device) for p in co.pairs]
    MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=100)
    torch.cuda.synchronize()

    K = len(pairs)
    N_pad = _band_key(max(p._ref_surf_t.shape[0] for p in pairs))
    M_pad = _band_key(max(p._fit_surf_t.shape[0] for p in pairs))
    ref = torch.zeros(K, N_pad, 3, device=device); fit = torch.zeros(K, M_pad, 3, device=device)
    Nr = torch.empty(K, dtype=torch.int32, device=device); Mr = torch.empty(K, dtype=torch.int32, device=device)
    for i, p in enumerate(pairs):
        n = p._ref_surf_t.shape[0]; m = p._fit_surf_t.shape[0]
        ref[i, :n] = p._ref_surf_t; fit[i, :m] = p._fit_surf_t; Nr[i] = n; Mr[i] = m
    VAA = _self_overlap_in_chunks(ref, Nr, 0.81); VBB = _self_overlap_in_chunks(fit, Mr, 0.81)
    sq, st = batched_seeds_torch(ref, fit, Nr, Mr, num_seeds=50)

    def run():
        coarse_fine_align_many(ref, fit, VAA, VBB, alpha=0.81, steps_fine=100,
                               N_real=Nr, M_real=Mr, early_stop_patience=10**9, seeds=(sq, st))

    run(); torch.cuda.synchronize()
    print(f"surf same {nb}, forced 100 steps, same cohort, 30 back-to-back reps")
    print(f'{"rep":>3} {"time s":>7} {"mol/s":>8} | {"SMclk":>6} {"MEMclk":>7} {"temp":>5} {"power":>6}')
    for i in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run(); torch.cuda.synchronize()
        t = time.perf_counter() - t0
        print(f'{i:3d} {t:7.3f} {nb/t:8.1f} | {gpu_stat()}', flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
