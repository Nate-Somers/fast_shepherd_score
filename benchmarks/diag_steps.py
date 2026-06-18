"""
Test whether the bimodal throughput across batch sizes is early-stopping running
a cohort-dependent number of fine-loop steps (NOT a size/memory effect).

For each size: time the SAME cohort with early-stop ON (default patience=5) vs
OFF (patience huge -> always the full steps_fine=100). If ON is bimodal and OFF
is flat, the variation is purely how many steps each random cohort needs.
"""
import time
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _self_overlap_in_chunks
from shepherd_score.alignment.utils.fast_common import batched_seeds_torch
from shepherd_score.container._core import _band_key


def build_pads(pairs, device):
    K = len(pairs)
    N_pad = _band_key(max(p._ref_surf_t.shape[0] for p in pairs))
    M_pad = _band_key(max(p._fit_surf_t.shape[0] for p in pairs))
    ref = torch.zeros(K, N_pad, 3, device=device)
    fit = torch.zeros(K, M_pad, 3, device=device)
    Nr = torch.empty(K, dtype=torch.int32, device=device)
    Mr = torch.empty(K, dtype=torch.int32, device=device)
    for i, p in enumerate(pairs):
        n = p._ref_surf_t.shape[0]; m = p._fit_surf_t.shape[0]
        ref[i, :n] = p._ref_surf_t; fit[i, :m] = p._fit_surf_t; Nr[i] = n; Mr[i] = m
    return ref, fit, Nr, Mr


def med_min(fn, reps=3):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    device = torch.device("cuda")
    print(f'{"batch":>6} | {"ES-on s":>8} {"ES-on mol/s":>11} | {"ES-off s":>8} {"ES-off mol/s":>12}')
    for nb in [4096, 6144, 8192, 10000, 14000]:
        co = make_real_cohort("surf", n_pairs=nb, bucket_kind="same", seed=3)
        pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=device) for p in co.pairs]
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=100)
        torch.cuda.synchronize()
        ref, fit, Nr, Mr = build_pads(pairs, device)
        VAA = _self_overlap_in_chunks(ref, Nr, 0.81)
        VBB = _self_overlap_in_chunks(fit, Mr, 0.81)
        sq, st = batched_seeds_torch(ref, fit, Nr, Mr, num_seeds=50)

        def run(patience):
            return coarse_fine_align_many(
                ref, fit, VAA, VBB, alpha=0.81, steps_fine=100,
                N_real=Nr, M_real=Mr, early_stop_patience=patience, seeds=(sq, st))

        t_on = med_min(lambda: run(5))
        t_off = med_min(lambda: run(10**9))
        print(f'{nb:6d} | {t_on:8.3f} {nb/t_on:11.1f} | {t_off:8.3f} {nb/t_off:12.1f}', flush=True)
        del pairs, co, ref, fit
        import shepherd_score.container._core as _cc
        _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
