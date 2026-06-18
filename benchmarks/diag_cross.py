"""
Diagnose why cross-bucket throughput is low / regresses at 10k.
For surf & esp cross-10k: band distribution, full-cohort time, and per-band
ISOLATED time+mol/s (so we see whether one big-N band dominates, whether
sum-of-bands == full (=> no dispatch overhead), and -- via SUBBATCH_DEBUG --
how many sub-batch chunks each band is split into).
"""
import os
os.environ["SUBBATCH_DEBUG"] = "1"          # print chosen chunk size per bucket
import time
from collections import defaultdict
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.container._core import _band_key


def mk(mode, nb):
    co = make_real_cohort(mode, n_pairs=nb, bucket_kind="cross", seed=3)
    return [MoleculePair(p.ref, p.fit, do_center=False, device=torch.device("cuda"))
            for p in co.pairs]


def align(mode, pairs):
    if mode == "surf":
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=100)
    else:
        MoleculePair.align_batch_esp(pairs, alpha=0.81, lam=0.3, num_repeats=50,
                                     topk=30, steps_fine=100, lr=0.075)


def band_of(p):
    return _band_key(p._ref_surf_t.shape[0])


def main():
    for mode in ["surf", "esp"]:
        nb = 10000
        print(f"\n{'='*64}\n{mode} cross {nb}\n{'='*64}")
        pairs = mk(mode, nb)
        align(mode, pairs); torch.cuda.synchronize()       # warmup + populate tensors
        by = defaultdict(list)
        for p in pairs:
            by[band_of(p)].append(p)
        print(f"{len(by)} bands: " + ", ".join(f"{b}:{len(by[b])}" for b in sorted(by)))

        print("\n--- SUBBATCH_DEBUG for one FULL cross align ---")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        align(mode, pairs); torch.cuda.synchronize()
        tf = time.perf_counter() - t0
        print(f"FULL cross: {tf:.3f}s -> {nb/tf:.1f} mol/s")

        print("\n--- per-band isolated ---")
        rows = []; tsum = 0.0
        for b in sorted(by):
            bp = by[b]
            align(mode, bp); torch.cuda.synchronize()       # warm this band
            torch.cuda.synchronize(); t0 = time.perf_counter()
            align(mode, bp); torch.cuda.synchronize()
            tb = time.perf_counter() - t0; tsum += tb
            rows.append((b, bp[0]._ref_surf_t.shape[0], len(bp), tb))
        print(f'{"band":>5} {"Npad":>5} {"pairs":>6} {"time s":>8} {"mol/s":>9} {"%time":>6}')
        for b, npad, k, tb in rows:
            print(f'{b:5d} {npad:5d} {k:6d} {tb:8.3f} {k/tb:9.1f} {100*tb/tsum:5.1f}%')
        print(f"SUM isolated: {tsum:.3f}s -> {nb/tsum:.1f} mol/s   (full {tf:.3f}s; "
              f"overhead {100*(tf-tsum)/tsum:+.0f}%)")

        del pairs, by
        import shepherd_score.container._core as _cc
        _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
