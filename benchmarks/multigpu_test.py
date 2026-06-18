"""
Validate the multi-GPU dispatcher MECHANICS on a single GPU.

We can't test true multi-device concurrency here, but we can verify the
dispatcher path itself: _run_distributed shards, spawns a worker thread per
shard, sets _DISPATCH_LOCAL.active (so the inner align takes the single-device
body, no recursion), moves the shard's tensors to the device, and writes results
in-place. With device_count()==1 it runs one shard/thread on cuda:0; results must
match a direct align exactly.
"""
import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
from shepherd_score.container._core import _run_distributed, _should_distribute


def main():
    dev = torch.device("cuda")
    print(f"device_count = {torch.cuda.device_count()}")
    co = make_real_cohort("surf", n_pairs=64, bucket_kind="cross", seed=3)
    p1 = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
    p2 = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]

    print(f"_should_distribute (expect False on 1 GPU) = {_should_distribute(p1)}")

    MoleculePair.align_batch_surf(p1, alpha=0.81, steps_fine=100)           # direct
    s1 = np.array([p.sim_aligned_surf for p in p1])

    _run_distributed(MoleculePair.align_batch_surf, p2, alpha=0.81, steps_fine=100)  # via dispatcher
    s2 = np.array([p.sim_aligned_surf for p in p2])

    d = np.abs(s1 - s2).max()
    print(f"dispatcher vs direct: max|delta| = {d:.2e} -> {'PASS' if d < 1e-5 else 'FAIL <<<'}")
    # also confirm the no-recursion guard reset
    print(f"_DISPATCH_LOCAL.active after = {getattr(__import__('shepherd_score.container._core', fromlist=['_DISPATCH_LOCAL'])._DISPATCH_LOCAL, 'active', False)}")


if __name__ == "__main__":
    raise SystemExit(main())
