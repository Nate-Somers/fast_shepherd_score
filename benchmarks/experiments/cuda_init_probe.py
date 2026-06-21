"""Find which step initializes CUDA in the parent (which would force the pool off
the fast fork path). Run on a GPU node."""
import torch


def s(label):
    print(f"  {label}: cuda_init={torch.cuda.is_initialized()}", flush=True)


s("start (just torch)")
import numpy  # noqa: F401
from shepherd_score.container import MoleculePair, MoleculePairBatch  # noqa: F401
s("after import shepherd_score.container")
from benchmarks.benchmark import make_real_cohort
s("after import benchmarks.benchmark")
co = make_real_cohort("vol", n_pairs=20, bucket_kind="same", seed=3)
s("after make_real_cohort")
p = MoleculePair(co.pairs[0].ref, co.pairs[0].fit, do_center=False, device=torch.device("cpu"))
s("after MoleculePair(device=cpu)")
import shepherd_score.container.multi_gpu as _mg  # noqa: F401
s("after import multi_gpu")
print(f"  device_count={torch.cuda.device_count()}  (after) cuda_init={torch.cuda.is_initialized()}",
      flush=True)
