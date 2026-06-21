"""Decompose a worker's cold start: heavy-import time vs CUDA context-init time.
One process; the slurm driver runs it 1x and 4x (concurrent + serial) to show how
the multi-GPU pool's spawn cost scales with worker count."""
import os
import time

t0 = time.time()
import numpy  # noqa: F401
import torch
import shepherd_score.container as _c  # noqa: F401
t_import = time.time() - t0

t1 = time.time()
torch.zeros(1, device="cuda:0").sum().item()
torch.cuda.synchronize()
t_cuda = time.time() - t1

cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"  [CVD={cvd} pid={os.getpid()}] import={t_import:.1f}s  "
      f"cuda_init={t_cuda:.1f}s  total={time.time() - t0:.1f}s", flush=True)
