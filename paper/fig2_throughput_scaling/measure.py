"""
Figure 2 (data) — measure fss alignment throughput vs batch size on THIS GPU, with
per-rep variance (mean ± SD), in a single controlled environment.

This replaces the old practice of stitching together saved runs from machines with different
torch/CUDA versions (a real confound for a launch-bound workload).  Run it once per GPU in
the SAME conda env; each writes data/<gpu>.json, and plot.py renders them together with error
bars.  Self-copy pairs (optimum = 1.0) — this measures kernel+launch throughput.

Run:  PYTHONPATH=. python paper/fig2_throughput_scaling/measure.py [--reps 7]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmarks.benchmark import make_real_cohort  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["vol", "surf", "esp", "pharm"]
SIZES = [1, 10, 100, 1000, 5000, 10000]
ALPHA, LAM = 0.81, 0.3


def align(mps, mode):
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(mps)
    if mode == "vol":
        b.align_with_vol(no_H=True, alpha=ALPHA, backend="triton", max_num_steps=100)
    elif mode == "surf":
        b.align_with_surf(alpha=ALPHA, backend="triton", num_repeats=50, max_num_steps=100)
    elif mode == "esp":
        b.align_with_esp(alpha=ALPHA, lam=LAM, backend="triton", num_repeats=50, max_num_steps=100)
    elif mode == "pharm":
        b.align_with_pharm(backend="triton", num_repeats=50, max_num_steps=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--modes", nargs="+", default=MODES)
    a = ap.parse_args()

    import torch
    from shepherd_score.container import MoleculePair
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    slug = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")

    out = {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
           "reps": a.reps, "modes": a.modes, "sizes": a.sizes, "data": {}}
    for mode in a.modes:
        out["data"][mode] = []
        for nb in a.sizes:
            co = make_real_cohort(mode, n_pairs=nb, bucket_kind="same", seed=3)
            mps = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
            align(mps, mode); torch.cuda.synchronize()                # warmup / autotune
            rates = []
            for _ in range(a.reps):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                align(mps, mode); torch.cuda.synchronize()
                rates.append(nb / (time.perf_counter() - t0))
            rates = np.array(rates)
            out["data"][mode].append({"n": nb, "mean": float(rates.mean()), "std": float(rates.std()),
                                      "max": float(rates.max()), "min": float(rates.min())})
            print(f"{gpu} {mode:5s} n={nb:<6d} {rates.mean():10.0f} ± {rates.std():8.0f} pairs/s "
                  f"(peak {rates.max():.0f})", flush=True)

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", f"{slug}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
