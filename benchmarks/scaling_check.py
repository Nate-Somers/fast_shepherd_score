"""
Throughput-vs-molecule-count scaling for the fast alignment paths.

Sweeps the number of pairs (molecule count) and reports, per mode/cohort, the
GPU batch path's throughput (pairs/s), per-pair latency, achieved mean score,
and bucket count. This is where the batched-seed + batch-kernel wins show up:
launch/seed overhead amortises as the molecule count grows, and a uniform
cohort stays a single bucket while a mixed cohort fans out.

It deliberately does NOT run the slow CPU reference at every size (that would
dominate wall time); score quality is checked against the noiseless ideal (1.0)
which every backend must approach, and full vs-reference parity is covered by
parity_check.py at a fixed size.
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.alignment_bench.workloads import make_cohort, MODES
from benchmarks.alignment_bench import backends as B


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_backend(backend, cohort, cfg, warmup, repeats):
    state = backend.prepare(cohort, cfg)
    for _ in range(warmup):
        out = backend.run(state); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = backend.run(state); sync()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "pharm"])
    ap.add_argument("--counts", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256])
    ap.add_argument("--backends", nargs="+",
                    default=["gpu_multi_batch", "gpu_single_fast"])
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps-fine", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)
    backends = {b.name: b for b in B.all_backends()}

    print(f"SCALING (num_repeats={args.num_repeats}, steps_fine={args.steps_fine}, "
          f"repeats={args.repeats})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    hdr = f'{"mode":5s} {"cohort":7s} {"backend":16s} {"n_pairs":>7s} {"pairs/s":>9s} {"ms/pair":>8s} {"ms_total":>9s} {"mean":>7s} {"bk":>4s}'
    print(hdr); print("-" * len(hdr))

    for mode in args.modes:
        for kind in ("uniform", "mixed"):
            for bname in args.backends:
                bk = backends.get(bname)
                ok, reason = bk.available()
                if not ok:
                    print(f"{mode:5s} {kind:7s} {bname:16s}  skipped: {reason}")
                    continue
                for n in args.counts:
                    cohort = make_cohort(mode, n_pairs=n, size_kind=kind,
                                         size=40, size_range=(12, 72),
                                         noise=0.0, seed=11)
                    out, ms = time_backend(bk, cohort, cfg, args.warmup, args.repeats)
                    thru = n / ms
                    lat = 1000.0 * ms / n
                    nb = out.n_buckets if out.n_buckets is not None else "-"
                    print(f'{mode:5s} {kind:7s} {bname:16s} {n:7d} {thru:9.1f} '
                          f'{lat:8.2f} {ms*1000:9.1f} {out.scores.mean():7.4f} {str(nb):>4s}')
                print()


if __name__ == "__main__":
    raise SystemExit(main())
