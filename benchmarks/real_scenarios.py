"""
REAL-molecule scenario sweep: original-repo speed paths vs fork speed paths,
over the BUCKET axis (same-bucket vs cross-bucket) x batch size.

Molecules: the curated real-drug set in real_workloads.py (see its docstring for
provenance). Each pair is (real molecule, rigid SE(3) copy). The original repo's
single-pair optimisers are O(n) loops (latency measured once per cohort and
scaled by n); the fork batch path and JAX are timed directly. We report, per
cell, speedup = (fastest original-repo time) / (fastest fork time), the fork
bucket count, and the fork's achieved score (so speed never hides quality loss).
"""
import argparse
import time

import numpy as np
import torch

from benchmarks.alignment_bench import jax_shmap
from benchmarks.alignment_bench import backends as B
from benchmarks.real_workloads import make_real_cohort, molecule_table

ORIG_LINEAR = ["cpu_single_torch", "cpu_single_analytical", "gpu_single_torch"]
FORK_BATCH = "gpu_multi_batch"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def loadavg():
    try:
        with open("/proc/loadavg") as fh:
            return fh.read().split()[0]
    except Exception:
        return "?"


# Contention-robust timing: we report the MINIMUM over many repeats. Background
# load (Dropbox/IDE/etc.) can only ADD time, so the minimum is the best estimate
# of the true uncontended compute time; p90/min exposes how noisy the cell was.
def time_backend(backend, cohort, cfg, warmup=1, repeats=9):
    state = backend.prepare(cohort, cfg)
    for _ in range(warmup):
        out = backend.run(state); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = backend.run(state); sync()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return out, {"min": float(a.min()), "med": float(np.median(a)),
                 "p90": float(np.percentile(a, 90))}


def time_jax(mode, cohort, cfg, warmup=1, repeats=9):
    if mode not in jax_shmap.SUPPORTED or jax_shmap.jax_device_count() == 0:
        return None
    try:
        state = jax_shmap.prepare_inputs(mode, cohort, cfg)
        for _ in range(warmup):
            jax_shmap.run(state, cfg)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); jax_shmap.run(state, cfg)
            ts.append(time.perf_counter() - t0)
        return float(np.min(ts))
    except Exception as e:
        print(f"      [jax {mode} skipped: {type(e).__name__}]")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp"])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
    # Time-budget guard: after measuring a batch, only escalate to the next
    # (larger) batch if that batch's fork-path measurement finished within this
    # many seconds. Single threshold applied to every size.
    ap.add_argument("--time-budget", type=float, default=15.0,
                    help="seconds; skip larger batches once a cell's fork measurement exceeds this")
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps-fine", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--lat-batch", type=int, default=8)
    ap.add_argument("--jax-max-batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timing repeats; the MIN is used (contention-robust)")
    ap.add_argument("--cal-repeats", type=int, default=2,
                    help="repeats for the cheap per-pair latency calibration of the linear paths")
    args = ap.parse_args()

    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)
    bk = {b.name: b for b in B.all_backends()}

    print("=" * 96)
    print("REAL-MOLECULE SCENARIO SWEEP: original-repo speed paths vs fork speed paths")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0),
              "| jax:", jax_shmap.jax_device_count(), jax_shmap.jax_platform())
    if jax_shmap.jax_platform() == "cpu":
        print("NOTE: jaxlib is CPU-only here -> the original repo's JAX path runs on CPU.")
    print(f"config: num_repeats={args.num_repeats}, steps={args.steps_fine}, "
          f"timing=MIN over {args.repeats} reps (contention-robust)")
    print("Molecules: curated real drugs (see benchmarks/real_workloads.py provenance).")
    print("=" * 96)

    from shepherd_score.container._core import _band_key
    for mode in args.modes:
        print(f"\n############################  mode = {mode}  ############################")
        tab = molecule_table(mode)
        print("  molecule pool (name heavy mode_count): "
              + ", ".join(f"{n}/{h}/{c}" for n, h, c in tab))
        import os
        for bucket in ("same", "cross"):
            print(f"  [loadavg(1m)={loadavg()} on {os.cpu_count()} cores]")
            # per-pair latency for the O(n) loop paths (measured once on this cohort)
            co_lat = make_real_cohort(mode, n_pairs=args.lat_batch, bucket_kind=bucket, seed=3)
            lat = {}
            for name in ORIG_LINEAR + ["gpu_single_fast"]:
                if not bk[name].available()[0]:
                    continue
                out, st = time_backend(bk[name], co_lat, cfg, warmup=1, repeats=args.cal_repeats)
                lat[name] = st["min"] / len(co_lat)   # uncontended per-pair time
            nbk = len(set(_band_key(p.n_ref) for p in co_lat.pairs))
            print(f"--- {mode}  {bucket}-bucket cohort  ({nbk} GPU bucket(s)) ---")
            print("  per-pair latency (ms, min): "
                  + ", ".join(f"{n.replace('cpu_single_','cpu.').replace('gpu_single_','gpu.')}"
                              f"={lat[n]*1000:.1f}" for n in lat))
            hdr = (f'{"batch":>5s} | {"orig_best":>22s} {"mol/s":>8s} | '
                   f'{"fork_best":>16s} {"mol/s":>8s} {"bk":>3s} {"noise":>6s} | '
                   f'{"SPEEDUP":>8s} | {"fork_score":>10s}')
            print(hdr); print("  " + "-" * (len(hdr) - 2))
            for nb in args.batches:
                cell_t0 = time.perf_counter()   # total wall-clock to test this batch
                orig = {n: lat[n] * nb * 1000.0 for n in ORIG_LINEAR if n in lat}
                if nb <= args.jax_max_batch:
                    co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=3)
                    jms = time_jax(mode, co, cfg, repeats=args.repeats)
                    if jms is not None:
                        orig[f"jax_{jax_shmap.jax_platform()}"] = jms * 1000.0
                fork = {}
                if "gpu_single_fast" in lat:
                    fork["gpu_single_fast"] = lat["gpu_single_fast"] * nb * 1000.0
                fscore, nbuckets, noise = float("nan"), "-", float("nan")
                if bk[FORK_BATCH].available()[0]:
                    co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=3)
                    out, st = time_backend(bk[FORK_BATCH], co, cfg, warmup=1, repeats=args.repeats)
                    fork[FORK_BATCH] = st["min"] * 1000.0
                    fscore = float(out.scores.mean())
                    nbuckets = out.n_buckets
                    noise = st["p90"] / st["min"]      # spread indicator (1.0 = perfectly stable)
                cell_elapsed = time.perf_counter() - cell_t0
                if not orig or not fork:
                    continue
                o = min(orig, key=orig.get); f = min(fork, key=fork.get)
                # molecules scored per second = pairs / wall-clock (each pair = one
                # fit molecule aligned/scored against the reference).
                o_mps = nb * 1000.0 / orig[o]
                f_mps = nb * 1000.0 / fork[f]
                print(f'{nb:5d} | {o:>22s} {o_mps:8.1f} | {f:>16s} {f_mps:8.1f} '
                      f'{str(nbuckets):>3s} {noise:5.2f}x | {f_mps/o_mps:7.1f}x | {fscore:10.4f}')
                # time-budget guard: only escalate to the NEXT (larger) batch if
                # THIS batch's full measurement finished within the budget. So
                # 64 runs only if 16 was < budget, 256 only if 64 was, etc.
                if cell_elapsed > args.time_budget:
                    print(f'    [batch {nb} took {cell_elapsed:.1f}s > {args.time_budget:.0f}s budget '
                          f'-> not escalating to larger batches for this cohort]')
                    break


if __name__ == "__main__":
    raise SystemExit(main())
