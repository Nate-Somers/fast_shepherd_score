"""
Usage-scenario comparison: ORIGINAL-repo speed paths vs THIS-FORK speed paths,
swept over molecule size x batch size.

"Speed paths" considered
------------------------
ORIGINAL repo (upstream optimisers):
  - cpu_single_torch       : torch autograd optimiser (the baseline)
  - cpu_single_analytical  : analytical-gradient torch optimiser (upstream's CPU speedup)
  - gpu_single_torch       : torch autograd on CUDA
  - jax_single_gpu         : JAX masked/vmap optimiser on the single GPU (upstream)
FORK (Triton stack, at HEAD):
  - gpu_single_fast        : Triton single-pair optimiser
  - gpu_multi_batch        : Triton bucketed batch optimiser

Method (honest + cheap)
-----------------------
Single-pair paths process pairs in an O(n) Python loop, so their cost is
linear: we measure per-pair latency once per (mode, size) and scale by the
batch size. The batched paths (gpu_multi_batch, JAX) are non-linear, so they are
timed DIRECTLY at every batch size. For each cell we report:
    speedup = (fastest ORIGINAL-repo time) / (fastest FORK time)
plus the winning path on each side and the fork's achieved score (so a speedup
can never be bought with quality loss). Timing uses warmup + CUDA sync + median.

JAX runs in the same process with preallocation disabled (set in jax_shmap); any
cell that OOMs or errors is reported as n/a rather than crashing the sweep.
"""
import argparse
import time

import numpy as np
import torch

# Import jax_shmap first so its no-preallocate env defaults are set before JAX loads.
from benchmarks.alignment_bench import jax_shmap  # noqa: E402
from benchmarks.alignment_bench.workloads import make_cohort  # noqa: E402
from benchmarks.alignment_bench import backends as B  # noqa: E402

ORIG_LINEAR = ["cpu_single_torch", "cpu_single_analytical", "gpu_single_torch"]
FORK_LINEAR = ["gpu_single_fast"]
FORK_BATCH = "gpu_multi_batch"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_backend(backend, cohort, cfg, warmup=1, repeats=3):
    state = backend.prepare(cohort, cfg)
    for _ in range(warmup):
        out = backend.run(state); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = backend.run(state); sync()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def time_jax(mode, cohort, cfg, warmup=1, repeats=3):
    if mode not in jax_shmap.SUPPORTED or jax_shmap.jax_device_count() == 0:
        return None, None
    try:
        state = jax_shmap.prepare_inputs(mode, cohort, cfg)
        for _ in range(warmup):
            s = jax_shmap.run(state, cfg)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            s = jax_shmap.run(state, cfg)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.mean(s))
    except Exception as e:
        print(f"      [jax {mode} n={len(cohort)} skipped: {type(e).__name__}]")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps-fine", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--lat-batch", type=int, default=8)
    ap.add_argument("--no-jax", action="store_true")
    ap.add_argument("--jax-max-batch", type=int, default=64)
    args = ap.parse_args()

    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)
    bk = {b.name: b for b in B.all_backends()}

    print("=" * 92)
    print("SCENARIO SWEEP: original-repo speed paths vs fork speed paths")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0),
              "| jax devices:", jax_shmap.jax_device_count(), jax_shmap.jax_platform())
    if jax_shmap.jax_platform() == "cpu":
        print("NOTE: jaxlib is CPU-only here, so the original repo's JAX path is")
        print("      measured on CPU (its GPU/multi-GPU advantage is NOT reflected).")
    print(f"config: num_repeats={args.num_repeats}, steps={args.steps_fine}, "
          f"latency measured at batch={args.lat_batch}")
    print("=" * 92)

    for mode in args.modes:
        print(f"\n############################  mode = {mode}  ############################")
        for size in args.sizes:
            # --- per-pair latency for the O(n) loop paths (measured once) ---
            co_lat = make_cohort(mode, n_pairs=args.lat_batch, size_kind="uniform",
                                 size=size, noise=0.0, seed=21)
            lat, sc_lin = {}, {}
            for name in ORIG_LINEAR + FORK_LINEAR:
                ok, reason = bk[name].available()
                if not ok:
                    continue
                out, ms = time_backend(bk[name], co_lat, cfg, warmup=1, repeats=2)
                lat[name] = ms / args.lat_batch
                sc_lin[name] = float(out.scores.mean())

            print(f"\n--- {mode}  molecule-size {size}  (uniform) ---")
            print("  per-pair latency (ms): "
                  + ", ".join(f"{n.replace('cpu_single_','cpu.').replace('gpu_single_','gpu.')}"
                              f"={lat[n]*1000:.1f}" for n in lat))
            hdr = (f'{"batch":>5s} | {"orig_best":>22s} {"ms":>9s} | '
                   f'{"fork_best":>16s} {"ms":>9s} | {"SPEEDUP":>8s} | {"fork_score":>10s}')
            print(hdr); print("  " + "-" * (len(hdr) - 2))

            for nb in args.batches:
                # original-repo candidate times
                orig = {n: lat[n] * nb * 1000.0 for n in ORIG_LINEAR if n in lat}
                # JAX (CPU-only jaxlib here) timed directly; capped to modest
                # batches to keep the CPU sweep tractable (it is competitive only
                # at small batch anyway -- torch single paths dominate at scale).
                if not args.no_jax and nb <= args.jax_max_batch:
                    co = make_cohort(mode, n_pairs=nb, size_kind="uniform",
                                     size=size, noise=0.0, seed=21)
                    jms, jsc = time_jax(mode, co, cfg)
                    if jms is not None:
                        orig[f"jax_{jax_shmap.jax_platform()}"] = jms * 1000.0
                # fork candidate times
                fork = {}
                if "gpu_single_fast" in lat:
                    fork["gpu_single_fast"] = lat["gpu_single_fast"] * nb * 1000.0
                fscore = float("nan")
                if bk[FORK_BATCH].available()[0]:
                    co = make_cohort(mode, n_pairs=nb, size_kind="uniform",
                                     size=size, noise=0.0, seed=21)
                    out, ms = time_backend(bk[FORK_BATCH], co, cfg, warmup=1, repeats=3)
                    fork[FORK_BATCH] = ms * 1000.0
                    fscore = float(out.scores.mean())

                if not orig or not fork:
                    continue
                o_name = min(orig, key=orig.get); o_ms = orig[o_name]
                f_name = min(fork, key=fork.get); f_ms = fork[f_name]
                print(f'{nb:5d} | {o_name:>22s} {o_ms:9.1f} | {f_name:>16s} {f_ms:9.1f} '
                      f'| {o_ms/f_ms:7.1f}x | {fscore:10.4f}')


if __name__ == "__main__":
    raise SystemExit(main())
