"""
Per-cell SUBPROCESS-ISOLATED speed scan.

Every (mode, bucket, batch) cell runs in a FRESH python process, so GPU allocator
state can't accumulate across cells and make late/large cells look artificially
slow -- the exact artifact that made cross-10k read as 705 mol/s in the in-process
scan when it really runs at ~3300. A per-cell wall-clock timeout is a true hard
cap on EVERYTHING (fork and JAX alike), unlike the in-process budget which only
projected the JAX path.

Molecules are built once into a shared on-disk cache (FSS_MOL_CACHE_DIR) so the
per-cell subprocesses don't each rebuild them; the per-cell timeout then bounds
roughly the align (+ a small fixed import/JIT slack).

  driver:   python -m benchmarks.bench_isolated --batches 16 ... 10000
  one cell: python -m benchmarks.bench_isolated --cell surf cross 10000
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile


def _run_one_cell(mode, bucket, batch, skip_jax):
    """Run a single cell in THIS (fresh) process; print FORK/JAX result lines."""
    import torch  # noqa: F401  (ensures CUDA init errors surface here)
    from benchmarks.alignment_bench import backends as B
    from benchmarks.real_workloads import make_real_cohort
    from benchmarks.triton_vs_jax import time_fork, time_jax

    cfg = B.BenchConfig(num_repeats=16, steps_fine=100, max_steps=100, topk=30)
    co = make_real_cohort(mode, n_pairs=batch, bucket_kind=bucket, seed=3)

    # Record the original-pharm-10k SKIP up front: the fork at this size can exceed
    # the cap and get this subprocess killed before any post-fork line would print.
    jax_handled = skip_jax
    if not skip_jax and mode == "pharm" and batch >= 10000:
        print("JAX " + json.dumps({"skip": "pharm10k"}), flush=True)
        jax_handled = True

    try:
        # extra warmup: a fresh process pays Triton compile/autotune + sub-batcher
        # footprint calibration cold, so warm twice before the timed MIN of 3.
        ft, fs = time_fork(mode, co, cfg, 3, warmup=2)
        print("FORK " + json.dumps({"s": ft, "mps": batch / ft, "score": fs}), flush=True)
    except Exception as e:                                   # e.g. CUDA OOM
        print("FORK " + json.dumps({"err": type(e).__name__}), flush=True)

    if jax_handled:
        return
    try:
        jt, js = time_jax(mode, co, cfg, 1)
        print("JAX " + json.dumps({"s": jt, "mps": batch / jt, "score": js}), flush=True)
    except Exception as e:                                   # e.g. JAX pharm degenerate-PCA
        print("JAX " + json.dumps({"err": type(e).__name__}), flush=True)


def _parse(stdout):
    fork = jax = None
    for line in (stdout or "").splitlines():
        if line.startswith("FORK "):
            fork = json.loads(line[5:])
        elif line.startswith("JAX "):
            jax = json.loads(line[4:])
    return fork, jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=3, metavar=("MODE", "BUCKET", "BATCH"))
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--buckets", nargs="+", default=["same", "cross"])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024, 4096, 10000])
    ap.add_argument("--skip-jax", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="hard wall-clock cap per cell (s); cell killed -> TIMEOUT")
    args = ap.parse_args()

    if args.cell:
        _run_one_cell(args.cell[0], args.cell[1], int(args.cell[2]), args.skip_jax)
        return

    cache_dir = os.path.join(tempfile.gettempdir(), "fss_mol_cache")
    os.makedirs(cache_dir, exist_ok=True)
    env = dict(os.environ, FSS_MOL_CACHE_DIR=cache_dir, JAX_PLATFORMS="cpu",
               PYTHONUNBUFFERED="1")

    # One-time molecule build into the shared disk cache so per-cell subprocesses
    # don't each rebuild (otherwise the timeout would mostly time the build).
    print(f"prebuilding molecule disk-cache in {cache_dir} (one-time) ...", flush=True)
    subprocess.run([sys.executable, "-c",
                    "from benchmarks.real_workloads import molecule_table; molecule_table('surf')"],
                   env=env, check=False)

    SLACK = 25.0   # imports + cached cohort build + JAX JIT, outside the align we cap
    print("=" * 104)
    print("FORK Triton-GPU  vs  ORIGINAL JAX-CPU  --  PER-CELL SUBPROCESS-ISOLATED (honest per cell)")
    print(f"fresh process per cell; hard {args.timeout:.0f}s cap on EVERYTHING incl. fork; "
          "JAX projected > cap = TIMEOUT (not run); original pharm 10k = SKIP")
    print("Speedup conflates implementation AND CPU->GPU (jaxlib is CPU-only here).")
    print("=" * 104)
    hdr = (f'{"mode":5s} {"bucket":6s} {"batch":>5s} | {"JAX-CPU s":>9s} {"JAX mol/s":>9s} {"jscore":>6s} | '
           f'{"Triton s":>8s} {"Trit mol/s":>10s} {"tscore":>6s} | {"speedup":>8s}')
    print(hdr); print("-" * len(hdr))
    for mode in args.modes:
        for bucket in args.buckets:
            last_jmps = None                                   # JAX mol/s last cell (~flat -> projects next)
            for batch in args.batches:
                # project JAX time from the prior cell's rate; skip launching it if
                # it would blow the cap (avoids running an over-budget JAX cell to its kill)
                proj_skip = (last_jmps is not None and batch / last_jmps > args.timeout)
                cmd = [sys.executable, "-m", "benchmarks.bench_isolated",
                       "--cell", mode, bucket, str(batch)]
                if args.skip_jax or proj_skip:
                    cmd.append("--skip-jax")
                killed = False
                try:
                    p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                       timeout=args.timeout + SLACK)
                    out = p.stdout
                except subprocess.TimeoutExpired as e:
                    out = e.stdout.decode() if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
                    killed = True

                fork, jax = _parse(out)
                if jax and "mps" in jax:
                    last_jmps = jax["mps"]

                # JAX cell
                if proj_skip:
                    jsec, jmps, jsc = "TIMEOUT", "-", "-"
                elif jax and "mps" in jax:
                    jsec, jmps, jsc = f"{jax['s']:9.3f}", f"{jax['mps']:9.1f}", f"{jax['score']:6.3f}"
                elif jax and jax.get("skip"):
                    jsec, jmps, jsc = "SKIP", "-", "-"
                elif jax and "err" in jax:
                    jsec, jmps, jsc = f"NA:{jax['err']}"[:9], "-", "-"
                elif args.skip_jax:
                    jsec, jmps, jsc = "-", "-", "-"
                else:
                    jsec, jmps, jsc = ("TIMEOUT" if killed else "-"), "-", "-"

                # fork cell
                sp = "-"
                if fork and "mps" in fork:
                    fsec, fmps, fsc = f"{fork['s']:8.3f}", f"{fork['mps']:10.1f}", f"{fork['score']:6.3f}"
                    if jax and "mps" in jax:
                        sp = f"{jax['s'] / fork['s']:7.1f}x"
                elif fork and "err" in fork:
                    fsec, fmps, fsc = f"ERR:{fork['err']}"[:8], "-", "-"
                else:
                    fsec, fmps, fsc = ("TIMEOUT" if killed else "-"), "-", "-"

                print(f'{mode:5s} {bucket:6s} {batch:5d} | {jsec:>9s} {jmps:>9s} {jsc:>6s} | '
                      f'{fsec:>8s} {fmps:>10s} {fsc:>6s} | {sp:>8s}', flush=True)
            print("-" * len(hdr))


if __name__ == "__main__":
    raise SystemExit(main())
