"""
CLI entry point for the alignment speed + accuracy benchmark.

Examples
--------
# Quick smoke run (small, fast) on whatever hardware is present:
python -m benchmarks.run_alignment_benchmark --quick

# Full run, write markdown + json:
python -m benchmarks.run_alignment_benchmark \
    --modes vol surf esp pharm --n-pairs 64 \
    --out-md bench.md --out-json bench.json

# Stress the bucketing axis with a wide mixed-size range:
python -m benchmarks.run_alignment_benchmark --mixed-range 8 120
"""
from __future__ import annotations

import argparse
import json
import sys

from benchmarks.alignment_bench.workloads import MODES
from benchmarks.alignment_bench import backends as B
from benchmarks.alignment_bench.runner import run_matrix, to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--uniform-size", type=int, default=30,
                    help="heavy-atom count for the uniform cohort")
    ap.add_argument("--mixed-range", type=int, nargs=2, default=(10, 60),
                    metavar=("LO", "HI"), help="heavy-atom range for the mixed cohort")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian position noise on the fit (0 => exact optimum=1.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-repeats", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=100, help="torch reference steps")
    ap.add_argument("--steps-fine", type=int, default=75, help="fast/batch fine steps")
    ap.add_argument("--cpu-workers", type=int, default=0, help="0 => os.cpu_count()")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="tiny/fast settings for a smoke check")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.n_pairs = min(args.n_pairs, 8)
        args.uniform_size = min(args.uniform_size, 16)
        args.mixed_range = (8, 24)
        args.num_repeats = min(args.num_repeats, 8)
        args.max_steps = min(args.max_steps, 40)
        args.steps_fine = min(args.steps_fine, 40)
        args.repeats = min(args.repeats, 2)

    cfg = B.BenchConfig(
        num_repeats=args.num_repeats,
        max_steps=args.max_steps,
        steps_fine=args.steps_fine,
        cpu_workers=args.cpu_workers,
    )

    results = run_matrix(
        modes=tuple(args.modes),
        n_pairs=args.n_pairs,
        uniform_size=args.uniform_size,
        mixed_range=tuple(args.mixed_range),
        noise=args.noise,
        seed=args.seed,
        cfg=cfg,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    md = to_markdown(results)
    print(md)

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n[wrote markdown -> {args.out_md}]", file=sys.stderr)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[wrote json -> {args.out_json}]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
