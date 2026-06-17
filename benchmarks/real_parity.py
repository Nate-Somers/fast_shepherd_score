"""Real-molecule parity: fork score vs ORIGINAL (torch reference) score on the
same real cohorts, so the speed claims can't hide a quality loss on real data."""
import argparse
import numpy as np

from benchmarks.alignment_bench import backends as B
from benchmarks.real_workloads import make_real_cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--num-repeats", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--steps-fine", type=int, default=100)
    args = ap.parse_args()
    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)
    bk = {b.name: b for b in B.all_backends()}
    ref = bk[B.REFERENCE_BACKEND]              # cpu_single_torch == original repo
    fork = bk["gpu_multi_batch"]

    print(f"REAL-MOLECULE PARITY (orig=cpu_single_torch vs fork=gpu_multi_batch)")
    print(f"config: n_pairs={args.n_pairs}, num_repeats={args.num_repeats}, "
          f"steps_fine={args.steps_fine}, max_steps={args.max_steps}\n")
    hdr = f'{"mode":5s} {"bucket":6s} {"orig_mean":>9s} {"fork_mean":>9s} {"mean_drop":>9s} {"worst_drop":>10s}'
    print(hdr); print("-" * len(hdr))
    for mode in args.modes:
        for bucket in ("same", "cross"):
            co = make_real_cohort(mode, n_pairs=args.n_pairs, bucket_kind=bucket, seed=5)
            so = ref.run(ref.prepare(co, cfg)).scores
            ok, reason = fork.available()
            if not ok:
                print(f"{mode:5s} {bucket:6s} fork unavailable: {reason}"); continue
            sf = fork.run(fork.prepare(co, cfg)).scores
            drop = so - sf
            print(f'{mode:5s} {bucket:6s} {so.mean():9.4f} {sf.mean():9.4f} '
                  f'{float(drop.mean()):9.4f} {float(drop.max()):10.4f}')


if __name__ == "__main__":
    raise SystemExit(main())
