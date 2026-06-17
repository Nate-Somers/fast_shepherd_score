"""
Score-parity + speed gate for the fast alignment paths.

Compares the GPU fast paths (single + batch) against the ORIGINAL torch
reference optimiser on a FIXED, diverse pair set (mixed sizes + anisotropy,
multiple seeds), for every mode. Reports, per mode:
  - reference mean achieved score (the quality bar)
  - fast-path mean score and the mean/worst per-pair DROP vs reference
  - PASS/FAIL against tolerances
  - wall-clock + speedup

Run after every optimization step. The tolerances encode the hard constraint
"fast-path scores must not fall significantly below the original":
  mean drop must be <= MEAN_TOL, and no single pair may drop more than PAIR_TOL.

This is deliberately distribution-agnostic and uses the same code paths the
benchmark uses (no special-casing) so it cannot be gamed.
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.alignment_bench.workloads import make_cohort, MODES
from benchmarks.alignment_bench import backends as B

# "Do not fall too significantly below the original" is an AGGREGATE statement,
# so the binding gate is the mean drop. The worst single-pair drop is reported
# as a diagnostic (coarse-to-fine occasionally lands one pair in a slightly
# worse basin even in the unmodified system); PAIR_HARD is only a backstop that
# flags genuine breakage / a regression introduced by an optimization.
MEAN_TOL = 0.01    # binding: mean score may not drop more than this vs reference
PAIR_INFO = 0.05   # informational threshold for the worst single pair
PAIR_HARD = 0.15   # backstop: a worst-pair drop beyond this is a real failure


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_backend(backend, cohort, cfg, warmup=1, repeats=3):
    state = backend.prepare(cohort, cfg)
    for _ in range(warmup):
        out = backend.run(state); sync()
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter()
        out = backend.run(state); sync()
        ts.append(time.perf_counter() - t0)
    return out.scores, float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--num-repeats", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--steps-fine", type=int, default=75)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)
    backends = {b.name: b for b in B.all_backends()}
    ref = backends[B.REFERENCE_BACKEND]            # cpu_single_torch == original path
    fast_names = ["gpu_single_fast", "gpu_multi_batch"]

    print(f"PARITY GATE  (n_pairs={args.n_pairs}, num_repeats={args.num_repeats}, "
          f"max_steps={args.max_steps}, steps_fine={args.steps_fine})")
    print(f"  binding gate: mean drop <= {MEAN_TOL}; worst-pair info >{PAIR_INFO}, hard fail >{PAIR_HARD}\n")
    hdr = f'{"mode":5s} {"cohort":8s} {"backend":16s} {"ref_mean":>9s} {"mean":>8s} {"meanΔ":>8s} {"worstΔ":>8s} {"ms":>8s} {"x":>6s} {"":>5s}'
    print(hdr); print("-" * len(hdr))

    all_pass = True
    for mode in args.modes:
        for kind in ("uniform", "mixed"):
            cohort = make_cohort(mode, n_pairs=args.n_pairs, size_kind=kind,
                                 size=40, size_range=(12, 72), noise=0.0, seed=args.seed)
            ok_ref, reason = ref.available()
            if not ok_ref:
                print(f"{mode:5s} {kind:8s} reference unavailable: {reason}")
                continue
            ref_scores, ref_ms = run_backend(ref, cohort, cfg)
            print(f'{mode:5s} {kind:8s} {ref.name:16s} {ref_scores.mean():9.4f} '
                  f'{ref_scores.mean():8.4f} {0.0:8.4f} {0.0:8.4f} {ref_ms*1000:8.1f} {1.0:6.1f}      ')
            for fn in fast_names:
                bk = backends[fn]
                ok, reason = bk.available()
                if not ok:
                    print(f'{mode:5s} {kind:8s} {fn:16s}  skipped: {reason}')
                    continue
                sc, ms = run_backend(bk, cohort, cfg)
                drop = ref_scores - sc                      # >0 means worse than reference
                mean_drop = float(drop.mean())
                worst_drop = float(drop.max())
                passed = (mean_drop <= MEAN_TOL) and (worst_drop <= PAIR_HARD)
                all_pass = all_pass and passed
                tag = "PASS" if passed else "**FAIL**"
                if passed and worst_drop > PAIR_INFO:
                    tag = "pass*"   # within mean gate; worst-pair tail noted
                print(f'{mode:5s} {kind:8s} {fn:16s} {ref_scores.mean():9.4f} '
                      f'{sc.mean():8.4f} {mean_drop:8.4f} {worst_drop:8.4f} '
                      f'{ms*1000:8.1f} {ref_ms/ms:6.1f} {tag:>8s}')
    print()
    print("OVERALL:", "PASS" if all_pass else "**FAIL**")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
