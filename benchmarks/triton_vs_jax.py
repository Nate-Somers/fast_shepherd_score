"""
Head-to-head: FORK Triton-GPU batch  vs  ORIGINAL JAX-CPU batch, scanning batch
size 16 -> 1024 across all alignment types (surf, esp, pharm), same- and
cross-bucket cohorts of real-drug self-SE(3)-copy pairs.

DEVICE NOTE (important, not apples-to-apples): the fork's `align_batch_*` runs on
the GPU (Triton/CUDA); the original's JAX batch (`shepherd_score._jax`) runs on
CPU here because jaxlib is CPU-only on this box. So "speedup" conflates the
implementation AND the CPU->GPU jump.

TIMEOUT semantics: JAX is timed at each size in increasing order; once a size
exceeds --budget seconds, that cell is kept (marked '*' = over budget) and all
LARGER sizes for that cohort are reported TIMEOUT (not run). The fork is timed at
every size regardless. 'NA' = JAX raised (e.g. pharm degenerate-PCA crash).
"""
import argparse
import time
import numpy as np
import torch

from benchmarks.alignment_bench import jax_shmap
from benchmarks.alignment_bench import backends as B
from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fork(mode, cohort, cfg, reps, warmup=1):
    dev = torch.device("cuda")
    pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev) for p in cohort.pairs]

    def run():
        if mode == "surf":
            MoleculePair.align_batch_surf(pairs, alpha=cfg.alpha, steps_fine=cfg.steps_fine)
        elif mode == "esp":
            MoleculePair.align_batch_esp(pairs, alpha=cfg.alpha, lam=cfg.lam,
                                         num_repeats=cfg.num_repeats, topk=cfg.topk,
                                         steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
        elif mode == "pharm":
            MoleculePair.align_batch_pharm(pairs, num_repeats=cfg.num_repeats, topk=cfg.topk,
                                           steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
    for _ in range(warmup):
        run(); _sync()
    ts = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter(); run(); _sync()
        ts.append(time.perf_counter() - t0)
    if mode == "surf":
        sc = np.array([p.sim_aligned_surf for p in pairs], dtype=float)
    elif mode == "esp":
        sc = np.array([p.sim_aligned_esp for p in pairs], dtype=float)
    else:
        sc = np.array([p.sim_aligned_pharm for p in pairs], dtype=float)
    return float(np.min(ts)), float(sc.mean())


def time_jax(mode, cohort, cfg, reps, warmup=1):
    """min timed run (post-JIT) + mean score, or raises."""
    state = jax_shmap.prepare_inputs(mode, cohort, cfg)
    for _ in range(warmup):
        jax_shmap.run(state, cfg)
    ts = []
    sc = None
    for _ in range(reps):
        t0 = time.perf_counter()
        sc = jax_shmap.run(state, cfg)
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), float(np.asarray(sc).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["surf", "esp", "pharm"])
    ap.add_argument("--buckets", nargs="+", default=["same", "cross"])
    ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--budget", type=float, default=15.0,
                    help="seconds; once a JAX size exceeds this, larger sizes are TIMEOUT")
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps-fine", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--fork-reps", type=int, default=2)
    ap.add_argument("--jax-reps", type=int, default=1)
    ap.add_argument("--skip-jax", action="store_true",
                    help="time only the Triton fork path (fast; for A/B-ing fork changes)")
    args = ap.parse_args()

    cfg = B.BenchConfig(num_repeats=args.num_repeats, max_steps=args.max_steps,
                        steps_fine=args.steps_fine, topk=30)

    print("=" * 104)
    print("FORK Triton-GPU batch  vs  ORIGINAL JAX-CPU batch  (real drugs, self-SE(3)-copy, optimum=1.0)")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), "| JAX:", jax_shmap.jax_device_count(),
              jax_shmap.jax_platform())
    print(f"config: num_repeats/restarts={args.num_repeats}, steps={args.steps_fine}, "
          f"JAX budget={args.budget:.0f}s, timing=MIN (fork {args.fork_reps} reps / jax {args.jax_reps} reps post-JIT)")
    print("Speedup conflates implementation AND CPU->GPU (jaxlib is CPU-only here).")
    print("=" * 104)
    hdr = (f'{"mode":5s} {"bucket":6s} {"batch":>5s} | {"JAX-CPU s":>10s} {"JAX mol/s":>9s} {"jscore":>6s} | '
           f'{"Triton s":>9s} {"Trit mol/s":>10s} {"tscore":>6s} | {"speedup":>8s}')
    print(hdr); print("-" * len(hdr))

    for mode in args.modes:
        for bucket in args.buckets:
            jax_status = "na" if args.skip_jax else "alive"   # alive | timeout | na
            for nb in args.batches:
                co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=3)
                try:
                    ftime, fscore = time_fork(mode, co, cfg, args.fork_reps)
                except Exception as e:  # e.g. CUDA OOM at very large batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f'{mode:5s} {bucket:6s} {nb:5d} |  fork ERR: {type(e).__name__}')
                    continue
                f_mps = nb / ftime

                jcell, jmps_s, jsc_s, sp_s = "TIMEOUT", "-", "-", "-"
                if jax_status == "alive":
                    try:
                        jtime, jscore = time_jax(mode, co, cfg, args.jax_reps)
                        over = jtime > args.budget
                        jcell = f"{jtime:9.3f}" + ("*" if over else " ")
                        jmps_s = f"{nb / jtime:9.1f}"
                        jsc_s = f"{jscore:6.3f}"
                        sp_s = f"{jtime / ftime:7.1f}x"
                        if over:
                            jax_status = "timeout"
                    except Exception as e:
                        jcell, jmps_s, jsc_s, sp_s = f"NA:{type(e).__name__}", "-", "-", "-"
                        jax_status = "na"
                elif jax_status == "na":
                    jcell = "NA"

                print(f'{mode:5s} {bucket:6s} {nb:5d} | {jcell:>10s} {jmps_s:>9s} {jsc_s:>6s} | '
                      f'{ftime:9.3f} {f_mps:10.1f} {fscore:6.3f} | {sp_s:>8s}')
                # Free per-cell GPU memory so a big batch can't starve the next.
                try:
                    import shepherd_score.container._core as _cc
                    _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
                except Exception:
                    pass
                del co
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("-" * len(hdr))


if __name__ == "__main__":
    raise SystemExit(main())
