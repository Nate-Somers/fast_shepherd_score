"""
Speed-task A/B lab (fork-only, no original-repo reruns).

This GPU's clock jitters ~2.5x, so absolute best-of-N is unreliable across phases.
We therefore PAIR: within each rep, time the baseline AND every variant back-to-back
(interleaved), then report the MEDIAN per-rep ratio baseline_time/variant_time. Both
configs see the same clock inside a rep, so the ratio is robust to slow drift.

Accuracy is deterministic, measured once per config: self-copy min (want 1.000) and
distinct-molecule-pair max|Δ| vs baseline (want 0). A change ships only if the paired
speedup > 1 AND accuracy is preserved.

  python -m benchmarks.speedlab --modes vol surf --batch 4096 --reps 8 --es-patience 3
"""
import argparse
import statistics
import time

import numpy as np
import torch

from benchmarks.real_workloads import make_real_cohort, _build_molecule, DRUGS
from shepherd_score.container import MoleculePair as MP
import shepherd_score.alignment.utils.fast_se3 as fse3
import shepherd_score.alignment.utils.fast_common as fcom
import shepherd_score.container._core as cc

dev = torch.device("cuda")
ATTR = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
        "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}
KW = {
    "vol":   dict(alpha=0.81, steps_fine=100),
    "surf":  dict(alpha=0.81, steps_fine=100),
    "esp":   dict(alpha=0.81, lam=0.3, num_repeats=50, topk=30, steps_fine=100, lr=0.075),
    "pharm": dict(num_repeats=50, topk=30, steps_fine=100, lr=0.1),
}
FN = {"vol": MP.align_batch_vol, "surf": MP.align_batch_surf,
      "esp": MP.align_batch_esp, "pharm": MP.align_batch_pharm}


def _align(mode, pairs):
    FN[mode](pairs, **KW[mode])
    return np.array([float(getattr(p, ATTR[mode])) for p in pairs])


def _self_pairs(mode, n, bucket="same"):
    co = make_real_cohort(mode, n_pairs=n, bucket_kind=bucket, seed=3)
    return [MP(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]


def _distinct_pairs(mode, n, seed=7):
    rng = np.random.default_rng(seed)
    smis = [s for _, s, _ in DRUGS]
    sel = []
    while len(sel) < n:
        i, j = int(rng.integers(len(smis))), int(rng.integers(len(smis)))
        if i != j:
            sel.append((smis[i], smis[j]))
    return [MP(_build_molecule(a), _build_molecule(b), do_center=False, device=dev) for a, b in sel]


def _time_once(mode, pairs):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _align(mode, pairs); torch.cuda.synchronize()
    return time.perf_counter() - t0


def ab(modes, batch, reps, strategies, ndist=40):
    self_pairs = {m: _self_pairs(m, batch) for m in modes}
    distinct_pairs = {m: _distinct_pairs(m, ndist) for m in modes}
    names = [n for n, _ in strategies]

    for name, setup in strategies:                                   # warm every (config, mode)
        setup()
        for m in modes:
            _align(m, self_pairs[m])
    torch.cuda.synchronize()

    t = {n: {m: [] for m in modes} for n in names}                   # interleaved paired timing
    for m in modes:
        for _ in range(reps):
            for name, setup in strategies:
                setup()
                t[name][m].append(_time_once(m, self_pairs[m]))

    acc = {}                                                          # deterministic accuracy, once
    for name, setup in strategies:
        setup()
        acc[name] = {m: (_align(m, self_pairs[m]), _align(m, distinct_pairs[m])) for m in modes}

    base = names[0]
    print(f"\n{'config':20s} {'mode':5s} {'pairs/s':>9s} {'speedup':>8s} {'self':>7s} {'dist|Δ|':>9s}")
    print("-" * 64)
    for name in names:
        for m in modes:
            tp = len(self_pairs[m]) / min(t[name][m])
            self_s, dist_s = acc[name][m]
            ratios = [tb / tn for tb, tn in zip(t[base][m], t[name][m])]
            sp = statistics.median(ratios)
            dd = float(np.abs(dist_s - acc[base][m][1]).max())
            print(f"{name:20s} {m:5s} {tp:9.0f} {sp:7.2f}x {self_s.min():7.4f} {dd:9.2e}", flush=True)
    return t, acc


_GRAPH_MAX_P_DEFAULT = fse3._GRAPH_MAX_P


def _off():
    fse3._PRUNE_AFTER = 0; fse3._PRUNE_KEEP = 0
    fse3._ES_PATIENCE = None; fse3._ES_TOL = None
    fcom.ES_PATIENCE_OVERRIDE = None                      # esp/pharm
    cc._NUM_SEEDS = None
    fse3._GRAPH_MAX_P = _GRAPH_MAX_P_DEFAULT


def _graphs():
    def f():
        _off(); fse3._GRAPH_MAX_P = 10 ** 9               # enable CUDA-graph fine loop at any batch
    return f


def _prune(after, keep):
    def f():
        _off(); fse3._PRUNE_AFTER = after; fse3._PRUNE_KEEP = keep
    return f


def _es(patience):
    def f():
        _off(); fse3._ES_PATIENCE = patience; fcom.ES_PATIENCE_OVERRIDE = patience
    return f


def _seeds(n):
    def f():
        _off(); cc._NUM_SEEDS = n
    return f


def _combo(patience, n):
    def f():
        _off(); fse3._ES_PATIENCE = patience; cc._NUM_SEEDS = n
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["vol", "surf"])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ndist", type=int, default=40)
    ap.add_argument("--prune", nargs=2, type=int, metavar=("AFTER", "KEEP"), action="append")
    ap.add_argument("--es-patience", type=int, action="append")
    ap.add_argument("--seeds", type=int, action="append")
    ap.add_argument("--combo", nargs=2, type=int, metavar=("PATIENCE", "SEEDS"), action="append")
    ap.add_argument("--graphs", action="store_true", help="A/B CUDA-graph fine loop at the test batch")
    args = ap.parse_args()
    strategies = [("baseline", _off)]
    if args.graphs:
        strategies.append(("cuda_graphs", _graphs()))
    for after, keep in (args.prune or []):
        strategies.append((f"prune@{after}/keep{keep}", _prune(after, keep)))
    for p in (args.es_patience or []):
        strategies.append((f"es_patience={p}", _es(p)))
    for n in (args.seeds or []):
        strategies.append((f"seeds={n}", _seeds(n)))
    for p, n in (args.combo or []):
        strategies.append((f"es{p}+seeds{n}", _combo(p, n)))
    ab(args.modes, args.batch, args.reps, strategies, args.ndist)


if __name__ == "__main__":
    raise SystemExit(main())
