"""
Figure 6 (data) — retrospective virtual-screening enrichment + the ESP ablation.

The decisive utility test for the ESP differentiator: does aligned shape+ESP retrieve
known actives better than shape alone? For a target with known actives + decoys we
align+score the library against query actives and compute enrichment
(ROC-AUC, EF1%, EF5%, BEDROC alpha=20).

This is a rewrite that fixes the methodology problems found in review:

  * FAIR BUDGET.  On the Triton backend ``align_with_surf`` ignores ``num_repeats``
    (it uses the ``FINE_NUM_SEEDS`` env, default 50) while ``align_with_esp`` honours
    ``num_repeats`` — so the old harness compared surf @50 seeds vs esp @16 seeds.
    Here we pin ``FINE_NUM_SEEDS`` AND pass an identical ``num_repeats``/``max_num_steps``
    to every mode, assert it, and record it, so surf-vs-esp is a real ablation.
  * MULTI-QUERY.  Single-query enrichment is extremely high variance.  We average over
    K query actives (fixed seed) and report mean +/- std + a bootstrap CI across queries,
    plus the paired esp-vs-surf delta.
  * CORRECT METRICS.  Tie-aware ROC-AUC (Mann-Whitney average-rank, matches sklearn for
    ties; the old np.trapz path both mishandled ties and crashes on numpy>=2.0).  EF and
    BEDROC(alpha=20) were verified correct against rdkit and Truchon&Bayly 2007 and kept.
  * CHECKPOINTED.  Conformer + xTB single point per molecule is the cost; each built
    molecule is cached to disk, so the expensive precompute is resumable and a crash at
    hour 10 of a long job is recoverable.  Precompute (CPU/xTB) is separable from the GPU
    screen.
  * lam SWEEP.  fig4 showed the ESP-Tanimoto term is shape-dominated at the default
    lam=0.3 and discriminates only at smaller lam (derived with MMFF charges).  Here we
    sweep lam with the SAME (xTB) charge model the screen uses and report enrichment vs lam,
    tying fig4's score-level finding to actual retrieval utility.

Run (GPU env, xTB on PATH):
    # build the molecule cache (CPU/xTB; resumable, parallel) then screen on the GPU:
    PYTHONPATH=. python paper/fig6_enrichment/run.py \
        --target ACES --actives data/ACES/ligands.smi --decoys data/ACES/decoys.smi \
        --n-decoys 3000 --queries 8 --modes surf esp --lams 0.3 0.1 0.03 0.01 0.003
    # CPU-only precompute (array-job friendly), no GPU needed:
    PYTHONPATH=. python paper/fig6_enrichment/run.py --target ACES --actives ... --decoys ... --build-only
    # tiny end-to-end smoke test on the curated drugs (NOT a benchmark):
    PYTHONPATH=. python paper/fig6_enrichment/run.py --smoke
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# xtb binary sits next to the python interpreter in the conda env
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fixed alignment knobs (recorded into the output) ----------------------
ALPHA = 0.81
NUM_SEEDS = 50          # SE(3) starts per pair; pins BOTH surf (via FINE_NUM_SEEDS) and esp (num_repeats)
MAX_STEPS = 200         # optimizer steps (repo default)
SURF_POINTS = 200       # fixed surface-point count for ALL molecules -> uniform, fair batches
SEED = 42               # conformer + query-sampling seed


# ===========================================================================
# Enrichment metrics
# ===========================================================================
def _avg_ranks(s: np.ndarray) -> np.ndarray:
    """1-indexed ranks of s with ties averaged (ascending)."""
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    r = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = np.empty(len(s), dtype=float)
    out[order] = r
    return out


def roc_auc(labels, scores) -> float:
    """Tie-aware ROC-AUC via the Mann-Whitney rank identity (== sklearn roc_auc_score,
    including correct averaging of tied scores; aligned 3D-similarity scores produce ties)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    P = int(y.sum()); Nn = int((1 - y).sum())
    if P == 0 or Nn == 0:
        return float("nan")
    ranks = _avg_ranks(s)
    return float((ranks[y == 1].sum() - P * (P + 1) / 2.0) / (P * Nn))


def enrichment_factor(labels, scores, frac) -> float:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = np.asarray(labels)[order]
    n = max(1, int(round(frac * len(y))))
    base = y.mean()
    return float(y[:n].mean() / base) if base > 0 else float("nan")


def bedroc(labels, scores, alpha=20.0) -> float:
    """Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly, JCIM 2007).
    Verified numerically against rdkit.ML.Scoring.CalcBEDROC."""
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = np.asarray(labels)[order]
    N = len(y); n = int(y.sum())
    if n == 0 or n == N:
        return float("nan")
    Ra = n / N
    ranks = np.where(y == 1)[0] + 1.0
    rie = (np.sum(np.exp(-alpha * ranks / N)) / n) / \
          ((1 - np.exp(-alpha)) / (N * (np.exp(alpha / N) - 1)))
    return float(rie * (Ra * np.sinh(alpha / 2)) /
                 (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
                 + 1.0 / (1 - np.exp(alpha * (1 - Ra))))


def all_metrics(labels, scores) -> dict:
    return {"auc": roc_auc(labels, scores),
            "ef1": enrichment_factor(labels, scores, 0.01),
            "ef5": enrichment_factor(labels, scores, 0.05),
            "bedroc": bedroc(labels, scores, 20.0)}


def boot_ci(vals, B=10000, seed=0):
    """Bootstrap (percentile) 95% CI of the mean of vals."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    if len(v) == 1:
        return [float(v[0]), float(v[0])]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ===========================================================================
# Molecule build + disk cache (conformer + xTB charges + surface + pharm)
# ===========================================================================
def _cache_path(cache_dir, smi, use_xtb):
    key = hashlib.md5(f"v2|{smi}|xtb={use_xtb}|sp={SURF_POINTS}|seed={SEED}".encode()).hexdigest()
    return os.path.join(cache_dir, key + ".pkl")


def build_realmol(smi, use_xtb=True):
    """Build a lightweight, picklable molecule (atom/surf/esp/pharm arrays) exactly like
    benchmarks.benchmark._RealMol, so it feeds MoleculePair directly.  Fixed SURF_POINTS
    for every molecule -> uniform surface arrays -> clean batched surf/esp alignment."""
    from rdkit import Chem
    from shepherd_score.conformer_generation import (
        embed_conformer_from_smiles, charges_from_single_point_conformer_with_xtb)
    from shepherd_score.container import Molecule
    from benchmarks.benchmark import _RealMol
    rd = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=SEED)
    if rd is None:
        return None
    q = charges_from_single_point_conformer_with_xtb(rd) if use_xtb else None
    m = Molecule(rd, num_surf_points=SURF_POINTS, partial_charges=q, pharm_multi_vector=False)
    return _RealMol(
        atom_pos=np.asarray(m.atom_pos, dtype=np.float64),
        surf_pos=np.asarray(m.surf_pos, dtype=np.float64),
        surf_esp=np.asarray(m.surf_esp, dtype=np.float64),
        partial_charges=np.asarray(m.partial_charges, dtype=np.float64),
        pharm_types=np.asarray(m.pharm_types, dtype=np.int64),
        pharm_ancs=np.asarray(m.pharm_ancs, dtype=np.float64),
        pharm_vecs=np.asarray(m.pharm_vecs, dtype=np.float64),
    )


def build_cached(args):
    """Build one molecule into the cache (resumable).  Returns (smi, status):
    status 'hit'|'built'|'fail'.  A deterministic failure (bad SMILES) is marked so it is
    not retried; transient exceptions are NOT cached so a re-run can retry them."""
    smi, cache_dir, use_xtb = args
    cp = _cache_path(cache_dir, smi, use_xtb)
    if os.path.exists(cp):
        return (smi, "hit")
    try:
        rm = build_realmol(smi, use_xtb=use_xtb)
        tmp = cp + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump(rm, fh)               # rm may be None (bad SMILES) -> cached as a skip
        os.replace(tmp, cp)
        return (smi, "built" if rm is not None else "fail")
    except Exception:
        traceback.print_exc()
        return (smi, "fail")


def load_cached(cache_dir, smi, use_xtb):
    cp = _cache_path(cache_dir, smi, use_xtb)
    if not os.path.exists(cp):
        return None
    try:
        with open(cp, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def precompute(smis, cache_dir, use_xtb, workers):
    """Build all molecules into cache_dir, in parallel, resumably."""
    os.makedirs(cache_dir, exist_ok=True)
    todo = [s for s in smis if not os.path.exists(_cache_path(cache_dir, s, use_xtb))]
    print(f"precompute: {len(smis)} molecules, {len(smis)-len(todo)} cached, {len(todo)} to build "
          f"(xtb={use_xtb}, workers={workers})", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    items = [(s, cache_dir, use_xtb) for s in todo]
    done = 0
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # avoid fork issues with open3d/rdkit
        try:
            with ctx.Pool(processes=workers, maxtasksperchild=25) as pool:
                for smi, status in pool.imap_unordered(build_cached, items, chunksize=1):
                    done += 1
                    if done % 50 == 0 or done == len(items):
                        print(f"  built {done}/{len(items)} ({time.perf_counter()-t0:.0f}s)", flush=True)
            return
        except Exception as e:
            print(f"  pool failed ({type(e).__name__}: {e}); falling back to sequential for the rest", flush=True)
    for it in items:
        if os.path.exists(_cache_path(cache_dir, it[0], use_xtb)):
            continue
        build_cached(it); done += 1
        if done % 50 == 0:
            print(f"  built {done} ({time.perf_counter()-t0:.0f}s)", flush=True)


# ===========================================================================
# GPU screen
# ===========================================================================
def screen_one(query, lib, mode, lam):
    """Align every lib molecule to `query` on the GPU; return aligned similarity per mode.
    Equal optimization budget across modes is enforced by the caller (FINE_NUM_SEEDS env +
    identical num_repeats/max_num_steps)."""
    import torch
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps = [MoleculePair(query, m, do_center=True, device=dev) for m in lib]
    b = MoleculePairBatch(mps)
    if mode == "surf":
        b.align_with_surf(alpha=ALPHA, backend="triton", num_repeats=NUM_SEEDS, max_num_steps=MAX_STEPS)
        return np.array([float(p.sim_aligned_surf) for p in mps])
    if mode == "esp":
        b.align_with_esp(alpha=ALPHA, lam=lam, backend="triton", num_repeats=NUM_SEEDS, max_num_steps=MAX_STEPS)
        return np.array([float(p.sim_aligned_esp) for p in mps])
    if mode == "pharm":
        b.align_with_pharm(backend="triton", num_repeats=NUM_SEEDS, max_num_steps=MAX_STEPS)
        return np.array([float(p.sim_aligned_pharm) for p in mps])
    raise ValueError(mode)


def read_smi(path, limit=None):
    out = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip().split()
            if ln:
                out.append(ln[0])
    return out[:limit] if limit else out


# ===========================================================================
# Driver
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="smoke")
    ap.add_argument("--actives"); ap.add_argument("--decoys")
    ap.add_argument("--n-decoys", type=int, default=3000, help="subsample decoys (keep ALL actives)")
    ap.add_argument("--limit-actives", type=int, default=None)
    ap.add_argument("--queries", type=int, default=8, help="number of query actives to average over")
    ap.add_argument("--modes", nargs="+", default=["surf", "esp"])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.3, 0.1, 0.03, 0.01, 0.003],
                    help="ESP weights to sweep (esp mode is re-aligned at each)")
    ap.add_argument("--charges", choices=["xtb", "mmff"], default="xtb")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--build-only", action="store_true", help="CPU/xTB precompute then exit (no GPU)")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline test on curated drugs (NOT a benchmark)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    use_xtb = (a.charges == "xtb")
    rng = np.random.default_rng(SEED)

    if a.smoke:
        from paper.common import DRUGS
        acts = [s for _, s, _ in DRUGS[:6]]
        decs = [s for _, s, _ in DRUGS[6:]]
        a.target, a.queries, a.lams = "smoke", 2, [0.3, 0.01]
        cache_dir = os.path.join(HERE, "cache_smoke")
    else:
        if not (a.actives and a.decoys):
            print("provide --actives and --decoys (see README), or --smoke"); return 0
        acts = read_smi(a.actives, a.limit_actives)
        decs = read_smi(a.decoys)
        if a.n_decoys and len(decs) > a.n_decoys:
            decs = [decs[int(i)] for i in rng.choice(len(decs), size=a.n_decoys, replace=False)]
        cache_dir = os.path.join(HERE, "data", a.target, f"cache_{a.charges}")

    print(f"=== target={a.target}  actives={len(acts)}  decoys={len(decs)}  charges={a.charges} ===", flush=True)

    # ---- precompute (resumable, parallel) ----
    t0 = time.perf_counter()
    precompute(acts + decs, cache_dir, use_xtb, a.workers)
    print(f"precompute done in {time.perf_counter()-t0:.0f}s", flush=True)
    if a.build_only:
        return 0

    # ---- load cache (drop failures) ----
    A = [(s, load_cached(cache_dir, s, use_xtb)) for s in acts]
    D = [(s, load_cached(cache_dir, s, use_xtb)) for s in decs]
    A = [(s, m) for s, m in A if m is not None]
    D = [(s, m) for s, m in D if m is not None]
    print(f"loaded: {len(A)} actives + {len(D)} decoys "
          f"(active:decoy = 1:{len(D)/max(1,len(A)):.0f})", flush=True)
    if len(A) < 2 or len(D) < 1:
        print("not enough molecules to screen"); return 1

    decoy_mols = [m for _, m in D]

    # ---- enforce + record equal optimization budget ----
    os.environ["FINE_NUM_SEEDS"] = str(NUM_SEEDS)   # surf reads this; esp/pharm read num_repeats=NUM_SEEDS
    import importlib
    import shepherd_score.container._batch_align as _ba
    importlib.reload(_ba)                            # re-read FINE_NUM_SEEDS at module level
    assert (_ba._NUM_SEEDS == NUM_SEEDS), f"FINE_NUM_SEEDS not honored: {_ba._NUM_SEEDS}"
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # ---- choose K query actives (fixed) ----
    K = min(a.queries, len(A))
    qidx = list(rng.choice(len(A), size=K, replace=False))
    print(f"screening with {K} query actives on {gpu}; "
          f"num_seeds={NUM_SEEDS} max_steps={MAX_STEPS} surf_points={SURF_POINTS}", flush=True)

    # build the list of (mode, lam) screens; surf/pharm are lam-independent (lam=None)
    screens = []
    for mode in a.modes:
        if mode == "esp":
            for lam in a.lams:
                screens.append(("esp", lam))
        else:
            screens.append((mode, None))

    # per-screen: list of per-query metric dicts + the raw scores/labels for the LAST query (for plots)
    results = {}
    for (mode, lam) in screens:
        tag = mode if lam is None else f"esp@{lam:g}"
        per_q = []
        sample = None
        ts = time.perf_counter()
        for qi in qidx:
            qsmi, qmol = A[qi]
            lib_actives = [m for j, (_, m) in enumerate(A) if j != qi]
            lib = lib_actives + decoy_mols
            labels = np.array([1] * len(lib_actives) + [0] * len(decoy_mols))
            sc = screen_one(qmol, lib, mode, lam)
            per_q.append(all_metrics(labels, sc))
            sample = {"scores": sc.tolist(), "labels": labels.tolist()}   # keep last query for the plot
        agg = {}
        for k in ("auc", "ef1", "ef5", "bedroc"):
            vals = [d[k] for d in per_q]
            agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                      "ci95": boot_ci(vals), "per_query": vals}
        results[tag] = {"mode": mode, "lam": lam, "metrics": agg, "sample": sample}
        print(f"  {tag:10s} AUC={agg['auc']['mean']:.3f}±{agg['auc']['std']:.3f}  "
              f"EF1%={agg['ef1']['mean']:.1f}  BEDROC={agg['bedroc']['mean']:.3f}  "
              f"({time.perf_counter()-ts:.0f}s)", flush=True)

    # ---- paired surf-vs-esp delta per query (the ablation) ----
    ablation = {}
    if "surf" in results:
        surf_q = {k: np.array(results["surf"]["metrics"][k]["per_query"]) for k in ("auc", "ef1", "bedroc")}
        for tag, r in results.items():
            if not tag.startswith("esp@"):
                continue
            d = {}
            for k in ("auc", "ef1", "bedroc"):
                delta = np.array(r["metrics"][k]["per_query"]) - surf_q[k]
                d[k] = {"mean_delta": float(np.nanmean(delta)), "ci95": boot_ci(delta),
                        "n_pos": int(np.sum(delta > 0)), "n_q": int(np.sum(np.isfinite(delta)))}
            ablation[tag] = d

    out = {
        "target": a.target, "gpu": gpu, "charges": a.charges,
        "n_actives": len(A), "n_decoys": len(D),
        "n_queries": K, "query_idx": [int(i) for i in qidx],
        "config": {"alpha": ALPHA, "num_seeds": NUM_SEEDS, "max_steps": MAX_STEPS,
                   "surf_points": SURF_POINTS, "lams": a.lams},
        "results": results, "ablation_vs_surf": ablation,
    }
    out_path = a.out or os.path.join(HERE, f"enrichment_{a.target}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # ---- human-readable ablation summary ----
    if ablation:
        print("\nESP ablation (esp - surf, paired across queries):")
        for tag, d in ablation.items():
            da = d["auc"]
            print(f"  {tag:10s} dAUC={da['mean_delta']:+.3f} "
                  f"CI[{da['ci95'][0]:+.3f},{da['ci95'][1]:+.3f}] "
                  f"(+ in {da['n_pos']}/{da['n_q']} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
