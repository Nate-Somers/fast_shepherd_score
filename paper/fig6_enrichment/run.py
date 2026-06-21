"""
Figure 6 (data) — retrospective virtual-screening enrichment + the ESP ablation.

The decisive utility test: does the package retrieve actives, and does adding ESP
to shape improve retrieval? For a target with known actives + decoys, pick a query
active, align+score every library molecule by each mode (shape-only `surf`,
shape+ESP `esp`, pharmacophore `pharm`), rank, and compute enrichment (ROC-AUC,
EF1%, EF5%, BEDROC). The ESP ablation = esp vs surf enrichment on the same library.

Datasets (provide SMILES; see README for download):
    --actives actives.smi  --decoys decoys.smi   [DUDE-Z / LIT-PCBA / DEKOIS]
Use --limit-decoys to subsample for a tractable first run; --modes to choose modes.
A --smoke mode runs the pipeline end-to-end on the repo's curated drugs (NOT a
benchmark — just proves the code path).

Run (GPU env, xTB on PATH):
    PYTHONPATH=. python paper/fig6_enrichment/run.py --actives a.smi --decoys d.smi --limit-decoys 1000
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, LAM = 0.81, 0.01      # lam=0.01: ESP-sensitive (see fig4); justify in the paper


# ---------------------------------------------------------------------------
# Enrichment metrics
# ---------------------------------------------------------------------------
def roc_auc(labels, scores):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    P, Nn = y.sum(), (1 - y).sum()
    if P == 0 or Nn == 0:
        return float("nan")
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    return float(np.trapz(tps / P, fps / Nn))


def enrichment_factor(labels, scores, frac):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    n = max(1, int(round(frac * len(y))))
    return float((y[:n].mean()) / (y.mean() + 1e-12))


def bedroc(labels, scores, alpha=20.0):
    """Boltzmann-Enhanced Discrimination of ROC (Truchon & Bayly, JCIM 2007)."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    N = len(y); n = int(y.sum())
    if n == 0 or n == N:
        return float("nan")
    Ra = n / N
    ranks = np.where(y == 1)[0] + 1.0                     # 1-indexed ranks of actives
    rie = (np.sum(np.exp(-alpha * ranks / N)) / n) / \
          ((1 - np.exp(-alpha)) / (N * (np.exp(alpha / N) - 1)))
    return float(rie * (Ra * np.sinh(alpha / 2)) /
                 (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
                 + 1.0 / (1 - np.exp(alpha * (1 - Ra))))


# ---------------------------------------------------------------------------
# Build + screen
# ---------------------------------------------------------------------------
def build(smi, use_xtb=True, surf_per_atom=3, seed=42):
    from rdkit import Chem
    from shepherd_score.conformer_generation import (
        embed_conformer_from_smiles, charges_from_single_point_conformer_with_xtb)
    from shepherd_score.container import Molecule
    rd = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=seed)
    if rd is None:
        return None
    q = charges_from_single_point_conformer_with_xtb(rd) if use_xtb else None
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    return Molecule(rd, num_surf_points=ns, partial_charges=q, pharm_multi_vector=False)


def screen(query, lib, mode):
    """Align every lib molecule to the query; return the aligned similarity per mode."""
    import torch
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mps = [MoleculePair(query, m, do_center=True, device=dev,
                        num_surf_points=query.num_surf_points) for m in lib]
    b = MoleculePairBatch(mps)
    if mode == "surf":
        b.align_with_surf(alpha=ALPHA, backend="triton", num_repeats=16, max_num_steps=100)
        return np.array([float(p.sim_aligned_surf) for p in mps])
    if mode == "esp":
        b.align_with_esp(alpha=ALPHA, lam=LAM, backend="triton", num_repeats=16, max_num_steps=100)
        return np.array([float(p.sim_aligned_esp) for p in mps])
    if mode == "pharm":
        b.align_with_pharm(backend="triton", num_repeats=16, max_num_steps=100)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actives"); ap.add_argument("--decoys")
    ap.add_argument("--query", default=None)
    ap.add_argument("--limit-decoys", type=int, default=None)
    ap.add_argument("--limit-actives", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["surf", "esp"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pipeline test on curated drugs (NOT a benchmark)")
    a = ap.parse_args()

    if a.smoke:
        from paper.common import DRUGS
        acts = [s for _, s, _ in DRUGS[:5]]; decs = [s for _, s, _ in DRUGS[5:]]
        print("SMOKE: 5 'actives' + decoys from curated drugs — pipeline test only.")
    else:
        if not (a.actives and a.decoys):
            print("provide --actives and --decoys (see README for DUDE-Z), or --smoke"); return 0
        acts = read_smi(a.actives, a.limit_actives)
        decs = read_smi(a.decoys, a.limit_decoys)

    print(f"building {len(acts)} actives + {len(decs)} decoys (conformers + xTB charges)...")
    t0 = time.perf_counter()
    A = [build(s) for s in acts]; D = [build(s) for s in decs]
    A = [m for m in A if m is not None]; D = [m for m in D if m is not None]
    lib = A[1:] + D                              # query = A[0]; rest of actives + decoys are the library
    labels = np.array([1] * (len(A) - 1) + [0] * len(D))
    query = A[0]
    print(f"built in {time.perf_counter()-t0:.0f}s; library = {len(lib)} ({labels.sum()} actives)")

    res = {"n_actives": int(labels.sum()), "n_library": len(lib), "lam": LAM, "metrics": {}}
    for mode in a.modes:
        sc = screen(query, lib, mode)
        m = {"auc": roc_auc(labels, sc), "ef1": enrichment_factor(labels, sc, 0.01),
             "ef5": enrichment_factor(labels, sc, 0.05), "bedroc": bedroc(labels, sc, 20.0),
             "scores": sc.tolist()}
        res["metrics"][mode] = m
        print(f"{mode:5s}  AUC={m['auc']:.3f}  EF1%={m['ef1']:.1f}  EF5%={m['ef5']:.1f}  "
              f"BEDROC={m['bedroc']:.3f}")
    res["labels"] = labels.tolist()
    with open(os.path.join(HERE, "enrichment.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {os.path.join(HERE, 'enrichment.json')}")
    if "esp" in res["metrics"] and "surf" in res["metrics"]:
        d = res["metrics"]["esp"]["auc"] - res["metrics"]["surf"]["auc"]
        print(f"\nESP ablation: AUC(esp) - AUC(surf) = {d:+.3f}  "
              f"({'ESP helps' if d > 0 else 'ESP does not help'} on this target)")


if __name__ == "__main__":
    raise SystemExit(main())
