"""
Speed benchmark for the new ``vol_color`` mode (ROCS/ROSHAMBO-style atom-Gaussian
shape + directionless pharmacophore "color"), measured the same way as the rest of
the suite: real drug molecules, each aligned to a rigid SE(3) self-copy (known
optimum 1.0), best-of-N wall-clock, throughput in **pair-alignments / second**.

IMPORTANT — scope of this number
--------------------------------
``vol_color`` v1 is the **per-pair torch optimizer** (``MoleculePair.align_with_vol_color``);
it does NOT yet have a batched GPU/Triton driver (that is the deferred v2 — the
``vol``/``surf``/``esp``/``pharm`` modes in ``benchmark.py`` are batched, ``vol_color`` is
not). So this measures the per-pair path: a Python loop over pairs, each running its own
multi-start SE(3) optimization. On a GPU it is therefore **host-bound** (like the per-pair
``pharm`` path), not representative of a batched kernel. Reported as honest per-pair
throughput; the batched driver is needed for the high-throughput regime.

PREP (build molecules + SE(3) copies + featurize with RDKit BaseFeatures) is separated
from COMPUTE (the timed alignment loop), with warmup + best-of-N.

Usage
-----
    python -m benchmarks.benchmark_vol_color                 # CPU (or CUDA if visible)
    python -m benchmarks.benchmark_vol_color --device cpu --n 30
    python -m benchmarks.benchmark_vol_color --device cuda --num-repeats 50 --steps 100
    python -m benchmarks.benchmark_vol_color --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from benchmarks.benchmark import DRUGS, _random_rotation  # reuse the curated drug set


def _rotated_rdmol(rd, R, t):
    """Rigid SE(3) copy of an RDKit mol's conformer (proper rotation + translation)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    m = Chem.Mol(rd)
    conf = m.GetConformer()
    pos = conf.GetPositions() @ R.T + t
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in pos[i]]))
    return m


def build_pairs(n, device, seed=3, feature_set="rdkit_base", directionless=True):
    """Build ``n`` (ref, SE(3)-self-copy) MoleculePairs, featurized for color."""
    import torch
    from rdkit import Chem
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule, MoleculePair

    rng = np.random.default_rng(seed)
    # embed each unique drug once
    base = {}
    for name, smi, _ in DRUGS:
        try:
            base[smi] = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=42)
        except Exception:
            continue
    smis = list(base.keys())

    pairs = []
    for _ in range(n):
        smi = smis[int(rng.integers(len(smis)))]
        rd = base[smi]
        R = _random_rotation(rng)
        t = rng.standard_normal(3)
        t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * 3.0)
        ref = Molecule(rd, pharm_multi_vector=False,
                       feature_set=feature_set, directionless=directionless)
        fit = Molecule(_rotated_rdmol(rd, R, t), pharm_multi_vector=False,
                       feature_set=feature_set, directionless=directionless)
        pairs.append(MoleculePair(ref, fit, do_center=True, device=device))
    return pairs


def align_all(pairs, color_weight, num_repeats, steps):
    for mp in pairs:
        mp.align_with_vol_color(color_weight=color_weight, num_repeats=num_repeats,
                                max_num_steps=steps)
    return np.array([float(mp.sim_aligned_vol_color) for mp in pairs])


def best_of_n(fn, reps, budget, sync):
    fn()  # warmup
    sync()
    best = float("inf"); n = 0; total = 0.0
    while n < reps and total < budget:
        sync(); t0 = time.perf_counter()
        fn(); sync()
        dt = time.perf_counter() - t0
        best = min(best, dt); total += dt; n += 1
    return best, n


def main():
    ap = argparse.ArgumentParser(description="vol_color per-pair throughput benchmark")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--n", type=int, default=30, help="number of pairs")
    ap.add_argument("--num-repeats", type=int, default=50)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--color-weight", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    if a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(a.device)

    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()

    gpu = torch.cuda.get_device_name(0) if (a.device == "cuda" and torch.cuda.is_available()) else None

    print(f"vol_color benchmark | device={a.device} {gpu or ''} | n={a.n} "
          f"num_repeats={a.num_repeats} steps={a.steps} color_weight={a.color_weight}", flush=True)

    t0 = time.perf_counter()
    pairs = build_pairs(a.n, dev, seed=a.seed)
    prep = time.perf_counter() - t0
    print(f"PREP: built {len(pairs)} pairs in {prep:.1f}s "
          f"(embed + RDKit BaseFeatures.fdef directionless featurize)", flush=True)

    best, nrep = best_of_n(
        lambda: align_all(pairs, a.color_weight, a.num_repeats, a.steps),
        a.reps, a.budget, sync)
    scores = align_all(pairs, a.color_weight, a.num_repeats, a.steps)

    n = len(pairs)
    pps = n / best
    out = {
        "mode": "vol_color", "device": a.device, "gpu": gpu, "cpu": platform.processor(),
        "host": platform.node(), "torch": torch.__version__,
        "n_pairs": n, "num_repeats": a.num_repeats, "steps": a.steps,
        "color_weight": a.color_weight, "reps": nrep,
        "compute_s": best, "pairs_per_s": pps,
        "self_score_mean": float(scores.mean()), "self_score_min": float(scores.min()),
    }
    print(f"\nvol_color: {pps:,.1f} pairs/s  (compute {best:.3f}s, best of {nrep})  "
          f"self-score mean={scores.mean():.3f} min={scores.min():.3f}", flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {a.json}", flush=True)
    return out


if __name__ == "__main__":
    main()
