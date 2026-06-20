"""Decisive gate (lever L10): is num_repeats reduction accuracy-safe for VOL on REAL molecules?

vol/vol_esp use heavy-atom positions (rdkit) -> NO open3d/jax needed, so this runs on the
dev box. Tests whether dropping num_repeats (50 -> 25/15/10/5) changes the FINAL best score
on real distinct drug pairs (the accuracy gate) -- and measures the single-core throughput at
each nr (the speed payoff). nr=5 is the upstream "adequate for non-surface" claim; this checks
it against the strict gate (self=1.0, distinct max|d|<1e-5, zero score-changes) on real shapes,
including pseudo-symmetric cases (benzene, caffeine) the audit flagged as the risk.

Uses the existing per-pair torch optimizer optimize_ROCS_overlay_analytical (the default CPU
vol path) -- nr=50 is the reference.

Run: python -m benchmarks.experiments.cpu_vol_nr_accuracy
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
from unittest.mock import MagicMock
# vol uses heavy-atom positions only; stub open3d so the package __init__ chain
# (generate_point_cloud -> open3d, used only for surfaces) imports on this box.
sys.modules.setdefault("open3d", MagicMock())
import numpy as np
import torch
torch.set_num_threads(1)
from rdkit import Chem
from rdkit.Chem import AllChem
from shepherd_score.alignment._torch_analytical import optimize_ROCS_overlay_analytical

ALPHA = 0.81
DRUGS = [
    ("benzene", "c1ccccc1"), ("phenol", "Oc1ccccc1"), ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("diphenhydramine", "O(CCN(C)C)C(c1ccccc1)c1ccccc1"),
    ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
]


def heavy_xyz(smiles, seed=42):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    xyz = m.GetConformer().GetPositions().astype(np.float32)
    return xyz - xyz.mean(0, keepdims=True)


def rand_rot(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def score_pair(ref, fit, nr):
    t = torch.from_numpy(ref); f = torch.from_numpy(fit)
    out = optimize_ROCS_overlay_analytical(t, f, ALPHA, num_repeats=nr, max_num_steps=200, lr=0.1)
    s = out[-1]
    return float(s.item() if hasattr(s, "item") else np.asarray(s).reshape(-1)[-1])


if __name__ == "__main__":
    print("Building real conformers (heavy atoms)...")
    mols = {n: heavy_xyz(s) for n, s in DRUGS}
    names = list(mols)
    print(f"{len(names)} molecules.\n")

    NRS = [50, 25, 15, 10, 5]

    # --- self-copy: each mol vs a rotated+translated copy (optimum = 1.0) ---
    print("SELF-COPY min score by nr (must stay ~1.000):")
    self_min = {nr: 1.0 for nr in NRS}
    for i, n in enumerate(names):
        ref = mols[n]
        fit = ref @ rand_rot(i).T + np.array([3.0, -2.0, 1.0], np.float32)
        for nr in NRS:
            self_min[nr] = min(self_min[nr], score_pair(ref, fit, nr))
    print("  " + "  ".join(f"nr={nr}:{self_min[nr]:.4f}" for nr in NRS) + "\n")

    # --- distinct pairs: ref_i vs fit_j (i!=j); reference = nr=50 ---
    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(40):
        i, j = rng.choice(len(names), 2, replace=False)
        pairs.append((names[i], names[j]))

    ref_scores = {}
    print("Computing nr=50 reference on 40 distinct pairs...")
    for k, (a, b) in enumerate(pairs):
        ref_scores[k] = score_pair(mols[a], mols[b], 50)

    print("\nDISTINCT-PAIR accuracy vs nr=50 reference:")
    print(f"{'nr':>4} | {'max|d|':>9} | {'#|d|>1e-5':>9} | {'#|d|>1e-3':>9} | {'mean|d|':>9}")
    for nr in NRS:
        if nr == 50:
            print(f"{nr:>4} | {'reference':>9} | {'-':>9} | {'-':>9} | {'-':>9}")
            continue
        diffs = []
        for k, (a, b) in enumerate(pairs):
            diffs.append(abs(score_pair(mols[a], mols[b], nr) - ref_scores[k]))
        diffs = np.array(diffs)
        print(f"{nr:>4} | {diffs.max():9.2e} | {int((diffs>1e-5).sum()):>9} | {int((diffs>1e-3).sum()):>9} | {diffs.mean():9.2e}")

    # --- single-core throughput at each nr (distinct pairs) ---
    print("\nSINGLE-CORE throughput (pairs/s/core, distinct pairs, max_num_steps=200):")
    for nr in NRS:
        t0 = time.perf_counter()
        for a, b in pairs:
            score_pair(mols[a], mols[b], nr)
        dt = time.perf_counter() - t0
        print(f"  nr={nr:>3}: {len(pairs)/dt:7.1f} pairs/s/core  ({dt/len(pairs)*1e3:.1f} ms/pair)")

    print("\nGATE: nr passes if self~1.0 AND distinct max|d|<1e-5 AND #changed=0.")
