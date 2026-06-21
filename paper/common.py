"""
Shared utilities for the paper figures.

Everything the figure scripts need that is common: locating the repo, building the
SAME real-drug molecules the repo's own benchmark uses (so every figure is on one
consistent molecule set), per-atom MMFF partial charges for the electrostatic
comparisons, a best-of-N timer, and a single production matplotlib style.

Run any figure with the repo's GPU env, e.g. (from repo root):
    PYTHONPATH=. python paper/figN_.../run.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# --- repo root on path so `import shepherd_score` / `import benchmarks...` work ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the repo benchmark's curated real-drug set + deterministic builders so the
# paper figures and the repo's own speed numbers are on the exact same molecules.
from benchmarks.benchmark import (  # noqa: E402
    DRUGS, _build_molecule, make_real_cohort, _RealMol, _random_rotation, _transform,
)

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# RDKit molecules + charges (for the third-party tools: O3A, USRCAT, ESP-Sim)
# ---------------------------------------------------------------------------
import functools  # noqa: E402


@functools.lru_cache(maxsize=None)
def build_fss_molecule(smiles: str, surf_per_atom: int = 3, seed: int = 42):
    """A real shepherd_score ``Molecule`` (surface points + MMFF ESP + pharmacophores),
    built with the same recipe as ``benchmarks.benchmark._build_molecule`` but
    returning the Molecule object itself, so it works with BOTH the JAX/_core
    single-pair path and the Triton batch path (the lightweight `_RealMol` only
    works with the latter)."""
    from rdkit import Chem
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    return Molecule(rd, num_surf_points=ns, pharm_multi_vector=False)


def build_rdkit_mol(smiles: str, seed: int = 42):
    """RDKit Mol with one ETKDG+MMFF94 conformer (explicit H), same recipe the
    repo uses to build its molecules. Returns the RDKit Mol (with Hs)."""
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    return embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)


def mmff_partial_charges(rdmol) -> np.ndarray:
    """Per-atom MMFF94 partial charges (same charge model the repo's surf_esp uses),
    in atom order including Hs. Falls back to Gasteiger if MMFF is unavailable."""
    from rdkit.Chem import AllChem
    props = AllChem.MMFFGetMoleculeProperties(rdmol)
    if props is not None:
        return np.array([props.GetMMFFPartialCharge(i) for i in range(rdmol.GetNumAtoms())],
                        dtype=float)
    AllChem.ComputeGasteigerCharges(rdmol)
    return np.array([float(a.GetProp("_GasteigerCharge")) for a in rdmol.GetAtoms()], dtype=float)


def transformed_rdkit_copy(rdmol, R: np.ndarray, t: np.ndarray):
    """A rigid SE(3) copy of an RDKit mol's conformer (R @ x + t). Used to build
    self-copy pairs with a known optimum (perfect overlap)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    m = Chem.Mol(rdmol)
    conf = m.GetConformer()
    pos = m.GetConformer().GetPositions()
    new = pos @ np.asarray(R).T + np.asarray(t)
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in new[i]]))
    return m


def drug_names():
    return [n for n, _, _ in DRUGS]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def best_of_n(fn, reps: int = 5, budget: float = 4.0, warmup: int = 1):
    """Best-of-N wall-clock of `fn` (seconds). One+ warmup (JIT/autotune/clock),
    then up to `reps` timed runs (stop once cumulative time exceeds `budget`),
    keep the fastest = least-throttled. Syncs CUDA around each timed run."""
    for _ in range(warmup):
        fn(); cuda_sync()
    best = float("inf"); total = 0.0; n = 0
    while n < reps and total < budget:
        cuda_sync(); t0 = time.perf_counter()
        fn(); cuda_sync()
        dt = time.perf_counter() - t0
        best = min(best, dt); total += dt; n += 1
    return best, n


# ---------------------------------------------------------------------------
# Plot style (one consistent production look across all figures)
# ---------------------------------------------------------------------------
# Mode palette matches the repo's existing plots for cross-figure consistency.
MODE_COLOR = {"vol": "#7b3294", "surf": "#1f6fb2", "esp": "#1a9850", "pharm": "#d9700a"}
# Per-tool palette for the competitor figures.
TOOL_COLOR = {
    "fast_shepherd_score": "#c0392b",
    "fss (GPU)": "#c0392b",
    "fss (CPU)": "#e08e85",
    "ESP-Sim": "#1a9850",
    "RDKit O3A": "#1f6fb2",
    "USRCAT": "#d9700a",
    "ROSHAMBO2": "#7b3294",
}


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })
    return plt


def save_fig(fig, out_stem: str):
    """Save a figure as both PNG (raster preview) and PDF (vector, for the paper)."""
    fig.savefig(out_stem + ".png", bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {out_stem}.png / .pdf")
