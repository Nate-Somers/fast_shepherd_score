"""
Figure 4 (data) — the electrostatic term carries information orthogonal to shape.

The clean test for "does ESP add anything": take molecules that are SHAPE-MATCHED
but electrostatically different — a benzene-ring analog series (benzene, toluene,
halobenzenes, phenol, aniline, pyridine, nitrobenzene, benzaldehyde). Shape ranks
them all as similar (~0.6-0.7 to benzene); a working ESP term should additionally
separate the nonpolar analogs (electrostatically like benzene) from the polar ones.

For each analog we align to benzene by SHAPE, then score ESP similarity at that
shape pose across a sweep of the ESP weight `lam` (smaller lam = sharper
electrostatic weighting). Charges are xTB partial charges (the package's intended,
physical charge model). This isolates ESP's contribution at fixed shape.

Run (repo root, GPU env):  PYTHONPATH=. python paper/fig4_esp_value/run.py
Writes analog_esp.json next to this file.
"""
import json
import os
import sys

# xtb binary (in the env) must be on PATH for the charge subprocess
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# (name, SMILES, polarity class for coloring)
REF = ("benzene", "c1ccccc1")
ANALOGS = [
    ("benzene", "c1ccccc1", "nonpolar"),
    ("toluene", "Cc1ccccc1", "nonpolar"),
    ("fluorobenzene", "Fc1ccccc1", "weak"),
    ("chlorobenzene", "Clc1ccccc1", "weak"),
    ("aniline", "Nc1ccccc1", "polar"),
    ("phenol", "Oc1ccccc1", "polar"),
    ("pyridine", "c1ccncc1", "polar"),
    ("benzaldehyde", "O=Cc1ccccc1", "strong"),
    ("nitrobenzene", "O=[N+]([O-])c1ccccc1", "strong"),
]
ALPHA = 0.81
LAMS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]   # default is 0.3; smaller = stronger ESP


def build_xtb(smi, surf_per_atom=4, seed=42):
    from rdkit import Chem
    from shepherd_score.conformer_generation import (
        embed_conformer_from_smiles, charges_from_single_point_conformer_with_xtb)
    from shepherd_score.container import Molecule
    rd = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=seed)
    q = charges_from_single_point_conformer_with_xtb(rd)
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    return Molecule(rd, num_surf_points=ns, partial_charges=q, pharm_multi_vector=False)


def main():
    import torch
    from shepherd_score.container import MoleculePair
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = build_xtb(REF[1])

    rows = []
    for name, smi, cls in ANALOGS:
        fit = build_xtb(smi)
        mp = MoleculePair(ref, fit, do_center=True, device=dev,
                          num_surf_points=ref.num_surf_points)
        mp.align_with_surf(alpha=ALPHA, num_repeats=16, lr=0.1, max_num_steps=100)
        shape = float(mp.sim_aligned_surf)
        tfit = mp.get_transformed_molecule(se3_transform=mp.transform_surf)
        mp2 = MoleculePair(ref, tfit, do_center=False, device=dev,
                           num_surf_points=ref.num_surf_points)
        esp = {l: float(mp2.score_with_esp(ALPHA, lam=l)) for l in LAMS}
        rows.append({"name": name, "cls": cls, "shape": shape, "esp": esp})
        print(f"{name:14s} ({cls:8s}) shape={shape:.3f}  " +
              " ".join(f"esp@{l}={esp[l]:.3f}" for l in LAMS))

    out = {"ref": REF[0], "lams": LAMS, "alpha": ALPHA, "charges": "xtb", "rows": rows}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analog_esp.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # quantify: ESP std vs shape std at each lam (discrimination), correlation of
    # esp-drop with polarity rank.
    shp = np.array([r["shape"] for r in rows])
    print(f"\nshape std across analogs: {shp.std():.4f}")
    for l in LAMS:
        e = np.array([r["esp"][l] for r in rows])
        print(f"  lam={l:<5g}: esp std {e.std():.4f}")
    print(f"\nwrote analog_esp.json")


if __name__ == "__main__":
    main()
