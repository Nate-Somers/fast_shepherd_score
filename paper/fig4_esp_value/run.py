"""
Figure 4 (data) — the electrostatic term carries information orthogonal to shape.

Clean test: take SHAPE-MATCHED but electrostatically different molecules — a benzene-ring
analog series — align each to benzene BY SHAPE, then score ESP similarity at that fixed
shape pose across a sweep of the ESP weight `lam`.  A working ESP term should separate the
nonpolar analogs (like benzene) from the polar ones, and only at small enough `lam` (the
package default 0.3 weights ESP very weakly).

This version adds what the figure needs to be load-bearing:
  * UNCERTAINTY.  Repeat the whole pipeline over R replicates with different conformer/seed
    and (nondeterministic) surface samplings, so every esp(lam) value carries a mean ± SD —
    the headline discrimination is no longer a single draw.
  * A POLARITY AXIS.  For each analog we also compute the molecular dipole magnitude from the
    xTB partial charges (|Σ q_i r_i|), giving a quantitative electrostatic axis to show the
    ESP signal tracks polarity (not residual shape).

Charges are xTB partial charges (the package's intended physical model).
Run (repo root, GPU env, xtb on PATH):  PYTHONPATH=. python paper/fig4_esp_value/run.py
"""
import json
import os
import sys

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
LAMS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
N_REP = 12                       # replicates (conformer seed + surface resampling)
HERE = os.path.dirname(os.path.abspath(__file__))


def build_xtb(smi, seed, surf_per_atom=4):
    """Returns (Molecule, dipole_Debye).  The dipole is computed from the RDKit conformer's
    ALL-atom positions × xTB all-atom charges (Molecule.atom_pos is heavy-only, so it can't
    be used for the dipole)."""
    from rdkit import Chem
    from shepherd_score.conformer_generation import (
        embed_conformer_from_smiles, charges_from_single_point_conformer_with_xtb)
    from shepherd_score.container import Molecule
    rd = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=seed)
    q = charges_from_single_point_conformer_with_xtb(rd)
    pos = rd.GetConformer().GetPositions()                 # (n_atoms_with_H, 3)
    q = np.asarray(q, float).ravel()
    n = min(len(q), len(pos))
    d = (q[:n, None] * pos[:n]).sum(axis=0)                # e·Å (neutral -> origin-independent)
    dipole = float(np.linalg.norm(d) * 4.80320)           # Debye
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    m = Molecule(rd, num_surf_points=ns, partial_charges=q, pharm_multi_vector=False)
    return m, dipole


def main():
    import torch
    from shepherd_score.container import MoleculePair
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # per (analog, rep): shape, esp[lam], dipole
    shape = {n: [] for n, _, _ in ANALOGS}
    esp = {n: {l: [] for l in LAMS} for n, _, _ in ANALOGS}
    dip = {n: [] for n, _, _ in ANALOGS}

    for rep in range(N_REP):
        seed = 42 + rep
        ref, _ = build_xtb(REF[1], seed)
        for name, smi, cls in ANALOGS:
            fit, fit_dip = build_xtb(smi, seed)
            dip[name].append(fit_dip)
            mp = MoleculePair(ref, fit, do_center=True, device=dev, num_surf_points=ref.num_surf_points)
            mp.align_with_surf(alpha=ALPHA, num_repeats=16, lr=0.1, max_num_steps=100)
            shape[name].append(float(mp.sim_aligned_surf))
            tfit = mp.get_transformed_molecule(se3_transform=mp.transform_surf)
            mp2 = MoleculePair(ref, tfit, do_center=False, device=dev, num_surf_points=ref.num_surf_points)
            for l in LAMS:
                esp[name][l].append(float(mp2.score_with_esp(ALPHA, lam=l)))
        print(f"rep {rep+1}/{N_REP} done", flush=True)

    rows = []
    for name, smi, cls in ANALOGS:
        rows.append({
            "name": name, "cls": cls,
            "shape_mean": float(np.mean(shape[name])), "shape_std": float(np.std(shape[name])),
            "dipole_mean": float(np.mean(dip[name])), "dipole_std": float(np.std(dip[name])),
            "esp_mean": {str(l): float(np.mean(esp[name][l])) for l in LAMS},
            "esp_std": {str(l): float(np.std(esp[name][l])) for l in LAMS},
            "shape_all": shape[name],
            "esp_all": {str(l): esp[name][l] for l in LAMS},
        })

    # discrimination: std ACROSS analogs at each lam, with its own uncertainty across reps
    disc = {}
    for l in LAMS:
        per_rep_std = [float(np.std([esp[n][l][r] for n, _, _ in ANALOGS])) for r in range(N_REP)]
        disc[str(l)] = {"mean": float(np.mean(per_rep_std)), "std": float(np.std(per_rep_std))}
    shape_disc_per_rep = [float(np.std([shape[n][r] for n, _, _ in ANALOGS])) for r in range(N_REP)]
    shape_disc = {"mean": float(np.mean(shape_disc_per_rep)), "std": float(np.std(shape_disc_per_rep))}

    out = {"ref": REF[0], "lams": LAMS, "alpha": ALPHA, "charges": "xtb", "n_rep": N_REP,
           "rows": rows, "discrimination": disc, "shape_discrimination": shape_disc,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with open(os.path.join(HERE, "analog_esp.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nshape discrimination (std across analogs): {shape_disc['mean']:.4f} ± {shape_disc['std']:.4f}")
    for l in LAMS:
        print(f"  lam={l:<5g}: esp discrimination {disc[str(l)]['mean']:.4f} ± {disc[str(l)]['std']:.4f}")
    print("wrote analog_esp.json")


if __name__ == "__main__":
    main()
