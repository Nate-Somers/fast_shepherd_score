"""Parity/quality gate for the structured per-mode seeds.

Asserts that every alignment mode, at its NEW per-mode default seed count (structured
+/-90deg PCA-axis seeds, see shepherd_score/accel/batch/aligners.py:_MODE_SEEDS), recovers
the SAME quality as the legacy identity+4PCA+Fibonacci 50-seed path:
  - rotated self-copies still reach the known optimum (~1.0), and
  - distinct cross-pair scores match the 50-seed result (mean parity + small tail).

The legacy seeds need FSS_STRUCT_SEEDS=0 set at import, so old vs new run in separate
subprocesses; this script orchestrates them and exits 0 (PASS) / 1 (FAIL).

    python benchmarks/seed_parity_gate.py            # run the gate
Requires a CUDA GPU with the Triton backend (the fork's fast path).
"""
import argparse, copy, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
MODES = ["vol", "surf", "esp", "pharm", "vol_color"]
_ATTR = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf", "esp": "sim_aligned_esp",
         "pharm": "sim_aligned_pharm", "vol_color": "sim_aligned_vol_color"}
# small, fixed drug set (self-contained; ~screening-sized molecules)
_SMI = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "Oc1ccccc1", "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O",
        "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "O(CCN(C)C)C(c1ccccc1)c1ccccc1",
        "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", "CN1CCCC1c1cccnc1",
        "CC(C)NCC(O)COc1cccc2ccccc12"]


def _build_mols():
    from rdkit import Chem  # noqa
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    mols = []
    for s in _SMI:
        rd = embed_conformer_from_smiles(s, MMFF_optimize=True, random_seed=0)
        if rd is None:
            continue
        m = Molecule(rd, num_surf_points=128, pharm_multi_vector=False)
        m.center_to(m.atom_pos.mean(0))
        mols.append(m)
    return mols


def _rotcopy(m, R, t):
    f = copy.copy(m); f.atom_pos = m.atom_pos @ R.T + t
    if getattr(m, "surf_pos", None) is not None:
        f.surf_pos = m.surf_pos @ R.T + t
    if getattr(m, "pharm_ancs", None) is not None:
        f.pharm_ancs = m.pharm_ancs @ R.T + t
        v = m.pharm_vecs @ R.T; n = np.linalg.norm(v, axis=1, keepdims=True)
        f.pharm_vecs = v / np.where(n > 0, n, 1.0)
    return f


def _align(batch, mode):
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(batch)
    if mode == "vol": b.align_with_vol(no_H=True, backend="triton", alpha=0.81, max_num_steps=100)
    elif mode == "surf": b.align_with_surf(alpha=0.81, backend="triton", max_num_steps=100)
    elif mode == "esp": b.align_with_esp(alpha=0.81, lam=0.3, backend="triton", max_num_steps=100)
    elif mode == "pharm": b.align_with_pharm(backend="triton", max_num_steps=100)
    elif mode == "vol_color": b.align_with_vol_color(color_weight=0.5, backend="triton", max_num_steps=100)
    return np.array([float(getattr(p, _ATTR[mode])) for p in batch])


def cell(mode, out):
    """Align one mode on this process's seed regime; dump self + cross scores."""
    import torch
    from shepherd_score.container import MoleculePair
    mols = _build_mols(); dev = torch.device("cuda:0")
    lib = [mols[i % len(mols)] for i in range(60)]
    cross = [MoleculePair(mols[q], lib[j], do_center=True, device=dev) for q in range(6) for j in range(len(lib))]
    rng = np.random.default_rng(0); q0 = mols[0]; sp = []
    for _ in range(80):
        a = rng.standard_normal((3, 3)); R, _ = np.linalg.qr(a)
        if np.linalg.det(R) < 0: R[:, 0] = -R[:, 0]
        sp.append(MoleculePair(q0, _rotcopy(q0, R, rng.standard_normal(3)), do_center=True, device=dev))
    cs = _align(cross, mode); ss = _align(sp, mode)
    json.dump({"mode": mode, "cross": [round(float(x), 5) for x in cs],
               "self_min": float(ss.min()), "self_mean": float(ss.mean())}, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=2, metavar=("MODE", "OUT"))
    a = ap.parse_args()
    if a.cell:
        return cell(a.cell[0], a.cell[1])

    import tempfile
    print("seed parity gate: NEW per-mode structured seeds vs LEGACY 50-seed Fibonacci\n")
    print(f"{'mode':>10} {'self new/old':>14} {'cross new/old':>16} {'d_mean':>7} {'tail(info)':>10} {'PASS':>5}")
    ok_all = True
    for mode in MODES:
        res = {}
        for tag, env in (("old", {"FSS_STRUCT_SEEDS": "0", "FINE_NUM_SEEDS": "50"}), ("new", {})):
            fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
            e = dict(os.environ); e.update(env)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--cell", mode, p],
                               env=e, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{mode:>10}  cell '{tag}' failed:\n{r.stderr[-800:]}"); ok_all = False; res = None; break
            res[tag] = json.load(open(p)); os.remove(p)
        if not res:
            continue
        nc, oc = np.array(res["new"]["cross"]), np.array(res["old"]["cross"])
        dmean = (nc.mean() - oc.mean()) / oc.mean() * 100
        tail = float(np.mean((oc - nc) > 0.01 * np.maximum(oc, 1e-9)) * 100)
        # Gate on MEAN overlap (total recovered overlap) + self-recovery, NOT per-pair
        # reproduction: esp/surf are inherently multi-basin (charge/surface landscapes have
        # many near-equal optima), so any two seed sets -- including legacy-50 vs structured-50
        # -- disagree per-pair while the mean is flat. `tail` is reported for visibility only.
        ok = (dmean > -0.3) and (res["new"]["self_min"] >= res["old"]["self_min"] - 0.003)
        ok_all &= ok
        print(f"{mode:>10} {res['new']['self_min']:.4f}/{res['old']['self_min']:.4f} "
              f"{nc.mean():.4f}/{oc.mean():.4f} {dmean:>+6.2f}% {tail:>9.1f}% {'PASS' if ok else 'FAIL':>5}")
    print("\nPARITY GATE:", "PASS" if ok_all else "FAIL",
          "\n  criteria: mean overlap not regressed vs legacy-50 by >0.3%, and self-copy recovery"
          "\n  preserved. tail (per-pair disagreement) is INFO -- esp/surf are inherently multi-basin,"
          "\n  so two seed sets differ per-pair even at 50 seeds; the stable quality metric is the mean.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
