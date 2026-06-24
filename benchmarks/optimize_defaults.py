"""Optimize the per-mode default (seed count, fine-step count) for all alignment modes.

For each of the 7 alignment modes we run a fixed set of *cross* pairs (distinct drug i
vs drug j) over a 2-D grid of (num_seeds, steps_fine), on the fast batched driver
(backend=numba on CPU -- alignment scores are hardware-independent, so the optimal counts
found here transfer to the Triton GPU path). For each mode we then find the cheapest
config (min seeds*steps) that still "captures all the alignment accuracy":

  - mean cross-overlap within `--mean-tol` (default 0.3%) of the per-pair ceiling
    (element-wise max over every config tested for that mode -- the true achievable
    optimum, robust to the multi-basin seed-set noise that makes per-pair reproduction
    unreliable for esp/surf/pharm; see seed_parity_gate.py), AND
  - rotated self-copy recovery min >= `--self-tol` below the best self-recovery seen, AND
  - per-pair tail (fraction of pairs > 1% below the ceiling) reported for visibility.

Everything runs in ONE warmed process: the numba kernels JIT-compile once, molecules build
once, and each cell just sets the seed count by mutating aligners._NUM_SEEDS (what
_seeds_for reads) and passes steps via max_num_steps. Structured PCA-axis seeds are ON
(the shipping default).

    python benchmarks/optimize_defaults.py                 # full sweep, all modes
    python benchmarks/optimize_defaults.py --modes vol surf --smoke
"""
import argparse, copy, json, os, sys, time

# Cap numba/BLAS threads BEFORE importing numba/torch (else numba grabs every physical
# core and throughput collapses -- see cluster-slurm numba oversubscription gotcha).
def _cap_threads(n):
    for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(n))


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ALL_MODES = ["vol", "surf", "esp", "vol_esp", "esp_combo", "pharm", "vol_color"]
_ATTR = {
    "vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf", "esp": "sim_aligned_esp",
    "vol_esp": "sim_aligned_vol_esp_noH", "esp_combo": "sim_aligned_esp_combo",
    "pharm": "sim_aligned_pharm", "vol_color": "sim_aligned_vol_color",
}
_DEFAULT_SMI = os.path.join(os.path.dirname(HERE), "..", "Shepherd-Score-Paper",
                            "overlap_speed", "drugs.smi")


def read_smiles(path, limit):
    out = []
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if p:
                out.append((p[0], p[1] if len(p) > 1 else p[0]))
    return out[:limit] if limit else out


def build_mols(smi_path, limit, n_surf):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    mols = []
    for s, name in read_smiles(smi_path, limit):
        rd = embed_conformer_from_smiles(s, MMFF_optimize=True, random_seed=0)
        if rd is None:
            continue
        m = Molecule(rd, num_surf_points=n_surf, pharm_multi_vector=False)
        m.center_to(m.atom_pos.mean(0))
        mols.append(m)
    return mols


def cross_pairs(mols, n, device):
    import numpy as np
    from shepherd_score.container import MoleculePair
    idx = [(i, j) for i in range(len(mols)) for j in range(len(mols)) if i != j]
    rng = np.random.default_rng(0); rng.shuffle(idx); idx = idx[:n]
    return [MoleculePair(mols[i], mols[j], do_center=False,
                         num_surf_points=mols[i].num_surf_points, device=device)
            for i, j in idx]


def _rotcopy(m, R, t):
    """A rigid SE(3) copy of a Molecule. Rotates every geometry array AND the RDKit
    conformer (vol_esp/esp_combo read atom positions straight from mol.GetConformer(),
    so a shallow copy would leave their shape channel unrotated and the self-copy score
    meaningless). Rotation-invariant arrays (charges, radii, surf_esp, pharm_types) are
    shared unchanged."""
    import numpy as np
    from rdkit import Chem
    f = copy.copy(m)
    f.atom_pos = m.atom_pos @ R.T + t
    if getattr(m, "surf_pos", None) is not None:
        f.surf_pos = m.surf_pos @ R.T + t
    if getattr(m, "pharm_ancs", None) is not None:
        f.pharm_ancs = m.pharm_ancs @ R.T + t
        v = m.pharm_vecs @ R.T; nrm = np.linalg.norm(v, axis=1, keepdims=True)
        f.pharm_vecs = v / np.where(nrm > 0, nrm, 1.0)
    if getattr(m, "mol", None) is not None:
        fm = Chem.Mol(m.mol)                       # deep copy
        conf = fm.GetConformer()
        pos = np.asarray(conf.GetPositions()) @ R.T + t
        for i in range(fm.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i].tolist())
        f.mol = fm
    return f


def self_pairs(mols, n, device):
    import numpy as np
    from shepherd_score.container import MoleculePair
    rng = np.random.default_rng(1); out = []
    for _ in range(n):
        m = mols[rng.integers(0, len(mols))]
        a = rng.standard_normal((3, 3)); R, _ = np.linalg.qr(a)
        if np.linalg.det(R) < 0: R[:, 0] = -R[:, 0]
        out.append(MoleculePair(m, _rotcopy(m, R, rng.standard_normal(3)),
                                do_center=False, num_surf_points=m.num_surf_points, device=device))
    return out


def align(batch, mode, steps, backend):
    import numpy as np
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(batch)
    k = dict(backend=backend, max_num_steps=steps)
    if mode == "vol": b.align_with_vol(no_H=True, alpha=0.81, **k)
    elif mode == "surf": b.align_with_surf(alpha=0.81, **k)
    elif mode == "esp": b.align_with_esp(alpha=0.81, lam=0.3, **k)
    elif mode == "vol_esp": b.align_with_vol_esp(lam=0.3, alpha=0.81, **k)
    elif mode == "esp_combo": b.align_with_esp_combo(alpha=0.81, lam=0.001, esp_weight=0.5, **k)
    elif mode == "pharm": b.align_with_pharm(similarity="tanimoto", **k)
    elif mode == "vol_color": b.align_with_vol_color(color_weight=0.5, alpha=0.81, **k)
    return np.array([float(getattr(p, _ATTR[mode])) for p in batch])


def analyze(mode, rows, mean_tol, self_tol):
    import numpy as np
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    cross = np.array([r["cross"] for r in rows])
    ceiling = cross.max(0); ceil_mean = float(ceiling.mean())
    ref_self = max(r["self_min"] for r in rows)
    recs = []
    for i, r in enumerate(rows):
        c = cross[i]; rec = float(c.mean()) / ceil_mean
        tail = float(np.mean((ceiling - c) > 0.01 * np.maximum(ceiling, 1e-9)) * 100)
        ok = (rec >= 1.0 - mean_tol) and (r["self_min"] >= ref_self - self_tol)
        recs.append({"seeds": r["seeds"], "steps": r["steps"], "compute": r["seeds"] * r["steps"],
                     "mean": float(c.mean()), "recovery": rec, "tail_pct": tail,
                     "self_min": r["self_min"], "seconds": r["seconds"], "ok": ok})
    passing = [x for x in recs if x["ok"]]
    best = min(passing, key=lambda x: (x["compute"], x["seeds"])) if passing else None
    return {"mode": mode, "ceiling_mean": ceil_mean, "ref_self_min": ref_self,
            "recommended": best, "grid": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=ALL_MODES)
    ap.add_argument("--seed-grid", type=int, nargs="+", default=[8, 12, 16, 20, 28, 40, 64])
    ap.add_argument("--step-grid", type=int, nargs="+", default=[40, 70, 100, 150, 250])
    ap.add_argument("--backend", default="numba")
    ap.add_argument("--n-cross", type=int, default=150)
    ap.add_argument("--n-self", type=int, default=60)
    ap.add_argument("--n-surf", type=int, default=200)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mean-tol", type=float, default=0.003)
    ap.add_argument("--self-tol", type=float, default=0.003)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "optimize_defaults_results.json"))
    ap.add_argument("--smiles", default=_DEFAULT_SMI)
    a = ap.parse_args()

    if a.smoke:
        a.seed_grid = [12, 40]; a.step_grid = [50, 150]
        a.n_cross = 24; a.n_self = 12; a.limit = 8

    _cap_threads(a.threads)
    import numpy as np
    import torch
    from shepherd_score.accel.batch import aligners
    dev = torch.device("cpu")

    print(f"building up to {a.limit} molecules (n_surf={a.n_surf}) ...", flush=True)
    t0 = time.perf_counter()
    mols = build_mols(a.smiles, a.limit, a.n_surf)
    print(f"built {len(mols)} molecules in {time.perf_counter()-t0:.1f}s; "
          f"cross={a.n_cross} self={a.n_self} threads={a.threads}", flush=True)

    results = {}
    for mode in a.modes:
        print(f"\n=== {mode} ===", flush=True)
        cp = cross_pairs(mols, a.n_cross, dev)
        sp = self_pairs(mols, a.n_self, dev)
        rows = []
        for steps in a.step_grid:
            for seeds in a.seed_grid:
                aligners._NUM_SEEDS = seeds          # what _seeds_for reads
                t1 = time.perf_counter()
                cs = align(cp, mode, steps, a.backend)
                ss = align(sp, mode, steps, a.backend)
                dt = time.perf_counter() - t1
                rows.append({"mode": mode, "seeds": seeds, "steps": steps,
                             "cross": [round(float(x), 6) for x in cs],
                             "self_min": float(ss.min()), "self_mean": float(ss.mean()),
                             "seconds": dt})
                print(f"    s={seeds:>3} st={steps:>3}  mean={cs.mean():.4f}  "
                      f"self_min={ss.min():.4f}  {dt:.1f}s", flush=True)
        results[mode] = analyze(mode, rows, a.mean_tol, a.self_tol)
        results["_raw_" + mode] = rows
        with open(a.out, "w") as fh:                  # save partial after each mode
            json.dump(results, fh, indent=2)

    print("\n\n================ RECOMMENDED PER-MODE DEFAULTS ================")
    print(f"{'mode':>10} {'seeds':>6} {'steps':>6} {'recov':>7} {'tail%':>6} {'self':>6} {'compute':>8}")
    for mode in a.modes:
        res = results.get(mode)
        if not res or not res["recommended"]:
            print(f"{mode:>10}   (no passing config)"); continue
        b = res["recommended"]
        print(f"{mode:>10} {b['seeds']:>6} {b['steps']:>6} {b['recovery']*100:>6.2f}% "
              f"{b['tail_pct']:>5.1f}% {b['self_min']:>6.3f} {b['compute']:>8}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
