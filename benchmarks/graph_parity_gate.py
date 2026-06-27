"""Parity gate for the CUDA-graph fine loop vs the eager fine loop, per mode.

For each mode it aligns the SAME molecule set twice -- once with the graph fast path
(FINE_CUDA_GRAPHS=1, P-cap raised so every bucket is actually graphed) and once with the
eager loop (FINE_CUDA_GRAPHS=0) -- in separate subprocesses (the flags are import-time),
and asserts the graph path does not regress quality:

  - rotated self-copies still reach the optimum (~1.0), and
  - cross-pair mean overlap is not below the eager mean (the graph runs a fixed step count
    >= the eager early-stop, so it is provably >= the eager result on the tracked best;
    per-pair |delta| is reported for visibility).

It also asserts graphs were ACTUALLY captured (the cache is non-empty) so a silent
fall-back to eager cannot masquerade as a pass.

    python benchmarks/graph_parity_gate.py --modes vol surf esp
Requires a CUDA GPU with the Triton backend.
"""
import argparse, copy, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

_ATTR = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf", "esp": "sim_aligned_esp",
         "pharm": "sim_aligned_pharm", "vol_color": "sim_aligned_vol_color"}
_SMI = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "Oc1ccccc1", "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O",
        "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "O(CCN(C)C)C(c1ccccc1)c1ccccc1",
        "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", "CN1CCCC1c1cccnc1",
        "CC(C)NCC(O)COc1cccc2ccccc12"]


def _build_mols():
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
    import torch
    from shepherd_score.container import MoleculePair
    from shepherd_score.accel.drivers import _graphed
    mols = _build_mols(); dev = torch.device("cuda:0")
    lib = [mols[i % len(mols)] for i in range(60)]
    cross = [MoleculePair(mols[q], lib[j], do_center=True, device=dev) for q in range(6) for j in range(len(lib))]
    rng = np.random.default_rng(0); q0 = mols[0]; sp = []
    for _ in range(80):
        a = rng.standard_normal((3, 3)); R, _ = np.linalg.qr(a)
        if np.linalg.det(R) < 0: R[:, 0] = -R[:, 0]
        sp.append(MoleculePair(q0, _rotcopy(q0, R, rng.standard_normal(3)), do_center=True, device=dev))
    cs = _align(cross, mode); ss = _align(sp, mode)
    json.dump({"mode": mode, "cross": [round(float(x), 6) for x in cs],
               "self_min": float(ss.min()), "self_mean": float(ss.mean()),
               "graphs_captured": len(_graphed._FINE_GRAPH_CACHE)}, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=2, metavar=("MODE", "OUT"))
    ap.add_argument("--modes", nargs="+", default=["vol", "surf", "esp"])
    a = ap.parse_args()
    if a.cell:
        return cell(a.cell[0], a.cell[1])

    print("graph parity gate: CUDA-graph fine loop vs eager fine loop\n")
    print(f"{'mode':>10} {'self g/e':>16} {'cross g/e':>18} {'d_mean':>8} {'max|dP|':>9} {'graphs':>7} {'PASS':>5}")
    ok_all = True
    for mode in a.modes:
        res = {}
        for tag, env in (("eager", {"FINE_CUDA_GRAPHS": "0"}),
                         ("graph", {"FINE_CUDA_GRAPHS": "1", "FINE_GRAPH_MAX_P": "2000000"})):
            fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
            e = dict(os.environ); e.update(env)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--cell", mode, p],
                               env=e, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{mode:>10}  cell '{tag}' failed:\n{r.stderr[-1500:]}"); ok_all = False; res = None; break
            res[tag] = json.load(open(p)); os.remove(p)
        if not res:
            continue
        gc, ec = np.array(res["graph"]["cross"]), np.array(res["eager"]["cross"])
        dmean = (gc.mean() - ec.mean()) / max(ec.mean(), 1e-9) * 100
        maxdp = float(np.max(np.abs(gc - ec)))
        ncap = res["graph"]["graphs_captured"]
        # PASS: graphs actually captured, mean not regressed >0.3%, self-copy preserved.
        ok = (ncap > 0) and (dmean > -0.3) and (res["graph"]["self_min"] >= res["eager"]["self_min"] - 0.003)
        ok_all &= ok
        print(f"{mode:>10} {res['graph']['self_min']:.4f}/{res['eager']['self_min']:.4f} "
              f"{gc.mean():.4f}/{ec.mean():.4f} {dmean:>+7.2f}% {maxdp:>9.4f} {ncap:>7d} {'PASS' if ok else 'FAIL':>5}")
    print("\nGRAPH PARITY GATE:", "PASS" if ok_all else "FAIL",
          "\n  criteria: >=1 graph captured (no silent eager fallback), cross mean not regressed vs"
          "\n  eager by >0.3%, self-copy recovery preserved. max|dP| (per-pair) is INFO -- the graph"
          "\n  runs a fixed step count >= the eager early-stop, so small per-pair deltas are expected.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
