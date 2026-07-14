"""Throughput: CUDA-graph fine loop vs eager fine loop, per mode and batch size.

For each (mode, N) it times the end-to-end batched align (best-of-3 after a warmup that
captures + caches the graph and JIT-compiles the kernels) twice -- graph on vs off (the
FINE_CUDA_GRAPHS flag is import-time, so each runs in its own subprocess) -- and reports
aligns/sec and the graph speedup. End-to-end timing (not just the fine loop) is the honest
metric; the speedup is diluted by the un-graphed seed/bucket/H2D overhead, which is the
point -- it shows the realized, user-visible gain.

    python benchmarks/graph_speed.py --modes vol vol_color vol_esp pharm --sizes 64 256 1024
Requires a CUDA GPU + Triton.
"""
import argparse, json, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

_SMI = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "Oc1ccccc1", "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O",
        "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "O(CCN(C)C)C(c1ccccc1)c1ccccc1",
        "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", "CN1CCCC1c1cccnc1",
        "CC(C)NCC(O)COc1cccc2ccccc12"]


def _build_lib():
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


def _align(batch, mode):
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(batch)
    if mode == "vol": b.align_with_vol(no_H=True, backend="triton", alpha=0.81)
    elif mode == "surf": b.align_with_surf(alpha=0.81, backend="triton")
    elif mode == "esp": b.align_with_esp(alpha=0.81, lam=0.3, backend="triton")
    elif mode == "vol_esp": b.align_with_vol_esp(alpha=0.81, lam=0.3, no_H=True, backend="triton")
    elif mode == "pharm": b.align_with_pharm(backend="triton")
    elif mode == "vol_color": b.align_with_vol_color(color_weight=0.5, backend="triton")
    elif mode in ("vol_and_surf_esp", "esp_combo"): b.align_with_vol_and_surf_esp(alpha=0.81, backend="triton")


def cell(mode, N, out):
    import torch
    from shepherd_score.container import MoleculePair
    lib = _build_lib(); dev = torch.device("cuda:0")
    rng = np.random.default_rng(0)
    idx = [(int(rng.integers(len(lib))), int(rng.integers(len(lib)))) for _ in range(N)]
    pairs = [MoleculePair(lib[a], lib[b], do_center=True, device=dev) for a, b in idx]
    # warmup: 2 aligns -> capture+cache the graph (or JIT the eager kernels)
    for _ in range(2):
        _align(pairs, mode); torch.cuda.synchronize()
    # best-of-3 wall-clock of the end-to-end align
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _align(pairs, mode); torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    json.dump({"mode": mode, "N": N, "sec": best, "aligns_per_sec": N / best}, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=3, metavar=("MODE", "N", "OUT"))
    ap.add_argument("--modes", nargs="+", default=["vol", "vol_color", "vol_esp", "pharm"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 1024])
    a = ap.parse_args()
    if a.cell:
        return cell(a.cell[0], int(a.cell[1]), a.cell[2])

    print("graph speed: end-to-end aligns/sec, CUDA-graph vs eager fine loop\n")
    print(f"{'mode':>16} {'N':>6} {'eager a/s':>11} {'graph a/s':>11} {'speedup':>8}")
    for mode in a.modes:
        for N in a.sizes:
            row = {}
            for tag, env in (("eager", {"FINE_CUDA_GRAPHS": "0"}),
                             ("graph", {"FINE_CUDA_GRAPHS": "1", "FINE_GRAPH_MAX_P": "2000000"})):
                fd, p = tempfile.mkstemp(suffix=".json"); os.close(fd)
                e = dict(os.environ); e.update(env)
                r = subprocess.run([sys.executable, os.path.abspath(__file__), "--cell", mode, str(N), p],
                                   env=e, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"{mode:>16} {N:>6}  cell '{tag}' failed:\n{r.stderr[-1200:]}"); row = None; break
                row[tag] = json.load(open(p)); os.remove(p)
            if not row:
                continue
            ea, ga = row["eager"]["aligns_per_sec"], row["graph"]["aligns_per_sec"]
            print(f"{mode:>16} {N:>6} {ea:>11.1f} {ga:>11.1f} {ga / ea:>7.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
