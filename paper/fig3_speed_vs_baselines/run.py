"""
Figure 3 (data) — fair speed + capability comparison vs open-source 3D-similarity tools.

ALL tools run in ONE process on ONE machine (so when this is run on a datacenter GPU node,
the CPU baselines run on that node's CPU and fss on its GPU — a real same-machine
comparison, not the old cross-hardware fss-on-L40S vs ESP-Sim-on-laptop footnote).

Tools (each doing its NATIVE per-pair 3D-similarity op, molecules/descriptors pre-built):
  USRCAT      RDKit alignment-free shape+pharmacophore moment descriptor  (no pose, no ESP)
  RDKit O3A   atom-mapping (MMFF) 3D alignment                            (pose; no ESP/Gaussian)
  ESP-Sim     O3A-align then Gaussian shape + ESP similarity (CPU)        (shape+ESP)
  fss (CPU)   surface shape + ESP, JAX backend on CPU
  fss (GPU)   surface shape + ESP + pharm, Triton backend (batched)

Fixes over the old harness: a DIVERSE library (DUD-E decoys, many unique molecules) instead
of 30 pairs tiled 136×; mean ± SD over reps instead of best-of-N min; n reported on the
figure; same machine for every tool.

Run:  PYTHONPATH=. python paper/fig3_speed_vs_baselines/run.py --lib <smi> --n-lib 300
Writes results.json next to this file.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import DRUGS, build_fss_molecule, build_rdkit_mol  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, LAM = 0.81, 0.3
CFG = dict(num_repeats=16, steps=100, lr=0.1)


def read_smi(path, limit=None):
    out = []
    with open(path) as fh:
        for ln in fh:
            p = ln.strip().split()
            if p:
                out.append(p[0])
    return out[:limit] if limit else out


def rate_stats(fn, items, reps=5):
    """Apply fn to every item, `reps` times; return per-rep throughput stats (items/s)."""
    rates = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        dt = time.perf_counter() - t0
        rates.append(len(items) / dt)
    rates = np.array(rates)
    return {"mean": float(rates.mean()), "std": float(rates.std()),
            "max": float(rates.max()), "n": len(items), "reps": reps}


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_rate_stats(align_fn, nb, reps=5):
    """Best-of and mean±SD throughput for a batched GPU align of nb pairs."""
    align_fn(); cuda_sync()                                  # warmup / autotune
    rates = []
    for _ in range(reps):
        cuda_sync(); t0 = time.perf_counter()
        align_fn(); cuda_sync()
        rates.append(nb / (time.perf_counter() - t0))
    rates = np.array(rates)
    return {"mean": float(rates.mean()), "std": float(rates.std()),
            "max": float(rates.max()), "n": nb, "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=None, help="diverse library SMILES (e.g. DUD-E decoys)")
    ap.add_argument("--n-lib", type=int, default=300, help="unique molecules to build")
    ap.add_argument("--n-cpu-pairs", type=int, default=200, help="pairs for the CPU baselines")
    ap.add_argument("--n-fss-cpu", type=int, default=16, help="pairs for the slow fss-CPU path")
    ap.add_argument("--gpu-batch", type=int, default=4096)
    a = ap.parse_args()

    import torch
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign, rdMolDescriptors
    import espsim
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    has_cuda = torch.cuda.is_available()
    rng = np.random.default_rng(0)
    if a.lib and os.path.exists(a.lib):
        smis = read_smi(a.lib)
        rng.shuffle(smis)
        smis = smis[:a.n_lib]
    else:
        print("no --lib; using curated DRUGS (small)")
        smis = [s for _, s, _ in DRUGS]

    # ---- build everything ONCE (setup, untimed) ----
    print(f"building {len(smis)} molecules (rdkit + fss surfaces) ...", flush=True)
    rd, fss = [], []
    for s in smis:
        try:
            m = build_rdkit_mol(s); f = build_fss_molecule(s)
        except Exception:
            continue
        rd.append(m); fss.append(f)
    L = len(rd)
    usrcat = [rdMolDescriptors.GetUSRCAT(m) for m in rd]
    print(f"built {L} molecules", flush=True)

    # unique pairs (no replacement) for CPU baselines; many distinct pairs for GPU
    all_pairs = list(itertools.combinations(range(L), 2))
    rng.shuffle(all_pairs)
    cpu_pairs = all_pairs[:a.n_cpu_pairs]
    fsscpu_pairs = all_pairs[:a.n_fss_cpu]
    gpu_pairs = [all_pairs[k % len(all_pairs)] for k in range(a.gpu_batch)]

    res = {"_meta": {"gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                     "cpu": None, "n_lib": L, "n_cpu_pairs": len(cpu_pairs),
                     "gpu_batch": a.gpu_batch, "n_unique_pairs_in_batch": len(set(gpu_pairs))},
           "throughput": {}}
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.lower().startswith("model name"):
                    res["_meta"]["cpu"] = ln.split(":", 1)[1].strip(); break
    except OSError:
        pass

    # ---- USRCAT ----
    res["throughput"]["USRCAT"] = rate_stats(
        lambda ij: rdMolDescriptors.GetUSRScore(usrcat[ij[0]], usrcat[ij[1]]), cpu_pairs, reps=5)
    print("USRCAT", res["throughput"]["USRCAT"]["mean"], flush=True)

    # ---- RDKit O3A ----
    def o3a_op(ij):
        prb = Chem.Mol(rd[ij[0]]); o = rdMolAlign.GetO3A(prb, rd[ij[1]]); o.Align(); return o.Score()
    res["throughput"]["RDKit O3A"] = rate_stats(o3a_op, cpu_pairs, reps=3)
    print("O3A", res["throughput"]["RDKit O3A"]["mean"], flush=True)

    # ---- ESP-Sim (CPU) ----
    def esp_op(ij):
        prb = Chem.Mol(rd[ij[0]])
        rdMolAlign.GetO3A(prb, rd[ij[1]]).Align()
        return espsim.GetShapeSim(prb, rd[ij[1]]), espsim.GetEspSim(prb, rd[ij[1]], partialCharges="mmff")
    res["throughput"]["ESP-Sim"] = rate_stats(esp_op, cpu_pairs, reps=3)
    print("ESP-Sim", res["throughput"]["ESP-Sim"]["mean"], flush=True)

    def make_mps(idxs, device):
        return [MoleculePair(fss[i], fss[j], do_center=False, device=device,
                             num_surf_points=fss[i].num_surf_points) for (i, j) in idxs]

    def align_batch(mps, mode, backend):
        b = MoleculePairBatch(mps)
        kw = dict(num_workers=1, use_shmap=False) if backend == "jax" else {}
        nr, st, lr = CFG["num_repeats"], CFG["steps"], CFG["lr"]
        if mode == "vol":
            b.align_with_vol(no_H=True, alpha=ALPHA, backend=backend, num_repeats=nr, lr=lr, max_num_steps=st, **kw)
        elif mode == "surf":
            b.align_with_surf(alpha=ALPHA, backend=backend, num_repeats=nr, lr=lr, max_num_steps=st, **kw)
        elif mode == "esp":
            b.align_with_esp(alpha=ALPHA, lam=LAM, backend=backend, num_repeats=nr, lr=lr, max_num_steps=st, **kw)
        elif mode == "pharm":
            b.align_with_pharm(backend=backend, num_repeats=nr, lr=lr, max_num_steps=st, **kw)

    # ---- fss (CPU) esp ----
    mps_cpu = make_mps(fsscpu_pairs, torch.device("cpu"))
    t0 = time.perf_counter(); align_batch(mps_cpu, "esp", "jax"); dt = time.perf_counter() - t0
    res["throughput"]["fss (CPU) esp"] = {"mean": len(fsscpu_pairs) / dt, "std": 0.0,
                                          "max": len(fsscpu_pairs) / dt, "n": len(fsscpu_pairs), "reps": 1}
    print("fss CPU esp", res["throughput"]["fss (CPU) esp"]["mean"], flush=True)

    # ---- fss (GPU) all modes, diverse batch ----
    if has_cuda:
        cuda = torch.device("cuda")
        for mode in ["vol", "surf", "esp", "pharm"]:
            mps = make_mps(gpu_pairs, cuda)
            res["throughput"][f"fss (GPU) {mode}"] = gpu_rate_stats(
                lambda: align_batch(mps, mode, "triton"), a.gpu_batch, reps=5)
            print(f"fss GPU {mode}", res["throughput"][f"fss (GPU) {mode}"]["mean"], flush=True)

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
