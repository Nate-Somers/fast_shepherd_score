"""
Figure 3 (data) — fair speed + capability comparison vs open-source CPU tools.

Tools compared (all open source), each doing its NATIVE per-pair 3D-similarity op
with molecules / conformers / descriptors pre-built (the realistic amortized
screening cost — you build a library once, then score many pairs):

  USRCAT      RDKit alignment-free shape+pharmacophore moment descriptor
              -> timed op: GetUSRScore(d_i, d_j)            (no pose, no ESP)
  RDKit O3A   atom-mapping (MMFF) 3D alignment
              -> timed op: GetO3A(prb,ref).Align().Score()  (pose; no ESP/Gaussian)
  ESP-Sim     Gaussian shape + electrostatic-potential similarity (RDKit-based)
              -> timed op: O3A-align then GetShapeSim+GetEspSim (shape+ESP; CPU)
  fss (CPU)   this package, surface shape + ESP, JAX backend on CPU
  fss (GPU)   this package, surface shape + ESP, Triton backend (batched)

Only the aligned shape+ESP tools (ESP-Sim, fss) do the same job; USRCAT and O3A
are faster but do NOT compute an electrostatic overlay (shown in the capability
matrix). fss is the only tool that is GPU-batched.

Run (repo root, GPU env):  PYTHONPATH=. python paper/fig3_speed_vs_baselines/run.py
Writes results.json next to this file.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import DRUGS, build_fss_molecule, build_rdkit_mol, best_of_n  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 30          # pairs used for the per-pair-rate tools
N_CPU_PAIRS = 16      # fewer for the slow fss-CPU path (rate is N-independent)
ALPHA, LAM = 0.81, 0.3
CFG = dict(num_repeats=16, steps=100, lr=0.1)


def time_loop(fn, items, reps=3):
    """Best-of-`reps` wall-clock of applying fn to every item; -> items/s."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return len(items) / best, best


def main():
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign, rdMolDescriptors
    import espsim

    has_cuda = torch.cuda.is_available()
    names = [n for n, _, _ in DRUGS]
    smis = [s for _, s, _ in DRUGS]

    # ---- build everything ONCE (setup, untimed) ----
    rd = [build_rdkit_mol(s) for s in smis]                 # Hs + conformer + MMFF-ready
    usrcat = [rdMolDescriptors.GetUSRCAT(m) for m in rd]    # precomputed descriptors
    fss = [build_fss_molecule(s) for s in smis]

    pair_idx = list(itertools.combinations(range(len(rd)), 2))[:N_PAIRS]
    cpu_idx = pair_idx[:N_CPU_PAIRS]

    res = {"_meta": {"gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                     "n_pairs": N_PAIRS, "n_cpu_pairs": N_CPU_PAIRS},
           "throughput": {}, "scores": {}}

    # ---- USRCAT: descriptor compare (no align, no ESP) ----
    rate, _ = time_loop(lambda ij: rdMolDescriptors.GetUSRScore(usrcat[ij[0]], usrcat[ij[1]]),
                        pair_idx, reps=5)
    res["throughput"]["USRCAT"] = rate
    print(f"USRCAT       {rate:12.1f} pairs/s")

    # ---- RDKit O3A: atom-mapping align + score ----
    def o3a_op(ij):
        prb = Chem.Mol(rd[ij[0]])                            # copy: Align mutates coords
        o = rdMolAlign.GetO3A(prb, rd[ij[1]])
        o.Align(); return o.Score()
    rate, _ = time_loop(o3a_op, pair_idx, reps=3)
    res["throughput"]["RDKit O3A"] = rate
    print(f"RDKit O3A    {rate:12.1f} pairs/s")

    # ---- ESP-Sim: O3A-align then Gaussian shape + ESP similarity ----
    def esp_op(ij):
        prb = Chem.Mol(rd[ij[0]])
        rdMolAlign.GetO3A(prb, rd[ij[1]]).Align()
        sh = espsim.GetShapeSim(prb, rd[ij[1]])
        ep = espsim.GetEspSim(prb, rd[ij[1]], partialCharges="mmff")
        return sh, ep
    rate, _ = time_loop(esp_op, pair_idx, reps=3)
    res["throughput"]["ESP-Sim"] = rate
    print(f"ESP-Sim      {rate:12.1f} pairs/s")

    # ---- fss: ALL modes (vol/surf = shape, esp = shape+ESP, pharm) ----
    # esp is fss's SLOWEST/least-favorable mode (surface points x electrostatics);
    # it is what the ESP-Sim comparison rests on, but the shape modes (vol/surf)
    # are the package's fast path. We report every mode so the comparison is not
    # pinned to the worst case.
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    def make_mps(idxs, device):
        return [MoleculePair(fss[i], fss[j], do_center=False, device=device,
                             num_surf_points=fss[i].num_surf_points) for (i, j) in idxs]

    def align_batch(mps, mode, backend):
        b = MoleculePairBatch(mps)
        kw = dict(num_workers=1, use_shmap=False) if backend == "jax" else {}
        nr, st, lr = CFG["num_repeats"], CFG["steps"], CFG["lr"]
        if mode == "vol":
            b.align_with_vol(no_H=True, alpha=ALPHA, backend=backend, num_repeats=nr,
                             lr=lr, max_num_steps=st, **kw)
        elif mode == "surf":
            b.align_with_surf(alpha=ALPHA, backend=backend, num_repeats=nr, lr=lr,
                              max_num_steps=st, **kw)
        elif mode == "esp":
            b.align_with_esp(alpha=ALPHA, lam=LAM, backend=backend, num_repeats=nr,
                             lr=lr, max_num_steps=st, **kw)
        elif mode == "pharm":
            b.align_with_pharm(backend=backend, num_repeats=nr, lr=lr,
                               max_num_steps=st, **kw)

    MODES_FSS = ["vol", "surf", "esp", "pharm"]

    # fss (CPU), esp mode (the mode comparable to ESP-Sim) -- batched JAX
    cpu = torch.device("cpu")
    mps_cpu = make_mps(cpu_idx, cpu)
    best, _ = best_of_n(lambda: align_batch(mps_cpu, "esp", "jax"), reps=1, budget=60, warmup=0)
    res["throughput"]["fss (CPU) esp"] = len(cpu_idx) / best
    print(f"fss (CPU) esp {len(cpu_idx)/best:12.1f} pairs/s")

    # fss (GPU), every mode -- batched Triton at nb (the package's intended use)
    if has_cuda:
        cuda = torch.device("cuda")
        nb = 4096
        fill = [pair_idx[k % len(pair_idx)] for k in range(nb)]
        for mode in MODES_FSS:
            mps = make_mps(fill, cuda)
            align_batch(mps, mode, "triton")               # warmup/autotune at this batch
            best, _ = best_of_n(lambda: align_batch(mps, mode, "triton"), reps=4, budget=8.0)
            res["throughput"][f"fss (GPU) {mode}"] = nb / best
            print(f"fss (GPU) {mode:4s} {nb / best:12.1f} pairs/s")

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'results.json')}")


if __name__ == "__main__":
    main()
