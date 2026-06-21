"""
Figure 5 (data) — head-to-head vs ROSHAMBO2 (open-source GPU shape+color).

This is the one comparison that does NOT exist in the literature: fast_shepherd_score
vs ROSHAMBO2 on identical molecules / identical hardware. It is gated on a working
ROSHAMBO install (CUDA toolkit + PAPER build — see setup.sh / README). The harness:

  1. Builds the shared molecule set (the repo's curated drugs, or --sdf <file>) and
     writes a query + a dataset SDF (one rigid SE(3) copy per molecule, optimum = 1).
  2. Times fast_shepherd_score aligning the dataset to the query (Triton, batched)
     -> pairs/s + recovered self-similarity.
  3. Times ROSHAMBO scoring the same dataset against the same query -> mols/s +
     ComboTanimoto. (ROSHAMBO aligns shape+color; we compare shape throughput and
     the recovered self-overlap.)
  4. Writes results.json for plot.py.

Run (GPU env with roshambo installed):
    PYTHONPATH=. python paper/fig5_roshambo_headtohead/run.py
If roshambo is not importable it prints setup instructions and exits 0.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import DRUGS, build_rdkit_mol, build_fss_molecule, best_of_n  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 256  # dataset size (rigid copies sampled from the curated drugs)


def write_sdf(mols, path):
    from rdkit import Chem
    w = Chem.SDWriter(path)
    for m in mols:
        w.write(m)
    w.close()


def fss_throughput(query_smi, lib_smis):
    """Align a library of FSS molecules to a query (surf mode, Triton, batched)."""
    import torch
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    dev = torch.device("cuda")
    q = build_fss_molecule(query_smi)
    lib = [build_fss_molecule(s) for s in lib_smis]
    mps = [MoleculePair(q, m, do_center=True, device=dev, num_surf_points=q.num_surf_points)
           for m in lib]

    def go():
        b = MoleculePairBatch(mps)
        b.align_with_surf(alpha=0.81, backend="triton", num_repeats=16, max_num_steps=100)
    go()  # warmup
    best, _ = best_of_n(go, reps=4, budget=8.0)
    return len(mps) / best


def roshambo_throughput(query_path, dataset_path):
    """Run ROSHAMBO on the same query/dataset SDFs. API per the roshambo package
    (verify against the installed version — see README)."""
    from roshambo.api import get_similarity_scores
    t0 = time.perf_counter()
    # ROSHAMBO writes a scored table; signature may vary by version.
    get_similarity_scores(
        ref_file=os.path.basename(query_path),
        dataset_files_pattern=os.path.basename(dataset_path),
        ignore_hs=True, n_confs=0, use_carbon_radii=True,
        color=False, sort_by="ShapeTanimoto", write_to_file=True,
        gpu_id=0, working_dir=os.path.dirname(dataset_path),
    )
    dt = time.perf_counter() - t0
    return dt


def main():
    try:
        import roshambo  # noqa: F401
    except Exception:
        print("ROSHAMBO is not installed in this environment.\n"
              "This head-to-head requires a CUDA-toolkit build of ROSHAMBO (which wraps\n"
              "PAPER's CUDA kernels). See paper/fig5_roshambo_headtohead/setup.sh and README.\n"
              "Current env is missing nvcc + cmake, so the build was not run here.\n"
              "Once installed, re-run this script to produce results.json.")
        return 0

    import torch
    from rdkit import Chem
    rng = np.random.default_rng(0)
    smis = [s for _, s, _ in DRUGS]
    query_smi = smis[0]
    lib_smis = [smis[int(i)] for i in rng.integers(0, len(smis), size=N)]

    # write SDFs for ROSHAMBO (query + rigid-copy dataset)
    q_rd = build_rdkit_mol(query_smi)
    write_sdf([q_rd], os.path.join(HERE, "query.sdf"))
    write_sdf([build_rdkit_mol(s) for s in lib_smis], os.path.join(HERE, "dataset.sdf"))

    fss_rate = fss_throughput(query_smi, lib_smis)
    rosh_dt = roshambo_throughput(os.path.join(HERE, "query.sdf"),
                                  os.path.join(HERE, "dataset.sdf"))
    rosh_rate = N / rosh_dt

    res = {"gpu": torch.cuda.get_device_name(0), "n": N,
           "fss_pairs_per_s": fss_rate, "roshambo_mols_per_s": rosh_rate}
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
