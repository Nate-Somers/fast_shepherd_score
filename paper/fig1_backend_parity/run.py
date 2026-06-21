"""
Figure 1 (data) — backend parity: do the fast Triton/GPU kernels reproduce the
reference JAX implementation?

Both code paths are the SAME public API (`MoleculePairBatch.align_with_*`); only
the `backend=` differs ("jax" = reference JAX/XLA, "triton" = the fork's GPU
kernels). We align every DISTINCT drug pair (optimum < 1, so scores span a real
range — a meaningful scatter, unlike self-copies that all sit at 1.0) with both
backends and record the per-pair aligned similarity for each mode.

If the GPU kernels preserve the math, the two backends agree pair-for-pair
(points on y = x, tiny mean|Δ|). Run (repo root, GPU env):
    PYTHONPATH=. python paper/fig1_backend_parity/run.py
Writes parity.json next to this file.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import DRUGS, build_fss_molecule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["vol", "surf", "esp", "pharm"]
SCORE_ATTR = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
              "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}
CFG = dict(num_repeats=16, steps=100, lr=0.1, alpha=0.81, lam=0.3)


def align(mode, pairs, backend):
    """Align a list of MoleculePair via the chosen backend; return per-pair scores."""
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(pairs)
    jax_kw = dict(num_workers=1, use_shmap=False) if backend == "jax" else {}
    if mode == "vol":
        b.align_with_vol(no_H=True, backend=backend, alpha=CFG["alpha"],
                         num_repeats=CFG["num_repeats"], lr=CFG["lr"],
                         max_num_steps=CFG["steps"], **jax_kw)
    elif mode == "surf":
        b.align_with_surf(alpha=CFG["alpha"], backend=backend,
                          num_repeats=CFG["num_repeats"], lr=CFG["lr"],
                          max_num_steps=CFG["steps"], **jax_kw)
    elif mode == "esp":
        b.align_with_esp(alpha=CFG["alpha"], lam=CFG["lam"], backend=backend,
                         num_repeats=CFG["num_repeats"], lr=CFG["lr"],
                         max_num_steps=CFG["steps"], **jax_kw)
    elif mode == "pharm":
        b.align_with_pharm(backend=backend, num_repeats=CFG["num_repeats"],
                           lr=CFG["lr"], max_num_steps=CFG["steps"], **jax_kw)
    # align_with_* writes the aligned score in place on each input MoleculePair.
    return np.array([float(getattr(p, SCORE_ATTR[mode])) for p in pairs])


def make_pairs(device):
    """All distinct (i<j) drug pairs as MoleculePair on the given device."""
    from shepherd_score.container import MoleculePair
    mols = [build_fss_molecule(smi) for _, smi, _ in DRUGS]
    names = [n for n, _, _ in DRUGS]
    pairs, labels = [], []
    for i, j in itertools.combinations(range(len(mols)), 2):
        # num_surf_points must be non-None for the surf/esp _core path (it is a
        # flag on the PAIR, not inferred from the molecules); molecules are
        # pre-built Molecule objects so this does not resample them.
        pairs.append(MoleculePair(mols[i], mols[j], do_center=False, device=device,
                                  num_surf_points=mols[i].num_surf_points))
        labels.append(f"{names[i]}~{names[j]}")
    return pairs, labels


def main():
    import torch
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        raise SystemExit("Triton backend needs CUDA; run in the GPU env.")
    out = {"_meta": {"n_pairs": None, "cfg": CFG,
                     "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}}
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    for mode in MODES:
        try:
            # Fresh pairs per backend (alignment writes results in place).
            pj, labels = make_pairs(cpu)
            sj = align(mode, pj, "jax")
            pt, _ = make_pairs(cuda)
            st = align(mode, pt, "triton")
            out[mode] = {"labels": labels, "jax": sj.tolist(), "triton": st.tolist()}
            d = np.abs(sj - st)
            print(f"{mode:5s} n={len(labels)}  mean|Δ|={d.mean():.2e}  "
                  f"max|Δ|={d.max():.2e}  jax∈[{sj.min():.3f},{sj.max():.3f}]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[mode] = {"err": f"{type(e).__name__}: {e}"}
    out["_meta"]["n_pairs"] = len(out.get("vol", {}).get("labels", []) or [])

    with open(os.path.join(HERE, "parity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'parity.json')}")


if __name__ == "__main__":
    main()
