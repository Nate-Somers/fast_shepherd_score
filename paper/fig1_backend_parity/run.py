"""
Figure 1 (data) — backend parity: do the fast Triton/GPU kernels reproduce the
reference math?

Two complementary experiments, written to parity.json:

  (A) ALIGNED parity.  Align every DISTINCT drug pair (optimum < 1, scores span a
      real range) with backend="jax" and backend="triton" and record the per-pair
      aligned Tanimoto for each mode.  This is the END-TO-END comparison: kernel
      arithmetic + the multi-start SE(3) optimizer.  Note the multi-start seeds are
      DETERMINISTIC and IDENTICAL across backends (identity + PCA + Fibonacci
      rotations; see alignment/utils/fast_common.py) — so any residual is NOT random
      restart noise; it is the optimizer trajectory diverging under different
      arithmetic, plus a small systematic precision offset.

  (B) FIXED-POSE parity.  Score every distinct pair at the SAME fixed (identity) pose
      with three scoring implementations — NumPy (fp64, reference), JAX, and PyTorch
      (fp32, the precision the Triton kernels use) — via score_with_*(use=...).  With
      the optimizer removed, this isolates pure scoring-kernel agreement and shows the
      precision floor that (A)'s optimizer then amplifies.

Run (repo root, GPU env with jax):
    PYTHONPATH=. python paper/fig1_backend_parity/run.py
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
    return np.array([float(getattr(p, SCORE_ATTR[mode])) for p in pairs])


def fixed_pose_scores(mode, pairs, use):
    """Score each pair at its CURRENT (identity / un-optimized) pose with the chosen
    scoring implementation (no alignment), via score_with_*(use=...)."""
    out = []
    for p in pairs:
        if mode == "vol":
            s = p.score_with_vol(CFG["alpha"], no_H=True, use=use)
        elif mode == "surf":
            s = p.score_with_surf(CFG["alpha"], use=use)
        elif mode == "esp":
            s = p.score_with_esp(CFG["alpha"], lam=CFG["lam"], use=use)
        elif mode == "pharm":
            s = p.score_with_pharm(use=use)
        out.append(float(np.asarray(s).ravel()[0]))
    return np.array(out)


def make_pairs(device):
    from shepherd_score.container import MoleculePair
    mols = [build_fss_molecule(smi) for _, smi, _ in DRUGS]
    names = [n for n, _, _ in DRUGS]
    pairs, labels = [], []
    for i, j in itertools.combinations(range(len(mols)), 2):
        pairs.append(MoleculePair(mols[i], mols[j], do_center=False, device=device,
                                  num_surf_points=mols[i].num_surf_points))
        labels.append(f"{names[i]}~{names[j]}")
    return pairs, labels


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Triton backend needs CUDA; run in the GPU env.")
    cuda = torch.device("cuda"); cpu = torch.device("cpu")
    out = {"_meta": {"cfg": CFG, "gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__}}
    try:
        import jax
        out["_meta"]["jax"] = jax.__version__
        out["_meta"]["jax_x64"] = bool(jax.config.read("jax_enable_x64"))
    except Exception as e:
        out["_meta"]["jax"] = f"unavailable: {e}"

    for mode in MODES:
        rec = {}
        # (A) aligned parity: jax backend vs triton backend
        try:
            pj, labels = make_pairs(cpu)
            sj = align(mode, pj, "jax")
            pt, _ = make_pairs(cuda)
            st = align(mode, pt, "triton")
            rec.update(labels=labels, jax=sj.tolist(), triton=st.tolist())
            d = np.abs(sj - st)
            print(f"[aligned] {mode:5s} n={len(labels)} mean|Δ|={d.mean():.2e} "
                  f"max|Δ|={d.max():.2e} signed={np.mean(st-sj):+.2e} "
                  f"triton>jax {int(np.sum(st>sj))}/{len(labels)}")
        except Exception as e:
            import traceback; traceback.print_exc()
            rec["aligned_err"] = f"{type(e).__name__}: {e}"
        # (B) fixed-pose scoring parity: np (fp64) vs jax vs torch (fp32)
        try:
            pf, _ = make_pairs(cuda)
            fnp = fixed_pose_scores(mode, pf, "np")
            ftorch = fixed_pose_scores(mode, pf, "torch")
            try:
                fjax = fixed_pose_scores(mode, pf, "jax")
            except Exception:
                fjax = np.full(len(pf), np.nan)
            rec.update(fixed_np=fnp.tolist(), fixed_jax=fjax.tolist(),
                       fixed_torch=ftorch.tolist())
            dt = np.abs(ftorch - fnp)
            print(f"[fixed]   {mode:5s} |torch-np| mean={np.nanmean(dt):.2e} "
                  f"max={np.nanmax(dt):.2e}")
        except Exception as e:
            import traceback; traceback.print_exc()
            rec["fixed_err"] = f"{type(e).__name__}: {e}"
        out[mode] = rec

    out["_meta"]["n_pairs"] = len(out.get("vol", {}).get("labels", []) or [])
    with open(os.path.join(HERE, "parity.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'parity.json')}")


if __name__ == "__main__":
    main()
