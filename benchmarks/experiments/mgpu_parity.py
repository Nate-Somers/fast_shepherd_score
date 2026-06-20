"""Validate the process-per-GPU path (_run_distributed_procs) is bit-exact vs the
direct single-GPU align. On 1 GPU it runs with one worker — still exercising the full
extract -> spawn -> worker-rebuild -> align -> write-back machinery."""
import numpy as np
import torch

from benchmarks.benchmark import _build_molecule, DRUGS
from shepherd_score.container import MoleculePair as MP
import shepherd_score.container._batch_align as cc

KW = {"vol": dict(alpha=0.81, steps_fine=100), "surf": dict(alpha=0.81, steps_fine=100),
      "esp": dict(alpha=0.81, lam=0.3, num_repeats=50, topk=30, steps_fine=100, lr=0.075),
      "pharm": dict(num_repeats=50, topk=30, steps_fine=100, lr=0.1)}
OUT = {"vol": ("transform_vol_noH", "sim_aligned_vol_noH"),
       "surf": ("transform_surf", "sim_aligned_surf"),
       "esp": ("transform_esp", "sim_aligned_esp"),
       "pharm": ("transform_pharm", "sim_aligned_pharm")}


def build_pairs(n=24):
    rng = np.random.default_rng(7)
    smis = [s for _, s, _ in DRUGS]
    sel = []
    while len(sel) < n:
        i, j = int(rng.integers(len(smis))), int(rng.integers(len(smis)))
        if i != j:
            sel.append((smis[i], smis[j]))
    dev = torch.device("cuda")
    return [MP(_build_molecule(a), _build_molecule(b), do_center=False, device=dev) for a, b in sel]


def collect(pairs, mode):
    tf, sc = OUT[mode]
    s = np.array([float(getattr(p, sc)) for p in pairs])
    t = np.stack([torch.as_tensor(getattr(p, tf)).detach().cpu().numpy().astype(np.float64) for p in pairs])
    return s, t


def main():
    for mode in ["vol", "surf", "esp", "pharm"]:
        fn = getattr(MP, "_align_batch_" + mode)
        a = build_pairs(); fn(a, **KW[mode]); ref_s, ref_t = collect(a, mode)
        b = build_pairs(); cc._run_distributed_procs(fn, b, **KW[mode]); got_s, got_t = collect(b, mode)
        ds = float(np.abs(ref_s - got_s).max())
        dt = float(np.abs(ref_t - got_t).max())
        ok = ds < 1e-6 and dt < 1e-5
        print(f"{mode:5s}: max|Δscore|={ds:.2e}  max|Δtransform|={dt:.2e}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
