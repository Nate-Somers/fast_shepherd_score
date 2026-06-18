"""Print distinct-molecule-pair alignment scores for all modes (deterministic).
Used to verify the overhead-vectorization is bit-identical: run on new code, git
stash, run on committed code, diff."""
import numpy as np
import torch

from benchmarks.real_workloads import _build_molecule, DRUGS
from shepherd_score.container import MoleculePair as MP

dev = torch.device("cuda")
rng = np.random.default_rng(7)
smis = [s for _, s, _ in DRUGS]
sel = []
while len(sel) < 30:
    i, j = int(rng.integers(len(smis))), int(rng.integers(len(smis)))
    if i != j:
        sel.append((smis[i], smis[j]))

KW = {"vol": dict(alpha=0.81), "surf": dict(alpha=0.81),
      "esp": dict(alpha=0.81, lam=0.3, num_repeats=50, topk=30, lr=0.075),
      "pharm": dict(num_repeats=50, topk=30, lr=0.1)}
FN = {"vol": MP.align_batch_vol, "surf": MP.align_batch_surf,
      "esp": MP.align_batch_esp, "pharm": MP.align_batch_pharm}
AT = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
      "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}

for mode in ("vol", "surf", "esp", "pharm"):
    pairs = [MP(_build_molecule(a), _build_molecule(b), do_center=False, device=dev) for a, b in sel]
    FN[mode](pairs, steps_fine=100, **KW[mode])
    sc = np.array([float(getattr(p, AT[mode])) for p in pairs])
    print(mode + " " + " ".join(f"{x:.6f}" for x in sc))
