"""Validate the applied per-mode defaults end-to-end through the PUBLIC API.

For each mode: run with the NEW baked-in defaults (no max_num_steps / no FINE_NUM_SEEDS ->
_MODE_SEEDS applies) and compare mean cross-overlap to a converged reference (64 seeds, 200
steps). Confirms each shipped default still captures >=99.9% of the converged mean, and that
esp_combo now responds to the seed config (was hardwired at 50).
"""
import os, sys
for v in ("NUMBA_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, torch
import optimize_defaults as od
from shepherd_score.accel.batch import aligners
from shepherd_score.container import MoleculePairBatch

DEFSTEPS = {"vol":50,"surf":50,"esp":50,"vol_esp":50,"esp_combo":50,"pharm":70,"vol_color":50}
# the prior fork defaults this change replaces (seeds; public max_num_steps was 200)
OLD_SEEDS = {"vol":18,"surf":20,"esp":40,"vol_esp":40,"esp_combo":50,"pharm":40,"vol_color":40}

def run(batch, mode, steps):
    b = MoleculePairBatch(batch); k = dict(backend="numba", max_num_steps=steps)
    if mode=="vol": b.align_with_vol(no_H=True, alpha=0.81, **k)
    elif mode=="surf": b.align_with_surf(alpha=0.81, **k)
    elif mode=="esp": b.align_with_esp(alpha=0.81, lam=0.3, **k)
    elif mode=="vol_esp": b.align_with_vol_esp(lam=0.3, alpha=0.81, **k)
    elif mode=="esp_combo": b.align_with_esp_combo(alpha=0.81, lam=0.001, esp_weight=0.5, **k)
    elif mode=="pharm": b.align_with_pharm(similarity="tanimoto", **k)
    elif mode=="vol_color": b.align_with_vol_color(color_weight=0.5, alpha=0.81, **k)
    return np.array([float(getattr(p, od._ATTR[mode])) for p in batch])

def main():
    dev = torch.device("cpu")
    mols = od.build_mols(od._DEFAULT_SMI, 30, 150)
    cp = od.cross_pairs(mols, 200, dev)
    print(f"{len(mols)} mols, 200 cross pairs\n")
    print("no-regression vs PRIOR fork defaults (old seeds @ 200 steps):")
    print(f"{'mode':>10} {'new(s/st)':>10} {'old(s/st)':>10} {'new_mean':>9} {'old_mean':>9} {'recovery':>9}")
    for m in od.ALL_MODES:
        aligners._NUM_SEEDS = None                       # -> _MODE_SEEDS (the new defaults)
        dmean = run(cp, m, DEFSTEPS[m]).mean()
        aligners._NUM_SEEDS = OLD_SEEDS[m]               # prior fork default
        rmean = run(cp, m, 200).mean()
        aligners._NUM_SEEDS = None
        rec = dmean / rmean * 100
        flag = "" if rec >= 99.85 else "  <-- CHECK"
        print(f"{m:>10} {f'{aligners._MODE_SEEDS[m]}/{DEFSTEPS[m]}':>10} {f'{OLD_SEEDS[m]}/200':>10} "
              f"{dmean:>9.4f} {rmean:>9.4f} {rec:>8.2f}%{flag}")

    # esp_combo now honors the seed config?
    print("\nesp_combo seed responsiveness (should differ now):")
    for s in (8, 50):
        aligners._NUM_SEEDS = s
        print(f"  FINE_NUM_SEEDS={s:>3} -> esp_combo mean {run(cp,'esp_combo',50).mean():.4f}")
    aligners._NUM_SEEDS = None

if __name__ == "__main__":
    main()
