"""esp_combo seed-sensitivity probe.

esp_combo's batched driver hardwires num_seeds=50 (the public entry never threads a seed
count through), so the main sweep couldn't vary it. Here we monkeypatch the seed count INTO
coarse_fine_esp_combo_align_many to measure the true seed knee (at a converged step count),
exactly as if esp_combo honored _seeds_for like the other modes.
"""
import os, sys, time
def _cap(n):
    for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
_cap(8)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for shepherd_score
sys.path.insert(0, HERE)                     # benchmarks/, for optimize_defaults
import numpy as np
import torch
import optimize_defaults as od  # reuse build_mols / cross_pairs / self_pairs / align

# --- monkeypatch esp_combo seed count -------------------------------------------------
import shepherd_score.accel.drivers.esp_combo as ec
_orig = ec.coarse_fine_esp_combo_align_many
_HOLDER = {"seeds": 50}
def _patched(*a, **k):
    k["num_seeds"] = _HOLDER["seeds"]
    return _orig(*a, **k)
ec.coarse_fine_esp_combo_align_many = _patched


def main():
    smi = od._DEFAULT_SMI
    dev = torch.device("cpu")
    mols = od.build_mols(smi, 30, 150)
    cp = od.cross_pairs(mols, 200, dev)
    sp = od.self_pairs(mols, 60, dev)
    steps = 70
    seed_grid = [5, 8, 12, 16, 20, 28, 40, 50]
    print(f"esp_combo seed probe: {len(mols)} mols, 200 cross, steps={steps}\n")
    print(f"{'seeds':>5} {'mean':>8} {'self_min':>9} {'sec':>6}")
    means = {}
    for s in seed_grid:
        _HOLDER["seeds"] = s
        t0 = time.perf_counter()
        cs = od.align(cp, "esp_combo", steps, "numba")
        ss = od.align(sp, "esp_combo", steps, "numba")
        dt = time.perf_counter() - t0
        means[s] = cs
        print(f"{s:>5} {cs.mean():>8.4f} {ss.min():>9.4f} {dt:>5.1f}s", flush=True)
    ref = means[50]; ref_mean = ref.mean()
    print(f"\nrecovery vs 50-seed (mean {ref_mean:.4f}):")
    for s in seed_grid:
        c = means[s]
        rec = c.mean() / ref_mean
        tail = np.mean((ref - c) > 0.01 * np.maximum(ref, 1e-9)) * 100
        print(f"  seeds={s:>3}  recov={rec*100:6.2f}%  tail={tail:4.1f}%")


if __name__ == "__main__":
    main()
