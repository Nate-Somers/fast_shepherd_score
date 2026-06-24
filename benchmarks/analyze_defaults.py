"""Re-analyze optimize_defaults_results.json -- knee finder vs the converged reference.

The reference per mode is the most-converged config in the grid (max seeds, max steps) --
i.e. what the fork produces today at full effort. For every config we report:
  recovery = mean(cand) / mean(ref)                 (fraction of converged overlap captured)
  tail     = % of pairs where ref - cand > 1% of ref (meaningful per-pair regressions)

We decompose the two knobs (they're ~separable):
  STEP knee : smallest steps s.t., at max seeds, recovery >= REC_TOL of the converged mean
  SEED knee : smallest seeds s.t., at the step knee, recovery >= REC_TOL and tail <= TAIL_TOL
The recommendation is (seed-knee, step-knee). Compute is shown as % of the current default.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC_TOL = float(os.environ.get("REC_TOL", 0.999))     # capture >= 99.9% of converged mean
TAIL_TOL = float(os.environ.get("TAIL_TOL", 3.0))     # <= 3% of pairs >1% below the ref config
SELF_TOL = 0.003

CUR_SEEDS = {"vol": 18, "surf": 20, "esp": 40, "vol_esp": 40, "esp_combo": 50,
             "pharm": 40, "vol_color": 40}
CUR_STEPS = 100


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "optimize_defaults_results.json")
    data = json.load(open(path))
    modes = [m for m in ["vol", "surf", "esp", "vol_esp", "esp_combo", "pharm", "vol_color"]
             if "_raw_" + m in data]

    print(f"reference = converged config (max seeds x max steps). "
          f"capture >= {REC_TOL*100:.2f}% of its mean, tail(>1%) <= {TAIL_TOL:.0f}%\n")
    print(f"{'mode':>10} {'rec_seeds':>9} {'rec_steps':>9} {'recov%':>7} {'tail%':>6} "
          f"{'self':>5} {'cur(s/st)':>10} {'cost%cur':>8}")
    print("-" * 78)

    summary = {}
    for m in modes:
        rows = data["_raw_" + m]
        seeds_vals = sorted({r["seeds"] for r in rows})
        steps_vals = sorted({r["steps"] for r in rows})
        smax, stmax = max(seeds_vals), max(steps_vals)
        tab = {(r["seeds"], r["steps"]): np.array(r["cross"]) for r in rows}
        selfm = {(r["seeds"], r["steps"]): r["self_min"] for r in rows}
        ref = tab[(smax, stmax)]
        ref_mean = float(ref.mean())
        ref_self = max(selfm.values())

        def rec(c): return float(c.mean()) / ref_mean
        def tail(c): return float(np.mean((ref - c) > 0.01 * np.maximum(ref, 1e-9)) * 100)

        # STEP knee at max seeds
        step_knee = stmax
        for st in steps_vals:
            if rec(tab[(smax, st)]) >= REC_TOL:
                step_knee = st; break
        # SEED knee at the step knee
        seed_knee = smax
        for sd in seeds_vals:
            c = tab[(sd, step_knee)]
            if rec(c) >= REC_TOL and tail(c) <= TAIL_TOL and selfm[(sd, step_knee)] >= ref_self - SELF_TOL:
                seed_knee = sd; break

        c = tab[(seed_knee, step_knee)]
        cost = (seed_knee * step_knee) / (CUR_SEEDS[m] * CUR_STEPS) * 100
        print(f"{m:>10} {seed_knee:>9} {step_knee:>9} {rec(c)*100:>6.2f}% {tail(c):>5.1f}% "
              f"{selfm[(seed_knee, step_knee)]:>5.2f} {f'{CUR_SEEDS[m]}/{CUR_STEPS}':>10} {cost:>7.0f}%")
        summary[m] = {"seeds": seed_knee, "steps": step_knee, "recovery": rec(c),
                      "tail_pct": tail(c), "ref_mean": ref_mean,
                      "cost_pct_of_current": cost, "current": [CUR_SEEDS[m], CUR_STEPS]}

    print("\nesp_combo: seed column is NOMINAL (driver hardwires 50 seeds; FINE_NUM_SEEDS is ignored).")
    json.dump(summary, open(os.path.join(HERE, "recommended_defaults.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
