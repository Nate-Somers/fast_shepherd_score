"""
Figure 5 — head-to-head vs ROSHAMBO2 (open-source GPU Gaussian shape).

Reads results.json (from run.py --combine) and renders:

  Left   Throughput (pairs/s, log) on the SAME GPU: fss `vol` (atomic-Gaussian volume,
         the representation-matched comparison to ROSHAMBO2) and fss `surf` vs ROSHAMBO2
         shape.  Compute-only (solid) and end-to-end incl. featurization (hatched).

  Right  Fairness anchor — recovered self-overlap on rigid SE(3) self-copies (optimum=1.0):
         every tool should recover ~1.0, confirming the throughput is measured at matched
         (solved) alignment quality, not by one tool cutting corners.

Run: PYTHONPATH=. python paper/fig5_roshambo_headtohead/plot.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import set_style, save_fig, TOOL_COLOR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# vol is the representation-matched comparison to ROSHAMBO2 (both = atom-centred Gaussian
# volume overlap).  fss's surface/ESP mode is a different, heavier representation (shown in
# Figs 2/3/6), so Fig 5 stays a clean apples-to-apples shape head-to-head.
SERIES = [  # (label, json-tool-key, throughput-mode-key, selfcopy-mode-key, color)
    ("fss · vol (atom Gaussian)", "fss", "vol", "vol", "#c0392b"),
    ("ROSHAMBO2 · shape", "roshambo2", "shape", "shape", "#7b3294"),
    ("ROSHAMBO2 · combo (shape+color)", "roshambo2", "combo", "combo", "#b07cc6"),
]


def main():
    plt = set_style()
    paths = sorted(glob.glob(os.path.join(HERE, "results*.json")))
    if not paths:
        raise SystemExit("no results.json — run fig5 run.py / sbatch first")
    with open(paths[0]) as fh:
        res = json.load(fh)
    gpu = (res.get("fss") or res.get("roshambo2") or {}).get("gpu", "?")

    rows = []
    for label, tool, tmode, smode, color in SERIES:
        d = res.get(tool)
        if not d:
            continue
        tp = d.get("throughput", {}).get(tmode)
        sc = d.get("selfcopy", {}).get(smode)
        if tp is None:
            continue
        rows.append(dict(label=label, color=color,
                         compute=tp["compute_pairs_per_s"], e2e=tp["endtoend_pairs_per_s"],
                         npairs=tp["n_pairs"],
                         self_mean=(sc or {}).get("mean", np.nan), self_min=(sc or {}).get("min", np.nan)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))[::-1]

    # ---- Left: throughput ----
    for yy, r in zip(y, rows):
        axL.barh(yy + 0.18, r["compute"], height=0.34, color=r["color"], label="compute-only" if yy == y[0] else None)
        axL.barh(yy - 0.18, r["e2e"], height=0.34, color=r["color"], alpha=0.45, hatch="//",
                 label="end-to-end" if yy == y[0] else None)
        axL.text(r["compute"] * 1.05, yy + 0.18, f"{r['compute']:,.0f}", va="center", fontsize=8.5)
        axL.text(r["e2e"] * 1.05, yy - 0.18, f"{r['e2e']:,.0f}", va="center", fontsize=8, color="#555")
    axL.set_yticks(y); axL.set_yticklabels([r["label"] for r in rows])
    axL.set_xscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter
    axL.xaxis.set_major_locator(LogLocator(base=10))
    axL.xaxis.set_minor_formatter(NullFormatter())
    axL.set_xlim(1e3, 2e5)
    axL.set_xlabel("throughput — query×dataset alignments / second (log)")
    axL.set_title("Shape-alignment throughput on one GPU")
    axL.legend(loc="lower right", fontsize=9)

    # speedup annotation: fss vol vs ROSHAMBO2 shape (matched) and vs combo (its typical mode)
    fv = next((r for r in rows if r["label"].startswith("fss · vol")), None)
    rsh = next((r for r in rows if r["label"] == "ROSHAMBO2 · shape"), None)
    rcb = next((r for r in rows if r["label"].startswith("ROSHAMBO2 · combo")), None)
    if fv:
        lines = []
        if rsh and rsh["compute"] > 0:
            lines.append(f"fss vol / ROSHAMBO2 shape = {fv['compute']/rsh['compute']:.1f}×  (matched)")
        if rcb and rcb["compute"] > 0:
            lines.append(f"fss vol / ROSHAMBO2 combo = {fv['compute']/rcb['compute']:.1f}×")
        if lines:
            axL.text(0.02, 0.02, "\n".join(lines), transform=axL.transAxes, fontsize=8.8,
                     color="#c0392b", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.9))

    # ---- Right: self-copy recovered overlap (quality anchor) ----
    for yy, r in zip(y, rows):
        axR.barh(yy, r["self_mean"], height=0.5, color=r["color"], alpha=0.9)
        if np.isfinite(r["self_min"]):
            axR.plot([r["self_min"], r["self_mean"]], [yy, yy], color="#333", lw=1.2)
        axR.text(min(r["self_mean"], 1.0) - 0.01, yy, f"{r['self_mean']:.3f}", va="center", ha="right",
                 fontsize=9, color="white", fontweight="bold")
    axR.set_yticks(y); axR.set_yticklabels([])
    axR.axvline(1.0, color="#444", lw=1, ls="--")
    axR.set_xlim(0.9, 1.005)
    axR.set_xlabel("recovered self-overlap Tanimoto (optimum = 1.0)")
    axR.set_title("Quality anchor — both tools solve the alignment")

    fss = res.get("fss", {}); rb2 = res.get("roshambo2", {})
    fig.suptitle("fast_shepherd_score vs ROSHAMBO2 — identical molecules, identical GPU",
                 fontsize=13.5, y=1.0, fontweight="bold")
    note = (f"{gpu} · {fss.get('n_queries','?')} queries × {fss.get('n_library','?')} library molecules · "
            f"fss: {fss.get('fss_seeds','?')} SE(3) seeds / {fss.get('steps','?')} steps · "
            f"ROSHAMBO2: start_mode={rb2.get('start_mode','?')} / {rb2.get('steps','?')} steps")
    fig.text(0.5, -0.03, note, ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig5_roshambo_headtohead"))
    plt.close(fig)


if __name__ == "__main__":
    main()
