"""
Figure 3 — speed + capability vs open-source 3D-similarity tools (same machine).

Left  : per-pair throughput (pairs/s, log, mean ± SD over reps) for every tool, ALL measured
        on the SAME node (datacenter GPU + that node's CPU).  The faster tools (USRCAT, O3A)
        do NOT compute an electrostatic overlay; the only other aligned shape+ESP tool
        (ESP-Sim) is CPU-bound; fss is the only correspondence-free shape+ESP+pharm tool that
        is GPU-batched — and even its slowest mode (esp) beats ESP-Sim on the same machine.
Right : capability matrix — what each tool actually computes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    plt = set_style()
    with open(os.path.join(HERE, "results.json")) as fh:
        res = json.load(fh)
    tp = res["throughput"]
    meta = res.get("_meta", {})

    def mean(k): return tp[k]["mean"] if isinstance(tp.get(k), dict) else tp.get(k, 0.0)
    def std(k):  return tp[k]["std"] if isinstance(tp.get(k), dict) else 0.0

    RED, REDL = "#c0392b", "#e08e85"
    bars = [
        ("USRCAT",           "USRCAT",        "#9e9e9e", "descriptor; no pose/ESP"),
        ("fss · vol (GPU)",  "fss (GPU) vol", RED,  "shape (atoms) — fss fast path"),
        ("fss · surf (GPU)", "fss (GPU) surf",RED,  "shape (surface)"),
        ("fss · pharm (GPU)","fss (GPU) pharm",RED, "pharmacophore"),
        ("fss · esp (GPU)",  "fss (GPU) esp", RED,  "shape+ESP — fss's slowest mode"),
        ("RDKit O3A",        "RDKit O3A",     "#5a7d9a", "atom-map align; no ESP"),
        ("ESP-Sim",          "ESP-Sim",       "#1a9850", "shape+ESP, CPU (O3A align)"),
        ("fss · esp (CPU)",  "fss (CPU) esp", REDL, "shape+ESP opt, CPU"),
    ]
    bars = [(lbl, mean(k), std(k), c, note) for (lbl, k, c, note) in bars if k in tp]
    bars.sort(key=lambda b: -b[1])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.8, 5.8), gridspec_kw={"width_ratios": [1.1, 1.0]})

    y = np.arange(len(bars))[::-1]
    axL.barh(y, [b[1] for b in bars], xerr=[b[2] for b in bars], color=[b[3] for b in bars],
             height=0.66, error_kw=dict(ecolor="#333", lw=0.8, capsize=2))
    for yi, (lbl, v, sd, c, note) in zip(y, bars):
        axL.text(v * 1.25, yi, f"{v:,.0f}/s", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#333")
        axL.text(v * 1.25, yi - 0.33, note, va="center", ha="left", fontsize=7.0, color="#888", style="italic")
    axL.set_yticks(y); axL.set_yticklabels([b[0] for b in bars], fontsize=9)
    axL.set_xscale("log"); axL.set_xlim(1, 3e6)
    axL.set_xlabel("pairs / second  (log, mean ± SD; molecules pre-built, batched op timed)")
    axL.set_title("Throughput — same machine, every tool", fontsize=11)
    axL.grid(True, axis="x", which="both", ls=":", color="#e5e5e5"); axL.grid(False, axis="y")

    # same-machine ESP ratio (the corrected headline for the ESP comparison)
    if "fss (GPU) esp" in tp and "ESP-Sim" in tp and mean("ESP-Sim") > 0:
        axL.text(0.5, 0.02, f"same-machine: fss esp (GPU) / ESP-Sim (CPU) = {mean('fss (GPU) esp')/mean('ESP-Sim'):,.0f}×",
                 transform=axL.transAxes, fontsize=8.5, color="#c0392b", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.9))

    # ---- capability matrix ----
    tools = ["USRCAT", "RDKit O3A", "ESP-Sim", "ROSHAMBO2", "fss (this work)"]
    feats = ["3D pose", "Gaussian\nshape", "ESP\noverlay", "pharmaco-\nphore", "corr.-free\nalign", "GPU", "open\nsource"]
    M = {
        "USRCAT":        [0,   0.5, 0,   0.5, np.nan, 0, 1],
        "RDKit O3A":     [1,   0,   0,   0,   0,      0, 1],
        "ESP-Sim":       [0.5, 1,   1,   0,   0,      0, 1],
        "ROSHAMBO2":     [1,   1,   0,   1,   1,      1, 1],
        "fss (this work)": [1, 1,   1,   1,   1,      1, 1],
    }
    nT, nF = len(tools), len(feats)
    for r, t in enumerate(tools):
        for c, val in enumerate(M[t]):
            yy = nT - 1 - r
            if np.isnan(val):
                axR.text(c, yy, "–", ha="center", va="center", fontsize=12, color="#bbb"); continue
            color = {1.0: "#1a9850", 0.5: "#e0a93b", 0.0: "#d64545"}[val]
            sym = {1.0: "●", 0.5: "◐", 0.0: "○"}[val]
            axR.text(c, yy, sym, ha="center", va="center", fontsize=15, color=color)
    axR.set_xticks(range(nF)); axR.set_xticklabels(feats, fontsize=8.0)
    axR.set_yticks(range(nT)); axR.set_yticklabels(tools[::-1], fontsize=9.5)
    axR.set_xlim(-0.5, nF - 0.5); axR.set_ylim(-0.5, nT - 0.5)
    axR.set_title("Capability matrix\n● yes   ◐ partial/indirect   ○ no", fontsize=11)
    axR.grid(False)
    for s in axR.spines.values():
        s.set_visible(False)
    axR.tick_params(length=0)
    axR.text(0.0, -1.15, "ROSHAMBO2: Gaussian shape benchmarked head-to-head in Fig 5 (no ESP overlay). "
             "ESP-Sim pose/align via RDKit O3A.", transform=axR.transAxes, ha="left", va="top",
             fontsize=7.2, color="#888")

    fig.suptitle("fast_shepherd_score vs open-source 3D-similarity tools — speed and capability",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.text(0.5, -0.05,
             f"all tools on ONE machine: {meta.get('gpu','GPU')} + {meta.get('cpu','CPU')}  ·  "
             f"{meta.get('n_lib','?')} unique molecules, {meta.get('n_cpu_pairs','?')} pairs (CPU tools), "
             f"GPU batch {meta.get('gpu_batch','?')} ({meta.get('n_unique_pairs_in_batch','?')} unique)",
             ha="center", fontsize=7.8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig3_speed_vs_baselines"))
    plt.close(fig)


if __name__ == "__main__":
    main()
