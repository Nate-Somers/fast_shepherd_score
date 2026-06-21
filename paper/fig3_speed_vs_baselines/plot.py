"""
Figure 3 — speed + capability vs open-source CPU tools.

Left  : per-pair throughput (pairs/s, log) for each tool, measured on this machine
        (RTX 4050 + Intel CPU), plus the package's datacenter-GPU ceiling for the
        same ESP mode from Fig 2 (real L40S / H100 numbers).
Right : capability matrix — what each tool actually computes. The point of the
        comparison: the faster tools (USRCAT, O3A) do NOT compute an electrostatic
        overlay; the only other aligned shape+ESP tool (ESP-Sim) is CPU-bound and
        aligns by atom correspondence (O3A); fast_shepherd_score is the only tool
        that is correspondence-free shape+ESP+pharmacophore AND GPU-accelerated.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import TOOL_COLOR, set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "benchmarks", "results")


def fig2_esp_peak(tag):
    """Peak esp throughput for a GPU from the repo's saved benchmark (Fig 2 data)."""
    with open(os.path.join(BENCH, tag, "plot_data.json")) as fh:
        d = json.load(fh)
    best = 0.0
    for b in ("same", "cross"):
        for _, v in d.get(f"esp|{b}", {}).get("fork", []):
            best = max(best, v)
    return best


def main():
    plt = set_style()
    with open(os.path.join(HERE, "results.json")) as fh:
        res = json.load(fh)
    tp = res["throughput"]

    # Bars: (label, pairs/s, color, note). Measured here unless noted.
    bars = [
        ("USRCAT",            tp["USRCAT"],            "#d9700a", "descriptor; no pose/ESP"),
        ("RDKit O3A",         tp["RDKit O3A"],         "#1f6fb2", "atom-map align; no ESP"),
        ("ESP-Sim",           tp["ESP-Sim"],           "#1a9850", "shape+ESP, CPU (O3A align)"),
        ("fss (CPU)",         tp["fss (CPU)"],         "#e08e85", "shape+ESP opt, CPU"),
        ("fss (GPU)\nRTX 4050", tp.get("fss (GPU) batch=4096", np.nan), "#c0392b",
         "shape+ESP opt, batched"),
    ]
    # Datacenter ceiling for the SAME esp mode, from Fig 2 (real saved runs).
    l40s = fig2_esp_peak("l40s_1gpu")
    h100 = fig2_esp_peak("h100")
    bars.append(("fss (GPU)\nL40S", l40s, "#7a241a", "datacenter GPU [Fig 2]"))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- Left: throughput bars ----
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    y = np.arange(len(bars))[::-1]
    axL.barh(y, vals, color=colors, height=0.66)
    for yi, (lbl, v, c, note) in zip(y, bars):
        axL.text(v * 1.15, yi, f"{v:,.0f}/s", va="center", ha="left", fontsize=9,
                 fontweight="bold", color="#333")
        axL.text(v * 1.15, yi - 0.30, note, va="center", ha="left", fontsize=7.3,
                 color="#888", style="italic")
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=9)
    axL.set_xscale("log")
    axL.set_xlim(1, 3e6)
    axL.set_xlabel("pairs / second  (log; molecules pre-built, per-pair op timed)")
    axL.set_title("Throughput — same hardware (RTX 4050 + Intel CPU)\n"
                  "shape+ESP task; faster tools (top) skip the ESP overlay", fontsize=11)
    axL.grid(True, axis="x", which="both", ls=":", color="#e5e5e5")
    axL.grid(False, axis="y")

    # ---- Right: capability matrix ----
    tools = ["USRCAT", "RDKit O3A", "ESP-Sim", "ROSHAMBO2", "fss (this work)"]
    feats = ["3D pose", "Gaussian\nshape", "ESP\noverlay", "pharmaco-\nphore",
             "corr.-free\nalign", "GPU", "open\nsource"]
    # 1 = yes, 0 = no, 0.5 = partial / indirect.
    M = {
        "USRCAT":        [0,   0.5, 0,   0.5, np.nan, 0, 1],   # no align needed
        "RDKit O3A":     [1,   0,   0,   0,   0,      0, 1],
        "ESP-Sim":       [0.5, 1,   1,   0,   0,      0, 1],   # pose via external O3A
        "ROSHAMBO2":     [1,   1,   0,   1,   1,      1, 1],   # GPL; not benchmarked here
        "fss (this work)": [1, 1,   1,   1,   1,      1, 1],
    }
    nT, nF = len(tools), len(feats)
    for r, t in enumerate(tools):
        for c, val in enumerate(M[t]):
            yy = nT - 1 - r
            if np.isnan(val):
                axR.text(c, yy, "–", ha="center", va="center", fontsize=12, color="#bbb")
                continue
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
    axR.text(0.0, -1.15, "ROSHAMBO2: from the literature (GPL; CUDA build not run here). "
             "ESP-Sim pose/align via RDKit O3A.", transform=axR.transData if False else axR.transAxes,
             ha="left", va="top", fontsize=7.2, color="#888")

    fig.suptitle("fast_shepherd_score vs open-source 3D-similarity tools — speed and capability",
                 fontsize=12.5, y=1.02)
    fig.text(0.5, -0.05,
             f"ESP mode; per-pair op timed with molecules/conformers/descriptors pre-built  ·  "
             f"fss GPU ceiling (esp, Fig 2): RTX 4050 {fig2_esp_peak('rtx4050'):,.0f} (same-size) · "
             f"L40S {l40s:,.0f} · H100 {h100:,.0f} pairs/s",
             ha="center", fontsize=7.8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig3_speed_vs_baselines"))
    plt.close(fig)


if __name__ == "__main__":
    main()
