"""
Combined cross-hardware speed plot.

Reads the per-hardware ``results/<tag>/plot_data.json`` files produced by
``benchmark.py`` and renders ONE figure with a panel per GPU, so the fork's
Triton/GPU throughput can be compared across hardware at a glance.

Each panel is the fork (Triton/GPU) throughput only — the recent multi-GPU
runs don't re-time the slow JAX/CPU "original" baseline (see WHATS_NEW.md for
the per-mode speedups over upstream measured on the laptop).

Usage:  python benchmarks/plot_all_hardware.py
Output: results/speed_all_hardware.png

(Distinct name from benchmark.py's per-run ``speed_plot.png`` so an untagged
benchmark run can't clobber the combined overview.)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Ordered (folder, display label) of the most-recent run per hardware option.
HARDWARE = [
    ("rtx4050",            "RTX 4050 laptop"),
    ("l40s_1gpu",          "L40S · 1 GPU"),
    ("l40s_4gpu",          "L40S · 4 GPU"),
    ("blackwell_rtx6000",  "RTX PRO 6000 Blackwell"),
    ("h100",               "H100 NVL"),
    ("h200",               "H200"),
]

MODES = ["vol", "surf", "esp", "pharm"]
BUCKETS = ["same", "cross"]

# Same palette/markers as benchmark.py's render_plot, for consistency.
COLOR = {"vol": "#7b3294", "surf": "#1f6fb2", "esp": "#1a9850", "pharm": "#d9700a"}
LS = {"same": "-", "cross": (0, (5, 2))}
MK = {"same": "o", "cross": "D"}


def load(tag):
    with open(os.path.join(RESULTS, tag, "plot_data.json"), encoding="utf-8") as fh:
        return json.load(fh)


def peak(data, mode):
    best = 0.0
    for bucket in BUCKETS:
        for _, v in data.get(f"{mode}|{bucket}", {}).get("fork", []):
            best = max(best, v)
    return best


def main():
    runs = [(tag, label, load(tag)) for tag, label in HARDWARE]

    ncols = 3
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 9.6),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (tag, label, data) in zip(axes, runs):
        for mode in MODES:
            for bucket in BUCKETS:
                pts = data.get(f"{mode}|{bucket}", {}).get("fork", [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=COLOR[mode], linestyle=LS[bucket],
                        marker=MK[bucket], markersize=6, linewidth=2.2,
                        clip_on=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="#cccccc", alpha=0.8)
        ax.grid(True, which="minor", ls=":", color="#e8e8e8", alpha=0.6)

        hw = data["_meta"]["hardware"]
        env = f"torch {hw.get('torch', '?')} / CUDA {hw.get('cuda', '?')}"
        ax.set_title(f"{label}\n{hw.get('gpu', '')}\n{env}",
                     fontsize=10, fontweight="bold", linespacing=1.45)
        ax.text(0.04, 0.97, f"peak vol {peak(data, 'vol') / 1000:.0f}k pairs/s",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                color="#444444")

    # Hide any unused panels.
    for ax in axes[len(runs):]:
        ax.set_visible(False)

    # Shared axis labels on the outer edges.
    for ax in axes[len(runs) - ncols:len(runs)]:
        ax.set_xlabel("batch size — pairs aligned per call (log)")
    for r in range(nrows):
        axes[r * ncols].set_ylabel("pair-alignments / s (log)")

    # One horizontal figure-level legend along the bottom (mode = color,
    # bucket = line/marker) so no entry overlaps a panel.
    handles = [Line2D([0], [0], color=COLOR[m], lw=2.6, label=f"{m}") for m in MODES]
    handles += [
        Line2D([0], [0], color="#555555", lw=2.2, linestyle=LS[b], marker=MK[b],
               markersize=6, label=f"bucket: {b}") for b in BUCKETS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.028),
               ncol=6, framealpha=0.95, fontsize=10, columnspacing=1.6,
               handlelength=2.4)

    fig.suptitle("Molecular-alignment throughput across GPUs — this fork · Triton/GPU\n"
                 "real drug self-copy pairs, isolated best-of-N",
                 fontweight="bold", fontsize=14, y=0.995, linespacing=1.4)
    stamps = ", ".join(sorted({r[2]["_meta"]["timestamp"][:10] for r in runs}))
    fig.text(0.5, 0.005,
             "fork runs the full size sweep; higher = faster   ·   "
             f"benchmark.py real-molecule sweep   ·   {stamps}",
             ha="center", va="bottom", fontsize=8.5, color="#666666")

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = os.path.join(RESULTS, "speed_all_hardware.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
