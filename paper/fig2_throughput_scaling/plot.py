"""
Figure 2 — fast_shepherd_score throughput across GPUs + batch-size scaling.

Pure plotting: reads the repo's existing, real benchmark results
(``benchmarks/results/<tag>/plot_data.json``, produced by ``benchmarks/benchmark.py``
on each GPU) and renders production figures. No re-computation, no fabricated data.

Outputs (this folder):
  fig2_throughput_scaling.{png,pdf}   main: scaling curve + peak-throughput bars
  fig2_supp_scaling_by_mode.{png,pdf} supplementary: throughput vs batch, per mode

Claim defended: the package delivers high alignment throughput on commodity *and*
datacenter GPUs, scales with batch size (amortizing host overhead), and scales
across GPU generations.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import MODE_COLOR, set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "benchmarks", "results")

# (folder, short label) ordered weakest -> strongest, matching the repo's set.
HARDWARE = [
    ("rtx4050",           "RTX 4050\nlaptop"),
    ("l40s_1gpu",         "L40S\n1 GPU"),
    ("l40s_4gpu",         "L40S\n4 GPU"),
    ("blackwell_rtx6000", "RTX PRO 6000\nBlackwell"),
    ("h100",              "H100 NVL"),
    ("h200",              "H200"),
]
HW_COLOR = {
    "rtx4050": "#9e9e9e", "l40s_1gpu": "#5aa9e6", "l40s_4gpu": "#1f6fb2",
    "blackwell_rtx6000": "#7b3294", "h100": "#1a9850", "h200": "#0b6e3b",
}
MODES = ["vol", "surf", "esp", "pharm"]
MODE_LABEL = {"vol": "shape (atoms)", "surf": "shape (surface)",
              "esp": "shape+ESP", "pharm": "pharmacophore"}


def load(tag):
    with open(os.path.join(RESULTS, tag, "plot_data.json"), encoding="utf-8") as fh:
        return json.load(fh)


def curve(data, mode, bucket="same"):
    pts = data.get(f"{mode}|{bucket}", {}).get("fork", [])
    return [p[0] for p in pts], [p[1] for p in pts]


def peak(data, mode):
    best = 0.0
    for bucket in ("same", "cross"):
        for _, v in data.get(f"{mode}|{bucket}", {}).get("fork", []):
            best = max(best, v)
    return best


def main():
    plt = set_style()
    runs = [(tag, label, load(tag)) for tag, label in HARDWARE]

    # ------------------------------------------------------------------ main
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Left: throughput vs batch size for the shape (surface) mode, all GPUs.
    for tag, label, data in runs:
        xs, ys = curve(data, "surf", "same")
        if not xs:
            continue
        axL.plot(xs, ys, marker="o", ms=5, lw=2.2, color=HW_COLOR[tag],
                 label=label.replace("\n", " "), clip_on=False)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("batch size — pairs aligned per call")
    axL.set_ylabel("pair-alignments / s")
    axL.set_title("Throughput scales with batch size\n(surface-shape mode, self-copy pairs)")
    axL.legend(title="GPU", fontsize=8.5, ncol=2, loc="upper left")
    axL.grid(True, which="minor", ls=":", color="#ececec")

    # Right: peak throughput per mode, grouped by GPU.
    x = np.arange(len(runs))
    w = 0.2
    for j, mode in enumerate(MODES):
        ys = [peak(d, mode) for _, _, d in runs]
        axR.bar(x + (j - 1.5) * w, ys, w, color=MODE_COLOR[mode],
                label=MODE_LABEL[mode])
    axR.set_yscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels([lbl for _, lbl, _ in runs], fontsize=8.5)
    axR.set_ylabel("peak pair-alignments / s")
    axR.set_title("Peak throughput across hardware\n(by alignment mode)")
    axR.legend(fontsize=8.5, ncol=2, loc="upper right")
    axR.grid(False, axis="x")

    fig.suptitle("fast_shepherd_score — GPU alignment throughput "
                 "(Triton kernels, real drug self-copy pairs, isolated best-of-N)",
                 fontsize=12.5, y=1.02)
    stamps = ", ".join(sorted({r[2]["_meta"]["timestamp"][:10] for r in runs}))
    fig.text(0.5, -0.04, f"source: benchmarks/results/*/plot_data.json  ·  {stamps}",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig2_throughput_scaling"))
    plt.close(fig)

    # ----------------------------------------------------------- supplementary
    fig2, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, mode in zip(axes.ravel(), MODES):
        for tag, label, data in runs:
            xs, ys = curve(data, mode, "same")
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=4, lw=2, color=HW_COLOR[tag],
                    label=label.replace("\n", " "), clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{MODE_LABEL[mode]}  ({mode})")
        ax.grid(True, which="minor", ls=":", color="#ececec")
    for ax in axes[-1]:
        ax.set_xlabel("batch size — pairs / call")
    for ax in axes[:, 0]:
        ax.set_ylabel("pair-alignments / s")
    axes[0, 0].legend(title="GPU", fontsize=8, ncol=2, loc="lower right")
    fig2.suptitle("fast_shepherd_score throughput vs batch size, per mode and GPU "
                  "(same-size bucket)", fontsize=12.5, y=0.995)
    fig2.tight_layout()
    save_fig(fig2, os.path.join(HERE, "fig2_supp_scaling_by_mode"))
    plt.close(fig2)

    # ------------------------------------------------------------ print summary
    print("\npeak pairs/s (max over buckets/sizes):")
    print(f"{'GPU':24s} " + " ".join(f"{m:>9s}" for m in MODES))
    for tag, label, data in runs:
        print(f"{label.replace(chr(10),' '):24s} " +
              " ".join(f"{peak(data,m):9.0f}" for m in MODES))


if __name__ == "__main__":
    main()
