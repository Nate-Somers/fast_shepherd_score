"""
Figure 2 — fast_shepherd_score throughput vs batch size, across datacenter GPUs.

Reads paper/fig2_throughput_scaling/data/<gpu>.json (from measure.py, run once per GPU in the
SAME conda env, with per-rep mean ± SD).  Renders:
  Left  throughput vs batch size (surface mode), mean ± SD band per GPU — showing throughput
        rises with batch then SATURATES (the workload is launch/host-bound at large batch).
  Right peak throughput per mode per GPU (mean ± SD bars).

Honest framing: throughput does NOT order cleanly by GPU "generation" (surf separates cards;
vol saturates ~equal across them) — so we claim batch-scaling + high absolute throughput,
not generational scaling.  Self-copy pairs (optimum=1.0): this is kernel+launch throughput.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import MODE_COLOR, set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["vol", "surf", "esp", "pharm"]
MODE_LABEL = {"vol": "shape (atoms)", "surf": "shape (surface)", "esp": "shape+ESP", "pharm": "pharmacophore"}
GPU_COLORS = ["#1f6fb2", "#1a9850", "#7b3294", "#d9700a", "#9e9e9e", "#0b6e3b"]


def short(gpu):
    return (gpu.replace("NVIDIA ", "").replace("Laptop GPU", "laptop")
            .replace(" NVL", "").strip())


def load_all():
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, "data", "*.json"))):
        with open(p) as fh:
            runs.append(json.load(fh))
    return runs


def series(run, mode):
    d = run["data"].get(mode, [])
    return ([x["n"] for x in d], np.array([x["mean"] for x in d]), np.array([x["std"] for x in d]))


def peak(run, mode):
    d = run["data"].get(mode, [])
    if not d:
        return 0.0, 0.0
    i = int(np.argmax([x["mean"] for x in d]))
    return d[i]["mean"], d[i]["std"]


def main():
    plt = set_style()
    runs = load_all()
    if not runs:
        raise SystemExit("no data/<gpu>.json — run measure.py on each GPU first")
    colors = {run["gpu"]: GPU_COLORS[i % len(GPU_COLORS)] for i, run in enumerate(runs)}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.5))

    # Left: surf throughput vs batch, mean +/- SD
    for run in runs:
        xs, ys, es = series(run, "surf")
        if not xs:
            continue
        c = colors[run["gpu"]]
        axL.plot(xs, ys, marker="o", ms=5, lw=2.2, color=c, label=short(run["gpu"]), clip_on=False)
        axL.fill_between(xs, ys - es, ys + es, color=c, alpha=0.15, lw=0)
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("batch size — pairs aligned per call")
    axL.set_ylabel("pair-alignments / s  (mean ± SD)")
    axL.set_title("Throughput rises with batch size, then saturates\n(surface-shape mode, self-copy pairs)")
    axL.legend(title="GPU", fontsize=8.5, ncol=2, loc="upper left")
    axL.grid(True, which="minor", ls=":", color="#ececec")

    # Right: peak per mode per GPU, with SD error bars
    x = np.arange(len(runs)); w = 0.2
    for j, mode in enumerate(MODES):
        means = [peak(r, mode)[0] for r in runs]
        stds = [peak(r, mode)[1] for r in runs]
        axR.bar(x + (j - 1.5) * w, means, w, yerr=stds, color=MODE_COLOR[mode],
                label=MODE_LABEL[mode], error_kw=dict(lw=0.7, ecolor="#444"))
    axR.set_yscale("log"); axR.set_xticks(x)
    axR.set_xticklabels([short(r["gpu"]) for r in runs], fontsize=8.5, rotation=12, ha="right")
    axR.set_ylabel("peak pair-alignments / s")
    axR.set_title("Peak throughput per mode\n(no clean generational ordering — see surf vs vol)")
    axR.legend(fontsize=8.5, ncol=2, loc="upper right")
    axR.grid(False, axis="x")

    envs = sorted({f"torch {r['torch']}/cu{r['cuda']}" for r in runs})
    fig.suptitle("fast_shepherd_score — GPU alignment throughput (batch scaling, controlled environment)",
                 fontsize=12.5, y=1.02, fontweight="bold")
    fig.text(0.5, -0.04,
             f"same conda env on every GPU ({', '.join(envs)}) · self-copy pairs (optimum=1.0) · "
             "throughput saturates at large batch (launch/host-bound), so absolute throughput + batch-scaling "
             "are claimed, not generational scaling",
             ha="center", fontsize=7.8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig2_throughput_scaling"))
    plt.close(fig)

    print("\npeak pairs/s (mean):")
    for r in runs:
        print(f"{short(r['gpu']):28s} " + " ".join(f"{m}={peak(r,m)[0]:.0f}" for m in MODES))


if __name__ == "__main__":
    main()
