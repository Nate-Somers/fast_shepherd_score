"""Big CPU speed comparison: one panel per core count, threads vs pool (mirrors plot_all_hardware.py)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODES = ["vol", "surf", "esp", "pharm"]
COLOR = {"vol": "#7b3294", "surf": "#1f6fb2", "esp": "#1a9850", "pharm": "#d9700a"}

th = json.load(open(sys.argv[1]))   # eng_threads plot_data
po = json.load(open(sys.argv[2]))   # eng_pool plot_data

def cores_present(d):
    cs = set()
    for k in d:
        if k.startswith("_"): continue
        e, m, b, pp = k.split("|")
        if e == "numba": cs.add(int(pp[1:]))
    return sorted(cs)

def pts(d, mode, p):
    return d.get("numba|%s|cross|p%d" % (mode, p), {}).get("pts", [])

cores = cores_present(th)
ncols = 4
nrows = (len(cores) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5.3 * nrows), sharex=True, sharey=True)
axes = axes.ravel()
for ax, p in zip(axes, cores):
    for mode in MODES:
        t = pts(th, mode, p); q = pts(po, mode, p)
        if t:
            ax.plot([x for x, _ in t], [y for _, y in t], color=COLOR[mode],
                    ls="-", marker="o", ms=6, lw=2.3, clip_on=False)
        if q:
            ax.plot([x for x, _ in q], [y for _, y in q], color=COLOR[mode],
                    ls=(0, (5, 2)), marker="D", ms=5, lw=2.3, clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color="#cccccc", alpha=0.8)
    ax.grid(True, which="minor", ls=":", color="#e8e8e8", alpha=0.6)
    ax.set_title("%d core%s" % (p, "" if p == 1 else "s"), fontsize=12.5, fontweight="bold")
    allpool = [y for mode in MODES for _, y in pts(po, mode, p)]
    if allpool:
        ax.text(0.04, 0.97, "peak %.1fk pairs/s" % (max(allpool) / 1000),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#444444")
for ax in axes[len(cores):]:
    ax.set_visible(False)
for ax in axes[max(0, len(cores) - ncols):len(cores)]:
    ax.set_xlabel("batch size — pairs aligned per call (log)")
for r in range(nrows):
    axes[r * ncols].set_ylabel("pair-alignments / s (log)")
handles = [Line2D([0], [0], color=COLOR[m], lw=2.6, label=m) for m in MODES]
handles += [Line2D([0], [0], color="#555555", lw=2.3, ls="-", marker="o", ms=6, label="threads"),
            Line2D([0], [0], color="#555555", lw=2.3, ls=(0, (5, 2)), marker="D", ms=5, label="pool")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=6,
           fontsize=11, framealpha=0.95, columnspacing=1.8, handlelength=2.6)
hw = th["_meta"]["hardware"]
fig.suptitle("CPU molecular-alignment throughput across core counts — this fork · numba (threads vs pool)\n"
             "%s · real drug self-copy pairs, isolated best-of-N" % hw.get("cpu", ""),
             fontweight="bold", fontsize=15, y=0.99, linespacing=1.4)
fig.text(0.5, 0.005, "AMD EPYC 9474F, MIT Engaging (exclusive node)   ·   benchmark_cpu.py   ·   %s"
         % th["_meta"]["timestamp"][:10], ha="center", va="bottom", fontsize=9, color="#666666")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[1]))),
                   "engaging", "speed_all_cores_cpu.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
