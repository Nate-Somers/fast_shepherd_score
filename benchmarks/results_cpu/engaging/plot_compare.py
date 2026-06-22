"""Overlay numba CPU throughput-vs-cores for multiple (machine, mode-of-parallelism) runs."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["vol", "surf", "esp", "pharm"]

def cpu_short(cpu):
    c = cpu or "?"
    if "185H" in c: return "Intel 185H (laptop)"
    if "9474F" in c: return "EPYC 9474F (cluster)"
    return c

def load(path):
    d = json.load(open(path))
    cpu = cpu_short(d["_meta"]["hardware"]["cpu"])
    nm = d["_meta"].get("numba_mode", "threads")
    return f"{cpu} · {nm}", d

def peak_by_cores(d, mode):
    out = {}
    for k, v in d.items():
        if k.startswith("_"): continue
        p = k.split("|")
        if len(p) != 4: continue
        eng, m, bucket, pp = p
        if eng != "numba" or m != mode: continue
        pts = v.get("pts", [])
        if pts: out[int(pp[1:])] = max(y for _, y in pts)
    return dict(sorted(out.items()))

paths = sys.argv[1:]
datasets = [load(p) for p in paths]
# style per series index: laptop-threads, cluster-threads, cluster-pool
COL = ["#1f77b4", "#7f7f7f", "#d62728"]
STY = ["--", "-", "-"]
MK  = ["s", "o", "^"]
LW  = [2.2, 2.2, 2.8]

fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
for ax, mode in zip(axes.flat, MODES):
    for i, (label, d) in enumerate(datasets):
        s = peak_by_cores(d, mode)
        if not s: continue
        xs = list(s); ys = [s[x] for x in xs]
        ax.plot(xs, ys, color=COL[i % 3], linestyle=STY[i % 3], marker=MK[i % 3],
                markersize=7, linewidth=LW[i % 3], label=label)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title(mode, fontweight="bold")
    ax.set_xlabel("CPU cores"); ax.set_ylabel("peak pairs/s (best over batch sizes)")
    ax.grid(True, which="major", alpha=0.7); ax.grid(True, which="minor", ls=":", alpha=0.4)
axes.flat[0].legend(loc="best", fontsize=8.5)
fig.suptitle("numba CPU alignment throughput vs cores — laptop vs Engaging cluster (threads vs pool)",
             fontweight="bold", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_png = sys.argv[0].rsplit("/", 1)[0] + "/compare_laptop_vs_cluster.png"
fig.savefig(out_png, dpi=200, facecolor="white")
print("wrote", out_png)

all_cores = sorted({c for _, d in datasets for mode in MODES for c in peak_by_cores(d, mode)})
lines = ["# numba CPU throughput — laptop vs cluster, threads vs pool (peak pairs/s over batch sizes)\n"]
for mode in MODES:
    lines.append("\n## %s\n" % mode)
    lines.append("| series | " + " | ".join("%dc" % c for c in all_cores) + " |")
    lines.append("|---|" + "".join("--:|" for _ in all_cores))
    for label, d in datasets:
        s = peak_by_cores(d, mode)
        cells = " | ".join(("%.0f" % s[c]) if c in s else "—" for c in all_cores)
        lines.append("| %s | %s |" % (label, cells))
txt = "\n".join(lines) + "\n"
out_md = sys.argv[0].rsplit("/", 1)[0] + "/compare_laptop_vs_cluster.md"
open(out_md, "w").write(txt)
print(txt); print("wrote", out_md)
