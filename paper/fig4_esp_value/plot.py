"""
Figure 4 — the electrostatic term carries information orthogonal to shape.

Left : ESP similarity of each benzene-analog (shape-aligned to benzene) vs the ESP
       weight. At the default lam=0.3 ESP ≈ shape (all analogs bunched); as the ESP
       weight increases (lam ↓) the lines fan out — polar analogs (nitrobenzene,
       benzaldehyde) are pushed down, nonpolar ones (toluene) stay put.
Right: shape vs ESP similarity at a discriminating weight (lam=0.003). Shape ranks
       every analog ~0.6-0.7 (all benzene-shaped); ESP pushes the polar analogs
       below the diagonal — electrostatic information shape cannot see.

Honest point: ESP's effect on the Tanimoto score is weight-dependent and modest at
the default lam; its discriminative value is real but must be surfaced by lam (and
physical xTB charges). The decisive utility test is retrieval enrichment (fig5).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLS_COLOR = {"nonpolar": "#1a9850", "weak": "#1f6fb2", "polar": "#e08e2a", "strong": "#c0392b"}
CLS_ORDER = ["nonpolar", "weak", "polar", "strong"]


def main():
    plt = set_style()
    with open(os.path.join(HERE, "analog_esp.json")) as fh:
        data = json.load(fh)
    rows = data["rows"]
    lams = data["lams"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- A: ESP vs weight (lam), one line per analog ----
    x = np.array(lams)
    for r in rows:
        ys = [r["esp"][str(l)] if str(l) in r["esp"] else r["esp"][l] for l in lams]
        # JSON keys may be floats stringified; handle both
        ys = [r["esp"].get(l, r["esp"].get(str(l))) for l in lams]
        axA.plot(x, ys, marker="o", ms=4, lw=1.8, color=CLS_COLOR[r["cls"]], alpha=0.9)
        axA.annotate(r["name"], (x[-1], ys[-1]), fontsize=7, color=CLS_COLOR[r["cls"]],
                     xytext=(3, 0), textcoords="offset points", va="center")
    axA.set_xscale("log")
    axA.invert_xaxis()                                   # stronger ESP (smaller lam) -> right
    axA.axvline(0.3, color="#888", ls=":", lw=1.2)
    axA.text(0.3, axA.get_ylim()[1], " default\n lam=0.3", fontsize=7.5, color="#888",
             va="top", ha="left")
    axA.set_xlabel("ESP weight:  lam (log, decreasing →  stronger electrostatics)")
    axA.set_ylabel("ESP similarity to benzene (at shape pose)")
    axA.set_title("ESP discrimination emerges as the ESP weight increases")
    handles = [plt.Line2D([0], [0], color=CLS_COLOR[c], lw=2.4, label=c) for c in CLS_ORDER]
    axA.legend(handles=handles, title="polarity", fontsize=8.5, loc="lower left")

    # ---- B: shape vs ESP at a discriminating lam ----
    lam_b = 0.003
    sh = np.array([r["shape"] for r in rows])
    ep = np.array([r["esp"].get(lam_b, r["esp"].get(str(lam_b))) for r in rows])
    lo, hi = min(sh.min(), ep.min()) - 0.03, max(sh.max(), ep.max()) + 0.03
    axB.plot([lo, hi], [lo, hi], color="#999", ls="--", lw=1.2, zorder=1)
    for r, s, e in zip(rows, sh, ep):
        axB.scatter(s, e, s=70, color=CLS_COLOR[r["cls"]], alpha=0.85, edgecolor="white",
                    linewidth=0.6, zorder=2)
        axB.annotate(r["name"], (s, e), fontsize=7, xytext=(5, -1),
                     textcoords="offset points", color="#444")
    axB.set_xlim(lo, hi); axB.set_ylim(lo, hi); axB.set_aspect("equal")
    axB.set_xlabel("shape similarity to benzene (surf Tanimoto)")
    axB.set_ylabel(f"shape+ESP similarity (esp Tanimoto, lam={lam_b})")
    axB.set_title("Shape ranks all analogs alike; ESP separates by polarity")
    axB.text(0.04, 0.96, "polar analogs fall\nbelow the diagonal\n(ESP penalty)",
             transform=axB.transAxes, ha="left", va="top", fontsize=8.5, color="#666",
             style="italic")

    fig.suptitle("The electrostatic term carries information orthogonal to shape "
                 "(benzene-analog series, xTB charges)", fontsize=12.5, y=1.01)
    fig.text(0.5, -0.04,
             "shape-matched analogs: shape similarity is ~uniform; ESP (appropriately "
             "weighted) separates nonpolar from polar.  Effect is modest at the default "
             "lam=0.3 — see README.",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig4_esp_value"))
    plt.close(fig)


if __name__ == "__main__":
    main()
