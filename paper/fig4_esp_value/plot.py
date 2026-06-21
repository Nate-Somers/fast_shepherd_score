"""
Figure 4 — ESP carries information orthogonal to shape (with uncertainty + a polarity axis).

  A  ESP similarity to benzene vs ESP weight λ, one line per shape-matched analog
     (mean ± SD over replicates), coloured by polarity.  Default λ=0.3 marked — there the
     lines are bunched (ESP ~inert); they fan out only at small λ.
  B  Discrimination = SD across analogs, for ESP(λ) (mean ± SD band) vs the shape-only
     baseline.  ESP exceeds shape only at small λ — the honest, quantified version of the
     "ESP adds information" claim.
  C  ESP signal (shape − ESP at the discriminating λ) vs molecular dipole magnitude
     (from xTB charges).  The separation tracks electrostatics, not residual shape.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CLS_COLOR = {"nonpolar": "#1a9850", "weak": "#66bd63", "polar": "#f46d43", "strong": "#a50026"}
DISC_LAM = 0.003   # the discriminating weight used for panel C


def main():
    plt = set_style()
    with open(os.path.join(HERE, "analog_esp.json")) as fh:
        d = json.load(fh)
    lams = d["lams"]; rows = d["rows"]; nrep = d.get("n_rep", 1)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---- A: esp vs lam per analog, mean +/- SD ----
    for r in rows:
        ys = np.array([r["esp_mean"][str(l)] for l in lams])
        es = np.array([r["esp_std"][str(l)] for l in lams])
        col = CLS_COLOR[r["cls"]]
        axA.plot(lams, ys, "-o", color=col, ms=3.5, lw=1.4, alpha=0.85)
        axA.fill_between(lams, ys - es, ys + es, color=col, alpha=0.13, lw=0)
        axA.text(lams[-1] * 0.92, ys[-1], r["name"], fontsize=7.2, va="center", ha="right", color=col)
    axA.set_xscale("log"); axA.invert_xaxis()
    axA.axvline(0.3, color="#444", ls="--", lw=1.1)
    axA.text(0.3, axA.get_ylim()[0], " default λ=0.3", fontsize=8, va="bottom", ha="left", color="#444")
    axA.set_xlabel("ESP weight λ  (→ stronger ESP)")
    axA.set_ylabel("ESP similarity to benzene")
    axA.set_title("A · ESP(λ) per shape-matched analog")

    # ---- B: discrimination vs lam ----
    dm = np.array([d["discrimination"][str(l)]["mean"] for l in lams])
    ds = np.array([d["discrimination"][str(l)]["std"] for l in lams])
    axB.plot(lams, dm, "-o", color="#1a9850", lw=1.8, label="ESP discrimination")
    axB.fill_between(lams, dm - ds, dm + ds, color="#1a9850", alpha=0.15, lw=0)
    sh = d["shape_discrimination"]
    axB.axhline(sh["mean"], color="#7b3294", ls="--", lw=1.4, label="shape-only baseline")
    axB.fill_between(lams, sh["mean"] - sh["std"], sh["mean"] + sh["std"], color="#7b3294", alpha=0.10, lw=0)
    axB.set_xscale("log"); axB.invert_xaxis()
    axB.axvline(0.3, color="#444", ls="--", lw=1.0)
    axB.set_xlabel("ESP weight λ  (→ stronger ESP)")
    axB.set_ylabel("discrimination = SD across analogs")
    axB.set_title("B · ESP out-discriminates shape only at small λ")
    axB.legend(fontsize=8.5, loc="upper left")

    # ---- C: ESP signal vs dipole ----
    x = np.array([r["dipole_mean"] for r in rows])
    xe = np.array([r["dipole_std"] for r in rows])
    drop = np.array([r["shape_mean"] - r["esp_mean"][str(DISC_LAM)] for r in rows])
    drop_e = []
    for r in rows:
        per = np.array(r["shape_all"]) - np.array(r["esp_all"][str(DISC_LAM)])
        drop_e.append(float(np.std(per)))
    drop_e = np.array(drop_e)
    for r, xi, xei, yi, yei in zip(rows, x, xe, drop, drop_e):
        col = CLS_COLOR[r["cls"]]
        axC.errorbar(xi, yi, xerr=xei, yerr=yei, fmt="o", color=col, ms=7, capsize=2)
        axC.annotate(r["name"], (xi, yi), fontsize=7, xytext=(4, 3), textcoords="offset points")
    if np.std(x) > 0:
        rho = float(np.corrcoef(x, drop)[0, 1])
        axC.text(0.04, 0.96, f"Pearson r = {rho:.2f}", transform=axC.transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
    axC.set_xlabel("molecular dipole magnitude (Debye, xTB)")
    axC.set_ylabel(f"ESP signal = shape − ESP@λ={DISC_LAM:g}")
    axC.set_title("C · the separation tracks electrostatics")

    fig.suptitle("ESP carries electrostatic information orthogonal to shape — weight-dependent, quantified",
                 fontsize=13, y=1.02, fontweight="bold")
    fig.text(0.5, -0.04,
             f"benzene-analog series · xTB charges · {nrep} replicates (conformer seed + surface resampling) · "
             f"shape-first alignment, ESP scored at the fixed shape pose · {d.get('gpu','')}",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig4_esp_value"))
    plt.close(fig)


if __name__ == "__main__":
    main()
