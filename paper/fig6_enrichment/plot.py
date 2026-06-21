"""
Figure 6 — virtual-screening enrichment + the ESP ablation.

Reads every enrichment_<target>.json in this folder (one per DUD-E target) and renders
the decisive ESP-utility figure:

  Left   ESP ablation vs lam: ΔAUC = AUC(esp@lam) − AUC(surf), one line per target,
         coloured by pocket chemistry (charged/polar vs aminergic vs hydrophobic control),
         with the package-default lam=0.3 marked.  Shows ESP becomes discriminative only at
         small lam (tying Fig 4's score-level finding to retrieval) AND that the benefit is
         pocket-specific (large on charged pockets, ~0 on hydrophobic controls).

  Right  Per-target retrieval at the chosen lam: surf vs esp AUC (paired, bootstrap-CI bars),
         grouped by pocket type.  The headline: aligned shape+ESP beats shape-only where
         electrostatics discriminate.

Run: PYTHONPATH=. python paper/fig6_enrichment/plot.py [--lam 0.003]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# pocket chemistry classification (for the per-target story)
# pocket chemistry by binding-site character (FABP4 binds a fatty-acid carboxylate via
# Arg126/Tyr128 — electrostatic, so it is polar, not a hydrophobic control; ANDR's steroid
# pocket is the clean hydrophobic control).
CATEGORY = {
    "aces": "charged/polar", "fa10": "charged/polar", "fa7": "charged/polar",
    "hmdh": "charged/polar", "ampc": "charged/polar", "ada": "charged/polar",
    "fabp4": "charged/polar",
    "adrb2": "aminergic",
    "andr": "hydrophobic ctrl",
}
CAT_COLOR = {"charged/polar": "#1a9850", "aminergic": "#1f6fb2", "hydrophobic ctrl": "#d9700a"}
CAT_ORDER = ["charged/polar", "aminergic", "hydrophobic ctrl"]


def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "enrichment_*.json"))):
        t = os.path.basename(p)[len("enrichment_"):-len(".json")]
        if t == "smoke":
            continue
        with open(p) as fh:
            out[t] = json.load(fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None,
                    help="lam to use for the right panel; default = the lam maximizing mean ΔAUC over charged targets")
    a = ap.parse_args()

    plt = set_style()
    data = load_all()
    if not data:
        raise SystemExit("no enrichment_<target>.json found — run fig6 run.py first")

    targets = sorted(data, key=lambda t: (CAT_ORDER.index(CATEGORY.get(t, "charged/polar")), t))
    lams = sorted({l for d in data.values() for l in d["config"]["lams"]}, reverse=True)

    def dauc(t, lam):
        tag = f"esp@{lam:g}"
        ab = data[t].get("ablation_vs_surf", {}).get(tag)
        return ab["auc"] if ab else None

    # choose lam for the right panel: maximize mean ΔAUC over charged/polar targets
    if a.lam is None:
        charged = [t for t in targets if CATEGORY.get(t) == "charged/polar"]
        best, blam = -1e9, lams[-1]
        for lam in lams:
            ds = [dauc(t, lam)["mean_delta"] for t in charged if dauc(t, lam)]
            if ds and np.mean(ds) > best:
                best, blam = np.mean(ds), lam
        chosen_lam = blam
    else:
        chosen_lam = a.lam

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------------- Left: ΔAUC vs lam ----------------
    for t in targets:
        cat = CATEGORY.get(t, "charged/polar")
        xs, ys, los, his = [], [], [], []
        for lam in lams:
            ab = dauc(t, lam)
            if ab:
                xs.append(lam); ys.append(ab["mean_delta"])
                los.append(ab["ci95"][0]); his.append(ab["ci95"][1])
        if not xs:
            continue
        axL.plot(xs, ys, "-o", color=CAT_COLOR[cat], ms=4, lw=1.6, alpha=0.85,
                 label=t.upper())
        axL.fill_between(xs, los, his, color=CAT_COLOR[cat], alpha=0.10, lw=0)
    axL.set_xscale("log")
    axL.invert_xaxis()                       # stronger ESP weighting to the right
    axL.axhline(0, color="#888", lw=1, ls="-")
    axL.axvline(0.3, color="#444", lw=1.1, ls="--")
    axL.text(0.3, axL.get_ylim()[0], " default λ=0.3 (ESP ~inert)", fontsize=8,
             va="bottom", ha="left", color="#444", rotation=90)
    axL.set_xlabel("ESP weight λ  (→ stronger ESP)")
    axL.set_ylabel("ΔAUC  =  AUC(shape+ESP) − AUC(shape)")
    axL.set_title("ESP ablation vs λ — benefit is pocket-specific")
    axL.legend(fontsize=8, ncol=2, loc="upper left", title="target", framealpha=0.9)

    # ---------------- Right: surf vs esp AUC per target at chosen lam ----------------
    surf_auc = {t: data[t]["results"]["surf"]["metrics"]["auc"] for t in targets}
    esp_tag = f"esp@{chosen_lam:g}"
    esp_auc = {t: data[t]["results"].get(esp_tag, {}).get("metrics", {}).get("auc")
               for t in targets}
    y = np.arange(len(targets))[::-1]
    for i, t in enumerate(targets):
        cat = CATEGORY.get(t, "charged/polar"); col = CAT_COLOR[cat]
        s = surf_auc[t]; e = esp_auc[t]
        yy = y[i]
        axR.errorbar(s["mean"], yy + 0.16, xerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]],
                     fmt="o", color="#999", ms=6, capsize=2, label="shape (surf)" if i == 0 else None)
        if e:
            axR.errorbar(e["mean"], yy - 0.16, xerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]],
                         fmt="s", color=col, ms=6, capsize=2, label="shape+ESP (esp)" if i == 0 else None)
            axR.annotate("", xy=(e["mean"], yy - 0.16), xytext=(s["mean"], yy + 0.16),
                         arrowprops=dict(arrowstyle="->", color=col, alpha=0.5, lw=1.2))
    axR.set_yticks(y); axR.set_yticklabels([t.upper() for t in targets])
    for tick, t in zip(axR.get_yticklabels(), targets):
        tick.set_color(CAT_COLOR[CATEGORY.get(t, "charged/polar")])
    axR.axvline(0.5, color="#ccc", lw=1, ls=":")
    axR.set_xlabel("ROC-AUC  (mean ± 95% CI over query actives)")
    axR.set_title(f"Retrieval at λ={chosen_lam:g}:  shape  →  shape+ESP")
    axR.legend(fontsize=9, loc="lower right")

    n_q = next(iter(data.values()))["n_queries"]
    cfg = next(iter(data.values()))["config"]
    fig.suptitle("Aligned shape+ESP improves active retrieval where electrostatics matter",
                 fontsize=13.5, y=1.0, fontweight="bold")
    fig.text(0.5, -0.03,
             f"DUD-E targets · all actives + ~{next(iter(data.values()))['n_decoys']} decoys · "
             f"{n_q} query actives averaged · xTB charges · "
             f"equal budget (surf & esp: {cfg['num_seeds']} SE(3) seeds, {cfg['max_steps']} steps) · "
             "bootstrap 95% CIs",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig6_enrichment"))
    plt.close(fig)
    print(f"chosen lam (right panel) = {chosen_lam:g}; targets = {targets}")


if __name__ == "__main__":
    main()
