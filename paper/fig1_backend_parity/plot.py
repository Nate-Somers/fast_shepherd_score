"""
Figure 1 — backend parity scatter (Triton/GPU vs JAX reference).

Reads parity.json (from run.py) and renders a 4-panel scatter (one per mode):
the fork's fast Triton/GPU aligned similarity vs the reference JAX/XLA aligned
similarity, for all 105 distinct drug pairs. Points on y = x ⇒ the GPU kernels
preserve the reference math. Annotates mean|Δ|, max|Δ|, and Spearman ρ.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paper.common import MODE_COLOR, set_style, save_fig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["vol", "surf", "esp", "pharm"]
MODE_LABEL = {"vol": "shape — atoms (vol)", "surf": "shape — surface (surf)",
              "esp": "shape + ESP (esp)", "pharm": "pharmacophore (pharm)"}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def main():
    plt = set_style()
    with open(os.path.join(HERE, "parity.json")) as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, mode in zip(axes, MODES):
        d = data.get(mode, {})
        if "jax" not in d:
            ax.set_visible(False); continue
        x = np.array(d["jax"]); y = np.array(d["triton"])
        c = MODE_COLOR[mode]
        lo = min(x.min(), y.min()) - 0.03
        hi = max(x.max(), y.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1.2, ls="--", zorder=1)
        ax.scatter(x, y, s=22, color=c, alpha=0.75, edgecolor="white",
                   linewidth=0.4, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("JAX reference — aligned Tanimoto")
        mad = float(np.abs(x - y).mean())
        mx = float(np.abs(x - y).max())
        rho = spearman(x, y)
        ax.text(0.04, 0.96,
                f"n = {len(x)}\nmean|Δ| = {mad:.1e}\nmax|Δ| = {mx:.1e}\nSpearman ρ = {rho:.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", alpha=0.9))
    axes[0].set_ylabel("Triton / GPU — aligned Tanimoto")

    fig.suptitle("Backend parity — the fast Triton/GPU kernels reproduce the JAX reference\n"
                 "all 105 distinct drug pairs, identical alignment config, "
                 "same MoleculePairBatch.align_with_* API",
                 fontsize=12.5, y=1.06)
    gpu = data.get("_meta", {}).get("gpu", "")
    fig.text(0.5, -0.06,
             f"residual = multi-start optimization noise, not kernel error  ·  {gpu}  ·  "
             "scoring-level bit-exactness gated separately by benchmarks/experiments/parity_scores.py",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig1_backend_parity"))
    plt.close(fig)


if __name__ == "__main__":
    main()
