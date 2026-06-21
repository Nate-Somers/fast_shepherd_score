"""
Figure 1 — backend parity (two rows):

  Row 1  ALIGNED end-to-end parity: backend="triton" vs backend="jax" aligned Tanimoto
         over all distinct drug pairs.  Annotated with the SIGNED bias + a sign test
         (the honest parity statistics), mean|Δ|, p95|Δ| — Spearman ρ demoted to a note.

  Row 2  FIXED-POSE scoring-kernel agreement: torch (fp32) and jax vs the NumPy fp64
         reference at one identical pose (optimizer removed).  Tight agreement here shows
         the kernels are faithful and that Row 1's larger residual is optimizer-trajectory
         divergence under fp32 arithmetic — NOT independent random-restart noise (the
         multi-start seeds are deterministic and identical across backends).
"""
import json
import math
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
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def sign_test_p(diffs):
    """Two-sided sign-test p that median(diff)=0 (normal approx, ties dropped)."""
    d = diffs[np.abs(diffs) > 0]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = int(np.sum(d > 0))
    z = (k - n / 2) / math.sqrt(n / 4)
    p = math.erfc(abs(z) / math.sqrt(2))            # 2*(1-Phi(|z|))
    return p, k, n


def main():
    plt = set_style()
    with open(os.path.join(HERE, "parity.json")) as fh:
        data = json.load(fh)
    gpu = data.get("_meta", {}).get("gpu", "")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))

    # ---------------- Row 1: aligned end-to-end parity ----------------
    for ax, mode in zip(axes[0], MODES):
        d = data.get(mode, {})
        if "jax" not in d:
            ax.set_visible(False); continue
        x = np.array(d["jax"]); y = np.array(d["triton"]); c = MODE_COLOR[mode]
        lo = min(x.min(), y.min()) - 0.03; hi = max(x.max(), y.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1.2, ls="--", zorder=1)
        ax.scatter(x, y, s=22, color=c, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("JAX reference — aligned Tanimoto")
        diff = y - x
        mad = float(np.abs(diff).mean()); p95 = float(np.percentile(np.abs(diff), 95))
        signed = float(diff.mean()); p, k, n = sign_test_p(diff); rho = spearman(x, y)
        ax.text(0.04, 0.96,
                f"n = {len(x)}\n"
                f"signed Δ = {signed:+.1e}\n"
                f"triton>jax: {k}/{n}  (p={p:.1e})\n"
                f"mean|Δ| = {mad:.1e}   p95 = {p95:.1e}\n"
                f"ρ = {rho:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", alpha=0.9))
    axes[0][0].set_ylabel("Triton / GPU — aligned Tanimoto")

    # ---------------- Row 2: fixed-pose scoring-kernel agreement ----------------
    for ax, mode in zip(axes[1], MODES):
        d = data.get(mode, {})
        if "fixed_np" not in d:
            ax.set_visible(False); continue
        ref = np.array(d["fixed_np"]); tor = np.array(d["fixed_torch"])
        jx = np.array(d.get("fixed_jax", [np.nan] * len(ref))); c = MODE_COLOR[mode]
        lo = min(ref.min(), tor.min()) - 0.03; hi = max(ref.max(), tor.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1.2, ls="--", zorder=1)
        ax.scatter(ref, tor, s=26, color=c, alpha=0.8, edgecolor="white", linewidth=0.4,
                   zorder=3, label="torch fp32")
        if np.isfinite(jx).any():
            ax.scatter(ref, jx, s=16, facecolor="none", edgecolor="#444", linewidth=0.7,
                       zorder=2, label="jax")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("NumPy fp64 — score at fixed pose")
        dt = np.abs(tor - ref); dj = np.abs(jx - ref)
        txt = f"|torch−fp64|: max {np.nanmax(dt):.1e}"
        if np.isfinite(dj).any():
            txt += f"\n|jax−fp64|: max {np.nanmax(dj):.1e}"
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", alpha=0.9))
        if mode == "vol":
            ax.legend(loc="lower right", fontsize=7.5)
    axes[1][0].set_ylabel("GPU kernels — score at fixed pose")

    fig.suptitle(
        "Backend parity — the GPU kernels reproduce the reference scoring\n"
        "Top: aligned (kernel + multi-start optimizer).  Bottom: a single fixed pose (optimizer removed).",
        fontsize=12.5, y=1.02, fontweight="bold")
    fig.text(0.5, -0.02,
             "Multi-start seeds are deterministic and identical across backends, so the small aligned "
             "residual (top) is optimizer-trajectory divergence under fp32 vs fp64 arithmetic — not random-restart "
             f"noise; at a fixed pose (bottom) the kernels agree to ~1e-6.   ·   {gpu}",
             ha="center", fontsize=8, color="#777")
    fig.tight_layout()
    save_fig(fig, os.path.join(HERE, "fig1_backend_parity"))
    plt.close(fig)


if __name__ == "__main__":
    main()
