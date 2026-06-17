"""
Accuracy metrics.

We treat accuracy as *delivered alignment quality*, not implementation equality:

* ``achieved`` -- the Tanimoto-style overlap each backend actually reached.
* ``gap_to_ideal`` -- only meaningful for noiseless cohorts, where the global
  optimum is a perfect overlap (Tanimoto = 1).  ``1 - achieved`` is then exactly
  how far the optimiser fell short of the recoverable optimum.
* ``vs_reference`` -- agreement with the canonical torch reference optimiser
  (``cpu_single_torch``): mean/RMS/max absolute per-pair score difference, a
  signed mean (did this backend do better or worse on average), the fraction of
  pairs within 1e-3, and Spearman rank correlation (does it rank poses the same
  way, which is what matters for screening).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def score_summary(scores: np.ndarray, noise: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    n_bad = int(np.sum(~np.isfinite(s)))
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return {"n_nonfinite": n_bad, "mean": float("nan")}
    out = {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "n_nonfinite": n_bad,
    }
    s = finite
    if noise == 0.0:
        # Noiseless optimum is a perfect overlap -> gap to ideal is 1 - score.
        out["gap_to_ideal_mean"] = float(np.mean(1.0 - s))
        out["gap_to_ideal_p90"] = float(np.percentile(1.0 - s, 90))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def accuracy_vs_reference(scores: np.ndarray,
                          ref_scores: Optional[np.ndarray]) -> Dict[str, float]:
    if ref_scores is None:
        return {}
    s = np.asarray(scores, dtype=float)
    r = np.asarray(ref_scores, dtype=float)
    d = s - r
    return {
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "rms_diff": float(np.sqrt(np.mean(d * d))),
        "max_abs_diff": float(np.max(np.abs(d))),
        "mean_signed_diff": float(np.mean(d)),     # >0 => beats reference on avg
        "frac_within_1e-3": float(np.mean(np.abs(d) <= 1e-3)),
        "spearman": _spearman(s, r),
    }
