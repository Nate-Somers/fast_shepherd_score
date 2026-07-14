"""
Surface diagnostics: detect atom-border "crimps" and quantify how much a surface point cloud
leaks atom positions.

The molecular surface is the outer boundary of the union of (vdW + probe) spheres. For a point
``x`` the *shell residual* to atom ``i`` is ``| ||x - c_i|| - a_i |`` with ``a_i = vdW_i + probe``
(the distance from ``x`` to atom ``i``'s sphere, NOT to its center). Then:

- ``leak``  : a point with shell residual ~0 to ONE sphere sits exactly on that atom's sphere, so
              the atom center is directly recoverable -> a leak.
- ``crimp`` : a point with shell residual ~0 to TWO spheres sits where two spheres intersect = an
              atom border / concave seam (plus a local normal discontinuity).

These functions need only numpy + scipy (no Open3D), so they run anywhere and are the validation
gate for any new surfacer: a low-leak surface spreads its shell-residual mass off zero, and a
smooth surfacer rounds the crimps (lower curvature there).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.linalg import lstsq
from scipy.spatial import cKDTree, distance


def _sorted_shell_residuals(points: np.ndarray,
                            centers: np.ndarray,
                            radii: np.ndarray,
                            probe_radius: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point residual ``| ||x-c_i|| - (vdW_i+probe) |`` to every atom, sorted ascending.

    Returns ``(residuals_sorted (M,N), owner_index_sorted (M,N))`` where column 0 is the nearest
    sphere (the one the point sits on).
    """
    a = np.asarray(radii) + probe_radius
    R = np.abs(distance.cdist(points, centers) - a)
    order = np.argsort(R, axis=1)
    return np.take_along_axis(R, order, axis=1), order


def leak_metrics(points: np.ndarray,
                 centers: np.ndarray,
                 radii: np.ndarray,
                 probe_radius: float = 1.2) -> Dict[str, float]:
    """How close the points lie to the atom spheres (lower residual => more leak).

    Returns median / mean shell residual (A) and the fraction of points within 0.05 / 0.10 A of a
    sphere. For scale: the ``method='mesh'`` (Open3D ball-pivoting + Poisson-disk) surfacer gives
    ~0.010 A median with 86-98% within 0.05 A; a surface whose points lie exactly on the
    (vdW + probe) spheres gives 0.000 A / ~100%.
    """
    Rs, _ = _sorted_shell_residuals(points, centers, radii, probe_radius)
    nearest = Rs[:, 0]
    return {
        "median_residual": float(np.median(nearest)),
        "mean_residual": float(np.mean(nearest)),
        "pct_within_0.05A": float(np.mean(nearest < 0.05) * 100.0),
        "pct_within_0.10A": float(np.mean(nearest < 0.10) * 100.0),
    }


def crimp_points(points: np.ndarray,
                 centers: np.ndarray,
                 radii: np.ndarray,
                 probe_radius: float = 1.2,
                 on_shell_tol: float = 0.10,
                 seam_tol: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean mask of atom-border (crimp) points + the owner atom index of every point.

    A crimp point sits on its owner sphere (``residual[0] < on_shell_tol``) AND close to a second
    atom's sphere (``residual[1] < seam_tol``) -- i.e. near where two spheres intersect.
    """
    Rs, order = _sorted_shell_residuals(points, centers, radii, probe_radius)
    is_crimp = (Rs[:, 0] < on_shell_tol) & (Rs[:, 1] < seam_tol)
    owner = order[:, 0]
    return is_crimp, owner


def _kasa_sphere_center(P: np.ndarray) -> np.ndarray:
    """Algebraic (Kasa) sphere fit; returns the center. Exact for points lying on a sphere."""
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    sol, *_ = lstsq(A, np.sum(P ** 2, axis=1), rcond=None)
    return sol[:3]


def center_recovery_attack(points: np.ndarray,
                           centers: np.ndarray,
                           radii: np.ndarray,
                           probe_radius: float = 1.2,
                           min_pts: int = 8) -> Dict[str, float]:
    """Strongest-case leak test: assume the attacker knows each point's owner atom, fit a sphere per
    atom, and recover its center. Returns the recovery error (A) -- HIGHER is safer.

    Exactly-on-sphere surfaces give ~0 (atoms fully recovered); a smooth off-sphere surface gives a
    large error (atoms hidden).
    """
    _, order = _sorted_shell_residuals(points, centers, radii, probe_radius)
    owner = order[:, 0]
    errs = []
    for i in range(len(centers)):
        Pi = points[owner == i]
        if len(Pi) < min_pts:
            continue
        errs.append(float(np.linalg.norm(_kasa_sphere_center(Pi) - centers[i])))
    errs = np.array(errs) if errs else np.array([np.nan])
    return {
        "median_center_error": float(np.nanmedian(errs)),
        "p90_center_error": float(np.nanpercentile(errs, 90)),
        "n_atoms_tested": int(np.sum(~np.isnan(errs))),
    }


def local_curvature(points: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-point local non-flatness: smallest / total eigenvalue of the kNN covariance (0 = flat,
    higher = more curved/faceted). Crimps are high-curvature; a smoother surface lowers it there.
    """
    tree = cKDTree(points)
    _, nbr = tree.query(points, k=min(k, len(points)))
    out = np.empty(len(points))
    for i in range(len(points)):
        Q = points[nbr[i]] - points[nbr[i]].mean(0)
        w = np.linalg.eigvalsh(Q.T @ Q / len(Q))
        out[i] = w[0] / (w.sum() + 1e-12)
    return out


def summarize(points: np.ndarray,
              centers: np.ndarray,
              radii: np.ndarray,
              probe_radius: float = 1.2) -> Dict[str, float]:
    """One-call report: leak metrics + crimp fraction + center-recovery error + curvature at vs off
    the crimps. Use this to gate any surfacer (mesh, smooth_sdf, ...)."""
    out: Dict[str, float] = {}
    out.update(leak_metrics(points, centers, radii, probe_radius))
    out.update(center_recovery_attack(points, centers, radii, probe_radius))
    is_crimp, _ = crimp_points(points, centers, radii, probe_radius)
    curv = local_curvature(points)
    out["crimp_fraction"] = float(np.mean(is_crimp))
    out["curvature_at_crimps"] = float(curv[is_crimp].mean()) if is_crimp.any() else float("nan")
    out["curvature_elsewhere"] = float(curv[~is_crimp].mean()) if (~is_crimp).any() else float("nan")
    return out
