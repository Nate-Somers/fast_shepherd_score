"""``vol_lipo`` driver entry points (shape + per-atom scalar field on the strict-heavy centres,
joint gradient). ``vol_mr`` and ``vol_fukui`` are the same objective over another field and
have their own registry specs; this module serves callers with padded lipo tensors."""
from __future__ import annotations

from ._common import apply_se3_transform, quaternion_to_rotation_matrix  # noqa: F401
from ._shim import batch, run


def _chans(centers_1, centers_2, pos_1, pos_2, val_1, val_2, N_real_centers, M_real_centers,
           N_real_field, M_real_field, field="lipo"):
    c = batch(centers_1, centers_2, N_real_centers, M_real_centers)
    f = batch(pos_1, pos_2, N_real_field, M_real_field)
    return {"atoms": c, f"{field}_pos": f, field: batch(val_1, val_2, f.n_real, f.m_real)}


def coarse_fine_vol_lipo_align_many(
        centers_1, centers_2, lipo_pos_1, lipo_pos_2, lipo_1, lipo_2, VAA=None, VBB=None, *,
        alpha: float = 0.81, lam: float = 0.1, lipo_weight: float = 0.5, num_seeds: int = 50,
        steps_fine: int = 100, lr: float = 0.075, N_real_centers=None, M_real_centers=None,
        N_real_lipo=None, M_real_lipo=None, seeds=None, early_stop_patience: int = 2,
        early_stop_tol: float = 1e-5, mode: str = "vol_lipo", **extra):
    chans = _chans(centers_1, centers_2, lipo_pos_1, lipo_pos_2, lipo_1, lipo_2, N_real_centers,
                   M_real_centers, N_real_lipo, M_real_lipo)
    return run(mode, chans, alpha=alpha, lam=lam, lipo_weight=lipo_weight, num_seeds=num_seeds,
               steps_fine=steps_fine, lr=lr, early_stop_patience=early_stop_patience,
               early_stop_tol=early_stop_tol, seeds=seeds, **extra)


def fast_optimize_vol_lipo_overlay_batch(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, *, alpha: float = 0.81, lam: float = 0.1,
        lipo_weight: float = 0.5, N_real_centers=None, M_real_centers=None, N_real_lipo=None,
        M_real_lipo=None, topk: int = 30, steps_fine: int = 100, lr: float = 0.075,
        num_seeds: int = 50, seeds=None):
    """Batched vol_lipo alignment: ``(aligned_fit_centers, q, t, scores)``."""
    scores, q, t = coarse_fine_vol_lipo_align_many(
        ref_centers_batch, fit_centers_batch, ref_lipo_pos_batch, fit_lipo_pos_batch,
        ref_lipo_batch, fit_lipo_batch, alpha=alpha, lam=lam, lipo_weight=lipo_weight,
        num_seeds=num_seeds, steps_fine=steps_fine, lr=lr, N_real_centers=N_real_centers,
        M_real_centers=M_real_centers, N_real_lipo=N_real_lipo, M_real_lipo=M_real_lipo,
        seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores
