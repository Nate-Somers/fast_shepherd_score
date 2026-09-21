"""``vol_avoid`` driver entry points: shape Tanimoto MINUS a linear hard-sphere excluded-volume
penalty against a FIXED avoid cloud in the reference frame (the one pair-level input)."""
from __future__ import annotations

import torch

from ._common import apply_se3_transform, quaternion_to_rotation_matrix  # noqa: F401
from ._shim import batch, run, se3_of
from .engine import Batch


def coarse_fine_vol_avoid_align_many(
        centers_1, centers_2, avoid_points, VAA=None, VBB=None, *, alpha: float = 0.81,
        avoid_min_dist: float = 2.0, avoid_weight: float = 1.0, num_seeds: int = 50,
        steps_fine: int = 100, lr: float = 0.075, N_real_centers=None, M_real_centers=None,
        K_real_avoid=None, early_stop_patience: int = 2, early_stop_tol: float = 1e-5, seeds=None):
    c = batch(centers_1, centers_2, N_real_centers, M_real_centers)
    if K_real_avoid is None:
        K_real_avoid = avoid_points.new_full((avoid_points.shape[0],), avoid_points.shape[1],
                                             dtype=torch.int32)
    chans = {"atoms": c, "avoid": Batch(avoid_points, None, K_real_avoid, None)}
    return run("vol_avoid", chans, alpha=alpha, avoid_min_dist=avoid_min_dist,
               avoid_weight=avoid_weight, num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
               early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol, seeds=seeds)


def fast_optimize_vol_avoid_overlay_batch(
        ref_centers_batch, fit_centers_batch, avoid_points_batch, *, alpha: float = 0.81,
        avoid_min_dist: float = 2.0, avoid_weight: float = 1.0, N_real_centers=None,
        M_real_centers=None, K_real_avoid=None, steps_fine: int = 100, lr: float = 0.075,
        num_seeds: int = 50, seeds=None):
    """Batched vol_avoid alignment: ``(aligned_fit_centers, q, t, scores)``."""
    scores, q, t = coarse_fine_vol_avoid_align_many(
        ref_centers_batch, fit_centers_batch, avoid_points_batch, alpha=alpha,
        avoid_min_dist=avoid_min_dist, avoid_weight=avoid_weight, num_seeds=num_seeds,
        steps_fine=steps_fine, lr=lr, N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        K_real_avoid=K_real_avoid, seeds=seeds)
    return apply_se3_transform(fit_centers_batch, q, t), q, t, scores


def fast_optimize_vol_avoid_overlay(ref_centers, fit_centers, avoid_points, *, alpha: float = 0.81,
                                    avoid_min_dist: float = 2.0, avoid_weight: float = 1.0,
                                    num_repeats: int = 50, steps_fine: int = 100, lr: float = 0.075,
                                    **kwargs):
    """Single-pair vol_avoid alignment: ``(aligned_fit_centers, SE3 (4,4), score)`` on CPU."""
    device = ref_centers.device
    rc = ref_centers.to(torch.float32).unsqueeze(0)
    fc = fit_centers.to(torch.float32).unsqueeze(0)
    av = avoid_points.to(torch.float32).unsqueeze(0)
    aligned, q, t, score = fast_optimize_vol_avoid_overlay_batch(
        rc, fc, av, alpha=alpha, avoid_min_dist=avoid_min_dist, avoid_weight=avoid_weight,
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        K_real_avoid=torch.tensor([avoid_points.shape[0]], device=device, dtype=torch.int32),
        steps_fine=steps_fine, lr=lr, num_seeds=num_repeats)
    return aligned[0].cpu(), se3_of(q[0], t[0]).cpu(), score[0].cpu()
