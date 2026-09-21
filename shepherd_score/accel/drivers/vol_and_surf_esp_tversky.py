"""``vol_and_surf_esp_tversky`` driver entry points (the ShaEP combo with a Tversky shape
channel; the ESP agreement is a masked potential average in [0, 1] and is unchanged)."""
from __future__ import annotations

import torch

from ._common import apply_se3_transform
from .esp_combo import (coarse_fine_esp_combo_align_many, _batch_esp_comparison,  # noqa: F401
                        _overlap_in_chunks_volumetric, _self_overlap_chunks, _ESP_STRIDE)


@torch.no_grad()
def _batch_esp_combo_tversky_score(
        centers_w_H_1, centers_w_H_2, points_1, points_2, partial_charges_1, partial_charges_2,
        point_charges_1, point_charges_2, radii_1, radii_2, lam, probe_radius, esp_weight, C, k,
        N_real_atoms_w_H_1, M_real_atoms_w_H_2, N_real_surf_1, M_real_surf_2, VAB_shape):
    """ESP-combo score with a TVERSKY shape channel at a pose whose fit clouds are transformed."""
    N_surf = N_real_surf_1.to(dtype=points_1.dtype)
    M_surf = M_real_surf_2.to(dtype=points_1.dtype)
    volumetric_sim = VAB_shape / (k * VAB_shape + C)
    esp_1 = _batch_esp_comparison(points_1, centers_w_H_2, partial_charges_2, point_charges_1,
                                  radii_2, M_real_atoms_w_H_2, N_real_surf_1, probe_radius, lam)
    esp_2 = _batch_esp_comparison(points_2, centers_w_H_1, partial_charges_1, point_charges_2,
                                  radii_1, N_real_atoms_w_H_1, M_real_surf_2, probe_radius, lam)
    electrostatic_sim = (esp_1 + esp_2) / (N_surf + M_surf)
    return esp_weight * electrostatic_sim + (1 - esp_weight) * volumetric_sim


def coarse_fine_esp_combo_tversky_align_many(
        centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
        partial_charges_1, partial_charges_2, point_charges_1, point_charges_2, radii_1, radii_2,
        VAA=None, VBB=None, *, alpha, lam=0.001, probe_radius=1.0, esp_weight=0.5,
        tversky_alpha=0.95, tversky_beta=0.05, num_seeds=50, topk=30, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None, N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None,
        N_real_surf_1=None, M_real_surf_2=None, early_stop_patience=5, early_stop_tol=1e-5,
        seeds=None):
    return coarse_fine_esp_combo_align_many(
        centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
        partial_charges_1, partial_charges_2, point_charges_1, point_charges_2, radii_1, radii_2,
        alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
        num_seeds=num_seeds, topk=topk, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1, M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_surf_1=N_real_surf_1, M_real_surf_2=M_real_surf_2,
        early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol, seeds=seeds,
        mode="vol_and_surf_esp_tversky", tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)


def fast_optimize_vol_and_surf_esp_tversky_overlay_batch(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch, alpha, *,
        lam=0.001, probe_radius=1.0, esp_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05,
        N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None, N_real_centers=None, M_real_centers=None,
        N_real_surf_1=None, M_real_surf_2=None, topk=30, steps_fine=100, num_seeds=50, lr=0.075,
        seeds=None):
    scores, q, t = coarse_fine_esp_combo_tversky_align_many(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch, alpha=alpha,
        lam=lam, probe_radius=probe_radius, esp_weight=esp_weight, tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta, num_seeds=num_seeds, topk=topk, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1, M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_surf_1=N_real_surf_1, M_real_surf_2=M_real_surf_2, seeds=seeds)
    return apply_se3_transform(fit_points_batch, q, t), q, t, scores
