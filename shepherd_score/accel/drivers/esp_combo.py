"""``vol_and_surf_esp`` driver entry points (ShaEP-style shape + surface-ESP agreement).

The pose is steered by the SHAPE gradient only; the ESP agreement enters the tracked score.
The eager loop scores the ESP term every ``_ESP_STRIDE`` steps (plus the last); the CUDA-graph
step scores it every step (see ``engine._GraphedFineTerms``).
"""
from __future__ import annotations

from typing import Optional

import torch

from ..kernels.dispatch import overlap_score_grad_se3_batch, _batch_self_overlap, esp_comparison_batch
from ._common import apply_se3_transform, quaternion_to_rotation_matrix  # noqa: F401
from ._shim import batch, run
from .engine import _ESP_STRIDE  # noqa: F401  (re-export)


@torch.no_grad()
def _overlap_in_chunks_volumetric(A, B, q, t, *, alpha: float, N_real, M_real, NEED_GRAD=True):
    """Evaluate the volumetric overlap kernel in grid-safe chunks."""
    K = A.shape[0]
    N_real = N_real.to(torch.int32).contiguous()
    M_real = M_real.to(torch.int32).contiguous()
    out_V = torch.empty(K, device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)
    CHUNK = 65_535
    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)
        V, dQ, dT = overlap_score_grad_se3_batch(
            A[start:end], B[start:end], q[start:end], t[start:end], alpha=alpha,
            N_real=N_real[start:end], M_real=M_real[start:end], NEED_GRAD=NEED_GRAD)
        out_V[start:end] = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT
    return out_V, out_dQ, out_dT


def _self_overlap_chunks(P_pad, N_real, alpha):
    K = P_pad.size(0)
    CHUNK = 65_535
    V_all = torch.empty(K, device=P_pad.device, dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(P_pad[s:e], N_real[s:e], alpha)
    return V_all


@torch.no_grad()
def _batch_esp_comparison(points_1, centers_w_H_2, partial_charges_2, points_charges_1, radii_2,
                          M_real_atoms, N_real_surf, probe_radius: float = 1.0, lam: float = 0.001):
    """ShaEP ESP comparison of molecule 1's surface against molecule 2's (world-frame) atoms."""
    return esp_comparison_batch(points_1, centers_w_H_2, partial_charges_2, points_charges_1,
                                radii_2, N_real=N_real_surf, M_real=M_real_atoms,
                                probe_radius=probe_radius, lam=lam)


@torch.no_grad()
def _batch_esp_combo_score(centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
                           partial_charges_1, partial_charges_2, point_charges_1, point_charges_2,
                           radii_1, radii_2, alpha, lam, probe_radius, esp_weight, VAA, VBB,
                           N_real_centers, M_real_centers, N_real_atoms_w_H_1, M_real_atoms_w_H_2,
                           N_real_surf_1, M_real_surf_2, VAB_shape=None):
    """The combined score at a pose whose fit clouds are ALREADY transformed (value only)."""
    B = centers_1.shape[0]
    if VAB_shape is None:
        VAB, _, _ = _overlap_in_chunks_volumetric(
            centers_1, centers_2, torch.tensor([[1., 0., 0., 0.]], device=centers_1.device).expand(B, 4),
            torch.zeros(B, 3, device=centers_1.device), alpha=alpha, N_real=N_real_centers,
            M_real=M_real_centers, NEED_GRAD=False)
    else:
        VAB = VAB_shape
    volumetric_sim = VAB / (VAA + VBB - VAB)
    esp_1 = _batch_esp_comparison(points_1, centers_w_H_2, partial_charges_2, point_charges_1,
                                  radii_2, M_real_atoms_w_H_2, N_real_surf_1, probe_radius, lam)
    esp_2 = _batch_esp_comparison(points_2, centers_w_H_1, partial_charges_1, point_charges_2,
                                  radii_1, N_real_atoms_w_H_1, M_real_surf_2, probe_radius, lam)
    electrostatic_sim = (esp_1 + esp_2) / (N_real_surf_1.to(centers_1.dtype)
                                           + M_real_surf_2.to(centers_1.dtype))
    return esp_weight * electrostatic_sim + (1 - esp_weight) * volumetric_sim


def _combo_chans(centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
                 partial_charges_1, partial_charges_2, point_charges_1, point_charges_2,
                 radii_1, radii_2, N_real_centers, M_real_centers, N_real_atoms_w_H_1,
                 M_real_atoms_w_H_2, N_real_surf_1, M_real_surf_2, alpha):
    """The engine's channel dict for the combo family. The spec selects the atom clouds as the
    shape centres at ``alpha == 0.81`` and the surfaces otherwise, so ``centers_*`` are stored
    under ``atoms`` and consulted only in the first case."""
    wh = batch(centers_w_H_1, centers_w_H_2, N_real_atoms_w_H_1, M_real_atoms_w_H_2)
    sf = batch(points_1, points_2, N_real_surf_1, M_real_surf_2)
    return {"cwh": wh, "partial": batch(partial_charges_1, partial_charges_2, wh.n_real, wh.m_real),
            "radii": batch(radii_1, radii_2, wh.n_real, wh.m_real),
            "surf": sf, "surf_esp": batch(point_charges_1, point_charges_2, sf.n_real, sf.m_real),
            "atoms": batch(centers_1, centers_2, N_real_centers, M_real_centers)}


def coarse_fine_esp_combo_align_many(
        centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
        partial_charges_1, partial_charges_2, point_charges_1, point_charges_2, radii_1, radii_2,
        VAA=None, VBB=None, *, alpha: float, lam: float = 0.001, probe_radius: float = 1.0,
        esp_weight: float = 0.5, num_seeds: int = 50, trans_centers=None, trans_centers_real=None,
        num_repeats_per_trans: int = 10, topk: int = 30, steps_fine: int = 100, lr: float = 0.075,
        N_real_centers=None, M_real_centers=None, N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None,
        N_real_surf_1=None, M_real_surf_2=None, seeds=None, early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5, mode: str = "vol_and_surf_esp", **extra):
    chans = _combo_chans(centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
                         partial_charges_1, partial_charges_2, point_charges_1, point_charges_2,
                         radii_1, radii_2, N_real_centers, M_real_centers, N_real_atoms_w_H_1,
                         M_real_atoms_w_H_2, N_real_surf_1, M_real_surf_2, alpha)
    return run(mode, chans, alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
               num_seeds=num_seeds, steps_fine=steps_fine, lr=lr,
               early_stop_patience=early_stop_patience, early_stop_tol=early_stop_tol, seeds=seeds,
               trans_centers=trans_centers, trans_centers_real=trans_centers_real,
               num_repeats_per_trans=num_repeats_per_trans, topk=topk, **extra)


def fast_optimize_esp_combo_score_overlay_batch(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch, alpha: float, *,
        lam: float = 0.001, probe_radius: float = 1.0, esp_weight: float = 0.5,
        N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None, N_real_centers=None, M_real_centers=None,
        N_real_surf_1=None, M_real_surf_2=None, trans_centers_batch=None, trans_centers_real=None,
        num_repeats_per_trans: int = 10, topk: int = 30, steps_fine: int = 100, num_seeds: int = 50,
        lr: float = 0.075, seeds=None):
    """Batched ESP-combo alignment: ``(aligned_fit_points, q, t, scores)``."""
    scores, q, t = coarse_fine_esp_combo_align_many(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch,
        alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
        trans_centers=trans_centers_batch, trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk, steps_fine=steps_fine,
        num_seeds=num_seeds, lr=lr, N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1, M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_surf_1=N_real_surf_1, M_real_surf_2=M_real_surf_2, seeds=seeds)
    return apply_se3_transform(fit_points_batch, q, t), q, t, scores
