# shepherd_score/accel/drivers/vol_and_surf_esp_tversky.py
# Fast batched ``vol_and_surf_esp_tversky`` alignment: the ShaEP-style vol_and_surf_esp mode with
# the SHAPE channel scored by TVERSKY instead of Tanimoto (the surface-ESP AGREEMENT channel is a
# masked point-to-point potential average in [0,1], not an overlap ratio, so Tversky does not apply
# to it -- it is unchanged).
#
#   score = esp_weight*esp_agreement + (1-esp_weight)*shape_Tversky
#   shape_Tversky = VAB / (k*VAB + C),  C = tα*VAA + tβ*VBB,  k = 1 - tα - tβ
#   d(shape_Tversky)/dVAB = C / (k*VAB + C)^2
#
# Identical plumbing to ``esp_combo`` (drivers/esp_combo.py) -- same shape kernel, same
# ``esp_comparison_batch`` ESP kernel, same sparse-ESP stride, pose steered by the SHAPE gradient
# only -- with the shape channel's Tanimoto reduction (score + gradient scale) swapped for the
# Tversky reduction proven in ``vol_tversky``. Matches the per-pair reference
# ``alignment._torch.optimize_vol_and_surf_esp_tversky_overlay``.

from __future__ import annotations

import torch
from typing import Tuple, Optional

from ..kernels.dispatch import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj)
from ._common import (
    batched_seeds_torch, apply_se3_transform, quaternion_to_rotation_matrix, _update_best)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from .esp_combo import (
    _overlap_in_chunks_volumetric, _self_overlap_chunks, _batch_esp_comparison, _ESP_STRIDE)


@torch.no_grad()
def _batch_esp_combo_tversky_score(
        centers_w_H_1, centers_w_H_2, points_1, points_2,
        partial_charges_1, partial_charges_2, point_charges_1, point_charges_2,
        radii_1, radii_2, lam, probe_radius, esp_weight, C, k,
        N_real_atoms_w_H_1, M_real_atoms_w_H_2, N_real_surf_1, M_real_surf_2, VAB_shape):
    """ESP-combo score with a TVERSKY shape channel (ESP agreement unchanged). ``VAB_shape`` is the
    shape overlap already computed for this pose by the caller; ``C``/``k`` are the Tversky combo of
    the shape self-overlaps and ``1-tα-tβ``."""
    N_surf = N_real_surf_1.to(dtype=points_1.dtype)
    M_surf = M_real_surf_2.to(dtype=points_1.dtype)
    volumetric_sim = VAB_shape / (k * VAB_shape + C)
    esp_1 = _batch_esp_comparison(points_1, centers_w_H_2, partial_charges_2,
                                  point_charges_1, radii_2, M_real_atoms_w_H_2, N_real_surf_1,
                                  probe_radius, lam)
    esp_2 = _batch_esp_comparison(points_2, centers_w_H_1, partial_charges_1,
                                  point_charges_2, radii_1, N_real_atoms_w_H_1, M_real_surf_2,
                                  probe_radius, lam)
    electrostatic_sim = (esp_1 + esp_2) / (N_surf + M_surf)
    return esp_weight * electrostatic_sim + (1 - esp_weight) * volumetric_sim


class _GraphedFineEspComboTversky(_GraphedFineBase):
    """CUDA-graph fine loop for vol_and_surf_esp_tversky (esp_combo with a Tversky shape channel)."""

    def __init__(self, Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
                 alpha, lam, probe_radius, esp_weight, ta, tb, lr, device):
        self.alpha = float(alpha); self.lam = float(lam)
        self.probe_radius = float(probe_radius); self.esp_weight = float(esp_weight)
        self.ta = float(ta); self.tb = float(tb); self.k = 1.0 - float(ta) - float(tb)
        self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        self.cent1 = f(P, Nc, 3); self.cent2 = f(P, Mc, 3)
        self.cwH1 = f(P, NwH, 3); self.cwH2 = f(P, MwH, 3)
        self.pts1 = f(P, Ns, 3); self.pts2 = f(P, Ms, 3)
        self.pc1 = f(P, NwH); self.pc2 = f(P, MwH)
        self.ptc1 = f(P, Ns); self.ptc2 = f(P, Ms)
        self.rad1 = f(P, NwH); self.rad2 = f(P, MwH)
        self.Nr = i(P); self.Mr = i(P)
        self.Natoms = i(P); self.Matoms = i(P)
        self.Nsurf = i(P); self.Msurf = i(P)
        self.C = f(P)                                        # tα*VAA + tβ*VBB (shape)
        self.qs = f(P, 4); self.ts = f(P, 3)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        cwH2_t = apply_se3_transform(self.cwH2, self.q, self.t)
        pts2_t = apply_se3_transform(self.pts2, self.q, self.t)
        VAB, dQ, dT = overlap_score_grad_se3_batch(
            self.cent1, self.cent2, self.q, self.t,
            alpha=self.alpha, N_real=self.Nr, M_real=self.Mr)
        score = _batch_esp_combo_tversky_score(
            self.cwH1, cwH2_t, self.pts1, pts2_t,
            self.pc1, self.pc2, self.ptc1, self.ptc2, self.rad1, self.rad2,
            self.lam, self.probe_radius, self.esp_weight, self.C, self.k,
            self.Natoms, self.Matoms, self.Nsurf, self.Msurf, VAB)
        denom = self.k * VAB + self.C
        scale = self.C / (denom * denom)
        better = score > self.best
        torch.where(better, score, self.best, out=self.best)
        bm = better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        g = (scale * (1.0 - self.esp_weight)).unsqueeze(1)
        fused_adam_qt_with_tangent_proj(self.q, self.t, -dQ * g, -dT * g,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2, rad1, rad2,
              Nr, Mr, Natoms, Matoms, Nsurf, Msurf, C, qs, ts):
        self.cent1.copy_(cent1); self.cent2.copy_(cent2)
        self.cwH1.copy_(cwH1); self.cwH2.copy_(cwH2)
        self.pts1.copy_(pts1); self.pts2.copy_(pts2)
        self.pc1.copy_(pc1); self.pc2.copy_(pc2)
        self.ptc1.copy_(ptc1); self.ptc2.copy_(ptc2)
        self.rad1.copy_(rad1); self.rad2.copy_(rad2)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Natoms.copy_(Natoms.to(torch.int32)); self.Matoms.copy_(Matoms.to(torch.int32))
        self.Nsurf.copy_(Nsurf.to(torch.int32)); self.Msurf.copy_(Msurf.to(torch.int32))
        self.C.copy_(C); self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_esp_combo_tversky(cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2,
                                   rad1, rad2, Nr, Mr, Natoms, Matoms, Nsurf, Msurf, C,
                                   q_seed, t_seed, alpha, lam, probe_radius, esp_weight, ta, tb, lr,
                                   steps, Nc, Mc, NwH, MwH, Ns, Ms, P, es_patience=0, es_tol=1e-5):
    key = (cent1.device.index, "vol_and_surf_esp_tversky", Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
           round(float(alpha), 4), round(float(lam), 6), round(float(probe_radius), 4),
           round(float(esp_weight), 4), round(float(ta), 4), round(float(tb), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineEspComboTversky(Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
                                            alpha, lam, probe_radius, esp_weight, ta, tb, lr, cent1.device),
        key, (cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2, rad1, rad2,
              Nr, Mr, Natoms, Matoms, Nsurf, Msurf, C, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_esp_combo_tversky_align_many(
        centers_w_H_1, centers_w_H_2, centers_1, centers_2, points_1, points_2,
        partial_charges_1, partial_charges_2, point_charges_1, point_charges_2,
        radii_1, radii_2, VAA, VBB, *,
        alpha, lam=0.001, probe_radius=1.0, esp_weight=0.5,
        tversky_alpha=0.95, tversky_beta=0.05,
        num_seeds=50, topk=30, steps_fine=100, lr=0.075,
        N_real_centers=None, M_real_centers=None,
        N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None,
        N_real_surf_1=None, M_real_surf_2=None,
        early_stop_patience=5, early_stop_tol=1e-5):
    """Batched vol_and_surf_esp_tversky alignment (seed path)."""
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad_centers, M_pad_centers = centers_1.shape[1], centers_2.shape[1]
    N_pad_w_H, M_pad_w_H = centers_w_H_1.shape[1], centers_w_H_2.shape[1]
    N_surf, M_surf = points_1.shape[1], points_2.shape[1]
    ta, tb = float(tversky_alpha), float(tversky_beta); k = 1.0 - ta - tb

    if N_real_centers is None:
        N_real_centers = centers_1.new_full((BATCH,), N_pad_centers, dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = centers_2.new_full((BATCH,), M_pad_centers, dtype=torch.int32)
    if N_real_atoms_w_H_1 is None:
        N_real_atoms_w_H_1 = centers_w_H_1.new_full((BATCH,), N_pad_w_H, dtype=torch.int32)
    if M_real_atoms_w_H_2 is None:
        M_real_atoms_w_H_2 = centers_w_H_2.new_full((BATCH,), M_pad_w_H, dtype=torch.int32)
    if N_real_surf_1 is None:
        N_real_surf_1 = points_1.new_full((BATCH,), N_surf, dtype=torch.int32)
    if M_real_surf_2 is None:
        M_real_surf_2 = points_2.new_full((BATCH,), M_surf, dtype=torch.int32)

    quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                         M_real_centers, num_seeds=num_seeds)
    P = quats.size(1)
    q_k = quats.clone().reshape(-1, 4).contiguous()
    t_k = t_seeds.clone().reshape(-1, 3).contiguous()

    def _exp3(x, D):
        return x.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, D, 3)

    def _exp2(x, D):
        return x.unsqueeze(1).expand(-1, P, -1).reshape(-1, D)

    centers_w_H_1_k = _exp3(centers_w_H_1, N_pad_w_H)
    centers_w_H_2_k = _exp3(centers_w_H_2, M_pad_w_H)
    centers_1_k = _exp3(centers_1, N_pad_centers)
    centers_2_k = _exp3(centers_2, M_pad_centers)
    points_1_k = _exp3(points_1, N_surf)
    points_2_k = _exp3(points_2, M_surf)
    partial_charges_1_k = _exp2(partial_charges_1, N_pad_w_H)
    partial_charges_2_k = _exp2(partial_charges_2, M_pad_w_H)
    point_charges_1_k = _exp2(point_charges_1, N_surf)
    point_charges_2_k = _exp2(point_charges_2, M_surf)
    radii_1_k = _exp2(radii_1, N_pad_w_H)
    radii_2_k = _exp2(radii_2, M_pad_w_H)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    N_atoms_k = N_real_atoms_w_H_1.repeat_interleave(P)
    M_atoms_k = M_real_atoms_w_H_2.repeat_interleave(P)
    N_surf_k = N_real_surf_1.repeat_interleave(P)
    M_surf_k = M_real_surf_2.repeat_interleave(P)
    C = (ta * VAA + tb * VBB).repeat_interleave(P)

    m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)
    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q = q_k.clone(); best_t = t_k.clone()
    prev_max_score = -float('inf'); no_improve_count = 0

    _graphed = None
    if (centers_1_k.is_cuda and centers_1_k.dtype == torch.float32
            and len(q_k) <= graph_cap(N_pad_centers * M_pad_centers, budget=8_000_000)):
        try:
            _graphed = _run_graphed_esp_combo_tversky(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                centers_w_H_1_k.contiguous(), centers_w_H_2_k.contiguous(),
                points_1_k.contiguous(), points_2_k.contiguous(),
                partial_charges_1_k, partial_charges_2_k,
                point_charges_1_k, point_charges_2_k, radii_1_k, radii_2_k,
                N_k, M_k, N_atoms_k, M_atoms_k, N_surf_k, M_surf_k, C, q_k, t_k,
                alpha, lam, probe_radius, esp_weight, ta, tb, lr,
                steps_fine, N_pad_centers, M_pad_centers, N_pad_w_H, M_pad_w_H, N_surf, M_surf, len(q_k))
        except Exception:
            _graphed = None
    if _graphed is not None:
        best_score, best_q, best_t = _graphed

    for step in range(steps_fine):
        if _graphed is not None:
            break
        VAB, dQ, dT = _overlap_in_chunks_volumetric(
            centers_1_k, centers_2_k, q_k, t_k, alpha=alpha, N_real=N_k, M_real=M_k)
        denom = k * VAB + C
        scale = C / (denom * denom)
        if (step % _ESP_STRIDE == 0) or (step == steps_fine - 1):
            centers_w_H_2_t = apply_se3_transform(centers_w_H_2_k, q_k, t_k)
            points_2_t = apply_se3_transform(points_2_k, q_k, t_k)
            score = _batch_esp_combo_tversky_score(
                centers_w_H_1_k, centers_w_H_2_t, points_1_k, points_2_t,
                partial_charges_1_k, partial_charges_2_k, point_charges_1_k, point_charges_2_k,
                radii_1_k, radii_2_k, lam, probe_radius, esp_weight, C, k,
                N_atoms_k, M_atoms_k, N_surf_k, M_surf_k, VAB)
            best_score, best_q, best_t = _update_best(score, q_k, t_k, best_score, best_q, best_t)
        if step % 5 == 0:
            current_max = best_score.max().item()
            if current_max - prev_max_score < early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    break
            else:
                no_improve_count = 0
                prev_max_score = current_max
        fused_adam_qt_with_tangent_proj(
            q_k, t_k, -dQ * scale.unsqueeze(1) * (1 - esp_weight),
            -dT * scale.unsqueeze(1) * (1 - esp_weight), m_q, v_q, m_t, v_t, lr)

    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P
    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_vol_and_surf_esp_tversky_overlay_batch(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch, alpha, *,
        lam=0.001, probe_radius=1.0, esp_weight=0.5, tversky_alpha=0.95, tversky_beta=0.05,
        N_real_atoms_w_H_1=None, M_real_atoms_w_H_2=None, N_real_centers=None, M_real_centers=None,
        N_real_surf_1=None, M_real_surf_2=None, topk=30, steps_fine=100, num_seeds=50, lr=0.075):
    """Batched vol_and_surf_esp_tversky alignment. Returns (aligned_fit_points, q, t, scores)."""
    BATCH = ref_centers_batch.shape[0]
    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full((BATCH,), ref_centers_batch.shape[1], dtype=torch.int32)
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full((BATCH,), fit_centers_batch.shape[1], dtype=torch.int32)

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_esp_combo_tversky_align_many(
        ref_centers_w_H_batch, fit_centers_w_H_batch, ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch, ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch, ref_radii_batch, fit_radii_batch, VAA, VBB,
        alpha=alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
        tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
        num_seeds=num_seeds, topk=topk, steps_fine=steps_fine, lr=lr,
        N_real_centers=N_real_centers, M_real_centers=M_real_centers,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1, M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_surf_1=N_real_surf_1, M_real_surf_2=M_real_surf_2)
    aligned_fit_points = apply_se3_transform(fit_points_batch, q_best, t_best)
    return aligned_fit_points, q_best, t_best, scores
