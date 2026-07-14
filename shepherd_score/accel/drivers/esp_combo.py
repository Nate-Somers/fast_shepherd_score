# shepherd_score/accel/drivers/esp_combo.py
# Fast GPU-accelerated ESP combo alignment.
# Combines shape overlap (using Triton kernel) with surface ESP comparison.

from __future__ import annotations

import os
import torch
from typing import Tuple, Optional

# Lever: sparse ESP scoring. vol_and_surf_esp steers the pose by the SHAPE gradient ONLY (the
# ESP enters just the TRACKED combo score, never the SE(3) derivative), so the optimisation
# trajectory is byte-identical whether or not the (expensive) ESP score is evaluated on a given
# step -- per step the ESP costs 2 surface-ESP kernels (N_surf x M_atoms) + 2 SE(3) cloud
# transforms, ~the bulk of the work, vs one small shape value+grad kernel. We can therefore
# evaluate the combo score every FINE_ESP_STRIDE steps (always including the final step) and
# track the best among those, skipping the ESP work on the off-steps. The pose path, gradient,
# and shape early-stop are unchanged; only best-pose *selection* is sampled more coarsely.
# Accuracy-gated (the trajectory converges, so the combo peak sits near the scored tail). =1
# reproduces the dense baseline exactly.
_ESP_STRIDE = max(1, int(os.environ.get("FINE_ESP_STRIDE", "5")))

# Lever (DEFAULT ON): seed from the VOLUME centers instead of the surface points. vol_and_surf_esp
# steers the pose by the volume (centers) shape gradient, but historically seeded from the surface
# clouds -- a mismatch: the surface is far more multi-basin than the heavy-atom volume (legacy
# standalone surf needed 64 seeds vs vol's 16), so surface seeds needed many more starts to be
# sure a volume basin was covered. Seeding from the centers aligns the seed PCA starts with the
# basins the optimiser actually descends, so far fewer seeds reach the same combo score: vol-seed
# @24 == surf-seed @64
# on the mean (99.7%; tested on the bench-380 and drugs.smi sets), at ~1.96x the throughput. The
# matching seed count (24, vs the legacy 64) lives in aligners._MODE_SEEDS. Set FSS_VASE_VOL_SEEDS=0
# to revert to legacy surface seeds (then bump the seed count back to ~64).
_VOL_SEEDS = os.environ.get("FSS_VASE_VOL_SEEDS", "1") != "0"

# Device-driven kernel dispatch (Triton on CUDA, numba on CPU); see kernel_dispatch.
# esp_combo reuses the SAME Gaussian shape kernel (vol overlap) as fast_se3/surface
# for the shape channel, and the fused value-only `esp_comparison_batch` kernel for
# the ShaEP ESP channel (replacing the per-step (B,N_surf,M_atoms) torch.cdist). Both
# kernels have Triton (CUDA) and numba (CPU) twins, so esp_combo runs on either backend.
from ..kernels.dispatch import (
    overlap_score_grad_se3_batch,
    fused_adam_qt_with_tangent_proj,
    _batch_self_overlap,
    esp_comparison_batch,
)
from ._common import (
    check_gpu_available,
    build_coarse_grid,
    batched_seeds_torch,
    apply_se3_transform,
    quaternion_to_rotation_matrix,
    _update_best,
    ES_PATIENCE_OVERRIDE,
)
from ._graphed import _GraphedFineBase, run_graphed, graph_cap, _FINE_GRAPHS, _GRAPH_MAX_P, _GRAPH_STEPS


@torch.no_grad()
def _overlap_in_chunks_volumetric(A, B, q, t, *, alpha: float,
                                   N_real: torch.Tensor,
                                   M_real: torch.Tensor,
                                   NEED_GRAD: bool = True,
                                   BLOCK: int | None = None):   # None -> kernel auto: BLOCK=16, 1 warp/CTA
    """Evaluate volumetric overlap kernel in chunks."""
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
            A[start:end], B[start:end],
            q[start:end], t[start:end],
            alpha=alpha,
            N_real=N_real[start:end],
            M_real=M_real[start:end],
            NEED_GRAD=NEED_GRAD,
            BLOCK=BLOCK)

        out_V[start:end] = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


def _self_overlap_chunks(P_pad, N_real, alpha):
    """Compute volumetric self-overlap in chunks."""
    K = P_pad.size(0)
    CHUNK = 65_535
    V_all = torch.empty(K, device=P_pad.device, dtype=P_pad.dtype)

    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(P_pad[s:e], N_real[s:e], alpha)

    return V_all


@torch.no_grad()
def _batch_esp_comparison(points_1: torch.Tensor,
                          centers_w_H_2: torch.Tensor,
                          partial_charges_2: torch.Tensor,
                          points_charges_1: torch.Tensor,
                          radii_2: torch.Tensor,
                          M_real_atoms: torch.Tensor,
                          N_real_surf: torch.Tensor,
                          probe_radius: float = 1.0,
                          lam: float = 0.001) -> torch.Tensor:
    """
    Batched ESP comparison on GPU.

    Computes the difference in ESP at surface points of molecule 1 for the ESP
    values generated by molecule 1 and molecule 2, with masking for overlapping
    volumes.

    Parameters
    ----------
    points_1 : torch.Tensor (B, N_surf, 3)
        Surface points of molecule 1
    centers_w_H_2 : torch.Tensor (B, M_atoms, 3)
        Atom coordinates (with H) of molecule 2
    partial_charges_2 : torch.Tensor (B, M_atoms)
        Partial charges of molecule 2
    points_charges_1 : torch.Tensor (B, N_surf)
        Pre-computed ESP at points_1 from molecule 1
    radii_2 : torch.Tensor (B, M_atoms)
        VdW radii of molecule 2 atoms
    probe_radius : float
        Probe radius for masking
    lam : float
        ESP weighting parameter

    Returns
    -------
    esp : torch.Tensor (B,)
        ESP comparison scores

    Notes
    -----
    Thin wrapper over the fused ``esp_comparison_batch`` kernel (Triton on CUDA,
    numba on CPU). The kernel computes the same masked Gaussian-of-ESP-difference
    sum as the original ``torch.cdist`` implementation, but without materializing
    the ``(B, N_surf, M_atoms)`` distance tensor -- which is what made the eager
    path the per-step bottleneck (and the CPU path untenable). ``points_1`` and
    ``centers_w_H_2`` are expected already in the world frame.
    """
    return esp_comparison_batch(
        points_1, centers_w_H_2, partial_charges_2, points_charges_1, radii_2,
        N_real=N_real_surf, M_real=M_real_atoms,
        probe_radius=probe_radius, lam=lam)


@torch.no_grad()
def _batch_esp_combo_score(
        centers_w_H_1: torch.Tensor,
        centers_w_H_2: torch.Tensor,
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        points_1: torch.Tensor,
        points_2: torch.Tensor,
        partial_charges_1: torch.Tensor,
        partial_charges_2: torch.Tensor,
        point_charges_1: torch.Tensor,
        point_charges_2: torch.Tensor,
        radii_1: torch.Tensor,
        radii_2: torch.Tensor,
        alpha: float,
        lam: float,
        probe_radius: float,
        esp_weight: float,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        N_real_centers: torch.Tensor,
        M_real_centers: torch.Tensor,
        N_real_atoms_w_H_1: torch.Tensor,
        M_real_atoms_w_H_2: torch.Tensor,
        N_real_surf_1: torch.Tensor,
        M_real_surf_2: torch.Tensor,
        VAB_shape: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute batched ESP combo score (shape + ESP similarity).

    ``VAB_shape`` (optional): the shape overlap VAB already computed for this pose
    by the caller. The fine loop computes VAB (with gradient) every step anyway, so
    passing it here skips a redundant value-only shape-kernel launch per step. When
    ``None`` (coarse-grid path) the shape overlap is recomputed from ``centers_*``.

    Returns
    -------
    score : torch.Tensor (B,)
        Combined similarity scores
    """
    BATCH = centers_1.shape[0]
    N_surf = N_real_surf_1.to(dtype=centers_1.dtype)
    M_surf = M_real_surf_2.to(dtype=centers_1.dtype)

    # Shape similarity using volumetric kernel (reuse the caller's VAB if given)
    if VAB_shape is None:
        VAB, _, _ = _overlap_in_chunks_volumetric(
            centers_1, centers_2,
            torch.tensor([[1., 0., 0., 0.]], device=centers_1.device).expand(BATCH, 4),
            torch.zeros(BATCH, 3, device=centers_1.device),
            alpha=alpha,
            N_real=N_real_centers,
            M_real=M_real_centers,
            NEED_GRAD=False)
    else:
        VAB = VAB_shape

    volumetric_sim = VAB / (VAA + VBB - VAB)

    # ESP similarity
    esp_1 = _batch_esp_comparison(
        points_1, centers_w_H_2, partial_charges_2,
        point_charges_1, radii_2,
        M_real_atoms_w_H_2, N_real_surf_1,
        probe_radius, lam)
    esp_2 = _batch_esp_comparison(
        points_2, centers_w_H_1, partial_charges_1,
        point_charges_2, radii_1,
        N_real_atoms_w_H_1, M_real_surf_2,
        probe_radius, lam)

    electrostatic_sim = (esp_1 + esp_2) / (N_surf + M_surf)

    # Combined score
    score = esp_weight * electrostatic_sim + (1 - esp_weight) * volumetric_sim

    return score


class _GraphedFineEspCombo(_GraphedFineBase):
    """CUDA-graph fine loop for vol_and_surf_esp -- the heaviest mode (per step: 1 shape
    value+grad kernel + 2 value-only ESP-comparison kernels + 2 SE(3) cloud transforms).
    The pose is steered purely by the shape gradient (scaled by 1-esp_weight); the ESP
    enters only the TRACKED combo score. The captured step mirrors the eager body verbatim
    with the best-update made in-place. Per-step temporaries (transformed clouds, ESP sums,
    score) are fresh, served from the graph's private pool on replay; only the loop-carried
    state (q/t, Adam moments, best*) is persistent. esp_combo is compute-bound, so graphing
    mainly UNIFIES the path; the eager loop remains the fallback for large P / capture
    failure."""

    def __init__(self, Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
                 alpha, lam, probe_radius, esp_weight, lr, device):
        self.alpha = float(alpha); self.lam = float(lam)
        self.probe_radius = float(probe_radius); self.esp_weight = float(esp_weight)
        self.lr = float(lr)
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        i = lambda *s: torch.empty(*s, device=device, dtype=torch.int32)
        self.cent1 = f(P, Nc, 3); self.cent2 = f(P, Mc, 3)           # shape centers
        self.cwH1 = f(P, NwH, 3); self.cwH2 = f(P, MwH, 3)           # ESP atoms (w/ H)
        self.pts1 = f(P, Ns, 3); self.pts2 = f(P, Ms, 3)            # surface points
        self.pc1 = f(P, NwH); self.pc2 = f(P, MwH)                   # atom partial charges
        self.ptc1 = f(P, Ns); self.ptc2 = f(P, Ms)                   # surface ESP per point
        self.rad1 = f(P, NwH); self.rad2 = f(P, MwH)                 # vdW radii
        self.Nr = i(P); self.Mr = i(P)                               # centers counts
        self.Natoms = i(P); self.Matoms = i(P)                      # atom-w-H counts
        self.Nsurf = i(P); self.Msurf = i(P)                        # surf-point counts
        self.VAA = f(P); self.VBB = f(P); self.normsum = f(P)        # shape self-overlaps
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
        score = _batch_esp_combo_score(
            self.cwH1, cwH2_t, self.cent1, self.cent2, self.pts1, pts2_t,
            self.pc1, self.pc2, self.ptc1, self.ptc2, self.rad1, self.rad2,
            self.alpha, self.lam, self.probe_radius, self.esp_weight,
            self.VAA, self.VBB, self.Nr, self.Mr, self.Natoms, self.Matoms,
            self.Nsurf, self.Msurf, VAB_shape=VAB)
        denom = self.normsum - VAB
        scale = self.normsum / (denom * denom)
        # best-pose tracking, in-place
        better = score > self.best
        torch.where(better, score, self.best, out=self.best)
        bm = better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        # pose steered by SHAPE gradient only, scaled by (1 - esp_weight)
        g = (scale * (1.0 - self.esp_weight)).unsqueeze(1)
        fused_adam_qt_with_tangent_proj(self.q, self.t, -dQ * g, -dT * g,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2, rad1, rad2,
              Nr, Mr, Natoms, Matoms, Nsurf, Msurf, VAA, VBB, normsum, qs, ts):
        self.cent1.copy_(cent1); self.cent2.copy_(cent2)
        self.cwH1.copy_(cwH1); self.cwH2.copy_(cwH2)
        self.pts1.copy_(pts1); self.pts2.copy_(pts2)
        self.pc1.copy_(pc1); self.pc2.copy_(pc2)
        self.ptc1.copy_(ptc1); self.ptc2.copy_(ptc2)
        self.rad1.copy_(rad1); self.rad2.copy_(rad2)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.Natoms.copy_(Natoms.to(torch.int32)); self.Matoms.copy_(Matoms.to(torch.int32))
        self.Nsurf.copy_(Nsurf.to(torch.int32)); self.Msurf.copy_(Msurf.to(torch.int32))
        self.VAA.copy_(VAA); self.VBB.copy_(VBB); self.normsum.copy_(normsum)
        self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_esp_combo(cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2,
                           rad1, rad2, Nr, Mr, Natoms, Matoms, Nsurf, Msurf, VAA, VBB,
                           normsum, q_seed, t_seed, alpha, lam, probe_radius, esp_weight, lr,
                           steps, Nc, Mc, NwH, MwH, Ns, Ms, P, es_patience=0, es_tol=1e-5):
    key = (cent1.device.index, "esp_combo", Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
           round(float(alpha), 4), round(float(lam), 6), round(float(probe_radius), 4),
           round(float(esp_weight), 4), round(float(lr), 5))
    return run_graphed(
        lambda: _GraphedFineEspCombo(Nc, Mc, NwH, MwH, Ns, Ms, P, steps,
                                     alpha, lam, probe_radius, esp_weight, lr, cent1.device),
        key, (cent1, cent2, cwH1, cwH2, pts1, pts2, pc1, pc2, ptc1, ptc2, rad1, rad2,
              Nr, Mr, Natoms, Matoms, Nsurf, Msurf, VAA, VBB, normsum, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol)


def coarse_fine_esp_combo_align_many(
        centers_w_H_1: torch.Tensor,
        centers_w_H_2: torch.Tensor,
        centers_1: torch.Tensor,
        centers_2: torch.Tensor,
        points_1: torch.Tensor,
        points_2: torch.Tensor,
        partial_charges_1: torch.Tensor,
        partial_charges_2: torch.Tensor,
        point_charges_1: torch.Tensor,
        point_charges_2: torch.Tensor,
        radii_1: torch.Tensor,
        radii_2: torch.Tensor,
        VAA: torch.Tensor,
        VBB: torch.Tensor,
        *,
        alpha: float,
        lam: float = 0.001,
        probe_radius: float = 1.0,
        esp_weight: float = 0.5,
        num_seeds: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_atoms_w_H_1: Optional[torch.Tensor] = None,
        M_real_atoms_w_H_2: Optional[torch.Tensor] = None,
        N_real_surf_1: Optional[torch.Tensor] = None,
        M_real_surf_2: Optional[torch.Tensor] = None,
        early_stop_patience: int = 5,
        early_stop_tol: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized ESP combo alignment over a batch of molecule pairs.

    Uses coarse-to-fine strategy with the combined shape + ESP scoring.

    Returns
    -------
    final_score : torch.Tensor (B,)
        Best combo scores
    q_best : torch.Tensor (B, 4)
        Best quaternions
    t_best : torch.Tensor (B, 3)
        Best translations
    """
    device = centers_1.device
    BATCH = centers_1.shape[0]
    N_pad_centers = centers_1.shape[1]
    M_pad_centers = centers_2.shape[1]
    N_pad_w_H = centers_w_H_1.shape[1]
    M_pad_w_H = centers_w_H_2.shape[1]
    N_surf = points_1.shape[1]
    M_surf = points_2.shape[1]

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

    # ------------------------------------------------------------------
    # 1) pose hypotheses
    # ------------------------------------------------------------------
    if trans_centers is None:
        # Reference seed set (identity + 4 PCA + Fibonacci, COM translation);
        # fine-optimise ALL seeds and take the per-pair max -- NO coarse-grid +
        # top-k pruning. Pruning on the raw (un-optimised) overlap dropped the
        # true basin for pseudo-symmetric molecules; see
        # fast_se3.coarse_fine_align_many for the full rationale. Seed from the
        # VOLUME centers (matches the shape gradient) when _VOL_SEEDS, else the
        # legacy surface point clouds.
        if _VOL_SEEDS:
            quats, t_seeds = batched_seeds_torch(centers_1, centers_2, N_real_centers,
                                                 M_real_centers, num_seeds=num_seeds)
        else:
            quats, t_seeds = batched_seeds_torch(points_1, points_2, N_real_surf_1,
                                                 M_real_surf_2, num_seeds=num_seeds)
        P = quats.size(1)
        q_best = quats.clone()
        t_best = t_seeds.clone()
    else:
        # Legacy translation-seeded path: coarse grid + top-k pruning (unchanged).
        q_grid, t_grid = build_coarse_grid(
            points_1, points_2, N_real_surf_1, M_real_surf_2, num_seeds=num_seeds,
            trans_centers_batch=trans_centers, trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans,
        )
        G = q_grid.size(1)
        ORI_CHUNK = 5_000  # Smaller chunks due to more data per evaluation
        coarse_score = torch.empty(BATCH, G, device=device, dtype=centers_1.dtype)
        for o0 in range(0, G, ORI_CHUNK):
            o1 = min(o0 + ORI_CHUNK, G)
            g_len = o1 - o0
            q_rep = q_grid[:, o0:o1].reshape(-1, 4).contiguous()
            t_rep = t_grid[:, o0:o1].reshape(-1, 3).contiguous()
            centers_w_H_2_exp = centers_w_H_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad_w_H, 3)
            centers_2_exp = centers_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_pad_centers, 3)
            points_2_exp = points_2.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, M_surf, 3)
            centers_w_H_2_t = apply_se3_transform(centers_w_H_2_exp, q_rep, t_rep)
            centers_2_t = apply_se3_transform(centers_2_exp, q_rep, t_rep)
            points_2_t = apply_se3_transform(points_2_exp, q_rep, t_rep)
            centers_w_H_1_exp = centers_w_H_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad_w_H, 3)
            centers_1_exp = centers_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_pad_centers, 3)
            points_1_exp = points_1.unsqueeze(1).expand(-1, g_len, -1, -1).reshape(-1, N_surf, 3)
            partial_charges_1_exp = partial_charges_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, N_pad_w_H)
            partial_charges_2_exp = partial_charges_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, M_pad_w_H)
            point_charges_1_exp = point_charges_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, N_surf)
            point_charges_2_exp = point_charges_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, M_surf)
            radii_1_exp = radii_1.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, N_pad_w_H)
            radii_2_exp = radii_2.unsqueeze(1).expand(-1, g_len, -1).reshape(-1, M_pad_w_H)
            VAA_exp = VAA.unsqueeze(1).expand(-1, g_len).reshape(-1)
            VBB_exp = VBB.unsqueeze(1).expand(-1, g_len).reshape(-1)
            N_real_exp = N_real_centers.repeat_interleave(g_len)
            M_real_exp = M_real_centers.repeat_interleave(g_len)
            N_atoms_exp = N_real_atoms_w_H_1.repeat_interleave(g_len)
            M_atoms_exp = M_real_atoms_w_H_2.repeat_interleave(g_len)
            N_surf_exp = N_real_surf_1.repeat_interleave(g_len)
            M_surf_exp = M_real_surf_2.repeat_interleave(g_len)
            scores_chunk = _batch_esp_combo_score(
                centers_w_H_1_exp, centers_w_H_2_t,
                centers_1_exp, centers_2_t,
                points_1_exp, points_2_t,
                partial_charges_1_exp, partial_charges_2_exp,
                point_charges_1_exp, point_charges_2_exp,
                radii_1_exp, radii_2_exp,
                alpha, lam, probe_radius, esp_weight,
                VAA_exp, VBB_exp, N_real_exp, M_real_exp,
                N_atoms_exp, M_atoms_exp, N_surf_exp, M_surf_exp)
            coarse_score[:, o0:o1] = scores_chunk.view(BATCH, g_len)
        best_idx = coarse_score.topk(k=topk, dim=1).indices
        q_best = torch.gather(q_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).clone()
        t_best = torch.gather(t_grid, 1, best_idx.unsqueeze(-1).expand(-1, -1, 3)).clone()
        P = topk

    # ------------------------------------------------------------------
    # 2) Fine optimization over ALL P poses
    # ------------------------------------------------------------------
    # For ESP combo, we use the shape gradient for optimization but score with combo
    q_k = q_best.reshape(-1, 4).contiguous()
    t_k = t_best.reshape(-1, 3).contiguous()

    # Expand all data for P poses
    centers_w_H_1_k = centers_w_H_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_w_H, 3)
    centers_w_H_2_k = centers_w_H_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_w_H, 3)
    centers_1_k = centers_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_pad_centers, 3)
    centers_2_k = centers_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_pad_centers, 3)
    points_1_k = points_1.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, N_surf, 3)
    points_2_k = points_2.unsqueeze(1).expand(-1, P, -1, -1).reshape(-1, M_surf, 3)
    partial_charges_1_k = partial_charges_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, N_pad_w_H)
    partial_charges_2_k = partial_charges_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, M_pad_w_H)
    point_charges_1_k = point_charges_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, N_surf)
    point_charges_2_k = point_charges_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, M_surf)
    radii_1_k = radii_1.unsqueeze(1).expand(-1, P, -1).reshape(-1, N_pad_w_H)
    radii_2_k = radii_2.unsqueeze(1).expand(-1, P, -1).reshape(-1, M_pad_w_H)

    N_k = N_real_centers.repeat_interleave(P)
    M_k = M_real_centers.repeat_interleave(P)
    N_atoms_k = N_real_atoms_w_H_1.repeat_interleave(P)
    M_atoms_k = M_real_atoms_w_H_2.repeat_interleave(P)
    N_surf_k = N_real_surf_1.repeat_interleave(P)
    M_surf_k = M_real_surf_2.repeat_interleave(P)
    VAA_k = VAA.repeat_interleave(P)
    VBB_k = VBB.repeat_interleave(P)
    VAA_plus_VBB = VAA_k + VBB_k

    # Adam state
    m_q = torch.zeros_like(q_k)
    v_q = torch.zeros_like(q_k)
    m_t = torch.zeros_like(t_k)
    v_t = torch.zeros_like(t_k)

    best_score = torch.full((len(q_k),), -float('inf'), device=device)
    best_q = q_k.clone()
    best_t = t_k.clone()

    prev_max_score = -float('inf')
    no_improve_count = 0

    # --- CUDA-graph fast path: capture the (shape-grad + 2 value-only ESP + 2 SE3) step and
    # replay it. esp_combo is compute-bound (heavy surface ESP), so this mainly UNIFIES the
    # path; the eager loop below remains the fallback for large P / capture failure. ---
    _graphed = None
    # Low work budget: esp_combo's per-step is heavy (2 surface-ESP kernels + 2 SE3 transforms)
    # and it runs full steps, so it crosses over sooner than the shape modes -> graph only the
    # clearly-winning small/mid-P regime, eager beyond (the baseline -- never a regression).
    if (_FINE_GRAPHS and centers_1_k.is_cuda and centers_1_k.dtype == torch.float32
            and len(q_k) <= graph_cap(N_pad_centers * M_pad_centers, budget=8_000_000)):
        try:
            _graphed = _run_graphed_esp_combo(
                centers_1_k.contiguous(), centers_2_k.contiguous(),
                centers_w_H_1_k.contiguous(), centers_w_H_2_k.contiguous(),
                points_1_k.contiguous(), points_2_k.contiguous(),
                partial_charges_1_k, partial_charges_2_k,
                point_charges_1_k, point_charges_2_k, radii_1_k, radii_2_k,
                N_k, M_k, N_atoms_k, M_atoms_k, N_surf_k, M_surf_k,
                VAA_k, VBB_k, VAA_plus_VBB, q_k, t_k,
                alpha, lam, probe_radius, esp_weight, lr,
                # NO blocked early-stop for esp_combo (es_patience defaults to 0 -> full
                # steps_fine). Its ESP landscape converges slowly AND its trajectory is
                # non-deterministic, so the trajectory-sensitive early-stop under-ran its
                # self-copies (0.8513 vs 0.8585). Full steps matches eager (which also runs
                # ~full here) and recovers parity; esp_combo is launch-heavy so it still wins.
                steps_fine,
                N_pad_centers, M_pad_centers, N_pad_w_H, M_pad_w_H, N_surf, M_surf, len(q_k))
        except Exception:
            _graphed = None
    if _graphed is not None:
        best_score, best_q, best_t = _graphed

    for step in range(steps_fine):
        if _graphed is not None:
            break
        # Shape value + gradient EVERY step -- this drives the trajectory AND the early-stop.
        # The shape "centers" of molecule 2 are NOT transformed here; the shape kernel rotates
        # them internally from (q_k, t_k), and its VAB is reused for the combo score below.
        VAB, dQ, dT = _overlap_in_chunks_volumetric(
            centers_1_k, centers_2_k, q_k, t_k,
            alpha=alpha, N_real=N_k, M_real=M_k)

        # Scale gradients for shape component (needed every step for the Adam update)
        denom = VAA_plus_VBB - VAB
        scale = VAA_plus_VBB / (denom * denom)

        # Sparse ESP: evaluate the full combo score (2 surface-ESP kernels + the 2 SE(3) cloud
        # transforms they need) only every _ESP_STRIDE steps, plus the final step. The pose path
        # is shape-driven, so off-step poses are still visited -- we just skip re-scoring them.
        # Track the best combo pose among the scored steps.
        if (step % _ESP_STRIDE == 0) or (step == steps_fine - 1):
            centers_w_H_2_t = apply_se3_transform(centers_w_H_2_k, q_k, t_k)
            points_2_t = apply_se3_transform(points_2_k, q_k, t_k)
            score = _batch_esp_combo_score(
                centers_w_H_1_k, centers_w_H_2_t,
                centers_1_k, centers_2_k,
                points_1_k, points_2_t,
                partial_charges_1_k, partial_charges_2_k,
                point_charges_1_k, point_charges_2_k,
                radii_1_k, radii_2_k,
                alpha, lam, probe_radius, esp_weight,
                VAA_k, VBB_k, N_k, M_k,
                N_atoms_k, M_atoms_k, N_surf_k, M_surf_k,
                VAB_shape=VAB)
            best_score, best_q, best_t = _update_best(score, q_k, t_k, best_score, best_q, best_t)

        # Early stopping check every 5 iterations to reduce GPU→CPU sync overhead
        if step % 5 == 0:
            current_max = best_score.max().item()
            if current_max - prev_max_score < early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= (ES_PATIENCE_OVERRIDE or early_stop_patience):
                    break
            else:
                no_improve_count = 0
                prev_max_score = current_max

        # Fused Adam with tangent-space projection (avoids intermediate dQ_tan tensor)
        fused_adam_qt_with_tangent_proj(
            q_k, t_k,
            -dQ * scale.unsqueeze(1) * (1 - esp_weight),
            -dT * scale.unsqueeze(1) * (1 - esp_weight),
            m_q, v_q, m_t, v_t, lr)

    # ------------------------------------------------------------------
    # 5) Gather final results
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, P)
    best = final_score.argmax(dim=1)
    sel = best + torch.arange(BATCH, device=device) * P

    return (final_score.flatten()[sel],
            best_q.view(BATCH, P, 4)[torch.arange(BATCH), best],
            best_t.view(BATCH, P, 3)[torch.arange(BATCH), best])


def fast_optimize_esp_combo_score_overlay(
        ref_centers_w_H: torch.Tensor,
        fit_centers_w_H: torch.Tensor,
        ref_centers: torch.Tensor,
        fit_centers: torch.Tensor,
        ref_points: torch.Tensor,
        fit_points: torch.Tensor,
        ref_partial_charges: torch.Tensor,
        fit_partial_charges: torch.Tensor,
        ref_surf_esp: torch.Tensor,
        fit_surf_esp: torch.Tensor,
        ref_radii: torch.Tensor,
        fit_radii: torch.Tensor,
        alpha: float,
        lam: float = 0.001,
        probe_radius: float = 1.0,
        esp_weight: float = 0.5,
        num_repeats: int = 50,
        trans_centers: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated ESP combo alignment.

    Drop-in replacement for optimize_esp_combo_score_overlay.

    Returns
    -------
    aligned_points : torch.Tensor (M, 3)
        Transformed fit surface points
    SE3_transform : torch.Tensor (4, 4)
        Best SE(3) transformation matrix
    score : torch.Tensor scalar
        Best combo score
    """
    if not check_gpu_available():
        from ...alignment._torch import optimize_esp_combo_score_overlay
        return optimize_esp_combo_score_overlay(
            ref_centers_w_H, fit_centers_w_H,
            ref_centers, fit_centers,
            ref_points, fit_points,
            ref_partial_charges, fit_partial_charges,
            ref_surf_esp, fit_surf_esp,
            ref_radii, fit_radii,
            alpha, lam, probe_radius, esp_weight,
            num_repeats,
            trans_centers=trans_centers,
            **kwargs)

    device = torch.device('cuda')

    # Move to GPU and add batch dimension
    ref_centers_w_H_gpu = ref_centers_w_H.to(device, dtype=torch.float32).unsqueeze(0)
    fit_centers_w_H_gpu = fit_centers_w_H.to(device, dtype=torch.float32).unsqueeze(0)
    ref_centers_gpu = ref_centers.to(device, dtype=torch.float32).unsqueeze(0)
    fit_centers_gpu = fit_centers.to(device, dtype=torch.float32).unsqueeze(0)
    ref_points_gpu = ref_points.to(device, dtype=torch.float32).unsqueeze(0)
    fit_points_gpu = fit_points.to(device, dtype=torch.float32).unsqueeze(0)
    ref_partial_gpu = ref_partial_charges.to(device, dtype=torch.float32).unsqueeze(0)
    fit_partial_gpu = fit_partial_charges.to(device, dtype=torch.float32).unsqueeze(0)
    ref_esp_gpu = ref_surf_esp.to(device, dtype=torch.float32).unsqueeze(0)
    fit_esp_gpu = fit_surf_esp.to(device, dtype=torch.float32).unsqueeze(0)
    ref_radii_gpu = ref_radii.to(device, dtype=torch.float32).unsqueeze(0)
    fit_radii_gpu = fit_radii.to(device, dtype=torch.float32).unsqueeze(0)

    trans_centers_batch = None
    trans_centers_real = None
    if trans_centers is not None:
        tc = trans_centers.to(device, dtype=torch.float32)
        trans_centers_batch = tc.unsqueeze(0)
        trans_centers_real = torch.tensor([tc.shape[0]], device=device, dtype=torch.int32)

    aligned_batch, q_best, t_best, score = fast_optimize_esp_combo_score_overlay_batch(
        ref_centers_w_H_gpu, fit_centers_w_H_gpu,
        ref_centers_gpu, fit_centers_gpu,
        ref_points_gpu, fit_points_gpu,
        ref_partial_gpu, fit_partial_gpu,
        ref_esp_gpu, fit_esp_gpu,
        ref_radii_gpu, fit_radii_gpu,
        alpha,
        lam=lam,
        probe_radius=probe_radius,
        esp_weight=esp_weight,
        N_real_atoms_w_H_1=torch.tensor([ref_centers_w_H.shape[0]], device=device, dtype=torch.int32),
        M_real_atoms_w_H_2=torch.tensor([fit_centers_w_H.shape[0]], device=device, dtype=torch.int32),
        N_real_centers=torch.tensor([ref_centers.shape[0]], device=device, dtype=torch.int32),
        M_real_centers=torch.tensor([fit_centers.shape[0]], device=device, dtype=torch.int32),
        N_real_surf_1=torch.tensor([ref_points.shape[0]], device=device, dtype=torch.int32),
        M_real_surf_2=torch.tensor([fit_points.shape[0]], device=device, dtype=torch.int32),
        trans_centers_batch=trans_centers_batch,
        trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk,
        steps_fine=steps_fine,
        lr=lr,
    )
    aligned = aligned_batch[0]

    # Build SE(3) matrix
    R = quaternion_to_rotation_matrix(q_best[0])
    SE3 = torch.eye(4, device=device)
    SE3[:3, :3] = R
    SE3[:3, 3] = t_best[0]

    return aligned.cpu(), SE3.cpu(), score[0].cpu()


def fast_optimize_esp_combo_score_overlay_batch(
        ref_centers_w_H_batch: torch.Tensor,
        fit_centers_w_H_batch: torch.Tensor,
        ref_centers_batch: torch.Tensor,
        fit_centers_batch: torch.Tensor,
        ref_points_batch: torch.Tensor,
        fit_points_batch: torch.Tensor,
        ref_partial_charges_batch: torch.Tensor,
        fit_partial_charges_batch: torch.Tensor,
        ref_surf_esp_batch: torch.Tensor,
        fit_surf_esp_batch: torch.Tensor,
        ref_radii_batch: torch.Tensor,
        fit_radii_batch: torch.Tensor,
        alpha: float,
        *,
        lam: float = 0.001,
        probe_radius: float = 1.0,
        esp_weight: float = 0.5,
        N_real_atoms_w_H_1: Optional[torch.Tensor] = None,
        M_real_atoms_w_H_2: Optional[torch.Tensor] = None,
        N_real_centers: Optional[torch.Tensor] = None,
        M_real_centers: Optional[torch.Tensor] = None,
        N_real_surf_1: Optional[torch.Tensor] = None,
        M_real_surf_2: Optional[torch.Tensor] = None,
        trans_centers_batch: Optional[torch.Tensor] = None,
        trans_centers_real: Optional[torch.Tensor] = None,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        num_seeds: int = 50,
        lr: float = 0.075) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU-accelerated batch ESP-combo alignment with padding-safe masks.

    Returns
    -------
    aligned_fit_points : torch.Tensor (B, M_surf_pad, 3)
    q_best : torch.Tensor (B, 4)
    t_best : torch.Tensor (B, 3)
    scores : torch.Tensor (B,)
    """
    device = ref_centers_batch.device
    BATCH = ref_centers_batch.shape[0]

    if N_real_centers is None:
        N_real_centers = ref_centers_batch.new_full(
            (BATCH,), ref_centers_batch.shape[1], dtype=torch.int32
        )
    if M_real_centers is None:
        M_real_centers = fit_centers_batch.new_full(
            (BATCH,), fit_centers_batch.shape[1], dtype=torch.int32
        )
    if N_real_atoms_w_H_1 is None:
        N_real_atoms_w_H_1 = ref_centers_w_H_batch.new_full(
            (BATCH,), ref_centers_w_H_batch.shape[1], dtype=torch.int32
        )
    if M_real_atoms_w_H_2 is None:
        M_real_atoms_w_H_2 = fit_centers_w_H_batch.new_full(
            (BATCH,), fit_centers_w_H_batch.shape[1], dtype=torch.int32
        )
    if N_real_surf_1 is None:
        N_real_surf_1 = ref_points_batch.new_full(
            (BATCH,), ref_points_batch.shape[1], dtype=torch.int32
        )
    if M_real_surf_2 is None:
        M_real_surf_2 = fit_points_batch.new_full(
            (BATCH,), fit_points_batch.shape[1], dtype=torch.int32
        )

    VAA = _self_overlap_chunks(ref_centers_batch, N_real_centers, alpha)
    VBB = _self_overlap_chunks(fit_centers_batch, M_real_centers, alpha)

    scores, q_best, t_best = coarse_fine_esp_combo_align_many(
        ref_centers_w_H_batch, fit_centers_w_H_batch,
        ref_centers_batch, fit_centers_batch,
        ref_points_batch, fit_points_batch,
        ref_partial_charges_batch, fit_partial_charges_batch,
        ref_surf_esp_batch, fit_surf_esp_batch,
        ref_radii_batch, fit_radii_batch,
        VAA, VBB,
        alpha=alpha,
        lam=lam,
        probe_radius=probe_radius,
        esp_weight=esp_weight,
        trans_centers=trans_centers_batch,
        trans_centers_real=trans_centers_real,
        num_repeats_per_trans=num_repeats_per_trans,
        topk=topk,
        steps_fine=steps_fine,
        num_seeds=num_seeds,
        lr=lr,
        N_real_centers=N_real_centers,
        M_real_centers=M_real_centers,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1,
        M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_surf_1=N_real_surf_1,
        M_real_surf_2=M_real_surf_2,
    )

    aligned_fit_points = apply_se3_transform(fit_points_batch, q_best, t_best)
    return aligned_fit_points, q_best, t_best, scores
