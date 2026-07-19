"""
Batched, multi-GPU GPU/Triton aligners for :class:`MoleculePair`.

Every function here is a *free function* that operates on duck-typed ``MoleculePair``
objects -- it only reads/writes their attributes. ``MoleculePair`` binds these as static
methods, so ``MoleculePair._align_batch_vol(pairs, ...)`` is the public seam.

This module must keep NO runtime dependency on ``_core`` (only a TYPE_CHECKING import), so
it imports cheaply and the worker processes stay picklable.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE
from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

if TYPE_CHECKING:                     # annotations only; never imported at runtime
    from shepherd_score.container._core import MoleculePair

from ._pad import _band_key, _subbatched_align, _scatter_fill, _BAND
from ._bucket import plan_buckets, PadSpec
from ._dispatch import _should_distribute, _run_distributed, _dev_idx


from .._modes import MODE_SEEDS as _MODE_SEEDS, MODE_STEPS as _MODE_STEPS

# ---- persistent, per-process caches (reused across calls) -------------------
_ALIGN_WORKSPACES: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
_INT_BUFFER_CACHE: dict[int, dict[str, torch.Tensor]] = {}


def _seeds_for(mode: str) -> int:
    """Per-mode default seed count from the mode registry."""
    return _MODE_SEEDS.get(mode, 50)


def _steps_for(mode: str) -> int:
    """Per-mode default fine-step count from the mode registry."""
    return _MODE_STEPS.get(mode, 50)


def _batch_upload(pairs, attr, src_fn, dtype, device):
    """Set ``p.<attr>`` for every ``p`` with ONE host concat + ONE ``.to(device)``
    view-split, instead of one ``torch.as_tensor(..., device=device)`` per pair.

    Rules this build MUST obey to stay bit-identical to a per-pair
    ``torch.as_tensor(src, dtype=..., device=...)``:

    (1) The dtype cast goes THROUGH TORCH (``from_numpy(flat).to(device=..., dtype=...)``),
        NEVER through numpy ``.astype``. numpy's float64->float32 rounding can differ from
        torch's by a ULP, and vol_esp / vol_and_surf_esp are fed float64 RDKit sources
        (``partial_charges``, ``mol.GetConformer().GetPositions()``), so a numpy-side cast
        would move their scores.
    (2) ``np.concatenate`` keeps the source dtype, so the concat itself never casts.
    (3) Each per-pair view is ``.clone()``d, so the cached ``_*_t`` tensor is its OWN
        contiguous allocation (separate storage, no shared-buffer aliasing across pairs).
    (4) Each call's ``src_fn`` must yield a single uniform-dtype attribute; mixed dtypes
        must be split into separate calls (as ``pharm_types``/int64 is).

    Only pairs whose ``<attr>`` is None (cold cache) get the batched upload; pairs already
    holding a same-device tensor are left untouched (so the screen path, which pre-warms
    these via build_fit, stays a no-op), and a wrong-device cached tensor is moved per-pair.
    ``src_fn`` performs any host-side numpy indexing (e.g. ``_nonH_atoms_idx``); that
    indexing must stay in numpy rather than becoming a device-side ``aten::to``.
    """
    cold = [p for p in pairs if getattr(p, attr, None) is None]
    if cold:
        arrs = [src_fn(p) for p in cold]                       # numpy, host, cheap
        sizes = [len(a) for a in arrs]
        flat = np.concatenate(arrs)                            # keep SOURCE dtype (no numpy cast)
        dev = torch.from_numpy(flat).to(device=device, dtype=dtype)  # torch does the cast (matches as_tensor)
        for p, t in zip(cold, dev.split(sizes)):
            setattr(p, attr, t.clone())                        # own allocation (no shared-buffer alias)
    for p in pairs:                                            # warm wrong-device path
        t = getattr(p, attr)
        if t.device != device:
            setattr(p, attr, t.to(device, non_blocking=True))


def _align_batch_vol(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
    """
    Batched alignment with workspace reuse & reduced per-pair transfers.
    """

    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol, pairs,
                                alpha=alpha, steps_fine=steps_fine)

    from shepherd_score.accel.drivers.shape import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.accel.drivers._common import batched_seeds_torch

    device = pairs[0].device
    # --- move coords once (skip if already there & right dtype) -------------
    # _ref_xyz_t/_fit_xyz_t are set in MoleculePair.__init__, so these are never cold here:
    # _batch_upload's cold branch is a no-op and only the per-pair device-move runs. atom_pos
    # is the same source the constructor uses, so a cold pair stays bit-identical too.
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)

    # --- result accumulators (GPU first; host copy only once) ---------------
    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # --- adaptive band bucketing (mode-agnostic cost-model planner) ----------
    _spec = PadSpec(merge={"ref": lambda p: p._ref_xyz_t.shape[0],
                           "fit": lambda p: p._fit_xyz_t.shape[0]},
                    seeds=_seeds_for("vol"))
    for _bk in plan_buckets(pairs, _spec, device):
        N_pad, M_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = _bk.K

        # ---- integer buffers (reuse) ---------------------------------------
        ib_key = (_dev_idx(device), K)
        int_buf = _INT_BUFFER_CACHE.get(ib_key)
        if int_buf is None:
            int_buf = {
                'N': torch.empty(K, dtype=torch.int32, device=device),
                'M': torch.empty(K, dtype=torch.int32, device=device),
            }
            _INT_BUFFER_CACHE[ib_key] = int_buf
        N_real = int_buf['N']
        M_real = int_buf['M']

        # Fill once from CPU lists (one H2D each) instead of per-element GPU writes
        ref_ts = [p._ref_xyz_t for p in bucket]
        fit_ts = [p._fit_xyz_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ts]
        m_list = [t.shape[0] for t in fit_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        # ---- workspaces (reuse & grow) -------------------------------------
        ws_key = (_dev_idx(device), N_pad, M_pad)
        ws = _ALIGN_WORKSPACES.get(ws_key)
        if ws is None or ws['ref'].shape[0] < K:
            # allocate at least K; allow some headroom (optional)
            ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
            _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
        ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
        fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

        # We only write the valid prefix; no need to .zero_ entire array
        # but we do clear the padding slices for deterministic results.
        ref_pad.zero_()
        fit_pad.zero_()
        # Batched scatter pad-fill: cat+scatter into the zero-init prefix -> bit-identical
        # to a per-pair pad_sequence fill, launch-count O(1) in K instead of O(K).
        _scatter_fill(ref_pad, ref_ts, n_list)
        _scatter_fill(fit_pad, fit_ts, m_list)

        # ---- self-overlaps (reused kernel) ---------------------------------
        # Screen query-reuse: when every pair in the bucket shares the SAME ref
        # tensor (the fast screen path sets one query ref on all pairs), the ref
        # self-overlap is identical for every row -- compute it once and broadcast
        # (bit-identical to the per-row kernel, but K-1 fewer self-overlaps).
        if K > 1 and all(p._ref_xyz_t is bucket[0]._ref_xyz_t for p in bucket):
            VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(K).contiguous()
        else:
            VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
        # memory-pressured chunking never re-pays the launch-bound seed-gen.
        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=_seeds_for("vol"))

        # ---- coarse + fine alignment, in GPU-memory-safe sub-batches -------
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(seeds_q[sl], seeds_t[sl]))
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("vol", N_pad, M_pad, _seeds_for("vol")), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    # ---- final host transfer (single) --------------------------------------
    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_noH = S
        p.sim_aligned_vol_noH = s

def _align_batch_surf(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
    """
    Batched alignment over *surface point clouds* using Gaussian-overlap
    surface similarity (ROCS-style), modeled after `_align_batch_vol`.

    Inputs
    ------
    pairs : list[MoleculePair]
        Each pair must provide surface point clouds for reference/fit:
        • prefer:   _ref_surf_t, _fit_surf_t  (torch.float32, (N/M, 3))
        • fallback: ref_molec.surf_pos, fit_molec.surf_pos (numpy, (N/M, 3))
    alpha : float
        Gaussian width parameter (same meaning as in `align_with_surf`).

    Side effects
    ------------
    Writes:
    • p.transform_surf      ← best SE(3) as 4×4 (via quaternion_to_SE3)
    • p.sim_aligned_surf    ← best Tanimoto surface score (float)
    """

    # Reuses the persistent per-process workspace / int-buffer caches (same ref/fit scratch
    # layout as _align_batch_vol; mutated in place -- accel.batch re-exports these objects --
    # never rebound). Buffers are zeroed before use, so cross-call and cross-mode reuse is safe.
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_surf, pairs,
                                alpha=alpha, steps_fine=steps_fine)

    from shepherd_score.accel.drivers.shape import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.accel.drivers._common import batched_seeds_torch

    device = pairs[0].device

    # --- ensure/prepare surface tensors on the right device --------------------
    # Validate the cold (build-from-numpy) pairs with the exact same guards/messages
    # as the per-pair build, preserving the "either missing -> rebuild both" coupling;
    # the two _batch_upload calls then do one batched H2D each (warm same-device pairs
    # untouched, warm wrong-device moved per-pair) -- bit-identical to the old loop.
    for p in pairs:
        if getattr(p, "_ref_surf_t", None) is None or getattr(p, "_fit_surf_t", None) is None:
            if not hasattr(p, "ref_molec") or not hasattr(p.ref_molec, "surf_pos"):
                raise ValueError(
                    "Surface points missing: MoleculePair must have _ref/_fit_surf_t "
                    "or ref_molec/fit_molec with .surf_pos."
                )
            if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
                raise ValueError("Surface points are None; cannot run _align_batch_surf.")
            # Rebuild BOTH when either is cold (matches the coupled original branch).
            p._ref_surf_t = None
            p._fit_surf_t = None
    _batch_upload(pairs, "_ref_surf_t", lambda p: p.ref_molec.surf_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_surf_t", lambda p: p.fit_molec.surf_pos, torch.float32, device)

    # --- result accumulators (GPU first; host copy only once) ------------------
    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # --- adaptive band bucketing (mode-agnostic cost-model planner) -----------
    _spec = PadSpec(merge={"ref": lambda p: p._ref_surf_t.shape[0],
                           "fit": lambda p: p._fit_surf_t.shape[0]},
                    seeds=_seeds_for("surf"))
    for _bk in plan_buckets(pairs, _spec, device):
        N_pad, M_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = _bk.K

        # ---- integer buffers (reuse) ------------------------------------------
        ib_key = (_dev_idx(device), K)
        int_buf = _INT_BUFFER_CACHE.get(ib_key)
        if int_buf is None:
            int_buf = {
                'N': torch.empty(K, dtype=torch.int32, device=device),
                'M': torch.empty(K, dtype=torch.int32, device=device),
            }
            _INT_BUFFER_CACHE[ib_key] = int_buf
        N_real = int_buf['N']
        M_real = int_buf['M']

        # Fill once from CPU lists (one H2D each) instead of per-element GPU scalar writes.
        ref_ts = [p._ref_surf_t for p in bucket]
        fit_ts = [p._fit_surf_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ts]
        m_list = [t.shape[0] for t in fit_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        # ---- workspaces (reuse & grow) ----------------------------------------
        ws_key = (_dev_idx(device), N_pad, M_pad)
        ws = _ALIGN_WORKSPACES.get(ws_key)
        if ws is None or ws['ref'].shape[0] < K:
            ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
            _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
        ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
        fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

        # Clear padding slices for determinism; write valid prefixes
        ref_pad.zero_()
        fit_pad.zero_()
        # Batched scatter pad-fill: cat+scatter into the zero-init prefix -> bit-identical
        # to a per-pair pad_sequence fill, launch-count O(1) in K instead of O(K).
        _scatter_fill(ref_pad, ref_ts, n_list)
        _scatter_fill(fit_pad, fit_ts, m_list)

        # ---- self-overlaps on surface point clouds ----------------------------
        # Screen query-reuse: a shared ref surface -> identical self-overlap per row;
        # compute once and broadcast (bit-identical). See _align_batch_vol.
        if K > 1 and all(p._ref_surf_t is bucket[0]._ref_surf_t for p in bucket):
            VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(K).contiguous()
        else:
            VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
        # memory-pressured chunking never re-pays the launch-bound seed-gen.
        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=_seeds_for("surf"))

        # ---- coarse + fine alignment (same engine as volumetric), processed in
        # GPU-memory-safe sub-batches sized per bucket (pairs are independent)
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(seeds_q[sl], seeds_t[sl]))
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("surf", N_pad, M_pad, _seeds_for("surf")), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    # ---- final host transfer (single) -----------------------------------------
    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_surf = S
        p.sim_aligned_surf = s

def _align_batch_surf_esp(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam: float,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched surface-ESP alignment using the fused ESP Triton kernel. ``surf_esp`` is
    the canonical name for the mode formerly called ``esp`` (legacy alias kept below).

    Side effects
    ------------
    Writes:
    - p.transform_surf_esp
    - p.sim_aligned_surf_esp
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_surf_esp, pairs,
                                alpha=alpha, lam=lam, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    from shepherd_score.score.constants import LAM_SCALING

    device = pairs[0].device
    lam_scaled = LAM_SCALING * lam

    # Ensure surface tensors (+ ESP) exist on device. Validate the cold pairs with the
    # same guards/messages and the same "any of the four missing -> rebuild all four"
    # coupling as the per-pair build; the four _batch_upload calls then do one batched
    # H2D per array (warm same-device untouched, warm wrong-device moved per-pair).
    for p in pairs:
        if (getattr(p, "_ref_surf_t", None) is None or getattr(p, "_fit_surf_t", None) is None
                or getattr(p, "_ref_surf_esp_t", None) is None
                or getattr(p, "_fit_surf_esp_t", None) is None):
            if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
                raise ValueError("Surface points are None; cannot run _align_batch_surf_esp.")
            if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
                raise ValueError("Surface ESP is None; cannot run _align_batch_surf_esp.")
            p._ref_surf_t = None
            p._fit_surf_t = None
            p._ref_surf_esp_t = None
            p._fit_surf_esp_t = None
    _batch_upload(pairs, "_ref_surf_t", lambda p: p.ref_molec.surf_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_surf_t", lambda p: p.fit_molec.surf_pos, torch.float32, device)
    _batch_upload(pairs, "_ref_surf_esp_t", lambda p: p.ref_molec.surf_esp, torch.float32, device)
    _batch_upload(pairs, "_fit_surf_esp_t", lambda p: p.fit_molec.surf_esp, torch.float32, device)

    # Translation centers must be available on device for trans_init. (Cold-only build,
    # no device-move -- matches the original; left unbatched to preserve those bytes.)
    if trans_init:
        for p in pairs:
            if getattr(p, "_ref_xyz_t", None) is None:
                p._ref_xyz_t = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=device)

    _esp_bucketed_align(
        pairs, alpha=alpha, lam_scaled=lam_scaled,
        ref_pts_attr="_ref_surf_t", fit_pts_attr="_fit_surf_t",
        ref_chg_attr="_ref_surf_esp_t", fit_chg_attr="_fit_surf_esp_t",
        out_tf_attr="transform_surf_esp", out_sc_attr="sim_aligned_surf_esp",
        subbatch_tag="surf_esp", trans_init=trans_init,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk,
        steps_fine=steps_fine, lr=lr,
    )

def _esp_bucketed_align(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam_scaled: float,
    ref_pts_attr: str,
    fit_pts_attr: str,
    ref_chg_attr: str,
    fit_chg_attr: str,
    out_tf_attr: str,
    out_sc_attr: str,
    subbatch_tag: str,
    trans_init: bool,
    num_repeats_per_trans: int,
    topk: int,
    steps_fine: int,
    lr: float,
) -> None:
    """
    Shared bucket -> pad -> fused-ESP-kernel -> SE(3) writeback core for the
    ESP-weighted batch aligners. ``_align_batch_surf_esp`` feeds surface points +
    surface ESP; ``_align_batch_vol_esp`` feeds (heavy-)atom coords + partial
    charges. The caller resolves ``lam_scaled`` (surf_esp applies ``LAM_SCALING``,
    vol_esp uses raw lam) and supplies the cached-tensor attribute names + the
    output attrs. Translation centers are always the ref atom coords
    (``_ref_xyz_t``), matching every per-pair ESP optimizer.
    """
    from shepherd_score.accel.drivers.esp import fast_optimize_ROCS_esp_overlay_batch

    device = pairs[0].device

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Bucket by padded point-cloud sizes; for translation-seeded mode also PARTITION by the
    # exact number of translation centers (it fixes the legacy 10*P+5 seed count, so it must
    # stay uniform within a bucket). Result-identical because the kernel masks padding to the
    # real counts.
    _tc_of = (lambda p: int(p._ref_xyz_t.shape[0])) if trans_init else (lambda p: 0)
    _spec = PadSpec(merge={"ref": lambda p: getattr(p, ref_pts_attr).shape[0],
                           "fit": lambda p: getattr(p, fit_pts_attr).shape[0]},
                    seeds=_seeds_for(subbatch_tag),
                    partition={"tc": _tc_of})

    # Workspace caches keyed by (N_pad, M_pad, K)
    workspaces: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
    int_buffers: dict[int, dict[str, torch.Tensor]] = {}

    for _bk in plan_buckets(pairs, _spec, device):
        N_pad, M_pad, tc = _bk.pad["ref"], _bk.pad["fit"], _bk.pad["tc"]
        bucket = _bk.members
        K = len(bucket)

        ib = int_buffers.get(K)
        if ib is None:
            ib = {
                "N": torch.empty(K, dtype=torch.int32, device=device),
                "M": torch.empty(K, dtype=torch.int32, device=device),
            }
            int_buffers[K] = ib
        N_real = ib["N"]
        M_real = ib["M"]

        ref_pts_ts = [getattr(p, ref_pts_attr) for p in bucket]
        fit_pts_ts = [getattr(p, fit_pts_attr) for p in bucket]
        ref_chg_ts = [getattr(p, ref_chg_attr) for p in bucket]
        fit_chg_ts = [getattr(p, fit_chg_attr) for p in bucket]
        n_list = [t.shape[0] for t in ref_pts_ts]
        m_list = [t.shape[0] for t in fit_pts_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        ws_key = (N_pad, M_pad, K)
        ws = workspaces.get(ws_key)
        if ws is None:
            ws = {
                "ref": torch.empty(K, N_pad, 3, device=device, dtype=torch.float32),
                "fit": torch.empty(K, M_pad, 3, device=device, dtype=torch.float32),
                "ref_c": torch.empty(K, N_pad, device=device, dtype=torch.float32),
                "fit_c": torch.empty(K, M_pad, device=device, dtype=torch.float32),
            }
            workspaces[ws_key] = ws

        ref_pad = ws["ref"]
        fit_pad = ws["fit"]
        ref_c_pad = ws["ref_c"]
        fit_c_pad = ws["fit_c"]

        ref_pad.zero_()
        fit_pad.zero_()
        ref_c_pad.zero_()
        fit_c_pad.zero_()
        # Batched scatter pad-fill: cat+scatter into the zero-init prefix -> bit-identical
        # to a per-pair slice-copy loop, O(1) launches in K instead of 4*K.
        _scatter_fill(ref_pad, ref_pts_ts, n_list)
        _scatter_fill(fit_pad, fit_pts_ts, m_list)
        _scatter_fill(ref_c_pad, ref_chg_ts, n_list)
        _scatter_fill(fit_c_pad, fit_chg_ts, m_list)

        trans_centers_batch = None
        trans_centers_real = None
        if trans_init:
            # NOTE: this bucket key uses exact translation center count (tc), so
            # the legacy seed count is identical for all pairs in this bucket.
            trans_centers_batch = torch.empty(K, tc, 3, device=device, dtype=torch.float32)
            for i, p in enumerate(bucket):
                trans_centers_batch[i] = p._ref_xyz_t
            trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

        # NOTE: seed-gen is intentionally NOT hoisted out of the sub-batch loop for the
        # ESP-family kernels. Unlike surf/vol (where hoisting helps under memory pressure),
        # the heavier per-chunk ESP footprint means held full-band seeds shave enough free
        # memory to tip the sub-batcher into OOM-retry thrash.
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
            tcr = trans_centers_real[sl] if trans_centers_real is not None else None
            _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                ref_pad[sl], fit_pad[sl], ref_c_pad[sl], fit_c_pad[sl],
                alpha=alpha, lam=lam_scaled,
                N_real=N_real[sl], M_real=M_real[sl],
                trans_centers_batch=tcb, trans_centers_real=tcr,
                num_repeats_per_trans=num_repeats_per_trans,
                num_seeds=_seeds_for(subbatch_tag),
                topk=topk, steps_fine=steps_fine, lr=lr,
            )
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=(subbatch_tag, N_pad, M_pad, _seeds_for(subbatch_tag)), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        setattr(p, out_tf_attr, S)
        setattr(p, out_sc_attr, s)

def _align_batch_vol_esp(
    pairs: list["MoleculePair"],
    *,
    lam: float,
    alpha: float = 0.81,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched volumetric-ESP alignment: heavy-atom Gaussian overlap weighted by
    partial charge. Reuses the fused ESP Triton kernel via ``_esp_bucketed_align``,
    fed atom coords + heavy-atom partial charges instead of surface points + ESP.
    Heavy-atom only (mirrors ``_align_batch_vol``); ``lam`` is RAW (no
    ``LAM_SCALING``) to match the per-pair ``align_with_vol_esp``.

    Side effects
    ------------
    Writes:
    - p.transform_vol_esp_noH
    - p.sim_aligned_vol_esp_noH
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_esp, pairs,
                                lam=lam, alpha=alpha, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    device = pairs[0].device

    # Ensure heavy-atom coords (+ heavy-atom partial charges) exist on device.
    # Each attr here follows the "cold -> build, warm wrong-device -> move"
    # pattern, which _batch_upload reproduces exactly (one batched H2D per array).
    for p in pairs:
        if p.ref_molec.partial_charges is None or p.fit_molec.partial_charges is None:
            raise ValueError("Partial charges are None; cannot run _align_batch_vol_esp.")

    # _ref_xyz_t / _fit_xyz_t (atom_pos) are kept only for trans_init seed
    # centers (matches per-pair vol_esp, which seeds from atom_pos).
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)

    # Gaussian centers MUST be the same strict heavy atoms as the charges
    # below (partial_charges[_nonH_atoms_idx]). atom_pos comes from
    # Chem.RemoveHs, which RETAINS some H (stereo/isotope/valence), so its
    # count can exceed len(_nonH_atoms_idx) -- a data-dependent off-by-a-few
    # that desyncs the bucketed scatter-fill (coords n_list != charge
    # length). Index the conformer by _nonH_atoms_idx so
    # centers stay 1:1 with the charges (bit-identical to atom_pos when
    # RemoveHs keeps no H, since center_to transforms both together). The
    # _nonH_atoms_idx numpy indexing stays on host inside the src_fn.
    _batch_upload(pairs, "_ref_xyz_noH_t",
                  lambda p: p.ref_molec.mol.GetConformer().GetPositions()[p.ref_molec._nonH_atoms_idx],
                  torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_noH_t",
                  lambda p: p.fit_molec.mol.GetConformer().GetPositions()[p.fit_molec._nonH_atoms_idx],
                  torch.float32, device)

    _batch_upload(pairs, "_ref_xyz_esp_t",
                  lambda p: p.ref_molec.partial_charges[p.ref_molec._nonH_atoms_idx],
                  torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_esp_t",
                  lambda p: p.fit_molec.partial_charges[p.fit_molec._nonH_atoms_idx],
                  torch.float32, device)

    _esp_bucketed_align(
        pairs, alpha=alpha, lam_scaled=lam,            # RAW lam (matches per-pair vol_esp)
        ref_pts_attr="_ref_xyz_noH_t", fit_pts_attr="_fit_xyz_noH_t",
        ref_chg_attr="_ref_xyz_esp_t", fit_chg_attr="_fit_xyz_esp_t",
        out_tf_attr="transform_vol_esp_noH", out_sc_attr="sim_aligned_vol_esp_noH",
        subbatch_tag="vol_esp", trans_init=trans_init,
        num_repeats_per_trans=num_repeats_per_trans, topk=topk,
        steps_fine=steps_fine, lr=lr,
    )

def _align_batch_vol_and_surf_esp(
    pairs: list["MoleculePair"],
    *,
    alpha: float,
    lam: float = 0.001,
    probe_radius: float = 1.0,
    esp_weight: float = 0.5,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched ESP-combo alignment (ShaEP-style) with padding-safe masks.
    ``vol_and_surf_esp`` is the canonical name for the mode formerly called
    ``esp_combo`` (legacy alias kept below).

    Side effects
    ------------
    Writes:
    - p.transform_vol_and_surf_esp
    - p.sim_aligned_vol_and_surf_esp
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_and_surf_esp, pairs,
                                alpha=alpha, lam=lam, probe_radius=probe_radius,
                                esp_weight=esp_weight, trans_init=trans_init,
                                num_repeats=num_repeats,
                                num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    from shepherd_score.accel.drivers.esp_combo import fast_optimize_esp_combo_score_overlay_batch

    device = pairs[0].device

    # Ensure required tensors exist on device. Each attr is independent (cold -> build,
    # warm wrong-device -> move), the same per-attr contract _batch_upload implements,
    # so the per-pair ensure loop collapses to one batched H2D per array name.
    for p in pairs:
        if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
            raise ValueError("Surface points are None; cannot run _align_batch_vol_and_surf_esp.")
        if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
            raise ValueError("Surface ESP is None; cannot run _align_batch_vol_and_surf_esp.")

    _batch_upload(pairs, "_ref_surf_t", lambda p: p.ref_molec.surf_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_surf_t", lambda p: p.fit_molec.surf_pos, torch.float32, device)
    _batch_upload(pairs, "_ref_surf_esp_t", lambda p: p.ref_molec.surf_esp, torch.float32, device)
    _batch_upload(pairs, "_fit_surf_esp_t", lambda p: p.fit_molec.surf_esp, torch.float32, device)
    _batch_upload(pairs, "_ref_centers_w_H_t",
                  lambda p: p.ref_molec.mol.GetConformer().GetPositions(), torch.float32, device)
    _batch_upload(pairs, "_fit_centers_w_H_t",
                  lambda p: p.fit_molec.mol.GetConformer().GetPositions(), torch.float32, device)
    _batch_upload(pairs, "_ref_partial_t", lambda p: p.ref_molec.partial_charges, torch.float32, device)
    _batch_upload(pairs, "_fit_partial_t", lambda p: p.fit_molec.partial_charges, torch.float32, device)
    _batch_upload(pairs, "_ref_radii_t", lambda p: p.ref_molec.radii, torch.float32, device)
    _batch_upload(pairs, "_fit_radii_t", lambda p: p.fit_molec.radii, torch.float32, device)

    if trans_init:
        _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
        _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Cost-model planner over the SIX cost dims (atoms-w-H, shape centers,
    # surface points -- ref & fit) + the exact tc partition. "centers" are volumetric atoms
    # when alpha==0.81, else the surface points (constant for the whole call). The work model
    # is the SUM of the three channel products (shape + the two ESP comparisons) -- exactly
    # why PadSpec.work is a callable, not a hardcoded N*M. Result-identical (kernel masks pad).
    _a0 = (alpha == 0.81)
    _tc_of = (lambda p: int(p._ref_xyz_t.shape[0])) if trans_init else (lambda p: 0)
    _spec = PadSpec(
        merge={
            "n_wH":   lambda p: p._ref_centers_w_H_t.shape[0],
            "m_wH":   lambda p: p._fit_centers_w_H_t.shape[0],
            "n_surf": lambda p: p._ref_surf_t.shape[0],
            "m_surf": lambda p: p._fit_surf_t.shape[0],
            "n_cent": (lambda p: p._ref_xyz_t.shape[0]) if _a0 else (lambda p: p._ref_surf_t.shape[0]),
            "m_cent": (lambda p: p._fit_xyz_t.shape[0]) if _a0 else (lambda p: p._fit_surf_t.shape[0]),
        },
        seeds=_seeds_for("vol_and_surf_esp"),
        partition={"tc": _tc_of},
        work=lambda pad: (pad["n_cent"] * pad["m_cent"]
                          + pad["n_surf"] * pad["m_wH"]
                          + pad["m_surf"] * pad["n_wH"]),
    )
    for _bk in plan_buckets(pairs, _spec, device):
        n_wH_pad, m_wH_pad = _bk.pad["n_wH"], _bk.pad["m_wH"]
        n_cent_pad, m_cent_pad = _bk.pad["n_cent"], _bk.pad["m_cent"]
        n_surf_pad, m_surf_pad = _bk.pad["n_surf"], _bk.pad["m_surf"]
        tc = _bk.pad["tc"]
        bucket = _bk.members
        K = len(bucket)

        # Allocate padded blocks
        centers_w_H_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
        centers_w_H_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)
        partial_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
        partial_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)
        radii_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
        radii_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)

        centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)

        points_1 = torch.zeros(K, n_surf_pad, 3, device=device, dtype=torch.float32)
        points_2 = torch.zeros(K, m_surf_pad, 3, device=device, dtype=torch.float32)
        point_charges_1 = torch.zeros(K, n_surf_pad, device=device, dtype=torch.float32)
        point_charges_2 = torch.zeros(K, m_surf_pad, device=device, dtype=torch.float32)

        N_real_atoms_w_H_1 = torch.empty(K, device=device, dtype=torch.int32)
        M_real_atoms_w_H_2 = torch.empty(K, device=device, dtype=torch.int32)
        N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        M_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        N_real_surf_1 = torch.empty(K, device=device, dtype=torch.int32)
        M_real_surf_2 = torch.empty(K, device=device, dtype=torch.int32)

        # Gather per-pair tensors once, then batched scatter-fill (was a per-pair
        # loop of ~10*K device slice-copies + per-element int scalar writes).
        ref_wH_ts = [p._ref_centers_w_H_t for p in bucket]
        fit_wH_ts = [p._fit_centers_w_H_t for p in bucket]
        ref_surf_ts = [p._ref_surf_t for p in bucket]
        fit_surf_ts = [p._fit_surf_t for p in bucket]
        n_wH_list = [t.shape[0] for t in ref_wH_ts]
        m_wH_list = [t.shape[0] for t in fit_wH_ts]
        n_surf_list = [t.shape[0] for t in ref_surf_ts]
        m_surf_list = [t.shape[0] for t in fit_surf_ts]

        N_real_atoms_w_H_1.copy_(torch.tensor(n_wH_list, dtype=torch.int32))
        M_real_atoms_w_H_2.copy_(torch.tensor(m_wH_list, dtype=torch.int32))
        N_real_surf_1.copy_(torch.tensor(n_surf_list, dtype=torch.int32))
        M_real_surf_2.copy_(torch.tensor(m_surf_list, dtype=torch.int32))

        _scatter_fill(centers_w_H_1, ref_wH_ts, n_wH_list)
        _scatter_fill(centers_w_H_2, fit_wH_ts, m_wH_list)
        _scatter_fill(partial_1, [p._ref_partial_t for p in bucket], n_wH_list)
        _scatter_fill(partial_2, [p._fit_partial_t for p in bucket], m_wH_list)
        _scatter_fill(radii_1, [p._ref_radii_t for p in bucket], n_wH_list)
        _scatter_fill(radii_2, [p._fit_radii_t for p in bucket], m_wH_list)
        _scatter_fill(points_1, ref_surf_ts, n_surf_list)
        _scatter_fill(points_2, fit_surf_ts, m_surf_list)
        _scatter_fill(point_charges_1, [p._ref_surf_esp_t for p in bucket], n_surf_list)
        _scatter_fill(point_charges_2, [p._fit_surf_esp_t for p in bucket], m_surf_list)

        # "centers" are volumetric atoms when alpha==0.81, else surface points
        # (constant for the whole call, so the branch is hoisted out of the bucket).
        if alpha == 0.81:
            ref_cent_ts = [p._ref_xyz_t for p in bucket]
            fit_cent_ts = [p._fit_xyz_t for p in bucket]
            n_cent_list = [t.shape[0] for t in ref_cent_ts]
            m_cent_list = [t.shape[0] for t in fit_cent_ts]
        else:
            ref_cent_ts, fit_cent_ts = ref_surf_ts, fit_surf_ts
            n_cent_list, m_cent_list = n_surf_list, m_surf_list
        _scatter_fill(centers_1, ref_cent_ts, n_cent_list)
        _scatter_fill(centers_2, fit_cent_ts, m_cent_list)
        N_real_centers.copy_(torch.tensor(n_cent_list, dtype=torch.int32))
        M_real_centers.copy_(torch.tensor(m_cent_list, dtype=torch.int32))

        trans_centers_batch = None
        trans_centers_real = None
        if trans_init:
            # All pairs in this bucket share exactly tc translation centers (bucket
            # key), so a single stack is equivalent to the per-pair fill.
            trans_centers_batch = torch.stack([p._ref_xyz_t for p in bucket])
            trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

        # The fine loop MUST go through _subbatched_align: the combo driver expands the bucket
        # by num_seeds internally, so on a large bucket its fine-loop tensors can exceed free
        # memory. _subbatched_align sizes each chunk from mem_get_info() free memory and
        # halves-and-retries on OOM, so the combo adapts to any GPU. Pairs are independent
        # (each result is its own max over seeds), so chunking is result-identical to one
        # big call.
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_esp_combo_score_overlay_batch(
                centers_w_H_1[sl], centers_w_H_2[sl], centers_1[sl], centers_2[sl],
                points_1[sl], points_2[sl], partial_1[sl], partial_2[sl],
                point_charges_1[sl], point_charges_2[sl], radii_1[sl], radii_2[sl],
                alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
                N_real_atoms_w_H_1=N_real_atoms_w_H_1[sl],
                M_real_atoms_w_H_2=M_real_atoms_w_H_2[sl],
                N_real_centers=N_real_centers[sl], M_real_centers=M_real_centers[sl],
                N_real_surf_1=N_real_surf_1[sl], M_real_surf_2=M_real_surf_2[sl],
                trans_centers_batch=(None if trans_centers_batch is None else trans_centers_batch[sl]),
                trans_centers_real=(None if trans_centers_real is None else trans_centers_real[sl]),
                num_repeats_per_trans=num_repeats_per_trans, topk=topk,
                steps_fine=steps_fine, lr=lr, num_seeds=_seeds_for("vol_and_surf_esp"))
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("vol_and_surf_esp", n_wH_pad, m_wH_pad, n_cent_pad,
                           m_cent_pad, n_surf_pad, m_surf_pad, _seeds_for("vol_and_surf_esp")),
            device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_and_surf_esp = S
        p.sim_aligned_vol_and_surf_esp = s

def _align_batch_pharm(
    pairs: list["MoleculePair"],
    *,
    similarity: _SIM_TYPE = "tanimoto",
    extended_points: bool = False,
    only_extended: bool = False,
    trans_init: bool = False,
    num_repeats: Optional[int] = None,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
):
    """
    Batched pharmacophore alignment using the fast GPU pathway when available.

    Writes per-pair:
    - p.transform_pharm (4x4)
    - p.sim_aligned_pharm (float)
    """
    if not pairs:
        return
    # Single source of truth: default the SO(3) multi-start count to the per-mode MODE_SEEDS
    # value (from the mode registry) so the fast batched kernel honors it. An explicit
    # num_repeats overrides.
    if num_repeats is None:
        num_repeats = _seeds_for("pharm")
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_pharm, pairs,
                                similarity=similarity, extended_points=extended_points,
                                only_extended=only_extended, trans_init=trans_init,
                                num_repeats=num_repeats, topk=topk,
                                steps_fine=steps_fine, lr=lr)

    # The batched pharm kernel runs on CUDA (Triton) and on CPU (numba) via device-driven
    # dispatch. There is no per-pair fallback, so on a CPU box without numba this raises rather
    # than silently producing a differently-computed result.
    if pairs[0].device.type != "cuda":
        try:
            import numba  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "batched pharm alignment on CPU requires numba; "
                "install numba (pip install numba) or run this mode on CUDA"
            ) from e

    from shepherd_score.accel.drivers.pharm import fast_optimize_pharm_overlay_batch

    device = pairs[0].device

    # Ensure per-pair cached tensors exist on the correct device. Each attr is the
    # independent "cold -> build, warm wrong-device -> move" pattern; one batched H2D
    # per array. pharm_types (int64) is its own _batch_upload call, kept out of the
    # float32 concat group.
    _batch_upload(pairs, "_ref_pharm_types_t", lambda p: p.ref_molec.pharm_types, torch.int64, device)
    _batch_upload(pairs, "_fit_pharm_types_t", lambda p: p.fit_molec.pharm_types, torch.int64, device)
    _batch_upload(pairs, "_ref_pharm_ancs_t", lambda p: p.ref_molec.pharm_ancs, torch.float32, device)
    _batch_upload(pairs, "_fit_pharm_ancs_t", lambda p: p.fit_molec.pharm_ancs, torch.float32, device)
    _batch_upload(pairs, "_ref_pharm_vecs_t", lambda p: p.ref_molec.pharm_vecs, torch.float32, device)
    _batch_upload(pairs, "_fit_pharm_vecs_t", lambda p: p.fit_molec.pharm_vecs, torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Merge by padded anchor counts, partition by exact tc.
    _tc_of = (lambda p: int(p._ref_pharm_ancs_t.shape[0])) if trans_init else (lambda p: 0)
    _spec = PadSpec(merge={"ref": lambda p: p._ref_pharm_ancs_t.shape[0],
                           "fit": lambda p: p._fit_pharm_ancs_t.shape[0]},
                    seeds=_seeds_for("pharm"),
                    partition={"tc": _tc_of})
    for _bk in plan_buckets(pairs, _spec, device):
        N_pad, M_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = len(bucket)

        ref_types = torch.zeros(K, N_pad, device=device, dtype=torch.int64)
        fit_types = torch.zeros(K, M_pad, device=device, dtype=torch.int64)
        ref_ancs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_ancs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)
        ref_vecs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
        fit_vecs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)

        N_real = torch.empty(K, device=device, dtype=torch.int32)
        M_real = torch.empty(K, device=device, dtype=torch.int32)

        ref_ancs_ts = [p._ref_pharm_ancs_t for p in bucket]
        fit_ancs_ts = [p._fit_pharm_ancs_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ancs_ts]
        m_list = [t.shape[0] for t in fit_ancs_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))   # one H2D, not per-element writes
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))
        # Batched scatter pad-fill: O(1) launches in K instead of 6*K device slice-copies.
        _scatter_fill(ref_types, [p._ref_pharm_types_t for p in bucket], n_list)
        _scatter_fill(fit_types, [p._fit_pharm_types_t for p in bucket], m_list)
        _scatter_fill(ref_ancs, ref_ancs_ts, n_list)
        _scatter_fill(fit_ancs, fit_ancs_ts, m_list)
        _scatter_fill(ref_vecs, [p._ref_pharm_vecs_t for p in bucket], n_list)
        _scatter_fill(fit_vecs, [p._fit_pharm_vecs_t for p in bucket], m_list)

        trans_centers_batch = ref_ancs if trans_init else None
        trans_centers_real = N_real if trans_init else None

        # GPU-memory-safe sub-batching per bucket (independent pairs). Pharm's
        # analytical fine loop has the largest (~N_pad*M_pad) footprint, so
        # this is where the dynamic cap matters most.
        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
            tcr = trans_centers_real[sl] if trans_centers_real is not None else None
            _, _, q, t, sc = fast_optimize_pharm_overlay_batch(
                ref_types[sl], fit_types[sl], ref_ancs[sl], fit_ancs[sl],
                ref_vecs[sl], fit_vecs[sl],
                similarity=similarity, extended_points=extended_points,
                only_extended=only_extended, num_repeats=num_repeats,
                trans_centers_batch=tcb, trans_centers_real=tcr,
                num_repeats_per_trans=10, N_real=N_real[sl], M_real=M_real[sl],
                topk=topk, steps_fine=steps_fine, lr=lr,
            )
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("pharm", N_pad, M_pad, num_repeats), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_pharm = S
        p.sim_aligned_pharm = s


def _align_batch_vol_color(
    pairs: list["MoleculePair"],
    *,
    alpha: float = 0.81,
    color_weight: float = 0.5,
    trans_init: bool = False,
    num_repeats: int = 50,
    num_repeats_per_trans: int = 10,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """
    Batched ROCS/ROSHAMBO-style vol_color alignment (atom Gaussian shape + directionless
    pharmacophore color). Shape uses the fused volumetric kernel (Triton on CUDA, numba on
    CPU); color uses the pure-torch directionless scorer. SE(3) gradient is shape-driven
    (FastROCS-style: shape-optimized, color-scored), matching the vol_and_surf_esp pattern.

    Writes per-pair: ``p.transform_vol_color`` (4x4), ``p.sim_aligned_vol_color`` (float).
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_color, pairs,
                                alpha=alpha, color_weight=color_weight, trans_init=trans_init,
                                num_repeats=num_repeats, num_repeats_per_trans=num_repeats_per_trans,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    for p in pairs:
        if p.ref_molec.pharm_types is None or p.fit_molec.pharm_types is None:
            raise ValueError("Pharmacophores are None; cannot run _align_batch_vol_color.")

    from shepherd_score.accel.drivers.vol_color import (
        fast_optimize_vol_color_overlay_batch, _PHARM_PAD_TYPE)

    device = pairs[0].device

    # Each attr is the independent "cold -> build, warm wrong-device -> move" pattern;
    # one batched H2D per array. pharm_types (int64) stays in its own _batch_upload
    # call, separate from the float32 coordinate/anchor concat group.
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_ref_pharm_types_t", lambda p: p.ref_molec.pharm_types, torch.int64, device)
    _batch_upload(pairs, "_fit_pharm_types_t", lambda p: p.fit_molec.pharm_types, torch.int64, device)
    _batch_upload(pairs, "_ref_pharm_ancs_t", lambda p: p.ref_molec.pharm_ancs, torch.float32, device)
    _batch_upload(pairs, "_fit_pharm_ancs_t", lambda p: p.fit_molec.pharm_ancs, torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Collapsed bucket key: merge by shape-center bands only, partition by exact tc. The
    # pharmacophore-anchor counts are deliberately NOT keyed -- their pad slots are Dummy-typed
    # (never scored) and masked by N_real_pharm, so two pairs that share a shape bucket but
    # differ in feature count stay together and their anchors are padded to the bucket's max
    # feature band (recomputed below). RESULT-IDENTICAL (the masked color kernel ignores the
    # extra slots), and it keeps vol_color on the vol shape-bucket granularity: keying the
    # anchor counts splits each shape bucket by two extra dims and starves occupancy.
    _tc_of = (lambda p: int(p._ref_xyz_t.shape[0])) if trans_init else (lambda p: 0)
    _spec = PadSpec(merge={"ref": lambda p: p._ref_xyz_t.shape[0],
                           "fit": lambda p: p._fit_xyz_t.shape[0]},
                    seeds=_seeds_for("vol_color"),
                    partition={"tc": _tc_of})
    for _bk in plan_buckets(pairs, _spec, device):
        n_cent_pad, m_cent_pad, tc = _bk.pad["ref"], _bk.pad["fit"], _bk.pad["tc"]
        bucket = _bk.members
        K = len(bucket)

        ref_cent_ts = [p._ref_xyz_t for p in bucket]
        fit_cent_ts = [p._fit_xyz_t for p in bucket]
        n_cent_list = [t.shape[0] for t in ref_cent_ts]
        m_cent_list = [t.shape[0] for t in fit_cent_ts]
        ref_anc_ts = [p._ref_pharm_ancs_t for p in bucket]
        fit_anc_ts = [p._fit_pharm_ancs_t for p in bucket]
        n_ph_list = [t.shape[0] for t in ref_anc_ts]
        m_ph_list = [t.shape[0] for t in fit_anc_ts]
        # Anchors are not keyed -> pad to this bucket's max feature band (>= every row's count).
        n_ph_pad = _band_key(max(n_ph_list))
        m_ph_pad = _band_key(max(m_ph_list))

        centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)
        # Pad pharm type slots with Dummy (index 8) so padded anchors are never scored.
        ref_types = torch.full((K, n_ph_pad), _PHARM_PAD_TYPE, device=device, dtype=torch.int64)
        fit_types = torch.full((K, m_ph_pad), _PHARM_PAD_TYPE, device=device, dtype=torch.int64)
        ref_ancs = torch.zeros(K, n_ph_pad, 3, device=device, dtype=torch.float32)
        fit_ancs = torch.zeros(K, m_ph_pad, 3, device=device, dtype=torch.float32)

        N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        M_real_centers = torch.empty(K, device=device, dtype=torch.int32)

        N_real_centers.copy_(torch.tensor(n_cent_list, dtype=torch.int32))
        M_real_centers.copy_(torch.tensor(m_cent_list, dtype=torch.int32))
        N_real_pharm = torch.tensor(n_ph_list, device=device, dtype=torch.int32)
        M_real_pharm = torch.tensor(m_ph_list, device=device, dtype=torch.int32)

        _scatter_fill(centers_1, ref_cent_ts, n_cent_list)
        _scatter_fill(centers_2, fit_cent_ts, m_cent_list)
        _scatter_fill(ref_types, [p._ref_pharm_types_t for p in bucket], n_ph_list)
        _scatter_fill(fit_types, [p._fit_pharm_types_t for p in bucket], m_ph_list)
        _scatter_fill(ref_ancs, ref_anc_ts, n_ph_list)
        _scatter_fill(fit_ancs, fit_anc_ts, m_ph_list)

        trans_centers_batch = None
        trans_centers_real = None
        if trans_init:
            trans_centers_batch = torch.stack([p._ref_xyz_t for p in bucket])
            trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

        _, q_batch, t_batch, scores = fast_optimize_vol_color_overlay_batch(
            centers_1, centers_2, ref_types, fit_types, ref_ancs, fit_ancs,
            alpha=alpha, color_weight=color_weight,
            N_real_centers=N_real_centers, M_real_centers=M_real_centers,
            N_real_pharm=N_real_pharm, M_real_pharm=M_real_pharm,
            trans_centers_batch=trans_centers_batch, trans_centers_real=trans_centers_real,
            num_repeats_per_trans=num_repeats_per_trans,
            topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=_seeds_for("vol_color"),
        )

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)
    scores_list = scores_cpu.tolist()
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_color = S
        p.sim_aligned_vol_color = s


def _align_batch_vol_tversky(pairs: list["MoleculePair"], *, alpha: float = 0.81,
                             tversky_alpha: float = 0.95, tversky_beta: float = 0.05,
                             steps_fine: int = 100):
    """Batched asymmetric ``vol_tversky`` shape alignment.

    Same heavy-atom Gaussian shape machinery as ``_align_batch_vol`` (reuses the shape
    value+grad kernel, the shared workspace/int-buffer caches, and the CUDA-graph fine loop),
    but scored with an asymmetric Tversky reduction
    ``AB / (AB + tversky_alpha*(AA-AB) + tversky_beta*(BB-AB))`` instead of Tanimoto. ``AA``/``BB``
    are pose-invariant and precomputed once per pair via the reused self-overlap kernel; only
    ``AB`` flows through the fine loop. The Tversky score is NOT bounded to [0, 1] and is never
    clamped here.

    Side effects: writes ``p.transform_vol_tversky`` / ``p.sim_aligned_vol_tversky``.
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_tversky, pairs,
                                alpha=alpha, tversky_alpha=tversky_alpha,
                                tversky_beta=tversky_beta, steps_fine=steps_fine)

    from shepherd_score.accel.drivers.vol_tversky import (
        coarse_fine_align_many_tversky, _self_overlap_in_chunks)
    from shepherd_score.accel.drivers._common import batched_seeds_torch

    device = pairs[0].device
    # Heavy-atom coords are the same source as _align_batch_vol (atom_pos == get_positions(no_H)),
    # so _ref_xyz_t/_fit_xyz_t are shared (already set in MoleculePair.__init__).
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    _seeds = _seeds_for("vol_tversky")
    _spec = PadSpec(merge={"ref": lambda p: p._ref_xyz_t.shape[0],
                           "fit": lambda p: p._fit_xyz_t.shape[0]},
                    seeds=_seeds)
    for _bk in plan_buckets(pairs, _spec, device):
        N_pad, M_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = _bk.K

        ib_key = (_dev_idx(device), K)
        int_buf = _INT_BUFFER_CACHE.get(ib_key)
        if int_buf is None:
            int_buf = {
                'N': torch.empty(K, dtype=torch.int32, device=device),
                'M': torch.empty(K, dtype=torch.int32, device=device),
            }
            _INT_BUFFER_CACHE[ib_key] = int_buf
        N_real = int_buf['N']
        M_real = int_buf['M']

        ref_ts = [p._ref_xyz_t for p in bucket]
        fit_ts = [p._fit_xyz_t for p in bucket]
        n_list = [t.shape[0] for t in ref_ts]
        m_list = [t.shape[0] for t in fit_ts]
        N_real.copy_(torch.tensor(n_list, dtype=torch.int32))
        M_real.copy_(torch.tensor(m_list, dtype=torch.int32))

        ws_key = (_dev_idx(device), N_pad, M_pad)
        ws = _ALIGN_WORKSPACES.get(ws_key)
        if ws is None or ws['ref'].shape[0] < K:
            ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
            _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
        ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
        fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

        ref_pad.zero_()
        fit_pad.zero_()
        _scatter_fill(ref_pad, ref_ts, n_list)
        _scatter_fill(fit_pad, fit_ts, m_list)

        # Self-overlaps (reused shape kernel). AA/BB are pose-invariant Tversky inputs.
        if K > 1 and all(p._ref_xyz_t is bucket[0]._ref_xyz_t for p in bucket):
            VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(K).contiguous()
        else:
            VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=_seeds)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many_tversky(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                steps_fine=steps_fine, seeds=(seeds_q[sl], seeds_t[sl]))
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("vol_tversky", N_pad, M_pad, _seeds), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_tversky = S
        p.sim_aligned_vol_tversky = s


def _align_batch_esp_field(
    pairs: list["MoleculePair"],
    *,
    field_weight: float = 0.5,
    alpha: float = 0.81,
    alpha_field: float = 0.81,
    lam: float = 0.1,
    num_repeats: int = 50,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """Batched Cresset-style ``esp_field`` alignment: atom-centred Gaussian *shape* +
    signed ESP *field-point* overlap, blended ``(1-field_weight)*shape + field_weight*field``.

    Two INDEPENDENT point sets are padded per bucket: the heavy-atom centres (own N_real, the
    bucket merge key, reusing the shape kernel) and the derived ESP field points (own M_real,
    variable length possibly 0, padded to the bucket's max field-point band and masked by the
    real count -- reusing the fused ESP kernel with the field-point SIGNS as its charges). Both
    fit sets transform under the same pose; the driver descends on the JOINT weighted gradient.

    Side effects: writes ``p.transform_esp_field`` / ``p.sim_aligned_esp_field``.
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_esp_field, pairs,
                                field_weight=field_weight, alpha=alpha,
                                alpha_field=alpha_field, lam=lam, num_repeats=num_repeats,
                                topk=topk, steps_fine=steps_fine, lr=lr)

    from shepherd_score.accel.drivers.esp_field import fast_optimize_esp_field_overlay_batch

    device = pairs[0].device

    # Heavy-atom centres (shape channel) -- same source as _align_batch_vol.
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)
    # ESP field points + signs (field channel). Derived point set (variable M, possibly 0);
    # get_field_points() returns (pos (M,3) float32, sign (M,) float32).
    _batch_upload(pairs, "_ref_fp_pos_t", lambda p: p.ref_molec.get_field_points()[0], torch.float32, device)
    _batch_upload(pairs, "_fit_fp_pos_t", lambda p: p.fit_molec.get_field_points()[0], torch.float32, device)
    _batch_upload(pairs, "_ref_fp_sign_t", lambda p: p.ref_molec.get_field_points()[1], torch.float32, device)
    _batch_upload(pairs, "_fit_fp_sign_t", lambda p: p.fit_molec.get_field_points()[1], torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Merge by shape-center bands only (like vol_color). Field-point counts are NOT keyed -- the
    # ESP kernel masks padded field slots by the real count, so two pairs sharing a shape bucket
    # but differing in field-point count stay together and their field points are padded to the
    # bucket's max field band (recomputed below). Result-identical.
    _spec = PadSpec(merge={"ref": lambda p: p._ref_xyz_t.shape[0],
                           "fit": lambda p: p._fit_xyz_t.shape[0]},
                    seeds=_seeds_for("esp_field"))
    for _bk in plan_buckets(pairs, _spec, device):
        n_cent_pad, m_cent_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = len(bucket)

        ref_cent_ts = [p._ref_xyz_t for p in bucket]
        fit_cent_ts = [p._fit_xyz_t for p in bucket]
        n_cent_list = [t.shape[0] for t in ref_cent_ts]
        m_cent_list = [t.shape[0] for t in fit_cent_ts]
        ref_fp_ts = [p._ref_fp_pos_t for p in bucket]
        fit_fp_ts = [p._fit_fp_pos_t for p in bucket]
        n_fp_list = [t.shape[0] for t in ref_fp_ts]
        m_fp_list = [t.shape[0] for t in fit_fp_ts]
        # Field points not keyed -> pad to this bucket's max field band (>= 1 slot so the
        # kernel always sees a well-formed tensor; zeroed slots are masked by the real count).
        n_fp_pad = _band_key(max(n_fp_list)) or _BAND
        m_fp_pad = _band_key(max(m_fp_list)) or _BAND

        centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)
        fp_pos_1 = torch.zeros(K, n_fp_pad, 3, device=device, dtype=torch.float32)
        fp_pos_2 = torch.zeros(K, m_fp_pad, 3, device=device, dtype=torch.float32)
        fp_sign_1 = torch.zeros(K, n_fp_pad, device=device, dtype=torch.float32)
        fp_sign_2 = torch.zeros(K, m_fp_pad, device=device, dtype=torch.float32)

        N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        M_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        N_real_centers.copy_(torch.tensor(n_cent_list, dtype=torch.int32))
        M_real_centers.copy_(torch.tensor(m_cent_list, dtype=torch.int32))
        N_real_fp = torch.tensor(n_fp_list, device=device, dtype=torch.int32)
        M_real_fp = torch.tensor(m_fp_list, device=device, dtype=torch.int32)

        _scatter_fill(centers_1, ref_cent_ts, n_cent_list)
        _scatter_fill(centers_2, fit_cent_ts, m_cent_list)
        _scatter_fill(fp_pos_1, ref_fp_ts, n_fp_list)
        _scatter_fill(fp_pos_2, fit_fp_ts, m_fp_list)
        _scatter_fill(fp_sign_1, [p._ref_fp_sign_t for p in bucket], n_fp_list)
        _scatter_fill(fp_sign_2, [p._fit_fp_sign_t for p in bucket], m_fp_list)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_esp_field_overlay_batch(
                centers_1[sl], centers_2[sl], fp_pos_1[sl], fp_pos_2[sl],
                fp_sign_1[sl], fp_sign_2[sl],
                alpha=alpha, alpha_field=alpha_field, lam=lam, field_weight=field_weight,
                N_real_centers=N_real_centers[sl], M_real_centers=M_real_centers[sl],
                N_real_fp=N_real_fp[sl], M_real_fp=M_real_fp[sl],
                topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=_seeds_for("esp_field"))
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("esp_field", n_cent_pad, m_cent_pad, n_fp_pad, m_fp_pad,
                           _seeds_for("esp_field")), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_esp_field = S
        p.sim_aligned_esp_field = s


def _align_batch_vol_lipo(
    pairs: list["MoleculePair"],
    *,
    lipo_weight: float = 0.5,
    alpha: float = 0.81,
    lam: float = 0.1,
    num_repeats: int = 50,
    topk: int = 30,
    steps_fine: int = 100,
    lr: float = 0.075,
) -> None:
    """Batched ``vol_lipo`` alignment: atom-centred Gaussian *shape* (volume) + per-atom
    *lipophilicity* overlap, blended ``(1-lipo_weight)*shape + lipo_weight*lipo``.

    Two INDEPENDENT point sets are padded per bucket, exactly as the reference passes
    ``ref_centers``/``fit_centers`` apart from ``ref_lipo_pos``/``fit_lipo_pos``:
      * the SHAPE centres -- ``atom_pos`` (the RemoveHs coordinate set, own N_real, the
        bucket merge key, reusing the shape kernel), and
      * the LIPO centres -- the TRUE-heavy atom coordinates
        ``mol.GetConformer().GetPositions()[_nonH_atoms_idx]`` (own M_real, reusing the fused
        ESP kernel with the per-atom Crippen logP as its charges, matched by value).
    The two sets CAN DIFFER IN LENGTH on isotope-labelled molecules (RemoveHs retains e.g.
    deuterium, so ``atom_pos`` is longer than the strict-heavy lipo set) -- the separate
    paddings keep them from desyncing. Both fit sets transform under the same pose; the driver
    descends on the JOINT weighted gradient. ``lam`` is RAW (atom-centred, no LAM_SCALING),
    matching the per-pair ``align_with_vol_lipo``.

    Side effects: writes ``p.transform_vol_lipo`` / ``p.sim_aligned_vol_lipo``.
    """
    if not pairs:
        return
    if _should_distribute(pairs):
        return _run_distributed(_align_batch_vol_lipo, pairs,
                                lipo_weight=lipo_weight, alpha=alpha, lam=lam,
                                num_repeats=num_repeats, topk=topk,
                                steps_fine=steps_fine, lr=lr)

    from shepherd_score.accel.drivers.vol_lipo import fast_optimize_vol_lipo_overlay_batch

    device = pairs[0].device

    # Shape centres (shape channel) -- same source as _align_batch_vol (RemoveHs atom_pos).
    _batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, device)
    _batch_upload(pairs, "_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32, device)
    # Lipo centres + per-atom logP (lipo channel). TRUE-heavy basis (get_lipo_positions()),
    # NOT atom_pos -- indexes the with-H conformer by _nonH_atoms_idx so positions stay 1:1 with
    # the logP even when RemoveHs retained an H (the retained-H trap). Both Molecule and the
    # RDKit-free MoleculeProfile expose get_lipo_positions() / get_lipophilicity(no_H=True), so
    # this duck-types identically on either input.
    _batch_upload(pairs, "_ref_lipo_pos_t", lambda p: p.ref_molec.get_lipo_positions(), torch.float32, device)
    _batch_upload(pairs, "_fit_lipo_pos_t", lambda p: p.fit_molec.get_lipo_positions(), torch.float32, device)
    _batch_upload(pairs, "_ref_lipo_t", lambda p: p.ref_molec.get_lipophilicity(no_H=True), torch.float32, device)
    _batch_upload(pairs, "_fit_lipo_t", lambda p: p.fit_molec.get_lipophilicity(no_H=True), torch.float32, device)

    all_pairs: list[MoleculePair] = []
    all_scores: list[torch.Tensor] = []
    all_q: list[torch.Tensor] = []
    all_t: list[torch.Tensor] = []

    # Merge by shape-centre bands only (like esp_field / vol_color). Lipo-centre counts are NOT
    # keyed -- the ESP kernel masks padded lipo slots by the real count, so two pairs sharing a
    # shape bucket but differing in lipo-centre count stay together and their lipo centres are
    # padded to the bucket's max lipo band (recomputed below). Result-identical.
    _spec = PadSpec(merge={"ref": lambda p: p._ref_xyz_t.shape[0],
                           "fit": lambda p: p._fit_xyz_t.shape[0]},
                    seeds=_seeds_for("vol_lipo"))
    for _bk in plan_buckets(pairs, _spec, device):
        n_cent_pad, m_cent_pad = _bk.pad["ref"], _bk.pad["fit"]
        bucket = _bk.members
        K = len(bucket)

        ref_cent_ts = [p._ref_xyz_t for p in bucket]
        fit_cent_ts = [p._fit_xyz_t for p in bucket]
        n_cent_list = [t.shape[0] for t in ref_cent_ts]
        m_cent_list = [t.shape[0] for t in fit_cent_ts]
        ref_lipo_ts = [p._ref_lipo_pos_t for p in bucket]
        fit_lipo_ts = [p._fit_lipo_pos_t for p in bucket]
        n_lipo_list = [t.shape[0] for t in ref_lipo_ts]
        m_lipo_list = [t.shape[0] for t in fit_lipo_ts]
        # Lipo centres not keyed -> pad to this bucket's max lipo band (>= 1 slot so the kernel
        # always sees a well-formed tensor; zeroed slots are masked by the real count).
        n_lipo_pad = _band_key(max(n_lipo_list)) or _BAND
        m_lipo_pad = _band_key(max(m_lipo_list)) or _BAND

        centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)
        lipo_pos_1 = torch.zeros(K, n_lipo_pad, 3, device=device, dtype=torch.float32)
        lipo_pos_2 = torch.zeros(K, m_lipo_pad, 3, device=device, dtype=torch.float32)
        lipo_1 = torch.zeros(K, n_lipo_pad, device=device, dtype=torch.float32)
        lipo_2 = torch.zeros(K, m_lipo_pad, device=device, dtype=torch.float32)

        N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        M_real_centers = torch.empty(K, device=device, dtype=torch.int32)
        N_real_centers.copy_(torch.tensor(n_cent_list, dtype=torch.int32))
        M_real_centers.copy_(torch.tensor(m_cent_list, dtype=torch.int32))
        N_real_lipo = torch.tensor(n_lipo_list, device=device, dtype=torch.int32)
        M_real_lipo = torch.tensor(m_lipo_list, device=device, dtype=torch.int32)

        _scatter_fill(centers_1, ref_cent_ts, n_cent_list)
        _scatter_fill(centers_2, fit_cent_ts, m_cent_list)
        _scatter_fill(lipo_pos_1, ref_lipo_ts, n_lipo_list)
        _scatter_fill(lipo_pos_2, fit_lipo_ts, m_lipo_list)
        _scatter_fill(lipo_1, [p._ref_lipo_t for p in bucket], n_lipo_list)
        _scatter_fill(lipo_2, [p._fit_lipo_t for p in bucket], m_lipo_list)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_vol_lipo_overlay_batch(
                centers_1[sl], centers_2[sl], lipo_pos_1[sl], lipo_pos_2[sl],
                lipo_1[sl], lipo_2[sl],
                alpha=alpha, lam=lam, lipo_weight=lipo_weight,
                N_real_centers=N_real_centers[sl], M_real_centers=M_real_centers[sl],
                N_real_lipo=N_real_lipo[sl], M_real_lipo=M_real_lipo[sl],
                topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=_seeds_for("vol_lipo"))
            return sc, q, t
        scores, q_batch, t_batch = _subbatched_align(
            _proc, K, key=("vol_lipo", n_cent_pad, m_cent_pad, n_lipo_pad, m_lipo_pad,
                           _seeds_for("vol_lipo")), device=device)

        all_pairs.extend(bucket)
        all_scores.append(scores)
        all_q.append(q_batch)
        all_t.append(t_batch)

    scores_cpu = torch.cat(all_scores).cpu()
    q_cpu = torch.cat(all_q).cpu()
    t_cpu = torch.cat(all_t).cpu()

    SE3_all = quaternions_to_SE3_batch(q_cpu, t_cpu)        # batched
    scores_list = scores_cpu.tolist()                       # one C call
    for p, s, S in zip(all_pairs, scores_list, SE3_all):
        p.transform_vol_lipo = S
        p.sim_aligned_vol_lipo = s


# --- legacy mode aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp) ----------
# Same function objects, so ``__name__`` stays canonical -- the multi-GPU dispatch
# (``align_fn.__name__.replace("_align_batch_", "")``) and _MODE_SPEC lookup are
# unaffected. Kept so external code / pickles referencing the old names still resolve.
_align_batch_esp = _align_batch_surf_esp
_align_batch_esp_combo = _align_batch_vol_and_surf_esp
