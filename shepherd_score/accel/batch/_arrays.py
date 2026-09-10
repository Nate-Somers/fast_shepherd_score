# shepherd_score/accel/batch/_arrays.py
"""Array-native screen path: the same alignment, with no per-molecule Python objects.

WHY THIS EXISTS. The batched aligners take a LIST OF PAIR OBJECTS -- a pairwise library's
contract -- and the streaming screen wears it. Screening K molecules manufactures K
``_FastPair`` instances and K ``torch.split`` views, bins them in a per-item Python loop,
rebuilds per-bucket lists from them, then ``cat``s the coordinates back into exactly the dense
padded array the store could have handed over directly. Measured at N=100,000 (L40S, mode vol):

    build_fit 2.59 + plan_buckets 1.38 + scatter_fill 0.62 + residual_inline 1.40
      = 5.99 us/mol, 59% of a 10.07 us/mol screen

ROSHAMBO2 runs the same algorithm through ONE native call with zero per-molecule Python, which
is the entire reason it is ~8x faster on the screen while having the SLOWER kernel per useful
atom pair (3.74 ps vs fss's 2.22).

A PARTIAL VERSION IS WORTH EXACTLY ZERO. That is measured, not cautionary: vectorising the
binning alone removed the O(K) loop and spent every microsecond back building the per-cell
Python lists that ``Bucket.members`` requires (0.89 vs 0.88 us/mol, reverted). The objects have
to stop existing along the whole path -- which is why bucket membership here is a SPAN over an
index array, and why the transforms come back as one (K,4,4) array instead of being written
onto K objects.

BIT-IDENTITY. Every step below is a re-expression, not a re-derivation:
  * the bucket PARTITION is reproduced exactly -- ``_merge_group`` and ``_cap_upfront`` are
    reused verbatim, so the merge policy cannot drift, and the cells they consume are built by
    the same band arithmetic (verified bit-identical over 300 randomized trials);
  * the padded workspace is filled from the same source memory by the same index arithmetic as
    ``_scatter_fill``, just gathering from the store's contiguous buffer instead of ``cat``ing
    K views of it;
  * the ref side is broadcast rather than replicated, which is what the existing VAA fast path
    already does for the identical reason (every screen pair shares one query tensor).

Enabled by FSS_SCREEN_ARRAYS=1. Default OFF until the gates pass.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ._bucket import Bucket, _cap_upfront, _merge_group, _min_wave, PadSpec
from ._pad import _band_key

ENABLED = os.environ.get("FSS_SCREEN_ARRAYS", "0") == "1"


class Span:
    """A contiguous ``[lo, hi)`` slice of a precomputed ordering, standing in for a Bucket's
    member LIST.

    ``Bucket.K`` is ``len(self.members)`` and ``_merge`` builds ``Bucket(a.members + b.members,
    ...)``, so supporting ``__len__``, ``__add__`` and slicing is the entire contract needed to
    reuse the real merge policy unchanged. Merges are provably adjacent: ``_merge_group`` sorts
    by pad and folds each bucket into its PREDECESSOR, and cells are emitted in ascending band
    order, so ``a.hi == b.lo`` always holds.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: int, hi: int):
        self.lo = int(lo)
        self.hi = int(hi)

    def __len__(self) -> int:
        return self.hi - self.lo

    def __add__(self, other: "Span") -> "Span":
        if self.hi != other.lo:                      # never silently produce a wrong partition
            raise AssertionError(f"non-adjacent span merge {self.hi} != {other.lo}")
        return Span(self.lo, other.hi)

    def __getitem__(self, sl: slice) -> "Span":
        start = 0 if sl.start is None else int(sl.start)
        stop = len(self) if sl.stop is None else min(int(sl.stop), len(self))
        return Span(self.lo + start, self.lo + stop)

    def idx(self, order: np.ndarray) -> np.ndarray:
        return order[self.lo:self.hi]


def plan_spans(m_sizes: np.ndarray, n_ref: int, seeds: int, device):
    """Partition K library molecules into padded buckets WITHOUT touching them individually.

    Returns ``(order, buckets)`` where ``order`` is a stable argsort by fit band and each
    bucket's ``.members`` is a :class:`Span` into it.

    On the screen path the ref size is constant (one query), so the cell key collapses to the
    fit band alone and the whole binning is ``((m+15)//16)*16`` plus one stable argsort --
    versus ~20 interpreted operations per molecule in ``plan_buckets``.
    """
    K = int(m_sizes.shape[0])
    bands = ((m_sizes.astype(np.int64) + 15) // 16) * 16          # == _band_key, vectorized
    order = np.argsort(bands, kind="stable")                      # stable => shard order within a cell
    sb = bands[order]
    # cell boundaries: ascending band order is also ascending PAD order, which is the order
    # _merge_group sorts into anyway, so emitting cells here cannot change its outcome.
    cuts = np.flatnonzero(np.diff(sb)) + 1
    starts = np.concatenate(([0], cuts, [K]))
    ref_pad = _band_key(int(n_ref))
    cells = [Bucket(Span(int(starts[i]), int(starts[i + 1])),
                    {"ref": ref_pad, "fit": int(sb[starts[i]])})
             for i in range(len(starts) - 1)]
    spec = PadSpec(merge={"ref": None, "fit": None}, seeds=int(seeds))
    return order, _cap_upfront(_merge_group(cells, spec, _min_wave(device)), spec, device)


def gather_fill(out: torch.Tensor, src: torch.Tensor,
                src_start: torch.Tensor, counts: torch.Tensor) -> None:
    """Fill a pre-zeroed ``(k, P_pad, 3)`` workspace directly from the store's contiguous
    coordinate buffer.

    Replaces ``torch.split`` into k views followed by ``_scatter_fill``'s ``torch.cat`` of
    those views -- a round trip back to the layout ``src`` already has. Same destination
    arithmetic as ``_scatter_fill``; the only change is that ``flat`` is an ``index_select``
    from ``src`` instead of a ``cat`` of slices of it, so it reads identical memory and the
    result is bit-identical.
    """
    k, P_pad = out.shape[0], out.shape[1]
    tot = int(counts.sum())
    if tot == 0:
        return
    dev = out.device
    seg_start = torch.cumsum(counts, 0) - counts                  # (k,) first flat row per molecule
    seg = torch.repeat_interleave(seg_start, counts)              # (tot,)
    local = torch.arange(tot, device=dev) - seg                   # (tot,) row within its molecule
    srcrow = torch.repeat_interleave(src_start, counts) + local   # (tot,) row in the store buffer
    dst = torch.repeat_interleave(torch.arange(k, device=dev) * P_pad, counts) + local
    out.view(k * P_pad, *out.shape[2:])[dst] = src.index_select(0, srcrow)


def align_batch_vol_arrays(ref_xyz: torch.Tensor, fit_flat: torch.Tensor,
                           fit_off: torch.Tensor, *, alpha: float = 0.81,
                           steps_fine: int = 100):
    """Array-native equivalent of ``_align_batch_vol`` for the screen path.

    Parameters
    ----------
    ref_xyz : (N, 3) float32 cuda -- the single query cloud, shared by every pair.
    fit_flat : (S, 3) float32 cuda -- every library molecule's atoms, concatenated (the store's
        own layout).
    fit_off : (K+1,) int64 cuda -- CSR offsets into ``fit_flat``.

    Returns ``(scores, SE3)`` in SHARD order: ``scores`` a float64 (K,) numpy vector and
    ``SE3`` a (K, 4, 4) float32 numpy array. Results come out of the buckets in ``order``
    sequence and are scattered back, which is what the object path achieved implicitly by
    writing each result onto its own pair.

    No ``_FastPair``, no ``torch.split``, no per-molecule Python anywhere in here.
    """
    from shepherd_score.accel.drivers.shape import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.accel.drivers._common import batched_seeds_torch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    device = fit_flat.device
    n_seeds = int(MODE_SEEDS["vol"])
    K = int(fit_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    counts_all = (fit_off[1:] - fit_off[:-1])                     # (K,) int64, on device
    m_sizes = counts_all.detach().cpu().numpy()
    N = int(ref_xyz.shape[0])
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]               # (k,) shard indices
        cnt = counts_all.index_select(0, rows)
        start = fit_off.index_select(0, rows)

        # ---- padded workspaces -------------------------------------------------------
        # ref is BROADCAST, not replicated: every screen pair shares one query, so the object
        # path filled k identical rows here. The existing VAA fast path already exploits the
        # same fact (aligners.py checks `all(p._ref_xyz_t is bucket[0]._ref_xyz_t ...)`).
        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_xyz
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(fit_pad, fit_flat, start, cnt)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = cnt.to(torch.int32)

        # ---- self-overlaps: ref computed once and broadcast (bit-identical) ------------
        VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(k).contiguous()
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ref_shared is STRUCTURAL here, not a property of the data: ref_pad is built by
        # broadcasting the single query cloud into all k rows a few lines above, exactly as the
        # VAA call already assumes. No identity predicate is needed or possible.
        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real,
                                               num_seeds=n_seeds, ref_shared=True)

        def _proc(_s, _k, _rp=ref_pad, _fp=fit_pad, _va=VAA, _vb=VBB,
                  _nr=N_real, _mr=M_real, _sq=seeds_q, _st=seeds_t):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                _rp[sl], _fp[sl], _va[sl], _vb[sl],
                N_real=_nr[sl], M_real=_mr[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(_sq[sl], _st[sl]))

        # same workspace/footprint key as the object path -> identical chunking
        sc, qb, tb = _subbatched_align(_proc, k, key=("vol", N_pad, M_pad, n_seeds),
                                        device=device)
        idx = rows
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, idx, qb)
        out_t.index_copy_(0, idx, tb)

    # ---- epilogue: one D2H, one batched SE(3), numpy out (see aligners.py:18) ----------
    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def align_batch_vol_color_arrays(ref_xyz: torch.Tensor, ref_types: torch.Tensor,
                                 ref_ancs: torch.Tensor, fit_flat: torch.Tensor,
                                 fit_off: torch.Tensor, fit_types_flat: torch.Tensor,
                                 fit_ancs_flat: torch.Tensor, ph_off: torch.Tensor,
                                 *, alpha: float = 0.81, color_weight: float = 0.5,
                                 num_repeats_per_trans: int = 10, topk: int = 30,
                                 steps_fine: int = 100, lr: float = 0.075):
    """Array-native equivalent of ``_align_batch_vol_color`` for the screen path.

    vol_color was the most host-bound mode measured: 53.0% of its screen is spent inside
    ``_build_fit_fast_pairs`` (in situ, N=1e5), because its object-path branch runs a 7-way zip
    doing three ``torch.split`` plus three ``np.split``, four attribute stores and one
    ``_ArrView`` allocation PER LIBRARY MOLECULE. None of that exists here.

    Two channels instead of vol's one: heavy-atom centers (CSR ``fit_off``) and directionless
    pharmacophore features -- types + anchors sharing their own CSR ``ph_off``. The two offset
    tables are independent on purpose; a molecule's feature count has no relation to its atom
    count.

    WHY ``plan_spans`` IS THE RIGHT PARTITIONER HERE. ``_align_batch_vol_color`` buckets with
    ``PadSpec(merge={ref,fit centers}, seeds=16, partition={"tc": ...})``. On the screen path
    ``trans_init`` is False, so ``tc`` is 0 for every pair and the partition never splits; and
    the ref is ONE query, so its band is constant too. ``plan_buckets`` keys on
    ``(banded-merge-dims, exact-partition-dims)`` (_bucket.py:199), so that key collapses to the
    fit band -- precisely what ``plan_spans`` computes. Seeds are passed as 16, not vol's 10,
    because ``_cap_upfront`` sizes the occupancy floor from ``K * seeds``.

    Anchor padding is NOT keyed, matching the object path: it pads to the bucket's max feature
    band and relies on Dummy-typed (``_PHARM_PAD_TYPE``) slots plus ``N_real_pharm`` masking, so
    two molecules sharing a shape bucket but differing in feature count stay together.

    Returns ``(scores, SE3)`` in SHARD order, like :func:`align_batch_vol_arrays`.
    """
    from shepherd_score.accel.drivers.vol_color import (
        fast_optimize_vol_color_overlay_batch, _PHARM_PAD_TYPE)
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from .._modes import MODE_SEEDS

    device = fit_flat.device
    n_seeds = int(MODE_SEEDS["vol_color"])
    K = int(fit_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt_c = fit_off[1:] - fit_off[:-1]                   # (K,) heavy-atom counts
    cnt_p = ph_off[1:] - ph_off[:-1]                     # (K,) pharmacophore counts
    m_sizes = cnt_c.detach().cpu().numpy()
    N = int(ref_xyz.shape[0])
    n_ph = int(ref_ancs.shape[0])
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    # One query for the whole screen, so its feature band is fixed across every bucket -- the
    # object path recomputes max(n_ph_list) per bucket over k copies of that same query.
    n_ph_pad = _band_key(n_ph)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]
        c_cnt = cnt_c.index_select(0, rows)
        c_start = fit_off.index_select(0, rows)
        p_cnt = cnt_p.index_select(0, rows)
        p_start = ph_off.index_select(0, rows)
        m_ph_pad = _band_key(int(p_cnt.max()))

        # ---- shape channel: ref broadcast, fit gathered from the store's own buffer --------
        centers_1 = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        centers_1[:, :N] = ref_xyz
        centers_2 = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(centers_2, fit_flat, c_start, c_cnt)

        # ---- colour channel: pad slots stay Dummy-typed so they are never scored -----------
        r_types = torch.full((k, n_ph_pad), _PHARM_PAD_TYPE, device=device, dtype=torch.int64)
        r_types[:, :n_ph] = ref_types
        f_types = torch.full((k, m_ph_pad), _PHARM_PAD_TYPE, device=device, dtype=torch.int64)
        gather_fill(f_types, fit_types_flat, p_start, p_cnt)

        r_ancs = torch.zeros(k, n_ph_pad, 3, device=device, dtype=torch.float32)
        r_ancs[:, :n_ph] = ref_ancs
        f_ancs = torch.zeros(k, m_ph_pad, 3, device=device, dtype=torch.float32)
        gather_fill(f_ancs, fit_ancs_flat, p_start, p_cnt)

        N_real_c = torch.full((k,), N, dtype=torch.int32, device=device)
        N_real_p = torch.full((k,), n_ph, dtype=torch.int32, device=device)

        _, qb, tb, sc = fast_optimize_vol_color_overlay_batch(
            centers_1, centers_2, r_types, f_types, r_ancs, f_ancs,
            alpha=alpha, color_weight=color_weight,
            N_real_centers=N_real_c, M_real_centers=c_cnt.to(torch.int32),
            N_real_pharm=N_real_p, M_real_pharm=p_cnt.to(torch.int32),
            trans_centers_batch=None, trans_centers_real=None,     # trans_init is False here
            num_repeats_per_trans=num_repeats_per_trans,
            topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=n_seeds)

        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3
