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

ON by default. The gates are tests/test_screen_arrays.py and tests/test_screen_pipeline.py
(26 tests); ``ENABLED`` survives only as the seam those parity tests flip to force the
object path for comparison -- it is not a runtime switch and nothing reads the environment.
"""
from __future__ import annotations

import numpy as np
import torch

from ._bucket import Bucket, _cap_upfront, _merge_group, _min_wave, PadSpec
from ._pad import _band_key

#: Test seam only -- see the module docstring. Production always takes this path.
ENABLED = True


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
                           steps_fine: int = 100, const_seeds=None):
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
    from ._pad import _subbatched_align, _FINE_CHUNK_POSES
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
        if const_seeds is not None:
            # CANONICAL store: every molecule is already in its principal frame, so the seeds are
            # one constant set broadcast over the bucket. This is the 44.1% of a vol screen that
            # batched_seeds_torch was spending on a per-molecule float64 eigensolve.
            # (k, S, 4) -- coarse_fine_align_many reads S from quats.size(1) and slices by PAIR
            seeds_q = const_seeds.unsqueeze(0).expand(k, -1, -1).contiguous()
            seeds_t = torch.zeros(k, n_seeds, 3, device=device, dtype=torch.float32)
        else:
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
        # ``pose_cap`` keeps every sub-batch
        # inside graph_cap so the fine loop is ALWAYS the CUDA-graph one. Without it the chunk
        # is sized from free memory alone and a big band silently falls back to the eager loop
        # -- a path that scores differently (it skips the graph's early-stop margin), so the
        # screen's results depended on allocator state. Opt-in per call site: every other
        # aligner keeps the schedule it has today.
        sc, qb, tb = _subbatched_align(_proc, k, key=("vol", N_pad, M_pad, n_seeds),
                                        device=device,
                                        pose_cap=_FINE_CHUNK_POSES, seeds=n_seeds)
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

        # NOT sub-batched, and that is measured, not an omission. vol_color's driver graphs only
        # below ``graph_cap(N*M, budget=30e6)`` = 29,296 poses (1,831 molecules at 16 seeds), so
        # chunking a screen-sized bucket small enough to reach the graph costs more in chunks
        # than the graph returns: wired up, it measured **0.6523x** at N=20,000 (job 22599113),
        # the same way pharm measured 0.6496x. One big call is the right shape here.
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


def align_batch_pharm_arrays(ref_types: torch.Tensor, ref_ancs: torch.Tensor,
                             ref_vecs: torch.Tensor, fit_types_flat: torch.Tensor,
                             fit_ancs_flat: torch.Tensor, fit_vecs_flat: torch.Tensor,
                             ph_off: torch.Tensor, *, similarity="tanimoto",
                             extended_points: bool = False, only_extended: bool = False,
                             num_repeats=None, topk: int = 30, steps_fine: int = 100,
                             lr: float = 0.075):
    """Array-native equivalent of ``_align_batch_pharm`` for the screen path.

    Three channels on ONE offset table: types, anchors and vectors all index by feature, so
    ``ph_off`` serves all three. ``_align_batch_pharm`` bands on the ANCHOR count, not the atom
    count, so that is what ``plan_spans`` bands here.

    NOTE the pad value. pharm zero-fills its type slots and relies on ``N_real``/``M_real``
    masking, where ``vol_color`` fills with ``_PHARM_PAD_TYPE``. Carrying vol_color's choice
    over would be a silent scoring change, so this mirrors the object path exactly.
    """
    from shepherd_score.accel.drivers.pharm import fast_optimize_pharm_overlay_batch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    device = fit_ancs_flat.device
    n_seeds = int(MODE_SEEDS["pharm"]) if num_repeats is None else int(num_repeats)
    K = int(ph_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt = ph_off[1:] - ph_off[:-1]
    m_sizes = cnt.detach().cpu().numpy()
    N = int(ref_ancs.shape[0])
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]
        c = cnt.index_select(0, rows)
        st = ph_off.index_select(0, rows)

        r_types = torch.zeros(k, N_pad, device=device, dtype=torch.int64)
        r_types[:, :N] = ref_types
        f_types = torch.zeros(k, M_pad, device=device, dtype=torch.int64)
        gather_fill(f_types, fit_types_flat, st, c)
        r_ancs = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        r_ancs[:, :N] = ref_ancs
        f_ancs = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(f_ancs, fit_ancs_flat, st, c)
        r_vecs = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        r_vecs[:, :N] = ref_vecs
        f_vecs = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(f_vecs, fit_vecs_flat, st, c)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = c.to(torch.int32)

        def _proc(_s, _k, _rt=r_types, _ft=f_types, _ra=r_ancs, _fa=f_ancs,
                  _rv=r_vecs, _fv=f_vecs, _nr=N_real, _mr=M_real):
            sl = slice(_s, _s + _k)
            _, _, q, t, sc = fast_optimize_pharm_overlay_batch(
                _rt[sl], _ft[sl], _ra[sl], _fa[sl], _rv[sl], _fv[sl],
                similarity=similarity, extended_points=extended_points,
                only_extended=only_extended, num_repeats=n_seeds,
                trans_centers_batch=None, trans_centers_real=None,
                num_repeats_per_trans=10, N_real=_nr[sl], M_real=_mr[sl],
                topk=topk, steps_fine=steps_fine, lr=lr)
            return sc, q, t

        # NO pose cap here: armed, pharm measured 0.6496x (job 22598857). Its driver graphs
        # only below graph_cap(N*M, budget=1e7) = 9,765 poses, so any cap loose enough to be
        # worth setting still never reaches a graph and only multiplies the chunk count.
        sc, qb, tb = _subbatched_align(_proc, k, key=("pharm", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def align_batch_vol_esp_arrays(ref_pts: torch.Tensor, ref_chg: torch.Tensor,
                               fit_pts_flat: torch.Tensor, fit_chg_flat: torch.Tensor,
                               off: torch.Tensor, *, alpha: float = 0.81, lam: float,
                               num_repeats_per_trans: int = 10, topk: int = 30,
                               steps_fine: int = 100, lr: float = 0.075):
    """Array-native equivalent of ``_esp_bucketed_align`` (``vol_esp``) for the screen path.

    Two channels on one offset table: strict-heavy centers and the heavy partial charges that
    are 1:1 with them. vol_esp reaches the shared ``_esp_bucketed_align``, whose PadSpec is the
    same simple ref/fit pair that ``vol`` uses, so ``plan_spans`` applies directly.

    ``lam`` is REQUIRED and passed RAW, matching the per-pair vol_esp path: vol_esp takes the
    raw value where surf_esp scales it internally, and the two are not interchangeable.
    """
    from shepherd_score.accel.drivers.esp import fast_optimize_ROCS_esp_overlay_batch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    device = fit_pts_flat.device
    n_seeds = int(MODE_SEEDS["vol_esp"])
    K = int(off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt = off[1:] - off[:-1]
    m_sizes = cnt.detach().cpu().numpy()
    N = int(ref_pts.shape[0])
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]
        c = cnt.index_select(0, rows)
        st = off.index_select(0, rows)

        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_pts
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(fit_pad, fit_pts_flat, st, c)
        ref_c = torch.zeros(k, N_pad, device=device, dtype=torch.float32)
        ref_c[:, :N] = ref_chg
        fit_c = torch.zeros(k, M_pad, device=device, dtype=torch.float32)
        gather_fill(fit_c, fit_chg_flat, st, c)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = c.to(torch.int32)

        def _proc(_s, _k, _rp=ref_pad, _fp=fit_pad, _rc=ref_c, _fc=fit_c,
                  _nr=N_real, _mr=M_real):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                _rp[sl], _fp[sl], _rc[sl], _fc[sl], alpha=alpha, lam=lam,
                N_real=_nr[sl], M_real=_mr[sl],
                trans_centers_batch=None, trans_centers_real=None,
                num_repeats_per_trans=num_repeats_per_trans, num_seeds=n_seeds,
                topk=topk, steps_fine=steps_fine, lr=lr)
            return sc, q, t

        # NO pose cap here: armed, vol_esp measured 0.9981x (job 22598857) -- the graph does
        # engage, but its per-step ESP kernel is heavy enough that the launch saving vanishes.
        sc, qb, tb = _subbatched_align(_proc, k, key=("vol_esp", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


class IdxSet:
    """Bucket members as an index ARRAY, for keys that are not one-dimensional.

    :class:`Span` is cheaper but requires every merge to be ADJACENT, which holds only because a
    1-D band key makes the emitted cell order identical to the order ``_merge_group`` sorts into.
    Once the key has several dims, ``_merge_group`` re-sorts after each fold and that guarantee is
    gone. ``IdxSet`` drops the requirement: ``__add__`` concatenates. The cost is one numpy concat
    per MERGE, over the small occupied-cell set (tens to hundreds), never per molecule -- so the
    "no per-molecule Python" property this module exists for is untouched.

    Contract needed by the planner is exactly ``__len__`` (Bucket.K), ``__add__`` (_merge) and
    slicing (_cap_upfront).
    """

    __slots__ = ("arr",)

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __add__(self, other: "IdxSet") -> "IdxSet":
        return IdxSet(np.concatenate([self.arr, other.arr]))

    def __getitem__(self, sl: slice) -> "IdxSet":
        return IdxSet(self.arr[sl])

    def idx(self, order=None) -> np.ndarray:
        """Absolute shard indices. ``order`` is accepted and ignored so a caller can treat this
        interchangeably with :meth:`Span.idx`, which resolves through the ordering."""
        return self.arr


def plan_spans_multi(fit_dims: dict, const_dims: dict, spec, device, partition: dict = None):
    """Multi-dimensional twin of :func:`plan_spans`, for modes whose PadSpec keys several dims.

    ``fit_dims``  : name -> (K,) int array of per-molecule sizes (the dims that VARY).
    ``const_dims``: name -> int, dims fixed for the whole screen (the single query's clouds, and
                    anything the store stores at a fixed width, e.g. surface points).
    ``partition`` : name -> exact value, uniform across the screen (``tc`` is 0 when trans_init
                    is False, which is always true on the screen path).

    Cells are keyed on the banded value of every merge dim, in ``spec.merge`` order, and merged
    with the REAL policy: ``_merge_group`` then ``_cap_upfront``, unchanged. Only ``spec.seeds``
    and the merge dim NAMES are consulted by that policy -- ``_should_merge`` is occupancy-only
    and never calls the merge callables, so a spec built for this path does not need them to be
    real accessors.
    """
    names = list(spec.merge)
    K = int(next(iter(fit_dims.values())).shape[0]) if fit_dims else int(next(iter(const_dims.values())))
    cols = []
    for n in names:
        if n in fit_dims:
            cols.append(((np.asarray(fit_dims[n], dtype=np.int64) + 15) // 16) * 16)
        else:
            cols.append(np.full(K, _band_key(int(const_dims[n])), dtype=np.int64))
    key = np.stack(cols, axis=1)                                  # (K, nm)
    # lexsort takes the LAST key as primary, so reverse to sort by names order left-to-right --
    # the same order _merge_group sorts buckets into.
    order = np.lexsort(tuple(key[:, i] for i in range(len(names) - 1, -1, -1)))
    sk = key[order]
    cuts = np.flatnonzero((np.diff(sk, axis=0) != 0).any(axis=1)) + 1
    starts = np.concatenate(([0], cuts, [K]))

    cells = []
    for i in range(len(starts) - 1):
        lo, hi = int(starts[i]), int(starts[i + 1])
        pad = {names[j]: int(sk[lo, j]) for j in range(len(names))}
        for pn, pv in (partition or {}).items():
            pad[pn] = pv
        cells.append(Bucket(IdxSet(order[lo:hi]), pad))
    return _cap_upfront(_merge_group(cells, spec, _min_wave(device)), spec, device)


def align_batch_vol_and_surf_esp_arrays(ref: dict, fit: tuple, *, alpha: float,
                                        lam: float = 0.001, probe_radius: float = 1.0,
                                        esp_weight: float = 0.5,
                                        num_repeats_per_trans: int = 10, topk: int = 30,
                                        steps_fine: int = 100, lr: float = 0.075):
    """Array-native equivalent of ``_align_batch_vol_and_surf_esp`` for the screen path.

    The heaviest object-path branch in the tree: nine splits, six attribute stores, an
    ``_ArrView`` AND a nested ``_MolShim`` per library molecule, for a measured 39.284 us/mol of
    fixed cost -- the largest of any mode.

    SIX cost dims, so :func:`plan_spans` (one banded fit dim) cannot reproduce the partition and
    :func:`plan_spans_multi` is used instead. On the screen path only TWO of the six actually
    vary: the ref is a single query, and the store writes surfaces at a FIXED width, so
    ``n_wH``/``n_surf``/``n_cent``/``m_surf`` are constant and the key is (m_wH, m_cent).

    Surfaces need no gather at all for the same reason -- they are already a dense ``(K, S, 3)``
    block in the store, so a row ``index_select`` replaces the object path's ``torch.unbind``.

    AT LIBRARY SCALE THIS MODE IS NOT BIT-IDENTICAL, AND NEITHER IS THE OBJECT PATH TO ITSELF.
    Measured at N=1e5 on an L40S: array-vs-object differs on 171/100,000 scores (max 1.265e-02),
    while the OBJECT path run twice against itself -- second run holding 12 GB of GPU memory --
    differs on 513/100,000 (max 1.757e-02). Cause is ``_subbatched_align``, which sizes chunks
    from ``mem_get_info()`` free memory: the schedule moved from (1024, 30326) to (22474, 8876)
    under pressure, and this mode is multi-basin, so a different batch shifts seed generation
    (``_masked_principal_axes`` is batch-size dependent) and a few molecules settle in a different
    basin. vol_esp and pharm chunk differently between paths TOO and stay bit-identical, so the
    sensitivity is this mode's, not the sub-batcher's. Top-1000 ids AND order were identical in
    both comparisons -- the ranking a screen delivers is stable. The 12-molecule fixture below
    fits one chunk, so it IS bit-identical there and the gate stays strict.
    """
    from shepherd_score.accel.drivers.esp_combo import fast_optimize_esp_combo_score_overlay_batch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._bucket import PadSpec
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    (cwh_flat, partial_flat, radii_flat, all_off,
     cent_flat, cent_off, surf_all, surf_esp_all) = fit
    device = cwh_flat.device
    n_seeds = int(MODE_SEEDS["vol_and_surf_esp"])
    K = int(all_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt_wH = all_off[1:] - all_off[:-1]
    cnt_ct = cent_off[1:] - cent_off[:-1]
    m_wH_sizes = cnt_wH.detach().cpu().numpy()
    m_ct_sizes = cnt_ct.detach().cpu().numpy()

    n_wH = int(ref["_ref_centers_w_H_t"].shape[0])
    n_surf = int(ref["_ref_surf_t"].shape[0])
    m_surf = int(surf_all.shape[1])
    a0 = (alpha == 0.81)
    n_cent = int(ref["_ref_xyz_t"].shape[0]) if a0 else n_surf

    # Same six names, same order, same seeds as _align_batch_vol_and_surf_esp's PadSpec. The
    # merge callables are never invoked on this path (see plan_spans_multi), so they are the
    # identity placeholders rather than pair accessors.
    spec = PadSpec(merge={"n_wH": None, "m_wH": None, "n_surf": None,
                          "m_surf": None, "n_cent": None, "m_cent": None},
                   seeds=n_seeds, partition={"tc": None})
    buckets = plan_spans_multi(
        fit_dims={"m_wH": m_wH_sizes, "m_cent": (m_ct_sizes if a0 else np.full(K, m_surf))},
        const_dims={"n_wH": n_wH, "n_surf": n_surf, "m_surf": m_surf, "n_cent": n_cent},
        spec=spec, device=device, partition={"tc": 0})

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)
    z = torch.zeros

    for bk in buckets:
        rows_np = bk.members.idx()
        rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
        k = bk.K
        n_wH_pad, m_wH_pad = int(bk.pad["n_wH"]), int(bk.pad["m_wH"])
        n_surf_pad, m_surf_pad = int(bk.pad["n_surf"]), int(bk.pad["m_surf"])
        n_cent_pad, m_cent_pad = int(bk.pad["n_cent"]), int(bk.pad["m_cent"])

        c_wH = cnt_wH.index_select(0, rows)
        s_wH = all_off.index_select(0, rows)
        c_ct = cnt_ct.index_select(0, rows)
        s_ct = cent_off.index_select(0, rows)

        # ---- ref side: one query broadcast into k rows ---------------------------------
        centers_w_H_1 = z(k, n_wH_pad, 3, device=device, dtype=torch.float32)
        centers_w_H_1[:, :n_wH] = ref["_ref_centers_w_H_t"]
        partial_1 = z(k, n_wH_pad, device=device, dtype=torch.float32)
        partial_1[:, :n_wH] = ref["_ref_partial_t"]
        radii_1 = z(k, n_wH_pad, device=device, dtype=torch.float32)
        radii_1[:, :n_wH] = ref["_ref_radii_t"]
        points_1 = z(k, n_surf_pad, 3, device=device, dtype=torch.float32)
        points_1[:, :n_surf] = ref["_ref_surf_t"]
        point_charges_1 = z(k, n_surf_pad, device=device, dtype=torch.float32)
        point_charges_1[:, :n_surf] = ref["_ref_surf_esp_t"]
        centers_1 = z(k, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_1[:, :n_cent] = ref["_ref_xyz_t"] if a0 else ref["_ref_surf_t"]

        # ---- fit side: CSR gathers for the ragged channels, row-select for the dense ----
        centers_w_H_2 = z(k, m_wH_pad, 3, device=device, dtype=torch.float32)
        gather_fill(centers_w_H_2, cwh_flat, s_wH, c_wH)
        partial_2 = z(k, m_wH_pad, device=device, dtype=torch.float32)
        gather_fill(partial_2, partial_flat, s_wH, c_wH)
        radii_2 = z(k, m_wH_pad, device=device, dtype=torch.float32)
        gather_fill(radii_2, radii_flat, s_wH, c_wH)

        points_2 = z(k, m_surf_pad, 3, device=device, dtype=torch.float32)
        points_2[:, :m_surf] = surf_all.index_select(0, rows)
        point_charges_2 = z(k, m_surf_pad, device=device, dtype=torch.float32)
        point_charges_2[:, :m_surf] = surf_esp_all.index_select(0, rows)

        centers_2 = z(k, m_cent_pad, 3, device=device, dtype=torch.float32)
        if a0:
            gather_fill(centers_2, cent_flat, s_ct, c_ct)
        else:
            centers_2[:, :m_surf] = surf_all.index_select(0, rows)

        i32 = torch.int32
        N_wH = torch.full((k,), n_wH, dtype=i32, device=device)
        M_wH = c_wH.to(i32)
        N_sf = torch.full((k,), n_surf, dtype=i32, device=device)
        M_sf = torch.full((k,), m_surf, dtype=i32, device=device)
        N_ct = torch.full((k,), n_cent, dtype=i32, device=device)
        M_ct = c_ct.to(i32) if a0 else M_sf

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_esp_combo_score_overlay_batch(
                centers_w_H_1[sl], centers_w_H_2[sl], centers_1[sl], centers_2[sl],
                points_1[sl], points_2[sl], partial_1[sl], partial_2[sl],
                point_charges_1[sl], point_charges_2[sl], radii_1[sl], radii_2[sl],
                alpha, lam=lam, probe_radius=probe_radius, esp_weight=esp_weight,
                N_real_atoms_w_H_1=N_wH[sl], M_real_atoms_w_H_2=M_wH[sl],
                N_real_centers=N_ct[sl], M_real_centers=M_ct[sl],
                N_real_surf_1=N_sf[sl], M_real_surf_2=M_sf[sl],
                trans_centers_batch=None, trans_centers_real=None,
                num_repeats_per_trans=num_repeats_per_trans, topk=topk,
                steps_fine=steps_fine, lr=lr, num_seeds=n_seeds)
            return sc, q, t

        sc, qb, tb = _subbatched_align(
            _proc, k, key=("vol_and_surf_esp", n_wH_pad, m_wH_pad, n_cent_pad,
                           m_cent_pad, n_surf_pad, m_surf_pad, n_seeds), device=device)
        out_scores[rows_np] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3
