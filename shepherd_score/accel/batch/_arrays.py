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
from ._pad import _band_key, _BAND

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


def _const_seed_batch(const_seeds: torch.Tensor, k: int, device):
    """A canonical store's constant seed set, broadcast over a bucket of ``k`` molecules.

    Returns ``(quats (k,S,4), trans (k,S,3))`` in the layout ``batched_seeds_torch`` produces, so
    a driver's ``seeds=`` argument takes either without knowing which. The translations are ZERO:
    a canonical store is pre-centred on the heavy-atom centroid and the query is centred the same
    way (screen.py::_centered_copy), so the COM-aligning translation the per-molecule generator
    computes is identically zero for every mode that seeds from the atom cloud -- which is what
    ``_modes.CONST_SEED_MODES`` lists. The expand is materialised (``.contiguous()``) because the
    drivers ``.reshape(-1, 4)`` the seeds into pose rows and a stride-0 view cannot serve that.
    """
    S = int(const_seeds.shape[0])
    return (const_seeds.unsqueeze(0).expand(k, -1, -1).contiguous(),
            torch.zeros(k, S, 3, device=device, dtype=torch.float32))


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
            seeds_q, seeds_t = _const_seed_batch(const_seeds, k, device)
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

        # same workspace/footprint key as the object path -> identical chunking.
        #
        # ``pose_cap`` makes the graph/eager DECISION DETERMINISTIC, not uniformly graphed.
        # The earlier claim here -- that the cap keeps every sub-batch inside graph_cap so the
        # fine loop is ALWAYS the CUDA-graph one -- is false, and measured false at this exact
        # call site (this is the only caller that passes pose_cap). A capped chunk is
        # P = 81,920 // seeds * seeds = 81,920 poses at vol's 10 seeds, while
        # graph_cap(work) = max(2000, min(262144, 300_000_000 // work)), so the capped chunk
        # fits only while N_pad*M_pad <= 3,662:
        #     band 48 (work 2,304)  -> cap 130,208  GRAPHS
        #     band 64 (work 4,096)  -> cap  73,242  EAGER
        #     band 112 (work 12,544)-> cap  23,915  EAGER
        # Verified by execution, 12 of 12 chunks over three bands, including the band-64 case
        # that straddles the boundary.
        #
        # What the cap DOES buy is the thing that mattered: without it the chunk is sized from
        # free memory alone, so whether a given bucket graphed depended on allocator state, and
        # the two paths score differently (the graph replay carries the early-stop margin). The
        # cap makes that choice a function of the band alone. Opt-in per call site: every other
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
                                 steps_fine: int = 100, lr: float = 0.075, const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_color`` for the screen path.

    ``const_seeds`` -- a canonical store's ``(S, 4)`` constant rotation set -- replaces the
    driver's per-molecule seed eigensolve, exactly as it does in :func:`align_batch_vol_arrays`;
    vol_color seeds from the SHAPE atom clouds (drivers/vol_color.py), the cloud the store
    canonicalises, so the same set applies. See ``_modes.CONST_SEED_MODES``.

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
            topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=n_seeds,
            seeds=None if const_seeds is None else _const_seed_batch(const_seeds, k, device))

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
                # the ref side here is ONE query materialised across the batch by
                # `r_ancs[:, :N] = ref_ancs` above, so its self-overlap is one scalar computed
                # BATCH times -- measured 64.6 ms, 10.2% of a 1500-pair screen (job 22653340)
                ref_shared=True,
                trans_centers_batch=None, trans_centers_real=None,
                num_repeats_per_trans=10, N_real=_nr[sl], M_real=_mr[sl],
                topk=topk, steps_fine=steps_fine, lr=lr)
            return sc, q, t

        # NO pose cap here: armed, pharm measured 0.6496x (job 22598857). THAT DATUM STANDS;
        # the arithmetic that used to explain it does not. The N_pad/M_pad this driver feeds
        # graph_cap are the PHARMACOPHORE ANCHOR pads (drivers/pharm.py:191-192), not the shape
        # band, so the threshold is far above the 9,765 poses once quoted here. Probed at the
        # real gate: drug-like molecules pad to 16 anchors -> work 256 -> cap 39,062 poses (job
        # 22637452); only peptide-sized feature counts reach work 1024 -> 9,765 (job 22637626).
        # So a cap here CAN reach a graph on ordinary ligands, and the 0.6496x has no explanation
        # yet. Keep the cap off until someone re-measures it, but not for the reason given.
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
                               steps_fine: int = 100, lr: float = 0.075, const_seeds=None):
    """Array-native equivalent of ``_esp_bucketed_align`` (``vol_esp``) for the screen path.

    ``const_seeds`` (a canonical store's constant rotation set) replaces the per-molecule seed
    eigensolve inside ``fast_optimize_ROCS_esp_overlay_batch``. vol_esp seeds from the
    strict-heavy centres, which the store rotates by the same ``rot`` as ``atom_pos``; see
    ``_modes.CONST_SEED_MODES`` for the retained-H caveat.

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

        # Seeds hoisted per BUCKET and sliced per chunk, as vol does; None keeps the driver's
        # own per-molecule generator.
        cs = None if const_seeds is None else _const_seed_batch(const_seeds, k, device)

        def _proc(_s, _k, _rp=ref_pad, _fp=fit_pad, _rc=ref_c, _fc=fit_c,
                  _nr=N_real, _mr=M_real, _cs=cs):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                _rp[sl], _fp[sl], _rc[sl], _fc[sl], alpha=alpha, lam=lam,
                N_real=_nr[sl], M_real=_mr[sl],
                trans_centers_batch=None, trans_centers_real=None,
                num_repeats_per_trans=num_repeats_per_trans, num_seeds=n_seeds,
                topk=topk, steps_fine=steps_fine, lr=lr,
                seeds=None if _cs is None else (_cs[0][sl], _cs[1][sl]))
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
                                        steps_fine: int = 100, lr: float = 0.075,
                                        const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_and_surf_esp`` for the screen path.

    ``const_seeds`` is honoured ONLY at alpha == 0.81: the driver seeds from ``centers_1/2``,
    which are the atom clouds there and the SURFACE clouds otherwise (drivers/esp_combo.py), and
    a canonical store canonicalises the atom frame alone. Off that alpha the per-molecule
    generator runs as before, whatever the caller passed.

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

        cs = None if (const_seeds is None or not a0) else _const_seed_batch(const_seeds, k, device)

        def _proc(_s, _k, _cs=cs):
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
                steps_fine=steps_fine, lr=lr, num_seeds=n_seeds,
                seeds=None if _cs is None else (_cs[0][sl], _cs[1][sl]))
            return sc, q, t

        # NO pose cap here, and UNTESTED -- do not read this as "measured neutral". The other
        # three array call sites each cite a measurement (vol 1.2985x kept, vol_esp 0.9981x,
        # pharm 0.6496x, job 22598857); this mode appears in none of them -- that job's listing
        # in batch/_pad.py names vol, vol_esp and pharm only, and no later run has armed a cap
        # here. The 2026-09 campaign measured this mode's DRIVER-level graph gate (jobs 22637452
        # / 22637626) but never its pose_cap. Arm it and measure before assuming either way.
        sc, qb, tb = _subbatched_align(
            _proc, k, key=("vol_and_surf_esp", n_wH_pad, m_wH_pad, n_cent_pad,
                           m_cent_pad, n_surf_pad, m_surf_pad, n_seeds), device=device)
        out_scores[rows_np] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


# =========================================================================================
# SECOND WAVE -- the six screen-capable modes that still ran the object path.
#
# WHY THESE SIX, AND WHAT THE GAIN ACTUALLY IS: the ``screen.py::_use_arrays`` docstring holds
# the whole record -- the per-mode speedups, the jobs that measured them, the bit-identity
# evidence, and why surf/surf_esp are a separate regime. It is deliberately the ONLY copy: this
# header used to carry its own restatement, asserted a ~2.6x vol_lipo ceiling that measurement
# later refuted, and the correction then had to be made in three files. Add measurements there.
#
# THE SIGNATURE IS ``(ref dict, fit tuple, *, per-mode kwargs)``, matching
# :func:`align_batch_vol_and_surf_esp_arrays` -- the one existing aligner here that already
# takes the dispatch-shaped pair. ``screen.py::_run_shards_inproc`` and ``_screen_worker`` call
# ``align(ref, tuple(fit), batch_kw)`` through the single ``_array_dispatch`` selector, and the
# ``_align_fast_arrays_<mode>`` wrapper on the other side of that table is what turns
# ``batch_kw`` into these keywords. The DEFAULTS below are the ones the ``_align_batch_<mode>``
# being replaced declares, so a direct call with no wrapper behaves like the object path too.
#
# EACH FIT TUPLE'S ARITY AND ORDER IS DERIVED FROM ITS OWN OBJECT PATH -- the ``p._fit_*_t``
# attributes that mode's ``_align_batch_<mode>`` actually reads, and the offset table
# ``screen.py::_build_fit_fast_pairs`` splits each of them by. It is stated per function and
# must not be inferred from a mode that looks similar: ``vol_fukui`` rides ``vol_lipo``'s driver
# on DIFFERENT arrays, and ``vol_esp_tversky`` reads ``_fit_xyz_noH_t``/``_fit_xyz_esp_t``
# (heavy_off) while ignoring the ``_fit_xyz_t`` its object-path builder also sets.
#
# NONE OF THEM PASSES A POSE CAP, and that is measured rather than stylistic. Armed on
# vol_lipo the cap measured 0.871x at 29,296 poses, 0.939x at 81,920 and 1.028x at 262,144
# (job 22637392) -- i.e. a loss at exactly the ``_FINE_CHUNK_POSES`` value ``vol`` ships, and a
# win only past the graph ceiling where no capture happens at all. On pharm it measured 0.6496x
# (job 22598857). ``vol`` is the only call site in this module that keeps its cap.
# =========================================================================================


def align_batch_vol_tversky_arrays(ref: dict, fit: tuple, *, alpha: float = 0.81,
                                   tversky_alpha: float = 0.95, tversky_beta: float = 0.05,
                                   steps_fine: int = 100, const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_tversky`` for the screen path.

    ``const_seeds`` replaces the hoisted ``batched_seeds_torch`` call below with a canonical
    store's constant set -- the same substitution :func:`align_batch_vol_arrays` makes, on the
    same atom clouds.

    fit tuple: ``(fit_flat, fit_off)`` -- heavy-atom coordinates concatenated, plus their CSR
    offsets. ONE ragged channel, because ``_align_batch_vol_tversky`` reads exactly one fit
    attribute (``p._fit_xyz_t``, which ``_build_fit_fast_pairs`` splits out of
    ``atom_pos``/``atom_off``). Identical in shape to ``vol``'s tuple, which is the point:
    vol_tversky is ``vol``'s shape machinery under an asymmetric reduction, so it buckets on the
    same band (``PadSpec(merge={ref: _ref_xyz_t, fit: _fit_xyz_t})``) and :func:`plan_spans`
    applies directly.

    Tversky lives entirely in the DRIVER: ``AA``/``BB`` are the same pose-invariant shape
    self-overlaps ``vol`` computes, and only the reduction differs. So the assembly here is
    :func:`align_batch_vol_arrays`'s, and the one substitution is
    ``coarse_fine_align_many_tversky`` for ``coarse_fine_align_many``.
    """
    from shepherd_score.accel.drivers.vol_tversky import (
        coarse_fine_align_many_tversky, _self_overlap_in_chunks)
    from shepherd_score.accel.drivers._common import batched_seeds_torch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    fit_flat, fit_off = fit
    ref_xyz = ref["_ref_xyz_t"]

    device = fit_flat.device
    n_seeds = int(MODE_SEEDS["vol_tversky"])
    K = int(fit_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    counts_all = fit_off[1:] - fit_off[:-1]
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
        rows = order_t[bk.members.lo:bk.members.hi]
        cnt = counts_all.index_select(0, rows)
        start = fit_off.index_select(0, rows)

        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_xyz
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(fit_pad, fit_flat, start, cnt)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = cnt.to(torch.int32)

        # ref BROADCAST, so its self-overlap is one row expanded. The object path takes this
        # same branch whenever ``K > 1``: its ``_ref_shared`` predicate is an identity test on
        # ``_ref_xyz_t`` and screen.py::_align_fast sets ONE ref tensor object on every pair.
        # At k == 1 the two spellings are the same call on the same memory.
        VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(k).contiguous()
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        # ref_shared=True unconditionally is EXACT here, not an approximation of the object
        # path's ``K > 1 and all(p._ref_xyz_t is ...)``: batched_seeds_torch computes
        # ``_dedup = bool(ref_shared) and K > 1`` internally (drivers/_common.py), so a k == 1
        # bucket takes the full solve either way. The guarantee is STRUCTURAL here -- ref_pad is
        # built by broadcasting the single query cloud into all k rows immediately above.
        if const_seeds is not None:
            seeds_q, seeds_t = _const_seed_batch(const_seeds, k, device)
        else:
            seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real,
                                                   num_seeds=n_seeds, ref_shared=True)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many_tversky(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                steps_fine=steps_fine, seeds=(seeds_q[sl], seeds_t[sl]))

        # NO pose cap -- see the section header (vol_lipo 0.871x/0.939x/1.028x job 22637392,
        # pharm 0.6496x job 22598857). Same sub-batch key tuple as the object path, so the two
        # share one ``_PAIR_FOOTPRINT_BYTES`` entry and chunk the same way.
        sc, qb, tb = _subbatched_align(_proc, k, key=("vol_tversky", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def align_batch_vol_esp_tversky_arrays(ref: dict, fit: tuple, *, lam: float = 0.1,
                                       alpha: float = 0.81, tversky_alpha: float = 0.95,
                                       tversky_beta: float = 0.05, steps_fine: int = 100,
                                       const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_esp_tversky`` for the screen path.

    ``const_seeds`` replaces the hoisted seed call below with a canonical store's constant set
    (strict-heavy centres, rotated with ``atom_pos``; see ``_modes.CONST_SEED_MODES``).

    fit tuple: ``(fit_pts, fit_chg, off)`` -- strict-heavy centers, the heavy partial charges
    that are 1:1 with them, and the ONE offset table (``heavy_off``) both index by. Identical to
    ``vol_esp``'s tuple, and that is not an analogy: ``screen.py::_build_fit_fast_pairs`` handles
    ``vol_esp`` and ``vol_esp_tversky`` in the SAME branch, and ``_align_batch_vol_esp_tversky``
    reads only ``_fit_xyz_noH_t`` and ``_fit_xyz_esp_t``. The ``_fit_xyz_t`` (RemoveHs
    ``atom_pos``) that branch also sets is trans-init-only, and this mode never reads it.

    ``lam`` is RAW (no ``LAM_SCALING``), matching ``_align_batch_vol_esp_tversky``.

    TWO DELIBERATE DIFFERENCES FROM :func:`align_batch_vol_tversky_arrays`, both derived from
    this mode's object path rather than carried over from its neighbour:

    * ``batched_seeds_torch`` is called WITHOUT ``ref_shared``. ``_align_batch_vol_esp_tversky``
      computes no ``_ref_shared`` predicate at all, so its seeds come from the full K-row
      reference eigensolve. That solve is documented as NOT bit-identical to the deduped one on
      CUDA (8 of 100,000 scores move, max 4.1723e-07 -- drivers/_common.py), so passing the flag
      here would be a scoring change, not a speedup.
    * ``VAA`` is computed over all k rows, not broadcast from row 0 -- same reason: the object
      path hands ``_self_overlap_esp_chunks`` the whole padded batch.

    The bucket band is the HEAVY count (``PadSpec(merge={ref: _ref_xyz_noH_t, fit:
    _fit_xyz_noH_t})``), which is exactly what ``off`` measures, so :func:`plan_spans` applies.
    """
    from shepherd_score.accel.drivers.vol_esp_tversky import (
        coarse_fine_esp_tversky_align_many, _self_overlap_esp_chunks)
    from shepherd_score.accel.drivers._common import batched_seeds_torch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    fit_pts_flat, fit_chg_flat, off = fit      # lam arrives RAW, as _align_batch_vol_esp_tversky
    ref_pts = ref["_ref_xyz_noH_t"]
    ref_chg = ref["_ref_xyz_esp_t"]

    device = fit_pts_flat.device
    n_seeds = int(MODE_SEEDS["vol_esp_tversky"])
    K = int(off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt_all = off[1:] - off[:-1]
    m_sizes = cnt_all.detach().cpu().numpy()
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
        c = cnt_all.index_select(0, rows)
        st = off.index_select(0, rows)

        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_pts
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        gather_fill(fit_pad, fit_pts_flat, st, c)
        ref_c_pad = torch.zeros(k, N_pad, device=device, dtype=torch.float32)
        ref_c_pad[:, :N] = ref_chg
        fit_c_pad = torch.zeros(k, M_pad, device=device, dtype=torch.float32)
        gather_fill(fit_c_pad, fit_chg_flat, st, c)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = c.to(torch.int32)

        # Full k-row ESP self-overlaps and NON-deduped seeds -- see the docstring. Mirroring the
        # object path IS the bit-identity argument here; the broadcast shortcut the shape modes
        # take is not available to a mode whose object path never took it.
        VAA = _self_overlap_esp_chunks(ref_pad, ref_c_pad, N_real, alpha, lam)
        VBB = _self_overlap_esp_chunks(fit_pad, fit_c_pad, M_real, alpha, lam)

        if const_seeds is not None:
            seeds_q, seeds_t = _const_seed_batch(const_seeds, k, device)
        else:
            seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real,
                                                   num_seeds=n_seeds)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_esp_tversky_align_many(
                ref_pad[sl], fit_pad[sl], ref_c_pad[sl], fit_c_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, lam=lam,
                tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                steps_fine=steps_fine, seeds=(seeds_q[sl], seeds_t[sl]))

        # NO pose cap -- see the section header. vol_esp, whose fused shape+ESP step this mode
        # shares, measured a flat 0.9981x with one armed (job 22598857): the graph engages, but
        # the per-step ESP kernel is heavy enough that the launch saving disappears.
        sc, qb, tb = _subbatched_align(_proc, k, key=("vol_esp_tversky", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def _align_batch_vol_lipo_family_arrays(ref_cent: torch.Tensor, ref_fpos: torch.Tensor,
                                        ref_fval: torch.Tensor, fit: tuple, *, tag: str,
                                        field_weight: float, alpha: float, lam: float,
                                        topk: int, steps_fine: int, lr: float,
                                        const_seeds=None):
    """Shared array-native body for the two modes that ride the ``vol_lipo`` driver.

    ``vol_lipo`` and ``vol_fukui`` are the same assembly over a DIFFERENT per-atom scalar field:
    ``_align_batch_vol_fukui`` is ``_align_batch_vol_lipo`` with ``lipo_pos``/``lipophilicity``
    replaced by ``fukui_pos``/``fukui`` and ``lipo_weight`` by ``fukui_weight``, feeding the same
    ``fast_optimize_vol_lipo_overlay_batch``. So the channel is a parameter here rather than a
    second copy of the body -- but ``tag`` is NOT cosmetic: it is the first element of the
    ``_subbatched_align`` key, and the per-shape footprint cache (``_pad._PAIR_FOOTPRINT_BYTES``)
    must stay keyed exactly as the object path keys it or the two paths chunk differently.

    fit tuple: ``(cent_flat, cent_off, fpos_flat, fval_flat, field_off)``.

    TWO INDEPENDENT OFFSET TABLES, and that is load-bearing rather than tidy. The SHAPE centres
    are ``atom_pos`` (the ``Chem.RemoveHs`` set, which RETAINS some H -- stereo/isotope/valence),
    while the field centres are the TRUE-heavy positions. The two counts diverge on exactly those
    molecules, so each channel is gathered by its own CSR table; sharing one would desync the
    fill on the first isotope-labelled molecule in the library.

    BUCKETING IS ON THE SHAPE BAND ONLY, matching ``PadSpec(merge={ref: _ref_xyz_t, fit:
    _fit_xyz_t})``. The field pads are NOT keyed: the object path pads them to the BUCKET'S max
    field band and lets ``N_real_lipo``/``M_real_lipo`` mask the rest, so two molecules sharing a
    shape bucket but differing in field count stay together. Reproduced literally, including the
    ``or _BAND`` floor that keeps a zero-length channel a well-formed tensor.
    """
    from shepherd_score.accel.drivers.vol_lipo import fast_optimize_vol_lipo_overlay_batch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    cent_flat, cent_off, fpos_flat, fval_flat, field_off = fit
    device = cent_flat.device
    # Seeds come from the registry, and ``num_repeats`` is not a parameter of this path at all.
    # The object path ACCEPTS it and never reads it -- measured on a 4,000-pair screen (job 22637761):
    # num_repeats 4, 16 and 32 return the SAME score vector (0 of 4,000 moved), while moving
    # MODE_SEEDS 16 -> 4 moves 2,604 of 4,000 (max 1.401e-01). Honouring the kwarg here would be
    # a behaviour change the object path does not make.
    n_seeds = int(MODE_SEEDS[tag])
    K = int(cent_off.shape[0]) - 1
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    cnt_c = cent_off[1:] - cent_off[:-1]                 # (K,) shape-centre counts
    cnt_f = field_off[1:] - field_off[:-1]               # (K,) field-centre counts
    m_sizes = cnt_c.detach().cpu().numpy()
    # Both count vectors come to the host HERE, in the one transfer the bucketer already forces,
    # because the per-bucket field pad below is derived from them. Reading it as
    # ``int(f_cnt.max())`` inside the loop instead put a device->host sync in every bucket, each
    # one landing after that bucket's predecessor kernels were enqueued and stalling the
    # pipeline; the object path derives the same pad from host-side Python lists and never
    # syncs. Same VALUE either way -- f_cnt is cnt_f.index_select(0, rows) and rows is
    # order_t[lo:hi], so the max is taken over exactly the same integers (verified bit-identical
    # on the CPU parity run for vol_lipo and vol_fukui: 0 of 12 scores moved, max|delta| 0.0).
    f_sizes = cnt_f.detach().cpu().numpy()
    n_cent = int(ref_cent.shape[0])
    n_field = int(ref_fpos.shape[0])
    order, buckets = plan_spans(m_sizes, n_cent, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    # One query for the whole screen, so its field band is fixed across every bucket -- the
    # object path recomputes max(n_field_list) per bucket over k copies of that same query.
    n_field_pad = _band_key(n_field) or _BAND

    for bk in buckets:
        n_cent_pad, m_cent_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]
        c_cnt = cnt_c.index_select(0, rows)
        c_start = cent_off.index_select(0, rows)
        f_cnt = cnt_f.index_select(0, rows)
        f_start = field_off.index_select(0, rows)
        # Host-side twin of ``int(f_cnt.max())`` -- see the f_sizes note above. No sync.
        m_field_pad = _band_key(int(f_sizes[order[bk.members.lo:bk.members.hi]].max())) or _BAND

        centers_1 = torch.zeros(k, n_cent_pad, 3, device=device, dtype=torch.float32)
        centers_1[:, :n_cent] = ref_cent
        centers_2 = torch.zeros(k, m_cent_pad, 3, device=device, dtype=torch.float32)
        gather_fill(centers_2, cent_flat, c_start, c_cnt)

        fpos_1 = torch.zeros(k, n_field_pad, 3, device=device, dtype=torch.float32)
        fpos_1[:, :n_field] = ref_fpos
        fpos_2 = torch.zeros(k, m_field_pad, 3, device=device, dtype=torch.float32)
        gather_fill(fpos_2, fpos_flat, f_start, f_cnt)
        fval_1 = torch.zeros(k, n_field_pad, device=device, dtype=torch.float32)
        fval_1[:, :n_field] = ref_fval
        fval_2 = torch.zeros(k, m_field_pad, device=device, dtype=torch.float32)
        gather_fill(fval_2, fval_flat, f_start, f_cnt)

        i32 = torch.int32
        N_real_centers = torch.full((k,), n_cent, dtype=i32, device=device)
        M_real_centers = c_cnt.to(i32)
        N_real_field = torch.full((k,), n_field, dtype=i32, device=device)
        M_real_field = f_cnt.to(i32)

        # No self-overlap and no seed call here: fast_optimize_vol_lipo_overlay_batch computes
        # both internally from the padded batch it is handed, on BOTH paths. There is no
        # ref_shared shortcut to take or to skip.
        # ``const_seeds`` (a canonical store's constant set, valid because this driver seeds
        # from the SHAPE atom clouds) is hoisted per bucket and sliced per chunk; None keeps the
        # driver's own per-molecule generator.
        cs = None if const_seeds is None else _const_seed_batch(const_seeds, k, device)

        def _proc(_s, _k, _cs=cs):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_vol_lipo_overlay_batch(
                centers_1[sl], centers_2[sl], fpos_1[sl], fpos_2[sl],
                fval_1[sl], fval_2[sl],
                alpha=alpha, lam=lam, lipo_weight=field_weight,
                N_real_centers=N_real_centers[sl], M_real_centers=M_real_centers[sl],
                N_real_lipo=N_real_field[sl], M_real_lipo=M_real_field[sl],
                topk=topk, steps_fine=steps_fine, lr=lr, num_seeds=n_seeds,
                seeds=None if _cs is None else (_cs[0][sl], _cs[1][sl]))
            return sc, q, t

        # NO pose cap, and for this driver the decline is DIRECTLY measured rather than
        # inherited: armed on vol_lipo it gave 0.871x at 29,296 poses, 0.939x at 81,920 (the
        # value ``vol`` ships) and only 1.028x at 262,144 (job 22637392). The driver graphs below
        # ``graph_cap(N_pad*M_pad, budget=30e6)``, so a cap tight enough to reach the graph costs
        # more in chunks than the graph returns -- the same shape of result as vol_color's
        # 0.6523x (job 22599113) and pharm's 0.6496x (job 22598857).
        sc, qb, tb = _subbatched_align(
            _proc, k, key=(tag, n_cent_pad, m_cent_pad, n_field_pad, m_field_pad, n_seeds),
            device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def align_batch_vol_lipo_arrays(ref: dict, fit: tuple, *, lipo_weight: float = 0.5,
                                alpha: float = 0.81, lam: float = 0.1, topk: int = 30,
                                steps_fine: int = 100, lr: float = 0.075, const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_lipo`` for the screen path.

    fit tuple: ``(atom_flat, atom_off, lipo_pos_flat, lipo_flat, lipo_off)`` -- the RemoveHs
    shape centres on ``atom_off``, and the TRUE-heavy lipophilicity centres plus their per-atom
    Crippen logP on ``lipo_off``. Those are exactly the three ``_fit_*_t`` attributes
    ``_align_batch_vol_lipo`` reads (``_fit_xyz_t``, ``_fit_lipo_pos_t``, ``_fit_lipo_t``) and
    the two tables ``screen.py::_build_fit_fast_pairs`` splits them by.

    ``lam`` is RAW (atom-centred, no ``LAM_SCALING``), matching the object path.

    ``num_repeats`` is deliberately absent from this signature. ``_align_batch_vol_lipo``
    accepts it and no line of its body reads it -- the seed count comes from ``MODE_SEEDS``
    (job 22637761) -- so accepting it here would advertise a knob that does nothing.
    """
    return _align_batch_vol_lipo_family_arrays(
        ref["_ref_xyz_t"], ref["_ref_lipo_pos_t"], ref["_ref_lipo_t"], fit,
        tag="vol_lipo", field_weight=lipo_weight, alpha=alpha, lam=lam,
        topk=topk, steps_fine=steps_fine, lr=lr, const_seeds=const_seeds)


def align_batch_vol_fukui_arrays(ref: dict, fit: tuple, *, fukui_weight: float = 0.5,
                                 alpha: float = 0.81, lam: float = 0.1, topk: int = 30,
                                 steps_fine: int = 100, lr: float = 0.075, const_seeds=None):
    """Array-native equivalent of ``_align_batch_vol_fukui`` for the screen path.

    fit tuple: ``(atom_flat, atom_off, fukui_pos_flat, fukui_flat, fukui_off)``.

    SAME DRIVER AS vol_lipo, DIFFERENT ARRAYS -- and that difference is the whole reason the
    channel is a parameter instead of a copied body. The store keeps ``fukui_pos``/``fukui`` on
    their OWN ``fukui_off``, never ``lipo_off``; the ref tensors are ``_ref_fukui_pos_t`` /
    ``_ref_fukui_t``; and the weight kwarg is ``fukui_weight``. Everything downstream
    (``fast_optimize_vol_lipo_overlay_batch``, the pad policy, the seed source) is shared with
    vol_lipo, but the sub-batch key tag is ``vol_fukui`` so the footprint cache stays split the
    way the object path splits it. The weight keyword is renamed at THIS boundary, exactly as
    ``_align_batch_vol_fukui`` renames it: only the shared driver call underneath still says
    ``lipo_weight=``.
    """
    return _align_batch_vol_lipo_family_arrays(
        ref["_ref_xyz_t"], ref["_ref_fukui_pos_t"], ref["_ref_fukui_t"], fit,
        tag="vol_fukui", field_weight=fukui_weight, alpha=alpha, lam=lam,
        topk=topk, steps_fine=steps_fine, lr=lr, const_seeds=const_seeds)


def align_batch_surf_arrays(ref: dict, fit: tuple, *, alpha: float = 0.81,
                            steps_fine: int = 100):
    """Array-native equivalent of ``_align_batch_surf`` for the screen path.

    fit tuple: ``(surf_all,)`` -- ONE dense ``(K, S, 3)`` block, not a flat buffer plus offsets.

    THE BUCKETING SPEC IS THE SAME SHAPE AS ``vol``'S BUT ON A DIFFERENT CLOUD, and that is
    derived, not assumed: ``_align_batch_surf`` builds ``PadSpec(merge={ref:
    _ref_surf_t.shape[0], fit: _fit_surf_t.shape[0]})`` -- SURFACE point counts, never
    ``_ref_xyz_t``. It keys its ``_ref_shared`` predicate and its seed eigensolve on the surface
    cloud for the same reason, and this does too.

    NO GATHER IS NEEDED. ``ProfileStore`` writes surfaces at a FIXED width
    (``out["surf_pos"] = np.stack(...)``), so every library molecule has exactly ``S`` points and
    the object path's per-molecule views come from ``torch.unbind`` rather than a split. A row
    ``index_select`` replaces that unbind, exactly as
    :func:`align_batch_vol_and_surf_esp_arrays` already does for its surface channel. The band
    key is therefore constant across the library, so the shard is one cell before
    ``_cap_upfront`` splits it -- which is what ``plan_buckets`` does on this mode too.

    ``mode="surf"`` IS LOAD-BEARING, not decoration. ``drivers/shape.py:_MODE_POSES`` is
    ``{"surf": 8}``, so that argument turns on the deduped multi-pose layout (``P_cta = 8``,
    molecule blocks indexed by ``pid // S`` instead of materialised S times). Dropping it -- as
    :func:`align_batch_vol_arrays` legitimately does, because ``vol`` is not in that table --
    would silently run a different kernel schedule here.
    """
    from shepherd_score.accel.drivers.shape import coarse_fine_align_many, _self_overlap_in_chunks
    from shepherd_score.accel.drivers._common import batched_seeds_torch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    (surf_all,) = fit
    ref_surf = ref["_ref_surf_t"]

    device = surf_all.device
    n_seeds = int(MODE_SEEDS["surf"])
    K = int(surf_all.shape[0])
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    S = int(surf_all.shape[1])
    N = int(ref_surf.shape[0])
    m_sizes = np.full(K, S, dtype=np.int64)
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]

        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_surf
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        fit_pad[:, :S] = surf_all.index_select(0, rows)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = torch.full((k,), S, dtype=torch.int32, device=device)

        # ref broadcast; see align_batch_vol_tversky_arrays for why ref_shared=True is exact
        # rather than approximate at every k, including k == 1.
        VAA = _self_overlap_in_chunks(ref_pad[:1], N_real[:1], alpha).expand(k).contiguous()
        VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

        seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real,
                                               num_seeds=n_seeds, ref_shared=True)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            return coarse_fine_align_many(
                ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                seeds=(seeds_q[sl], seeds_t[sl]), mode="surf")

        # NO pose cap -- see the section header. NOT MEASURED FOR THIS MODE: no job has armed a
        # cap on surf. What is measured is every OTHER mode a cap was armed on -- vol_color
        # 0.6523x (job 22599113), pharm 0.6496x (job 22598857), vol_esp 0.9981x, vol_lipo
        # 0.871x/0.939x at 29,296/81,920 poses (job 22637392) -- so ``vol`` remains the only mode
        # a cap has ever helped, and surf pads WIDER than any of them (surface clouds are
        # hundreds of points, which shrinks ``graph_cap(N_pad*M_pad)`` further). Arm it and
        # measure before assuming either way.
        sc, qb, tb = _subbatched_align(_proc, k, key=("surf", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def align_batch_surf_esp_arrays(ref: dict, fit: tuple, *, alpha: float, lam: float,
                                num_repeats_per_trans: int = 10, topk: int = 30,
                                steps_fine: int = 100, lr: float = 0.075):
    """Array-native equivalent of ``_align_batch_surf_esp`` for the screen path.

    fit tuple: ``(surf_all, surf_esp_all)`` -- two dense blocks, ``(K, S, 3)`` and ``(K, S)``,
    the two ``_fit_*_t`` attributes ``_align_batch_surf_esp`` uploads (``_fit_surf_t``,
    ``_fit_surf_esp_t``). No offset table: the store writes surfaces at a fixed width, so the
    object path unbinds rather than splits here too.

    ``lam`` IS SCALED, AND THAT IS THE ONE THING NOT TO COPY FROM ``vol_esp``.
    ``_align_batch_surf_esp`` resolves ``lam_scaled = LAM_SCALING * lam`` BEFORE reaching the
    shared ``_esp_bucketed_align``, where ``_align_batch_vol_esp`` passes ``lam`` raw. The two
    are not interchangeable (``LAM_SCALING`` is ~207), so the scaling is applied here for the
    same reason: this aligner stands in for ``_align_batch_surf_esp``, not for the shared core.

    BUCKETING IS ON THE SURFACE CLOUDS -- ``_esp_bucketed_align`` is handed
    ``ref_pts_attr="_ref_surf_t"`` / ``fit_pts_attr="_fit_surf_t"``, so its ``PadSpec`` merge
    dims are surface point counts, and its ``partition={"tc": ...}`` collapses to 0 because the
    screen always calls with ``trans_init=False`` (screen.py::_fast_batch_kwargs pins it). One
    partition value means no split, so :func:`plan_spans` reproduces the partition exactly, and
    ``trans_centers_batch``/``trans_centers_real`` are ``None`` as they are on the object path.

    No ``VAA``/``VBB`` here: ``fast_optimize_ROCS_esp_overlay_batch`` computes the ESP
    self-overlaps internally, on both paths.
    """
    from shepherd_score.accel.drivers.esp import fast_optimize_ROCS_esp_overlay_batch
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch
    from shepherd_score.score.constants import LAM_SCALING
    from ._pad import _subbatched_align
    from .._modes import MODE_SEEDS

    surf_all, surf_esp_all = fit
    ref_surf = ref["_ref_surf_t"]
    ref_surf_esp = ref["_ref_surf_esp_t"]
    # ``alpha`` and ``lam`` are REQUIRED keywords, mirroring _align_batch_surf_esp, which
    # declares a default for neither. lam arrives RAW and is scaled HERE -- see the docstring.
    lam_scaled = LAM_SCALING * lam

    device = surf_all.device
    # Registry seeds, and no ``num_repeats`` parameter: _esp_bucketed_align reads
    # ``_seeds_for(subbatch_tag)`` and never the ``num_repeats`` _align_batch_surf_esp accepts --
    # the same accepted-and-ignored kwarg vol_lipo has (job 22637761).
    n_seeds = int(MODE_SEEDS["surf_esp"])
    K = int(surf_all.shape[0])
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    S = int(surf_all.shape[1])
    N = int(ref_surf.shape[0])
    m_sizes = np.full(K, S, dtype=np.int64)
    order, buckets = plan_spans(m_sizes, N, n_seeds, device)
    order_t = torch.as_tensor(order, device=device, dtype=torch.long)

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)

    for bk in buckets:
        N_pad, M_pad = int(bk.pad["ref"]), int(bk.pad["fit"])
        k = bk.K
        rows = order_t[bk.members.lo:bk.members.hi]

        ref_pad = torch.zeros(k, N_pad, 3, device=device, dtype=torch.float32)
        ref_pad[:, :N] = ref_surf
        fit_pad = torch.zeros(k, M_pad, 3, device=device, dtype=torch.float32)
        fit_pad[:, :S] = surf_all.index_select(0, rows)
        ref_c_pad = torch.zeros(k, N_pad, device=device, dtype=torch.float32)
        ref_c_pad[:, :N] = ref_surf_esp
        fit_c_pad = torch.zeros(k, M_pad, device=device, dtype=torch.float32)
        fit_c_pad[:, :S] = surf_esp_all.index_select(0, rows)

        N_real = torch.full((k,), N, dtype=torch.int32, device=device)
        M_real = torch.full((k,), S, dtype=torch.int32, device=device)

        def _proc(_s, _k):
            sl = slice(_s, _s + _k)
            _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                ref_pad[sl], fit_pad[sl], ref_c_pad[sl], fit_c_pad[sl],
                alpha=alpha, lam=lam_scaled,
                N_real=N_real[sl], M_real=M_real[sl],
                trans_centers_batch=None, trans_centers_real=None,   # trans_init is False here
                num_repeats_per_trans=num_repeats_per_trans, num_seeds=n_seeds,
                topk=topk, steps_fine=steps_fine, lr=lr)
            return sc, q, t

        # NO pose cap -- see the section header. Not measured for surf_esp itself; what is
        # measured is its sibling on the SAME kernel, vol_esp, at 0.9981x with one armed (job
        # 22598857). surf_esp pads to surface clouds rather than heavy-atom counts, so its
        # per-step kernel is the heavier of the two and a launch saving has less to win back --
        # an argument for keeping the cap off, not a measurement of it.
        sc, qb, tb = _subbatched_align(_proc, k, key=("surf_esp", N_pad, M_pad, n_seeds),
                                       device=device)
        out_scores[bk.members.idx(order)] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3
