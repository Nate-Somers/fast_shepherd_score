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
    the same band arithmetic;
  * the padded workspace is filled from the same source memory by the same index arithmetic as
    ``_scatter_fill``, just gathering from the store's contiguous buffer instead of ``cat``ing
    K views of it;
  * the ref side is broadcast rather than replicated, which is what the existing VAA fast path
    already does for the identical reason (every screen pair shares one query tensor).

ONE ALIGNER FOR EVERY MODE. :func:`align_arrays` reads the mode's :class:`ModeSpec` channels,
so a mode joins this path by existing in the registry rather than by gaining a hand-written
body here; ``align_batch_<mode>_arrays`` wrappers are generated for callers that name one.

ON by default. The gates are tests/test_screen_arrays.py and tests/test_screen_pipeline.py;
``ENABLED`` survives only as the seam those parity tests flip to force the object path for
comparison -- it is not a runtime switch and nothing reads the environment.
"""
from __future__ import annotations

import numpy as np
import torch

from ._bucket import Bucket, _cap_upfront, _merge_group, _min_wave, PadSpec
from ._pad import _band_key, _BAND, _subbatched_align, _FINE_CHUNK_POSES
from .._modes import SPECS, MODE_SEEDS
from ..channels import CHANNELS

#: Test seam only -- see the module docstring. Production always takes this path.
ENABLED = True

#: Modes whose fine-loop sub-batch is capped at ``_FINE_CHUNK_POSES``. Measured per mode, and
#: ``vol`` is the only one it has ever helped: vol 1.2985x kept; vol_esp 0.9981x; pharm 0.6496x
#: (job 22598857); vol_color 0.6523x (job 22599113); vol_lipo 0.871x/0.939x/1.028x at
#: 29,296/81,920/262,144 poses (job 22637392). What the cap buys ``vol`` is a graph/eager
#: decision that is a function of the band rather than of allocator state.
_POSE_CAP_MODES = ("vol",)


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


class IdxSet:
    """Bucket members as an index ARRAY, for keys that are not one-dimensional.

    :class:`Span` is cheaper but requires every merge to be ADJACENT, which holds only because a
    1-D band key makes the emitted cell order identical to the order ``_merge_group`` sorts into.
    Once the key has several dims, ``_merge_group`` re-sorts after each fold and that guarantee is
    gone. ``IdxSet`` drops the requirement: ``__add__`` concatenates. The cost is one numpy concat
    per MERGE, over the small occupied-cell set, never per molecule.
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
        interchangeably with :meth:`Span.idx`."""
        return self.arr


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
    cuts = np.flatnonzero(np.diff(sb)) + 1
    starts = np.concatenate(([0], cuts, [K]))
    ref_pad = _band_key(int(n_ref))
    cells = [Bucket(Span(int(starts[i]), int(starts[i + 1])),
                    {"ref": ref_pad, "fit": int(sb[starts[i]])})
             for i in range(len(starts) - 1)]
    spec = PadSpec(merge={"ref": None, "fit": None}, seeds=int(seeds))
    return order, _cap_upfront(_merge_group(cells, spec, _min_wave(device)), spec, device)


def plan_spans_multi(fit_dims: dict, const_dims: dict, spec, device, partition: dict = None):
    """Multi-dimensional twin of :func:`plan_spans`, for modes whose PadSpec keys several dims.

    ``fit_dims``  : name -> (K,) int array of per-molecule sizes (the dims that VARY).
    ``const_dims``: name -> int, dims fixed for the whole screen (the single query's clouds, and
                    anything the store stores at a fixed width, e.g. surface points).
    ``partition`` : name -> exact value, uniform across the screen.

    Cells are keyed on the banded value of every merge dim, in ``spec.merge`` order, and merged
    with the REAL policy: ``_merge_group`` then ``_cap_upfront``, unchanged.
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


def gather_fill(out: torch.Tensor, src: torch.Tensor,
                src_start: torch.Tensor, counts: torch.Tensor) -> None:
    """Fill a pre-zeroed ``(k, P_pad, ...)`` workspace directly from the store's contiguous
    buffer.

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
    the engine's ``seeds=`` argument takes either without knowing which. The translations are
    ZERO: a canonical store is pre-centred on the heavy-atom centroid and the query is centred
    the same way, so the COM-aligning translation the per-molecule generator computes is
    identically zero for every mode that seeds from the atom cloud -- which is what
    ``_modes.CONST_SEED_MODES`` lists. The expand is materialised because the engine reshapes
    the seeds into pose rows and a stride-0 view cannot serve that.
    """
    S = int(const_seeds.shape[0])
    return (const_seeds.unsqueeze(0).expand(k, -1, -1).contiguous(),
            torch.zeros(k, S, 3, device=device, dtype=torch.float32))


# =============================================================================================
# the generic array aligner
# =============================================================================================
def _counts_of(fit: dict, name: str, K: int, device):
    """``(counts (K,), starts (K,) or None, source tensor)`` for one channel's fit side."""
    flat, off = fit[name]
    if off is None:                                               # dense (K, S, ...) block
        S = int(flat.shape[1])
        return torch.full((K,), S, device=device, dtype=torch.long), None, flat
    return off[1:] - off[:-1], off, flat


def align_arrays(mode: str, ref: dict, fit: dict, *, params: dict, steps_fine: int,
                 num_seeds=None, const_seeds=None, avoid=None, early_stop_patience=None,
                 early_stop_tol: float = 1e-5):
    """Array-native alignment of one shard against one query, for any registry mode.

    ``ref``: concrete channel name -> the query's (N, ...) device tensor.
    ``fit``: concrete channel name -> ``(flat, off)`` (CSR) or ``(dense, None)``.
    ``avoid``: the query-side ``(K_a, 3)`` avoid cloud for ``vol_avoid``, else None.

    Returns ``(scores (K,) float64 numpy, SE3 (K,4,4) float32 numpy)`` in SHARD order.
    """
    from ..drivers.engine import Batch, align, term_self_overlaps
    from shepherd_score.alignment.utils.se3 import quaternions_to_SE3_batch

    spec = SPECS[mode]
    names = []
    for c in spec.channels:
        r = spec.resolve_channel(c, params)
        if r not in names and not CHANNELS[r].is_pair:
            names.append(r)
    device = next(iter(fit.values()))[0].device
    n_seeds = int(MODE_SEEDS[mode] if num_seeds is None else num_seeds)
    es_patience = int(spec.patience if early_stop_patience is None else early_stop_patience)

    any_ch = names[0]
    K = int(fit[any_ch][1].shape[0] - 1) if fit[any_ch][1] is not None else int(fit[any_ch][0].shape[0])
    if K == 0:
        return np.empty(0, dtype=float), np.empty((0, 4, 4), dtype=np.float32)

    # per-channel fit counts / starts / source, and the host-side count vectors the pads need
    cnt, start, src, host = {}, {}, {}, {}
    for n in names:
        c, s, f = _counts_of(fit, n, K, device)
        cnt[n], start[n], src[n] = c, s, f
        host[n] = c.detach().cpu().numpy()

    # ---- bucket on the spec's cost dims -----------------------------------------------------
    bch = []
    for b in spec.bucket:
        r = spec.resolve_channel(b, params)
        if r not in bch:
            bch.append(r)
    if len(bch) == 1:
        order, buckets = plan_spans(host[bch[0]], int(ref[bch[0]].shape[0]), n_seeds, device)
        pad_of = lambda bk, b, side: int(bk.pad["ref" if side == "ref" else "fit"])   # noqa: E731
    else:
        merge = {}
        for b in bch:
            merge[f"n_{b}"] = None
            merge[f"m_{b}"] = None
        pspec = PadSpec(merge=merge, seeds=n_seeds)
        fit_dims = {f"m_{b}": host[b] for b in bch}
        const_dims = {f"n_{b}": int(ref[b].shape[0]) for b in bch}
        buckets = plan_spans_multi(fit_dims, const_dims, pspec, device)
        order = None
        pad_of = lambda bk, b, side: int(bk.pad[f"n_{b}" if side == "ref" else f"m_{b}"])  # noqa: E731

    out_scores = np.empty(K, dtype=float)
    out_q = torch.empty(K, 4, device=device)
    out_t = torch.empty(K, 3, device=device)
    order_t = None if order is None else torch.as_tensor(order, device=device, dtype=torch.long)
    # the query's own channel widths are constant across the whole screen
    n_pads = {n: _band_key(int(ref[n].shape[0])) or _BAND for n in names}
    pose_cap = _FINE_CHUNK_POSES if mode in _POSE_CAP_MODES else 0

    for bk in buckets:
        if order_t is None:
            rows_np = bk.members.idx()
            rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
        else:
            rows = order_t[bk.members.lo:bk.members.hi]
            rows_np = bk.members.idx(order)
        k = bk.K
        chans = {}
        pads = []
        for n in names:
            ch = CHANNELS[n]
            dt = torch.int64 if ch.dtype == "int64" else torch.float32
            in_bucket = any(CHANNELS[b].basis == ch.basis for b in bch)
            if in_bucket:
                b = next(b for b in bch if CHANNELS[b].basis == ch.basis)
                n_pad, m_pad = pad_of(bk, b, "ref"), pad_of(bk, b, "fit")
            else:
                n_pad = n_pads[n]
                m_pad = _band_key(int(host[n][rows_np].max())) or _BAND
            N = int(ref[n].shape[0])
            feat = tuple(ref[n].shape[1:])
            if ch.pad == 0:
                r_pad = torch.zeros((k, n_pad) + feat, device=device, dtype=dt)
                f_pad = torch.zeros((k, m_pad) + feat, device=device, dtype=dt)
            else:
                r_pad = torch.full((k, n_pad) + feat, ch.pad, device=device, dtype=dt)
                f_pad = torch.full((k, m_pad) + feat, ch.pad, device=device, dtype=dt)
            r_pad[:, :N] = ref[n]                              # BROADCAST: one query, k rows
            c = cnt[n].index_select(0, rows)
            if start[n] is None:                               # dense block: a row select
                S = int(src[n].shape[1])
                f_pad[:, :S] = src[n].index_select(0, rows)
            else:
                gather_fill(f_pad, src[n], start[n].index_select(0, rows), c)
            chans[n] = Batch(r_pad, f_pad,
                             torch.full((k,), N, dtype=torch.int32, device=device),
                             c.to(torch.int32))
            pads += [n_pad, m_pad]
        if avoid is not None:
            Ka = int(avoid.shape[0])
            a_pad = torch.zeros(k, _band_key(Ka) or _BAND, 3, device=device, dtype=torch.float32)
            a_pad[:, :Ka] = avoid
            chans["avoid"] = Batch(a_pad, None,
                                   torch.full((k,), Ka, dtype=torch.int32, device=device), None)
        cs = None if const_seeds is None else _const_seed_batch(const_seeds, k, device)
        so = term_self_overlaps(spec, chans, params, ref_shared=True)   # once per bucket

        def _proc(_s, _k, _ch=chans, _cs=cs, _so=so):
            sl = slice(_s, _s + _k)
            sub = {n: Batch(b.ref[sl], None if b.fit is None else b.fit[sl], b.n_real[sl],
                            None if b.m_real is None else b.m_real[sl])
                   for n, b in _ch.items()}
            return align(spec, sub, params=params, num_seeds=n_seeds, steps_fine=steps_fine,
                         lr=float(params["lr"]), early_stop_patience=es_patience,
                         early_stop_tol=early_stop_tol, ref_shared=True,
                         seeds=None if _cs is None else (_cs[0][sl], _cs[1][sl]),
                         self_overlaps=[None if p is None else (p[0][sl], p[1][sl])
                                        for p in _so])
        sc, qb, tb = _subbatched_align(_proc, k, key=(mode,) + tuple(pads) + (n_seeds,),
                                       device=device, pose_cap=pose_cap, seeds=n_seeds)
        out_scores[rows_np] = sc.detach().cpu().numpy().astype(float)
        out_q.index_copy_(0, rows, qb)
        out_t.index_copy_(0, rows, tb)

    SE3 = quaternions_to_SE3_batch(out_q.cpu(), out_t.cpu()).detach().numpy()
    return out_scores, SE3


def _make_array_aligner(mode: str):
    def fn(ref: dict, fit: dict, **kw):
        return align_arrays(mode, ref, fit, **kw)
    fn.__name__ = fn.__qualname__ = f"align_batch_{mode}_arrays"
    fn.__doc__ = (f"Array-native ``{mode}`` aligner: ``(ref channel tensors, fit "
                  "(flat, off) pairs) -> (scores, SE3)``. A thin name over align_arrays.")
    return fn


for _m in SPECS:
    globals()[f"align_batch_{_m}_arrays"] = _make_array_aligner(_m)
del _m
