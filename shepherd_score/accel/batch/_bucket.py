"""Mode-agnostic adaptive bucketer for the batched aligners (speed_plan.md P1-4).

The batched aligners group same-size pairs into a padded workspace + one kernel launch.
Historically every ``_align_batch_<mode>`` hand-rolled that grouping as a fixed 16-atom band
key (see ``ADDING_A_FAST_MODE.md`` step 11). A *fixed* band over-fragments a wide size
distribution into many under-occupied launches, and over-pads a tight one.

This module replaces that hand-rolling with ONE planner that works for any mode -- the seven
that exist today and any added later -- driven by a small declarative :class:`PadSpec` the
driver supplies inline. A future mode gets adaptive bucketing by calling :func:`plan_buckets`
with its own field layout; no change to this file is needed.

Why it is RESULT-IDENTICAL (the safety property that makes a cost model legitimate)
----------------------------------------------------------------------------------
The overlap kernels are one-CTA-per-pose and mask padding to the real counts
(``N_real``/``M_real``); seeds come from ``batched_seeds_torch`` keyed on the *real* counts,
not the pad width. So padding two different-sized molecules into the SAME bucket is
bit-identical -- the only cost is masked lanes still executing the tile loop. Bucketing is
therefore a pure optimisation with no correctness constraint:

    minimise   sum_b  K_b * work(pad_b)        (total padded compute)
             + lambda * (number of buckets)    (launch / occupancy penalty)

subject to every pad snapping to a ``_band_key`` multiple (so the Triton autotune cache,
keyed on ``(N_pad, M_pad)``, stays small).

Three field roles (a :class:`PadField` via the three dicts on :class:`PadSpec`)
------------------------------------------------------------------------------
* ``merge``     -- cost-driving, keyed, and adaptively merged across (e.g. ref/fit cloud
                   sizes). These enter ``work()``.
* ``partition`` -- must be uniform within a bucket but is NOT a cost dim and is NEVER merged
                   across (e.g. the translation-center count ``tc``, which fixes the seed
                   count). Buckets only merge with others sharing the same partition bands.
* ``masked``    -- cheap, padded to the bucket's max, not part of the key (e.g. vol_color
                   pharmacophore anchors, which are Dummy-typed and masked out).

Env knobs (all default to preserving/raising current behaviour):
* ``FINE_ADAPTIVE_BUCKETS=0``    -> legacy: one bucket per exact band cell (no merging).
* ``FINE_BUCKET_MAX_PAD_WASTE``  -> opportunistic-merge cap on added padded work (default 0.15).
* ``FINE_BUCKET_MIN_WAVE`` / ``FINE_BUCKET_CTAS_PER_SM`` -> occupancy floor (force-merge a
  bucket whose pose count ``K*seeds`` is below one CTA wave).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from ._pad import _band_key

# --- env knobs --------------------------------------------------------------------------
_ADAPTIVE = os.environ.get("FINE_ADAPTIVE_BUCKETS", "1") != "0"
_MAX_PAD_WASTE = float(os.environ.get("FINE_BUCKET_MAX_PAD_WASTE", "0.15"))
_CTAS_PER_SM = int(os.environ.get("FINE_BUCKET_CTAS_PER_SM", "16"))
# Cap a merged bucket at this many CTA waves. Merging exists ONLY to fill under-occupied
# buckets toward full GPU occupancy; once a bucket fills ~this many waves we stop, so we never
# (a) merge two already-full buckets -- at compute-bound large batch that only adds masked-lane
# padding waste (measured ~5-10% large-N regression), nor (b) grow an unbounded giant bucket
# that wastes compute and stresses the heavy kernels' indexing/memory (an illegal-access crash
# was observed for vol_and_surf_esp screen @100k before this cap).
_MERGE_MAX_WAVES = float(os.environ.get("FINE_BUCKET_MAX_WAVES", "2"))
# Cap the UPFRONT held memory (padded clouds + hoisted seeds) per bucket. Occupancy merging
# never splits a full same-size cell, so at large N one cell becomes a single giant bucket whose
# hoisted pad+seeds -- held resident across the WHOLE fine loop -- starve _subbatched_align's
# chunks: the held bytes inflate the measured per-pair footprint and shrink each chunk, an
# occupancy death-spiral (measured surf 100k=11.5k/s, fp=1.6MB -> 300k=6.9k/s, fp=4.1MB -> 1M OOM).
# Splitting an oversized bucket into <=cap-byte pieces is result-identical (pairs are independent,
# same as the sub-batcher) and bounds the upfront so chunks stay large.
#
# The cap is ADAPTIVE: a fraction of the device's FREE memory read at call time (mem_get_info),
# NOT a fixed literal -- so buckets shrink on a smaller/busier GPU (or under multi-process
# sharing) and grow on a big idle one, and the upfront always leaves the rest for the fine loop.
# FINE_BUCKET_UPFRONT_FRAC sets the fraction (default 0.25); FINE_BUCKET_MAX_UPFRONT_GB, if set,
# is an optional HARD CEILING on top of the adaptive value (0 disables capping entirely).
# The fraction bounds the bucket's PEAK (upfront held + seed-gen float64 transient, see per_pair
# below) -- both freed before the fine loop, which _subbatched_align budgets separately -- so
# 0.25 leaves ample room and keeps the light modes' 1M in a single bucket (no over-split dip)
# while sizing the heavy modes (surf) small enough that their seed-gen transient can't OOM.
_UPFRONT_FRAC = float(os.environ.get("FINE_BUCKET_UPFRONT_FRAC", "0.25"))
_UPFRONT_CEIL_GB = os.environ.get("FINE_BUCKET_MAX_UPFRONT_GB")   # None -> no fixed ceiling
_UPFRONT_FLOOR_BYTES = 256 * (1024 ** 2)                          # never split below 256 MB


def _max_upfront_bytes(device) -> float:
    """Per-bucket upfront-memory cap, derived from the device's current FREE memory."""
    if _UPFRONT_CEIL_GB is not None and float(_UPFRONT_CEIL_GB) <= 0:
        return float("inf")                                       # capping disabled
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free = 8 * (1024 ** 3)                                    # conservative if unavailable
    cap = _UPFRONT_FRAC * float(free)
    if _UPFRONT_CEIL_GB is not None:
        cap = min(cap, float(_UPFRONT_CEIL_GB) * (1024 ** 3))
    return max(_UPFRONT_FLOOR_BYTES, cap)


def _cap_upfront(buckets: list, spec: "PadSpec", device) -> list:
    """Split any bucket whose hoisted pad+seed footprint exceeds the adaptive upfront cap."""
    if getattr(device, "type", None) != "cuda":
        return buckets
    max_bytes = _max_upfront_bytes(device)
    if max_bytes == float("inf"):
        return buckets
    out: list = []
    for b in buckets:
        # Peak bytes/pair = UPFRONT held (padded coords 12 B/pt + seed q/t 28 B) PLUS the
        # SEED-GEN transient. batched_seeds_torch converts the clouds to float64 and materializes
        # a 4x fit-cloud expansion (A64 + B64 + fit4 + masks) -- a large transient run on the WHOLE
        # bucket that is NOT sub-batched, so it must bound the bucket size too. Omitting it OOMed
        # surf kernel_screen @1M once the adaptive cap grew past the old fixed 1.5 GB. ~176 B per
        # point of the LARGEST cloud covers that float64 transient (24*N + 152*M, upper-bounded by
        # the max merge pad); it scales with cloud size, so heavy modes (surf) get small buckets
        # and light modes (vol) stay large.
        merge_pads = [int(b.pad[n]) for n in spec.merge]
        per_pair = sum(merge_pads) * 12 + int(spec.seeds) * 28 + max(merge_pads) * 176
        max_k = max(1, int(max_bytes // max(1, per_pair)))
        if b.K <= max_k:
            out.append(b)
        else:
            for s in range(0, b.K, max_k):
                out.append(Bucket(b.members[s:s + max_k], dict(b.pad)))
    return out


def _min_wave(device) -> int:
    """Pose count that fills one CTA wave on ``device``. A bucket below this underfills the
    GPU, so it is force-merged. Overridable with ``FINE_BUCKET_MIN_WAVE``; on CPU it is 1
    (no occupancy concept), so the floor never fires and only the pad-waste cap merges."""
    env = os.environ.get("FINE_BUCKET_MIN_WAVE")
    if env is not None:
        return max(1, int(env))
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        return max(1, sm * _CTAS_PER_SM)
    except Exception:
        return 2048


# --- spec + bucket ----------------------------------------------------------------------
@dataclass
class PadSpec:
    """Declarative description of how a mode pads + buckets. Built inline by each
    ``_align_batch_<mode>`` at the same place the old hand-rolled band key lived.

    Parameters
    ----------
    merge : dict[str, Callable]
        name -> ``f(item) -> int``. Cost-driving cloud sizes; keyed and adaptively merged.
    seeds : int
        Poses per pair (``num_seeds``). Used only for the occupancy floor (``K*seeds``).
    work : Callable | None
        ``f(pad: dict[str, int]) -> float`` estimating per-pose kernel cost from the bucket's
        pad sizes. Default = product of the ``merge`` pad sizes (the right model for the
        single-channel ``N_pad*M_pad`` kernels). Combos pass a sum-of-channel-products.
    partition : dict[str, Callable]
        name -> ``f(item) -> int``. Uniform-required, non-cost dims (e.g. ``tc``). Never
        merged across.
    masked : dict[str, Callable]
        name -> ``f(item) -> int``. Cheap dims padded to the bucket max, not keyed.
    """
    merge: dict
    seeds: int
    work: Callable | None = None
    partition: dict = field(default_factory=dict)
    masked: dict = field(default_factory=dict)

    def _all(self):
        return (*self.merge, *self.partition, *self.masked)

    def sizes(self, item) -> dict:
        out = {}
        for n, fn in self.merge.items():
            out[n] = int(fn(item))
        for n, fn in self.partition.items():
            out[n] = int(fn(item))
        for n, fn in self.masked.items():
            out[n] = int(fn(item))
        return out

    def work_of(self, pad: dict) -> float:
        if self.work is not None:
            return float(self.work(pad))
        v = 1.0
        for n in self.merge:
            v *= max(1, int(pad[n]))
        return v


@dataclass
class Bucket:
    members: list
    pad: dict                       # field name -> padded size (a _band_key multiple)

    @property
    def K(self) -> int:
        return len(self.members)


# --- the planner ------------------------------------------------------------------------
def plan_buckets(items, spec: PadSpec, device) -> list[Bucket]:
    """Partition ``items`` into padded buckets minimising padded work + launch penalty.

    Result-identical to any finer partition (kernels mask padding). The per-item binning is
    O(N) and on the host, so it is written TIGHT (no per-item dict, banding inlined): for the
    host-sensitive screen path the planner must add ~no overhead over the legacy band key.
    The merge then runs on the small OCCUPIED-CELL set (tens-hundreds even for 1e6 items).

    A cell is stored as ``[members, merge_max(list), partition_exact(tuple), masked_max(list)]``.
    merge dims key on their band; partition dims key on their EXACT value (they must be uniform
    within a bucket -- e.g. tc fixes the seed count -- so banding them would wrongly group
    tc=17 with tc=20).
    """
    if not items:
        return []

    mfns = list(spec.merge.values()); mnames = list(spec.merge)
    pfns = list(spec.partition.values()); pnames = list(spec.partition)
    xfns = list(spec.masked.values()); xnames = list(spec.masked)
    nm = len(mfns); nx = len(xfns)
    B = 16                                              # == _pad._BAND; inlined for the hot loop

    cells: dict = {}
    for it in items:
        msz = [fn(it) for fn in mfns]
        mk = tuple(((v + B - 1) // B) * B for v in msz)
        pk = tuple(fn(it) for fn in pfns)
        key = (mk, pk)
        c = cells.get(key)
        if c is None:
            cells[key] = [[it], msz, pk, [fn(it) for fn in xfns]]
        else:
            c[0].append(it)
            mm = c[1]
            for i in range(nm):
                if msz[i] > mm[i]:
                    mm[i] = msz[i]
            if nx:
                xm = c[3]
                for i in range(nx):
                    v = xfns[i](it)
                    if v > xm[i]:
                        xm[i] = v

    def _pad_of(c):
        mm, pk, xm = c[1], c[2], c[3]
        pad = {mnames[i]: _band_key(mm[i]) for i in range(nm)}
        for i in range(len(pnames)):
            pad[pnames[i]] = pk[i]
        for i in range(nx):
            pad[xnames[i]] = _band_key(xm[i])
        return pad

    if not _ADAPTIVE:                                   # legacy: one bucket per exact cell
        return _cap_upfront([Bucket(c[0], _pad_of(c)) for c in cells.values()], spec, device)

    # group by partition signature (never merge across different partition values), then
    # greedy-merge within each group along the merge dims.
    groups: dict = {}
    for (mk, pk), c in cells.items():
        groups.setdefault(pk, []).append(Bucket(c[0], _pad_of(c)))

    mw = _min_wave(device)
    out: list[Bucket] = []
    for buckets in groups.values():
        out.extend(_merge_group(buckets, spec, mw))
    return _cap_upfront(out, spec, device)


def _merge_group(buckets: list[Bucket], spec: PadSpec, min_wave: int) -> list[Bucket]:
    """Repeatedly sort buckets by their merge-dim pads (so neighbours are shape-similar) and
    fold each into its predecessor while the merge is profitable, until nothing changes."""
    merge_names = list(spec.merge)
    changed = True
    while changed and len(buckets) > 1:
        changed = False
        buckets.sort(key=lambda b: tuple(b.pad[n] for n in merge_names))
        folded: list[Bucket] = [buckets[0]]
        for b in buckets[1:]:
            a = folded[-1]
            if _should_merge(a, b, spec, min_wave):
                folded[-1] = _merge(a, b, spec)
                changed = True
            else:
                folded.append(b)
        buckets = folded
    return buckets


def _should_merge(a: Bucket, b: Bucket, spec: PadSpec, min_wave: int) -> bool:
    a_poses = a.K * spec.seeds
    b_poses = b.K * spec.seeds
    # Merge ONLY two UNDER-occupied buckets, to fill the GPU toward a CTA wave. If EITHER is
    # already full (>= one wave), leave it: merging a full bucket gives no launch/occupancy
    # benefit and only adds masked-lane padding waste at compute-bound (large) batch -- the
    # ~5-10% large-N regression -- and can pad-up a saturated kernel (which crashed
    # vol_and_surf_esp screen @100k). So full buckets fall back to legacy bucketing.
    if a_poses >= min_wave or b_poses >= min_wave:
        return False
    # Bound the merged size to ~_MERGE_MAX_WAVES waves: enough to fill the GPU, never an
    # unbounded giant bucket. NO pad-waste gate here -- both buckets are under-occupied, so the
    # poses (and thus the masked-lane waste) are few, and the occupancy win dominates. A waste
    # gate here would block exactly the size-diverse merges that fill the wave (it cost the
    # small-batch wins when tried). The greedy sort merges size-adjacent buckets first, so the
    # pad-up stays gradual.
    return a_poses + b_poses <= _MERGE_MAX_WAVES * min_wave


def _merge(a: Bucket, b: Bucket, spec: PadSpec) -> Bucket:
    return Bucket(a.members + b.members, {n: max(a.pad[n], b.pad[n]) for n in a.pad})
