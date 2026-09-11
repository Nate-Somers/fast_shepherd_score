from __future__ import annotations

import os
import torch
# Kernels are dispatched per-call by tensor device (Triton on CUDA, numba on CPU), so one
# process can run both -- e.g. backend="numba" runs CPU tensors through the numba kernels
# even on a GPU box.
from ..kernels.dispatch import (
    overlap_score_grad_se3_batch, fused_adam_qt_with_tangent_proj,
    _batch_self_overlap,
)
from ._common import batched_seeds_torch, _update_best
from ._graphed import _GraphedFineBase, run_graphed, graph_cap
from typing import Optional
from .._stats import record as _record_steps

torch.backends.cuda.matmul.allow_tf32 = True


#: Read each molecule's coordinates ONCE instead of once per seed. The fine loop expands
#: (K, N_pad, 3) -> (K*S, N_pad, 3) and materialises it, so a 100k x 10-seed screen holds ~384 MB
#: per coordinate buffer of ten identical copies and re-reads all of it every step; the kernel can
#: index molecule ``pid // S`` instead. Same bytes, same arithmetic, different address --
#: bit-identical by construction. Opt-in so it is trivial to A/B and revert.
_DEDUP_SEED_COORDS = os.environ.get("FSS_SEED_COORD_DEDUP", "0") == "1"

#: Poses of ONE molecule per CTA. The single-pose kernel is latency/occupancy-bound -- ~5% of an
#: L40S fp32 peak, ~17% of its SFU, ~1024 pair-evals per CTA with <= 4 warps -- and removing its
#: coordinate traffic was worth only 1.01x, which ruled memory out. This gives each CTA POSES times
#: the work, amortising the quaternion->rotmat build, the tile loads and the loop overhead. Needs
#: the deduped layout (implies it below) and SEEDS % POSES == 0. 1 = the original one-CTA-per-pose
#: kernel, untouched.
_POSES_PER_CTA = (int(os.environ["FSS_POSES_PER_CTA"])
                  if os.environ.get("FSS_POSES_PER_CTA") else None)

#: Per-mode default poses-per-CTA, applied when FSS_POSES_PER_CTA is unset. ENABLED ONLY WHERE
#: MEASURED TO PAY, and it is not free: multi-pose is NOT bit-identical (the 3-D reduction rounds
#: differently), so this trades score stability for speed.
#:   surf  POSES=8 -> 1.70x (21,072 -> 35,842 aligns/s), divergence 1.16e-03
#:   vol   OMITTED: only 1.04-1.07x, for a LARGER 1.60e-02 divergence -- not worth it
#: surf_esp / vol_esp are on the ESP kernel and measured NEGATIVE (0.92-0.94x), so they are not
#: here either. Set FSS_POSES_PER_CTA=1 to force the legacy one-CTA-per-pose kernel everywhere.
_MODE_POSES = {"surf": 8}

#: Run the whole fine step -- overlap, Tanimoto, best-pose tracking and the tangent-projected
#: Adam -- as ONE kernel launch instead of seventeen, by giving the overlap kernel a scalar
#: epilogue. MEASURED AND REJECTED; kept DEFAULT OFF because the null result is the useful part.
#:
#: L40S, vol screen N=100,000, same process, alternating A/B (job 22594730):
#:     launches   5,938 -> 508        (the fusion does what it says)
#:     device     87.7 ms -> 88.8 ms  (the fused kernel alone is 83.2 ms vs 76 ms for the
#:                                     seventeen it replaces)
#:     wall       723,077 -> 726,211 aligns/s = 1.004x
#:     parity     74,180 / 100,000 scores move, max|d| 6.5e-03, one molecule past 1%
#:
#: So the fine step was NEVER LAUNCH-BOUND: its 0.143 us/mol of tail was real, well-parallelised
#: bandwidth work, and moving it inside the overlap CTA makes ~128 threads redundantly execute a
#: per-pose scalar epilogue (7 sqrt, 3 divides, 20 loads/stores) that one thread per pose used to
#: do. The launch count fell 12x and bought nothing, which is the measurement worth keeping.
#: ``_tanimoto_best_adam`` (kernels/shape_triton.py) is the version that DOES pay: same fusion,
#: one thread per pose, no redundancy.
_FUSED_STEP = os.environ.get("FSS_FUSED_STEP", "0") == "1"

#: Run the fine step's TAIL -- Tanimoto, best-pose tracking, tangent-projected Adam -- as one
#: pose-parallel kernel instead of twelve torch elementwise ops plus ``_adam_qt``. Unlike
#: ``_FUSED_STEP`` above this keeps one thread per pose, so it adds no redundant work; it only
#: stops seven per-pose intermediates making a round trip to HBM.
#:
#: DEFAULT OFF, and the reason is not speed. It works -- 1.012x alone and ~1.045x once the
#: sub-batch pose cap raises the launch count (L40S, vol screen; jobs 22595396/22596016) -- but
#: it is NOT bit-identical: 2,412 of 100,000 scores move by up to 2.1e-04 on its own, and with
#: the pose cap on, 74,148 move by up to 6.5e-03. The arithmetic is the same operations in the
#: same order, so the cause is a last-place rounding difference between torch's and Triton's
#: float divide; what turns 1 ulp into 6.5e-03 is that 30 Adam steps from a marginally different
#: gradient can land a seed in a different local optimum, and the per-pair best is a max over
#: seeds. Everything else in this campaign is bit-identical, so this stays opt-in rather than
#: spending that property for 4%.
_FUSED_TAIL = os.environ.get("FSS_FUSED_TAIL", "0") == "1"

#: Hand the overlap kernel its coordinates as (K, 3, P_pad) -- the three coordinate planes
#: contiguous -- instead of the interleaved (K, P_pad, 3). Nsight Compute on the shipped kernel
#: (L40S, job 22595748) reports the L1/TEX pipe 91.1% busy while DRAM sits at 5.0%, and names
#: the cause outright: "only 8.9 of the 32 bytes transmitted per sector are utilized by each
#: thread ... caused by a stride between threads", estimated speedup 65%. A tile load of BLOCK
#: interleaved points touches BLOCK*3 floats' worth of sectors to use BLOCK of them; the same
#: load on a coordinate plane is contiguous.
#:
#: MEASURED AND REJECTED (job 22596016); DEFAULT OFF, kept because the null is informative.
#: Bit-identical as predicted -- V, dQ and dT all matched to 0.000e+00 over 81,920 poses -- and
#: SLOWER: 0.902x at 32x32 (0.132 -> 0.146 ms) and 0.974x at 32x48. So the sector-utilisation
#: figure ncu reports does NOT translate into time here: at BLOCK=16 a tile's 16 interleaved
#: points are 192 contiguous bytes that L1 serves at an 86.5% hit rate, while the three
#: coordinate planes of the SoA layout are three separate 128-byte lines. The "wasted" bytes
#: per sector are bytes the very next lane consumes.
#:
#: The transpose is paid ONCE per bucket, in the graph's ``_load``, against 30 fine steps that
#: read the result -- so the cost is the kernel's, not the layout change's.
_SOA_COORDS = os.environ.get("FSS_SOA_COORDS", "0") == "1"

_FUSED_FN = None
_FUSED_TAIL_FN = None


def _fused_tail_fn():
    """Resolve the fused Tanimoto/best/Adam tail lazily (Triton/CUDA only, like below)."""
    global _FUSED_TAIL_FN
    if _FUSED_TAIL_FN is None:
        from ..kernels.shape_triton import tanimoto_best_adam
        _FUSED_TAIL_FN = tanimoto_best_adam
    return _FUSED_TAIL_FN


def _fused_step_fn():
    """Resolve the fused single-launch fine step lazily.

    ``kernels.dispatch`` deliberately routes by tensor device and has no CPU twin for this
    one (the numba fine loop is already fully fused -- see ``cpu_fused.cpu_fused_shape``), so
    importing it here rather than at module scope keeps this module importable on a box with
    no Triton, exactly as the dispatch indirection does for every other kernel."""
    global _FUSED_FN
    if _FUSED_FN is None:
        from ..kernels.shape_triton import fused_fine_step
        _FUSED_FN = fused_fine_step
    return _FUSED_FN


@torch.no_grad()
def _overlap_in_chunks(A, B, q, t, *, alpha: float = 0.81,
                       N_real: torch.Tensor | None = None,
                       M_real: torch.Tensor | None = None,
                       NEED_GRAD = True, seeds_per_mol: int = 1,
                       poses_per_cta: int = 1):
    """
    Evaluate the fused overlap kernel on an arbitrary-long list of
    orientations, slicing the list so that each launch respects the
    CUDA `grid.z ≤ 65 535` limit.

    Parameters
    ----------
    A, B : (K,N,3) / (K,M,3)   – padded coordinate blocks
    q, t : (K,4) / (K,3)       – quaternions & translations
    N_real, M_real : (K,) int32 tensors holding the *true* atom counts
                     (rows beyond those indices are padding).  If None,
                     we assume no padding.
    Returns
    -------
    VAB : (K,)    dQ : (K,4)    dT : (K,3)     — all contiguous on GPU
    """
    K = q.shape[0]                       # POSES; A has K // seeds_per_mol molecules
    if N_real is not None:
        N_real = N_real.to(torch.float32).contiguous()
    if M_real is not None:
        M_real = M_real.to(torch.float32).contiguous()

    out_V  = torch.empty(K,        device=A.device, dtype=A.dtype)
    out_dQ = torch.empty_like(q)
    out_dT = torch.empty_like(t)

    S = int(seeds_per_mol)
    # Chunk on POSE index, but keep every molecule's seed group whole: a boundary mid-group
    # would make ``pid // S`` inside the launch address the wrong molecule.
    CHUNK = 65_535 if S == 1 else max(S, (65_535 // S) * S)

    for start in range(0, K, CHUNK):
        end = min(start + CHUNK, K)
        ms, me = start // S, end // S          # molecule slice for this pose slice

        extra = {} if S == 1 else {"seeds_per_mol": S}   # CPU kernels take no such kwarg
        if int(poses_per_cta) > 1:
            extra["poses_per_cta"] = int(poses_per_cta)
        V, dQ, dT = overlap_score_grad_se3_batch(
            A[ms:me], B[ms:me],
            q[start:end], t[start:end],
            alpha=alpha,
            N_real=N_real[ms:me],
            M_real=M_real[ms:me],
            NEED_GRAD=NEED_GRAD, **extra)

        out_V[start:end]  = V
        out_dQ[start:end] = dQ
        out_dT[start:end] = dT

    return out_V, out_dQ, out_dT


def _self_overlap_in_chunks(P_pad, N_real, alpha=0.81):
    K = P_pad.size(0)
    CHUNK = 65_535                     # hardware limit
    V_all = torch.empty(K,
                        device=P_pad.device,
                        dtype=P_pad.dtype)
    for s in range(0, K, CHUNK):
        e = min(s + CHUNK, K)
        V_all[s:e] = _batch_self_overlap(
            P_pad[s:e], N_real[s:e], alpha)   # ← original function
    return V_all


class _GraphedFineSurf(_GraphedFineBase):
    """Capture one in-place surf/vol fine step into a CUDA graph; replay = N steps.

    All loop-carried state (q/t, Adam moments, best_*) lives in persistent buffers
    updated in place; per-step temporaries use out= so there are no host syncs and
    no reallocation -- the requirements for graph capture. Inputs (A/B/seeds/...)
    are copied into the buffers before each replay, so one captured graph serves
    every bucket of the same (N_pad, M_pad, P) shape. capture()/run() are inherited
    from _GraphedFineBase.
    """

    def __init__(self, N_pad, M_pad, P, steps, alpha, lr, device, seeds=1, poses=1,
                 fused=False, soa=False):
        self.alpha = float(alpha); self.lr = float(lr)
        self.fused = bool(fused)
        # SoA is a property of the OVERLAP kernel only: the multi-pose variant and the fused
        # single-launch step both read the interleaved layout, so they keep it.
        self.soa = bool(soa) and not self.fused and max(1, int(poses)) == 1
        f = lambda *s: torch.empty(*s, device=device, dtype=torch.float32)
        # Coordinate + real-count buffers are per MOLECULE; pose state stays per pose.
        self.S = max(1, int(seeds))
        self.P_cta = max(1, int(poses))
        nm = P // self.S
        if self.soa:
            self.A = f(nm, 3, N_pad); self.B = f(nm, 3, M_pad)
        else:
            self.A = f(nm, N_pad, 3); self.B = f(nm, M_pad, 3)
        self.Nr = torch.empty(nm, device=device, dtype=torch.int32)
        self.Mr = torch.empty(nm, device=device, dtype=torch.int32)
        self.norm = f(P)
        self.qs = f(P, 4); self.ts = f(P, 3)            # seeds (replay start state)
        self.q = f(P, 4); self.t = f(P, 3)
        self.mq = f(P, 4); self.vq = f(P, 4); self.mt = f(P, 3); self.vt = f(P, 3)
        self.best = f(P); self.bq = f(P, 4); self.bt = f(P, 3)
        self.denom = f(P); self.d2 = f(P); self.score = f(P); self.scale = f(P)
        self.better = torch.empty(P, device=device, dtype=torch.bool)
        self.gq = f(P, 4); self.gt = f(P, 3)
        super().__init__(steps)

    def _step(self):
        if self.fused:
            # ONE launch: the overlap kernel's own epilogue does the Tanimoto, the best-pose
            # update and the Adam step, so dQ/dT never reach memory. Same buffers, same
            # order of operations -- see _gauss_overlap_se3_fused_step.
            _fused_step_fn()(
                self.A, self.B, self.q, self.t, self.norm,
                self.best, self.bq, self.bt, self.mq, self.vq, self.mt, self.vt,
                alpha=self.alpha, N_real=self.Nr, M_real=self.Mr, lr=self.lr,
                seeds_per_mol=self.S)
            return
        extra = {} if self.S == 1 else {"seeds_per_mol": self.S}
        if self.P_cta > 1:
            extra["poses_per_cta"] = self.P_cta
        if self.soa:
            extra["soa"] = True
        VAB, dQ, dT = overlap_score_grad_se3_batch(
            self.A, self.B, self.q, self.t, alpha=self.alpha, N_real=self.Nr, M_real=self.Mr,
            **extra)
        self._tanimoto_adam_tail(VAB, dQ, dT)

    def _tanimoto_adam_tail(self, VAB, dQ, dT):
        """Single-channel Tanimoto score + best-pose tracking + tangent-projected Adam, all
        in-place into persistent buffers. Shared by surf/vol and the ESP subclass (whose
        only difference is the fused shape+ESP overlap kernel that produces VAB/dQ/dT)."""
        if _FUSED_TAIL and VAB.is_cuda:
            # Same twelve ops plus the Adam, one thread per pose, one launch. Every
            # intermediate below (denom/score/d2/scale/better/gq/gt) is a full (P,) or (P,4)
            # HBM round trip in the unfused form; measured at 0.143 us/mol of device time on
            # a vol screen at N=100,000 against 0.4977 for the overlap kernel itself.
            _fused_tail_fn()(VAB, dQ, dT, self.norm, self.q, self.t,
                             self.best, self.bq, self.bt,
                             self.mq, self.vq, self.mt, self.vt, self.lr)
            return
        torch.sub(self.norm, VAB, out=self.denom)
        torch.div(VAB, self.denom, out=self.score)
        torch.mul(self.denom, self.denom, out=self.d2)
        torch.div(self.norm, self.d2, out=self.scale)
        torch.gt(self.score, self.best, out=self.better)
        torch.where(self.better, self.score, self.best, out=self.best)
        bm = self.better.unsqueeze(1)
        torch.where(bm, self.q, self.bq, out=self.bq)
        torch.where(bm, self.t, self.bt, out=self.bt)
        torch.mul(dQ, self.scale.unsqueeze(1), out=self.gq); self.gq.neg_()
        torch.mul(dT, self.scale.unsqueeze(1), out=self.gt); self.gt.neg_()
        fused_adam_qt_with_tangent_proj(self.q, self.t, self.gq, self.gt,
                                        self.mq, self.vq, self.mt, self.vt, self.lr)

    def _load(self, A, B, Nr, Mr, norm, qs, ts):
        # One transposing copy per BUCKET against 30 fine steps that read the result; the
        # driver keeps handing us (K, P_pad, 3) so nothing upstream has to know about SoA.
        if self.soa:
            self.A.copy_(A.transpose(1, 2)); self.B.copy_(B.transpose(1, 2))
        else:
            self.A.copy_(A); self.B.copy_(B)
        self.Nr.copy_(Nr.to(torch.int32)); self.Mr.copy_(Mr.to(torch.int32))
        self.norm.copy_(norm); self.qs.copy_(qs); self.ts.copy_(ts)

    def _reset(self):
        self.q.copy_(self.qs); self.t.copy_(self.ts)
        self.mq.zero_(); self.vq.zero_(); self.mt.zero_(); self.vt.zero_()
        self.best.fill_(-float('inf')); self.bq.copy_(self.qs); self.bt.copy_(self.ts)

    def _result(self):
        return self.best, self.bq, self.bt


def _run_graphed_fine(A_k, B_k, q_seed, t_seed, N_k, M_k, norm, alpha, lr, steps, N_pad, M_pad, P,
                      es_patience=0, es_tol=1e-5, es_seeds=0, seeds=1, poses=1, fused=False,
                      soa=False):
    # ``fused`` joins the key: the two variants capture DIFFERENT graphs (one kernel vs
    # seventeen) into the same process-wide cache, so a run that toggles the flag must not
    # replay the other one's graph.
    # ``_FUSED_TAIL`` joins it too: the tail variant is baked in at CAPTURE time, so a cached
    # graph would keep replaying whichever one was live when it was captured.
    key = (A_k.device.index, "surf", N_pad, M_pad, P, steps, round(float(alpha), 4),
           round(float(lr), 5), int(seeds), int(poses), bool(fused), bool(_FUSED_TAIL),
           bool(soa))
    return run_graphed(
        lambda: _GraphedFineSurf(N_pad, M_pad, P, steps, alpha, lr, A_k.device, seeds=seeds,
                                 poses=poses, fused=fused, soa=soa),
        key, (A_k, B_k, N_k, M_k, norm, q_seed, t_seed),
        es_patience=es_patience, es_tol=es_tol, es_seeds=es_seeds)


def coarse_fine_align_many(
        A_batch, B_batch, VAA, VBB, *,
        alpha: float = 0.81,
        num_seeds: int = 50,
        steps_fine: int = 100,
        lr: float = 0.075,
        N_real: torch.Tensor | None = None,
        M_real: torch.Tensor | None = None,
        early_stop_patience: int = 2,   # vol/surf converge fast; esp/pharm use 5
        early_stop_tol: float = 1e-5,
        seeds: tuple | None = None,
        prune_after: int = 0, prune_keep: int = 0,
        mode: str | None = None):
    """
    Vectorised padding-aware alignment over a batch of (A, B) pairs.

    Strategy (matches the reference optimiser, fully batched): build the
    reference seed set -- identity + 4 principal-axis quaternions + Fibonacci
    rotations, each with a COM-aligning translation -- then fine-optimise EVERY
    seed and take the per-pair best.

    No coarse-grid top-k prune
    --------------------------
    Seeds are never ranked by their raw, un-optimised overlap: that score does
    not predict a seed's post-optimisation result, so ranking on it discards the
    seed sitting in the true basin for pseudo-symmetric molecules and keeps
    decoys that plateau at a lower local optimum. Every seed therefore enters the
    fine loop, and the per-pair max is taken.

    Parameters
    ----------
    A_batch, B_batch : (B, N_pad, 3) / (B, M_pad, 3)  atom coordinates
    VAA, VBB         : (B,)  pre-computed Gaussian self-overlaps
    num_seeds        : int   reference seed count (identity + 4 PCA + Fibonacci)
    N_real, M_real   : (B,)  optional true atom counts
    early_stop_patience : int  consecutive checks (one every 5 steps) in which NO pair
                       improved, before the loop stops. A pair improves when its own
                       best -- the max over its own seeds -- gains more than the tol.
    early_stop_tol : float  minimum improvement threshold
    """
    device = A_batch.device
    BATCH, N_pad, _ = A_batch.shape
    _,     M_pad, _ = B_batch.shape

    if N_real is None:
        N_real = A_batch.new_full((BATCH,), N_pad, dtype=torch.int32)
    if M_real is None:
        M_real = A_batch.new_full((BATCH,), M_pad, dtype=torch.int32)

    # ------------------------------------------------------------------
    # 1) reference seed set (no coarse grid, no flips, no 2nd translation)
    # ------------------------------------------------------------------
    # Seeds may be precomputed once per band and passed in, so the sub-batcher
    # can chunk ONLY the (memory-heavy) fine loop without re-paying the
    # launch-bound seed-gen per chunk. Slicing precomputed per-pair seeds is
    # exactly equivalent to recomputing per chunk (seeds are per-pair independent).
    if seeds is None:
        quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real,
                                             num_seeds=num_seeds)
    else:
        quats, t_seeds = seeds
    S = quats.size(1)

    # ------------------------------------------------------------------
    # 2) fine polishing (Adam-like) on EVERY seed
    # ------------------------------------------------------------------
    # DEDUP: hand the kernel the molecule blocks as they already are and let it index
    # ``pid // S``. The expand+reshape below MATERIALISES S copies of every molecule (and
    # :240 then calls .contiguous() again), which is the traffic this avoids. CUDA + fp32 only:
    # the CPU fused path and fp64 keep the replicated layout.
    # An explicit FSS_POSES_PER_CTA wins; otherwise the per-mode default applies. Multi-pose
    # REQUIRES the deduped layout (the reuse is the shared molecule), so asking for poses turns
    # dedup on rather than silently falling back to the slow path.
    _want_poses = _POSES_PER_CTA if _POSES_PER_CTA is not None else _MODE_POSES.get(mode, 1)
    _dedup = ((_DEDUP_SEED_COORDS or _want_poses > 1) and A_batch.is_cuda
              and A_batch.dtype == torch.float32 and S > 1)
    if _dedup:
        A_k, B_k = A_batch, B_batch
    else:
        A_k = A_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, N_pad, 3)
        B_k = B_batch.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, M_pad, 3)
    q_seed = quats.reshape(-1, 4).contiguous()
    t_seed = t_seeds.reshape(-1, 3).contiguous()

    # real-counts follow the coordinates: per molecule when deduped, per pose otherwise
    N_k = N_real if _dedup else N_real.repeat_interleave(S)
    M_k = M_real if _dedup else M_real.repeat_interleave(S)
    S_fine = S if _dedup else 1
    # POSES needs the deduped layout AND must divide the seed count evenly, so that every pose a
    # CTA handles belongs to the same molecule -- that shared molecule is the reuse being bought.
    P_cta = _want_poses if (_dedup and _want_poses > 1 and S % _want_poses == 0) else 1
    VAA_plus_VBB = (VAA + VBB).repeat_interleave(S)        # invariant in loop
    P = q_seed.shape[0]

    best_score = best_q = best_t = None

    # --- CUDA-graph fast path for the launch-bound regime --------------------
    # Compute-aware cap: one formula covers vol and surf, which share this kernel -- see
    # graph_cap.
    if (A_batch.is_cuda and P <= graph_cap(N_pad * M_pad)
            and A_batch.dtype == torch.float32):
        try:
            best_score, best_q, best_t = _run_graphed_fine(
                A_k.contiguous(), B_k.contiguous(), q_seed, t_seed, N_k, M_k,
                VAA_plus_VBB, alpha, lr, steps_fine, N_pad, M_pad, P,
                es_patience=early_stop_patience, es_tol=early_stop_tol,
                es_seeds=S, seeds=S_fine, poses=P_cta,
                fused=(_FUSED_STEP and P_cta == 1),
                soa=(_SOA_COORDS and P_cta == 1 and not _FUSED_STEP))
        except Exception:
            best_score = None                              # capture failed -> eager

    # --- opt-in CPU (numba) fast path: fully-fused fine loop, NO torch in the hot loop ------
    # Skipped for fp64; any failure falls through to the eager loop. Same seeds/steps as
    # every other path.
    if (best_score is None and not A_batch.is_cuda
            and A_batch.dtype == torch.float32):
        try:
            from ..kernels.cpu_fused import cpu_fused_shape
            best_score, best_q, best_t = cpu_fused_shape(
                A_k, B_k, q_seed, t_seed, N_k, M_k, VAA_plus_VBB, alpha, lr, steps_fine,
                early_stop_patience, early_stop_tol, n_seeds=S)
        except Exception:
            best_score = None                              # fused failed -> eager

    # --- eager fall-back (large P, non-CUDA, fp64, or capture failure) -------
    if best_score is None:
        q_k = q_seed.clone()
        t_k = t_seed.clone()
        m_q = torch.zeros_like(q_k); v_q = torch.zeros_like(q_k)
        m_t = torch.zeros_like(t_k); v_t = torch.zeros_like(t_k)

        best_score = torch.full((len(q_k),), -float('inf'), device=device)
        best_q = q_k.clone()
        best_t = t_k.clone()

        # Early stopping state
        es_patience = early_stop_patience
        es_tol = early_stop_tol
        # Per-pair early-stop baseline: prev_best[k] is pair k's best score as of its
        # last recorded improvement. One entry per PAIR, not per pose.
        prev_best = torch.full((BATCH,), -float('inf'), device=device)
        no_improve_count = 0

        # The fused kernel does overlap + best-update + Adam in one launch, so this loop's
        # per-step torch tail and its ``fused_adam_qt_with_tangent_proj`` are both skipped.
        # The early-stop check still sees exactly the eager value: the Adam update the fused
        # kernel folds in cannot change ``best_score``, which is written before it.
        _eager_fused = (_FUSED_STEP and P_cta == 1 and A_batch.is_cuda
                        and A_batch.dtype == torch.float32)
        _fstep = _fused_step_fn() if _eager_fused else None
        if _eager_fused:
            best_score = best_score.contiguous()
            q_k = q_k.contiguous(); t_k = t_k.contiguous()

        for step in range(steps_fine):
            if _eager_fused:
                _fstep(A_k, B_k, q_k, t_k, VAA_plus_VBB,
                       best_score, best_q, best_t, m_q, v_q, m_t, v_t,
                       alpha=alpha, N_real=N_k, M_real=M_k, lr=lr, seeds_per_mol=S_fine)
            else:
                VAB, dQ, dT = _overlap_in_chunks(
                    A_k, B_k, q_k, t_k,
                    alpha=alpha, N_real=N_k, M_real=M_k, seeds_per_mol=S_fine,
                    poses_per_cta=P_cta)

                denom = VAA_plus_VBB - VAB
                score = VAB / denom
                scale = VAA_plus_VBB / (denom * denom)

                best_score, best_q, best_t = _update_best(
                    score, q_k, t_k, best_score, best_q, best_t)

            # Early-stop check every 5 steps, so the host sync it needs costs one sync per
            # 5 steps, not one per step. Gating only makes the early stop LESS aggressive.
            if step % 5 == 0:
                # PER-PAIR convergence test. `best_score` is (BATCH*S,) laid out
                # pair-major, so .view(BATCH, S) row k is pair k -- the same reshape the
                # result gather uses below. A pair has converged when ITS OWN best (max over
                # its own seeds) stops improving, and the loop may break only once EVERY pair
                # has stalled. A bucket-global max would let one converged pair halt the
                # optimisation of every other pair sharing the bucket.
                cur = best_score.view(BATCH, S).amax(dim=1)
                improved = (cur - prev_best) > es_tol
                # amax stays on-device; this .any() is the ONE host sync per check, exactly
                # where the old .max().item() sync was. No per-pair sync is introduced.
                if not improved.any():
                    no_improve_count += 1
                    if no_improve_count >= es_patience:
                        break
                else:
                    no_improve_count = 0
                # Advance a pair's baseline only where that pair actually improved, as the old
                # rule did. In every measured case this runs at least as long as the bucket-global
                # test it replaces; it is NOT provably never-earlier. A trajectory whose per-check
                # gains straddle es_tol can spend a baseline reset the global rule still holds, and
                # stop one check block (5 steps) sooner. Never seen on real molecules.
                prev_best = torch.where(improved, cur, prev_best)

            if not _eager_fused:
                fused_adam_qt_with_tangent_proj(
                    q_k, t_k,
                    -dQ * scale.unsqueeze(1),
                    -dT * scale.unsqueeze(1),
                    m_q, v_q, m_t, v_t, lr
                )

        # One record per eager fine-loop invocation: value+grad evaluations actually
        # executed (the loop breaks AFTER an evaluation, before that step's Adam update)
        # against the configured budget. No-op unless _stats recording was enabled.
        _ran = (step + 1) if steps_fine else 0
        _record_steps(_ran, steps_fine, _ran < steps_fine)

    # ------------------------------------------------------------------
    # 3) gather final results (using already-tracked best scores)
    # ------------------------------------------------------------------
    final_score = best_score.view(BATCH, S)

    best = final_score.argmax(dim=1)
    sel  = best + torch.arange(BATCH, device=device) * S

    return final_score.flatten()[sel], \
           best_q.view(BATCH, S, 4)[torch.arange(BATCH), best], \
           best_t.view(BATCH, S, 3)[torch.arange(BATCH), best]
