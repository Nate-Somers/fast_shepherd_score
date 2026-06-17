"""
Backend adapters: a uniform ``prepare`` / ``run`` interface over every
(mode x device x granularity) alignment path.

Each backend implements:
    available()        -> (bool, reason)         is this runnable here?
    prepare(cohort)    -> state                  one-time data prep (timed)
    run(state)         -> np.ndarray[n_pairs]     per-pair achieved score (timed)

Fairness notes baked into the adapters
--------------------------------------
* ``prepare`` does the host->device transfer / batch-padding / MoleculePair
  construction.  It is timed once and reported separately, so transfer/padding
  costs are attributed to the path that pays them (the GPU paths) and not hidden
  inside the compute number.

* All single-pair backends share the *same* per-pair config (num_repeats,
  steps, lr family) and the multi/batch backends use the matching batch config.
  The fast/Triton optimiser and the reference torch optimiser use genuinely
  different algorithms (coarse-grid+top-k+fine vs random-restart autograd); we
  do not pretend otherwise -- we report both the achieved score *and* the time
  so the speed/accuracy trade-off is explicit.

* The GPU batch backend goes through the real container ``align_batch_*`` methods
  so it exercises the production size-bucketing.  We additionally record the
  number of buckets the cohort induces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from benchmarks.alignment_bench.workloads import Cohort, PairSpec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BenchConfig:
    alpha: float = 0.81
    lam: float = 0.3                 # ESP weight (held constant across backends)
    num_repeats: int = 32            # random restarts / coarse seeds
    max_steps: int = 100             # torch reference optimiser steps
    steps_fine: int = 75             # fast/batch fine-tuning steps
    topk: int = 30                   # fast/batch: top-k seeds kept after coarse
    lr_cpu: float = 0.1              # torch reference learning rate
    lr_fast: float = 0.075           # fast/batch learning rate
    similarity: str = "tanimoto"
    trans_init: bool = False         # translation-seeded initialisation
    cpu_workers: int = 0             # 0 -> os.cpu_count()


@dataclass
class BackendOutput:
    scores: np.ndarray               # (n_pairs,) achieved score per pair
    n_buckets: Optional[int] = None  # GPU-multi only
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cuda_ok() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def _fast_ok() -> bool:
    if not _cuda_ok():
        return False
    try:
        from shepherd_score.alignment import _torch as _t
        return bool(getattr(_t, "FAST_ALIGNMENT_AVAILABLE", False))
    except Exception:
        return False


def _t(arr, device, dtype=torch.float32):
    return torch.as_tensor(arr, dtype=dtype, device=device)


def _score_to_float(score) -> float:
    if _HAS_TORCH and isinstance(score, torch.Tensor):
        return float(score.detach().reshape(-1)[0].item())
    return float(np.asarray(score).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Per-mode single-pair call wrappers
# ---------------------------------------------------------------------------
def _single_reference(mode: str, pre: dict, cfg: BenchConfig, *, analytical: bool):
    """Call the torch reference optimiser (autograd or analytical) for one pair.

    ``pre`` holds device tensors for this pair.  Returns achieved score (float).
    """
    from shepherd_score import alignment as A
    nr = cfg.num_repeats
    tc = pre.get("trans_centers") if cfg.trans_init else None

    if mode in ("vol", "surf"):
        fn = A.optimize_ROCS_overlay_analytical if analytical else A.optimize_ROCS_overlay
        out = fn(pre["ref_pts"], pre["fit_pts"], cfg.alpha,
                 num_repeats=nr, trans_centers=tc,
                 lr=cfg.lr_cpu, max_num_steps=cfg.max_steps)
        return _score_to_float(out[2])

    if mode == "esp":
        fn = A.optimize_ROCS_esp_overlay_analytical if analytical else A.optimize_ROCS_esp_overlay
        kw = {} if analytical else {"use_fast": False}
        out = fn(pre["ref_pts"], pre["fit_pts"], pre["ref_chg"], pre["fit_chg"],
                 cfg.alpha, cfg.lam, num_repeats=nr, trans_centers=tc,
                 lr=cfg.lr_cpu, max_num_steps=cfg.max_steps, **kw)
        return _score_to_float(out[2])

    if mode == "pharm":
        fn = A.optimize_pharm_overlay_analytical if analytical else A.optimize_pharm_overlay
        kw = {} if analytical else {"use_fast": False}
        out = fn(pre["ref_ph"], pre["fit_ph"], pre["ref_anc"], pre["fit_anc"],
                 pre["ref_vec"], pre["fit_vec"], similarity=cfg.similarity,
                 num_repeats=nr, trans_centers=tc,
                 lr=cfg.lr_cpu, max_num_steps=cfg.max_steps, **kw)
        return _score_to_float(out[3])

    raise ValueError(mode)


def _single_fast(mode: str, pre: dict, cfg: BenchConfig):
    """Call the Triton fast single-pair optimiser for one pair."""
    if mode in ("vol", "surf"):
        from shepherd_score.alignment.utils.fast_surface_se3 import fast_optimize_ROCS_overlay
        out = fast_optimize_ROCS_overlay(
            pre["ref_pts"], pre["fit_pts"], cfg.alpha,
            num_repeats=cfg.num_repeats, topk=cfg.topk,
            steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
        return _score_to_float(out[2])

    if mode == "esp":
        from shepherd_score.alignment.utils.fast_esp_se3 import fast_optimize_ROCS_esp_overlay
        tc = pre.get("trans_centers") if cfg.trans_init else None
        out = fast_optimize_ROCS_esp_overlay(
            pre["ref_pts"], pre["fit_pts"], pre["ref_chg"], pre["fit_chg"],
            cfg.alpha, cfg.lam, num_repeats=cfg.num_repeats, trans_centers=tc,
            topk=cfg.topk, steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
        return _score_to_float(out[2])

    if mode == "pharm":
        from shepherd_score.alignment.utils.fast_pharm_se3 import fast_optimize_pharm_overlay
        tc = pre.get("trans_centers") if cfg.trans_init else None
        out = fast_optimize_pharm_overlay(
            pre["ref_ph"], pre["fit_ph"], pre["ref_anc"], pre["fit_anc"],
            pre["ref_vec"], pre["fit_vec"], similarity=cfg.similarity,
            num_repeats=cfg.num_repeats, trans_centers=tc,
            topk=cfg.topk, steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
        return _score_to_float(out[3])

    raise ValueError(mode)


def _prep_pair(p: PairSpec, mode: str, device, *, pharm_int: bool) -> dict:
    """Move one pair's mode-relevant arrays onto ``device`` as torch tensors."""
    d: dict = {}
    if mode == "vol":
        d["ref_pts"] = _t(p.ref.atom_pos, device)
        d["fit_pts"] = _t(p.fit.atom_pos, device)
        d["trans_centers"] = _t(p.ref.atom_pos, device)
    elif mode in ("surf", "esp"):
        d["ref_pts"] = _t(p.ref.surf_pos, device)
        d["fit_pts"] = _t(p.fit.surf_pos, device)
        d["trans_centers"] = _t(p.ref.atom_pos, device)
        if mode == "esp":
            d["ref_chg"] = _t(p.ref.surf_esp, device)
            d["fit_chg"] = _t(p.fit.surf_esp, device)
    elif mode == "pharm":
        pdt = torch.int64 if pharm_int else torch.float32
        d["ref_ph"] = _t(p.ref.pharm_types, device, dtype=pdt)
        d["fit_ph"] = _t(p.fit.pharm_types, device, dtype=pdt)
        d["ref_anc"] = _t(p.ref.pharm_ancs, device)
        d["fit_anc"] = _t(p.fit.pharm_ancs, device)
        d["ref_vec"] = _t(p.ref.pharm_vecs, device)
        d["fit_vec"] = _t(p.fit.pharm_vecs, device)
        d["trans_centers"] = _t(p.ref.pharm_ancs, device)
    return d


# ===========================================================================
# Backend classes
# ===========================================================================
class Backend:
    name: str = "base"
    device: str = "cpu"
    granularity: str = "single"      # "single" | "multi"

    def available(self) -> Tuple[bool, str]:
        return True, ""

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        raise NotImplementedError

    def run(self, state) -> BackendOutput:
        raise NotImplementedError


class _TorchSingle(Backend):
    """One pair per call on CPU or GPU via the torch reference optimiser."""

    def __init__(self, device: str, analytical: bool = False, threads: Optional[int] = None):
        self.device = device
        self.analytical = analytical
        self.threads = threads
        self.granularity = "single"
        kind = "analytical" if analytical else "torch"
        self.name = f"{'gpu' if device.startswith('cuda') else 'cpu'}_single_{kind}"

    def available(self):
        if self.device.startswith("cuda") and not _cuda_ok():
            return False, "no CUDA device"
        return True, ""

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        if self.threads is not None and _HAS_TORCH:
            torch.set_num_threads(self.threads)
        dev = torch.device(self.device)
        pre = [_prep_pair(p, cohort.mode, dev, pharm_int=False) for p in cohort.pairs]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (cohort.mode, pre, cfg)

    def run(self, state) -> BackendOutput:
        mode, pre, cfg = state
        scores = np.empty(len(pre), dtype=float)
        for i, d in enumerate(pre):
            scores[i] = _single_reference(mode, d, cfg, analytical=self.analytical)
        return BackendOutput(scores=scores)


class _TorchCpuMulti(Backend):
    """Many pairs in parallel on CPU via a thread pool of single-pair aligns.

    Parallelism comes from running ``cpu_workers`` pairs concurrently; each
    worker pins torch to a single intra-op thread to avoid oversubscription.
    Torch's C++ kernels release the GIL, so the heavy linear algebra runs truly
    in parallel; the Python optimisation loop is the serial remainder (reported
    honestly -- this is why CPU-multi scaling is sub-linear, see README).
    """

    granularity = "multi"
    device = "cpu"
    name = "cpu_multi_torch"

    def __init__(self, analytical: bool = False):
        self.analytical = analytical

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        dev = torch.device("cpu")
        pre = [_prep_pair(p, cohort.mode, dev, pharm_int=False) for p in cohort.pairs]
        workers = cfg.cpu_workers or (os.cpu_count() or 1)
        return (cohort.mode, pre, cfg, workers)

    def run(self, state) -> BackendOutput:
        from concurrent.futures import ThreadPoolExecutor
        mode, pre, cfg, workers = state
        prev = torch.get_num_threads() if _HAS_TORCH else 1
        torch.set_num_threads(1)
        try:
            def work(d):
                return _single_reference(mode, d, cfg, analytical=self.analytical)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scores = list(ex.map(work, pre))
        finally:
            torch.set_num_threads(prev)
        return BackendOutput(scores=np.asarray(scores, dtype=float),
                             extra={"workers": workers})


class _TorchCpuIntraop(Backend):
    """Many pairs on CPU, parallelised *within* each align via torch intra-op
    threads (BLAS / ATen).  Pairs run sequentially but each one uses all cores
    for its linear algebra.  This is the GIL-free complement to the thread-pool
    variant: it scales with molecule *size* (bigger matmuls) rather than with the
    number of pairs.  Reported as the second honest face of "CPU multi"."""

    granularity = "multi"
    device = "cpu"
    name = "cpu_multi_intraop"

    def __init__(self, analytical: bool = False):
        self.analytical = analytical

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        # Intra-op (BLAS) threading on the small matmuls of a single alignment
        # suffers badly from oversubscription past a handful of threads, so the
        # default is capped.  Override with --cpu-workers to probe scaling.
        workers = cfg.cpu_workers or min(8, os.cpu_count() or 1)
        if _HAS_TORCH:
            torch.set_num_threads(workers)
        dev = torch.device("cpu")
        pre = [_prep_pair(p, cohort.mode, dev, pharm_int=False) for p in cohort.pairs]
        return (cohort.mode, pre, cfg, workers)

    def run(self, state) -> BackendOutput:
        mode, pre, cfg, workers = state
        scores = np.empty(len(pre), dtype=float)
        for i, d in enumerate(pre):
            scores[i] = _single_reference(mode, d, cfg, analytical=self.analytical)
        return BackendOutput(scores=scores, extra={"intraop_threads": workers})


class _FastSingle(Backend):
    """One pair per call on GPU via the Triton fast optimiser."""

    device = "cuda"
    granularity = "single"
    name = "gpu_single_fast"

    def available(self):
        if not _cuda_ok():
            return False, "no CUDA device"
        if not _fast_ok():
            return False, "FAST_ALIGNMENT unavailable (triton import failed)"
        return True, ""

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        dev = torch.device("cuda")
        pre = [_prep_pair(p, cohort.mode, dev, pharm_int=True) for p in cohort.pairs]
        torch.cuda.synchronize()
        return (cohort.mode, pre, cfg)

    def run(self, state) -> BackendOutput:
        mode, pre, cfg = state
        scores = np.empty(len(pre), dtype=float)
        for i, d in enumerate(pre):
            scores[i] = _single_fast(mode, d, cfg)
        return BackendOutput(scores=scores)


class _FastBatch(Backend):
    """Many pairs per call on GPU via the container align_batch_* (bucketed)."""

    device = "cuda"
    granularity = "multi"
    name = "gpu_multi_batch"

    def available(self):
        if not _cuda_ok():
            return False, "no CUDA device"
        if not _fast_ok():
            return False, "FAST_ALIGNMENT unavailable (triton import failed)"
        return True, ""

    def _build_pairs(self, cohort: Cohort):
        from shepherd_score.container import MoleculePair
        dev = torch.device("cuda")
        pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=dev)
                 for p in cohort.pairs]
        return pairs

    def _n_buckets(self, cohort: Cohort) -> int:
        from shepherd_score.container._core import _band_key
        keys = set()
        for p in cohort.pairs:
            if cohort.mode in ("surf", "esp"):
                n = p.ref.surf_pos.shape[0]
                m = p.fit.surf_pos.shape[0]
            elif cohort.mode == "pharm":
                n = p.ref.pharm_ancs.shape[0]
                m = p.fit.pharm_ancs.shape[0]
            else:
                n = p.ref.atom_pos.shape[0]
                m = p.fit.atom_pos.shape[0]
            keys.add((_band_key(n), _band_key(m)))
        return len(keys)

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        pairs = self._build_pairs(cohort)
        torch.cuda.synchronize()
        return (cohort.mode, pairs, cfg, self._n_buckets(cohort))

    def run(self, state) -> BackendOutput:
        from shepherd_score.container import MoleculePair
        mode, pairs, cfg, n_buckets = state
        if mode == "vol":
            MoleculePair.align_batch_vol(pairs, alpha=cfg.alpha, steps_fine=cfg.steps_fine)
            # align_batch_vol stores into the no-hydrogen volume slot.
            scores = [p.sim_aligned_vol_noH for p in pairs]
        elif mode == "surf":
            MoleculePair.align_batch_surf(pairs, alpha=cfg.alpha, steps_fine=cfg.steps_fine)
            scores = [p.sim_aligned_surf for p in pairs]
        elif mode == "esp":
            MoleculePair.align_batch_esp(pairs, alpha=cfg.alpha, lam=cfg.lam,
                                         num_repeats=cfg.num_repeats, topk=cfg.topk,
                                         steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
            scores = [p.sim_aligned_esp for p in pairs]
        elif mode == "pharm":
            MoleculePair.align_batch_pharm(pairs, similarity=cfg.similarity,
                                           num_repeats=cfg.num_repeats, topk=cfg.topk,
                                           steps_fine=cfg.steps_fine, lr=cfg.lr_fast)
            scores = [p.sim_aligned_pharm for p in pairs]
        else:
            raise ValueError(mode)
        return BackendOutput(scores=np.asarray(scores, dtype=float),
                             n_buckets=n_buckets)


class _JaxShmapMultiDevice(Backend):
    """Many pairs across **multiple devices** via JAX ``shard_map``.

    On a multi-GPU host this shards the pair axis across the GPUs; on a single
    device it is skipped (sharding over one device is just the single-device
    path).  To validate the multi-device code path on a 1-GPU/CPU box, launch a
    fresh process with ``JAX_PLATFORMS=cpu`` and
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` set before import --
    see ``benchmarks/validate_multigpu.py``.
    """

    granularity = "multi"
    name = "gpu_multi_jax_shmap"

    def __init__(self):
        # device label is resolved lazily (depends on the JAX platform present)
        self.device = "multi"

    def available(self):
        # Route through jax_shmap so its no-preallocate env defaults are applied
        # before JAX initialises (lets JAX coexist with torch/Triton on the GPU).
        from benchmarks.alignment_bench import jax_shmap
        n = jax_shmap.jax_device_count()
        if n == 0:
            return False, "JAX not importable"
        if n <= 1:
            return False, f"need >1 JAX device (found {n}); set XLA_FLAGS to simulate"
        self.device = f"{jax_shmap.jax_platform()}x{n}"
        return True, ""

    def prepare(self, cohort: Cohort, cfg: BenchConfig):
        from benchmarks.alignment_bench import jax_shmap
        if cohort.mode not in jax_shmap.SUPPORTED:
            raise ValueError(f"jax_shmap unsupported mode {cohort.mode}")
        state = jax_shmap.prepare_inputs(cohort.mode, cohort, cfg)
        return (state, cfg)

    def run(self, state) -> BackendOutput:
        from benchmarks.alignment_bench import jax_shmap
        st, cfg = state
        scores = jax_shmap.run(st, cfg)
        return BackendOutput(scores=scores,
                             extra={"n_devices": st["n_dev"], "total_padded": st["total"]})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def all_backends() -> List[Backend]:
    """Every backend, in a sensible report order. Reference path first."""
    return [
        _TorchSingle("cpu", analytical=False, threads=1),   # accuracy reference
        _TorchSingle("cpu", analytical=True, threads=1),
        _TorchCpuMulti(analytical=False),                   # pair-level threads
        _TorchCpuIntraop(analytical=False),                 # op-level threads
        _TorchSingle("cuda", analytical=False),
        _FastSingle(),
        _FastBatch(),
        _JaxShmapMultiDevice(),     # multi-GPU (auto-skips unless >1 device)
    ]


REFERENCE_BACKEND = "cpu_single_torch"
