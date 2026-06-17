"""
Honest timing utilities.

Principles
----------
* **Warmup is discarded.** The first call to any GPU/JAX path pays for kernel
  autotuning, JIT compilation and lazy allocation.  That cost is real but it is
  *amortised*, so we measure it separately (``prepare`` / cold cost) and report
  steady-state compute from post-warmup repeats.

* **GPU work is synchronised.** CUDA kernels are asynchronous; without
  ``torch.cuda.synchronize()`` you would time the launch, not the work.  Every
  measured region is bracketed by a device sync when a CUDA device is involved.

* **Robust statistics.** We report the median and the 10th/90th percentiles over
  repeats rather than the mean, so a single scheduling hiccup does not dominate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _sync(device: Optional[str]) -> None:
    if _HAS_TORCH and device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()


@dataclass
class TimeStats:
    median_s: float
    mean_s: float
    p10_s: float
    p90_s: float
    min_s: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _stats(samples) -> TimeStats:
    a = np.asarray(samples, dtype=float)
    return TimeStats(
        median_s=float(np.median(a)),
        mean_s=float(np.mean(a)),
        p10_s=float(np.percentile(a, 10)),
        p90_s=float(np.percentile(a, 90)),
        min_s=float(np.min(a)),
        n=int(a.size),
    )


@dataclass
class TimedResult:
    """Outcome of timing a callable, separating cold vs steady-state cost."""

    prepare_s: float           # one-time prep (transfer / pad / compile / autotune)
    cold_s: float              # prepare + first (warmup) compute -- the true first-call cost
    compute: TimeStats         # steady-state compute over repeats
    result: object             # value returned by the run fn on the last repeat


def time_prepare_run(
    prepare: Callable[[], object],
    run: Callable[[object], object],
    *,
    device: Optional[str] = None,
    warmup: int = 1,
    repeats: int = 5,
) -> TimedResult:
    """Time a prepare/run split.

    ``prepare()`` is executed once (its cost is the device transfer / padding /
    JIT-compile bucket).  ``run(state)`` is executed ``warmup`` times (discarded,
    but the first warmup time is folded into ``cold_s``) then ``repeats`` times
    (recorded for steady-state stats).
    """
    # ---- prepare (one-time) ----
    _sync(device)
    t0 = time.perf_counter()
    state = prepare()
    _sync(device)
    prepare_s = time.perf_counter() - t0

    # ---- warmup (first warmup time counts toward the cold/first-call cost) ----
    cold_compute = 0.0
    last = None
    for i in range(max(1, warmup)):
        _sync(device)
        t0 = time.perf_counter()
        last = run(state)
        _sync(device)
        dt = time.perf_counter() - t0
        if i == 0:
            cold_compute = dt

    # ---- steady-state ----
    samples = []
    for _ in range(max(1, repeats)):
        _sync(device)
        t0 = time.perf_counter()
        last = run(state)
        _sync(device)
        samples.append(time.perf_counter() - t0)

    return TimedResult(
        prepare_s=prepare_s,
        cold_s=prepare_s + cold_compute,
        compute=_stats(samples),
        result=last,
    )
