"""
Matrix runner + report generation.

Orchestrates  modes x cohorts x backends, times each cell honestly (prepare vs
steady-state compute), scores accuracy against the torch reference, and emits a
human-readable markdown report plus a machine-readable JSON dump.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks.alignment_bench.workloads import Cohort, make_cohort, MODES
from benchmarks.alignment_bench.timing import time_prepare_run
from benchmarks.alignment_bench import backends as B
from benchmarks.alignment_bench import metrics as M

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class Record:
    mode: str
    cohort: str
    size_kind: str
    n_pairs: int
    backend: str
    device: str
    granularity: str
    available: bool
    skip_reason: str = ""
    # timing
    prepare_s: float = float("nan")
    cold_s: float = float("nan")
    compute_median_s: float = float("nan")
    throughput_pairs_s: float = float("nan")
    latency_ms_per_pair: float = float("nan")
    # accuracy
    score: Dict[str, float] = field(default_factory=dict)
    vs_ref: Dict[str, float] = field(default_factory=dict)
    # extras
    n_buckets: Optional[int] = None
    extra: Dict[str, object] = field(default_factory=dict)


def system_info() -> Dict[str, object]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if _HAS_TORCH:
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["n_gpus"] = torch.cuda.device_count()
        try:
            import os
            info["cpu_count"] = os.cpu_count()
        except Exception:
            pass
        try:
            from shepherd_score.alignment import _torch as _t
            info["fast_alignment_available"] = bool(getattr(_t, "FAST_ALIGNMENT_AVAILABLE", False))
        except Exception:
            info["fast_alignment_available"] = False
    return info


def run_cohort(cohort: Cohort, cfg: B.BenchConfig, *,
               warmup: int, repeats: int) -> List[Record]:
    """Run every backend on one cohort; reference backend first for accuracy."""
    records: List[Record] = []
    ref_scores: Optional[np.ndarray] = None

    backend_list = B.all_backends()
    # ensure reference backend is first
    backend_list.sort(key=lambda b: 0 if b.name == B.REFERENCE_BACKEND else 1)

    for backend in backend_list:
        ok, reason = backend.available()
        rec = Record(
            mode=cohort.mode, cohort=cohort.name, size_kind=cohort.size_kind,
            n_pairs=len(cohort), backend=backend.name, device=backend.device,
            granularity=backend.granularity, available=ok, skip_reason=reason,
        )
        if not ok:
            records.append(rec)
            continue

        dev = "cuda" if backend.device.startswith("cuda") else None
        try:
            timed = time_prepare_run(
                prepare=lambda b=backend: b.prepare(cohort, cfg),
                run=lambda state, b=backend: b.run(state),
                device=dev, warmup=warmup, repeats=repeats,
            )
        except Exception as e:  # a backend blowing up should not kill the matrix
            rec.available = False
            rec.skip_reason = f"ERROR: {type(e).__name__}: {e}"
            records.append(rec)
            continue

        out: B.BackendOutput = timed.result
        n = len(cohort)
        rec.prepare_s = timed.prepare_s
        rec.cold_s = timed.cold_s
        rec.compute_median_s = timed.compute.median_s
        rec.throughput_pairs_s = n / timed.compute.median_s if timed.compute.median_s > 0 else float("inf")
        rec.latency_ms_per_pair = 1000.0 * timed.compute.median_s / n
        rec.score = M.score_summary(out.scores, cohort.noise)
        rec.n_buckets = out.n_buckets
        rec.extra = dict(out.extra)
        rec.extra["compute_stats"] = timed.compute.as_dict()

        if backend.name == B.REFERENCE_BACKEND:
            ref_scores = out.scores
        rec.vs_ref = M.accuracy_vs_reference(out.scores, ref_scores)
        records.append(rec)

    return records


def run_matrix(
    *,
    modes: Tuple[str, ...] = MODES,
    n_pairs: int = 32,
    uniform_size: int = 30,
    mixed_range: Tuple[int, int] = (10, 60),
    noise: float = 0.0,
    seed: int = 0,
    cfg: Optional[B.BenchConfig] = None,
    warmup: int = 1,
    repeats: int = 3,
) -> Dict[str, object]:
    cfg = cfg or B.BenchConfig()
    all_records: List[Record] = []
    cohort_meta: List[Dict[str, object]] = []

    for mode in modes:
        for size_kind in ("uniform", "mixed"):
            cohort = make_cohort(
                mode, n_pairs=n_pairs, size_kind=size_kind,
                size=uniform_size, size_range=mixed_range,
                noise=noise, seed=seed,
            )
            cohort_meta.append({
                "mode": mode, "cohort": cohort.name, "size_kind": size_kind,
                "n_pairs": len(cohort), "size_hist": cohort.meta.get("size_hist"),
            })
            all_records.extend(run_cohort(cohort, cfg, warmup=warmup, repeats=repeats))

    return {
        "system": system_info(),
        "config": asdict(cfg),
        "params": {
            "n_pairs": n_pairs, "uniform_size": uniform_size,
            "mixed_range": list(mixed_range), "noise": noise, "seed": seed,
            "warmup": warmup, "repeats": repeats,
        },
        "cohorts": cohort_meta,
        "records": [asdict(r) for r in all_records],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
FAIRNESS_LEDGER = """\
### Fairness ledger (read before trusting any number)

* **prepare(ms)** is the one-time cost of moving data to the device / padding
  batches / building MoleculePairs.  It is *excluded* from the steady-state
  compute number and reported on its own.  **cold(ms)** = prepare + first
  (warmup) compute, i.e. the true first-call latency including Triton autotune
  and any JIT.
* **throughput / latency** are the median over post-warmup repeats with CUDA
  synchronised around each measured region.
* **single vs multi**: single backends process pairs one at a time (latency
  oriented); multi/batch backends process the whole cohort per call (throughput
  oriented).  Both are summarised as pairs/s so they are comparable.
* **buckets**: the GPU batch path groups pairs into padded size buckets.  A
  *uniform* cohort collapses to a single bucket (best case); a *mixed* cohort
  spreads across many buckets (more padding waste).  Both are reported so the
  bucketing advantage is not cherry-picked.
* **accuracy** is delivered overlap, not implementation equality.  The fast path
  is a different optimiser; for noiseless cohorts the recoverable optimum is 1.0
  so ``gap_to_ideal`` is directly interpretable.  ``vs_ref`` compares against the
  torch reference optimiser.
"""


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:  # nan
            return "-"
        if x == float("inf"):
            return "inf"
        return f"{x:.{nd}g}"
    return str(x)


def to_markdown(results: Dict[str, object]) -> str:
    sysinfo = results["system"]
    lines: List[str] = []
    lines.append("# Alignment speed + accuracy benchmark\n")
    lines.append("## System\n")
    for k, v in sysinfo.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Config\n")
    lines.append("```json")
    lines.append(json.dumps({"config": results["config"], "params": results["params"]}, indent=2))
    lines.append("```\n")
    lines.append(FAIRNESS_LEDGER)
    lines.append("")

    # group records by (mode, cohort)
    recs = results["records"]
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in recs:
        groups.setdefault((r["mode"], r["cohort"]), []).append(r)

    header = ("| backend | dev | gran | pairs/s | lat ms/pair | prep ms | cold ms "
              "| mean | gap→ideal | vs-ref |Δ| | spearman | buckets |")
    sep = "|" + "|".join(["---"] * 11) + "|"

    for (mode, cohort), rows in groups.items():
        n_pairs = rows[0]["n_pairs"]
        lines.append(f"### mode=`{mode}`  cohort=`{cohort}`  (n_pairs={n_pairs})\n")
        lines.append(header)
        lines.append(sep)
        for r in rows:
            if not r["available"]:
                lines.append(f"| {r['backend']} | {r['device']} | {r['granularity']} "
                             f"| _skipped: {r['skip_reason']}_ |||||||||")
                continue
            sc = r["score"]
            vr = r["vs_ref"]
            lines.append(
                f"| {r['backend']} | {r['device']} | {r['granularity']} "
                f"| {_fmt(r['throughput_pairs_s'])} | {_fmt(r['latency_ms_per_pair'])} "
                f"| {_fmt(r['prepare_s']*1000)} | {_fmt(r['cold_s']*1000)} "
                f"| {_fmt(sc.get('mean'))} | {_fmt(sc.get('gap_to_ideal_mean'))} "
                f"| {_fmt(vr.get('mean_abs_diff'))} | {_fmt(vr.get('spearman'))} "
                f"| {_fmt(r['n_buckets'])} |"
            )
        lines.append("")

    return "\n".join(lines)
