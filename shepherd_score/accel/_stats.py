"""Opt-in fine-loop schedule recorder (off by default).

WHY THIS EXISTS. A throughput number for a mode is only a number for the optimizer that mode
was CONFIGURED with, and nothing in the result can tell you which optimizer actually ran. The
score cannot: across a 51.6x change in how the fine loop was partitioned into buckets, the
returned ``overlap_score`` stayed BIT-IDENTICAL, so a score check certifies a truncated or
differently-scheduled cell as healthy. The schedule can: measured twice at N=1e5 on the same
node against the same library, fine-loop CALLS PER REP predicted throughput 6 modes out of 6 --
matching calls/rep gave rates within 1% (pharm 4 vs 4; vol_color 3 vs 3), while 20-50x apart
gave 1.13-3.14x apart (surf 4 vs 94; surf_esp 6 vs 175; vol_lipo 5 vs 258; vol_and_surf_esp
13 vs 127). The partition is the discriminator, so the partition is what has to be recorded.

Every fine loop -- the eager torch loop in ``drivers/*.py`` and the CUDA-graph replay loop in
``drivers/_graphed.py`` -- calls :func:`record` exactly once as it leaves its loop, reporting
how many value+grad evaluations it executed against the budget it was configured with, and
which of the two paths it was. That is what makes an early stop visible (a mode configured for
50 steps that reports 11 is stopping early) and what makes the graph/eager split visible: the
two paths have different per-step cost and a DIFFERENT early-stop schedule (the replay loop
carries an extra patience margin, see ``_GRAPH_ES_MARGIN``), so ``calls`` alone cannot say
whether a rate moved because the schedule changed or because the dispatch fell back to eager.

Off by default and zero-overhead when off: :func:`record` reads one module-global boolean and
returns. Nothing is called per step, only once per fine-loop invocation.

    from shepherd_score.accel import _stats
    _stats.reset()                     # clear counters AND enable recording
    ...                                # run an alignment / a benchmark cell
    _stats.summary()                   # {'calls': 12, 'graphed': 12, 'steps_min': 50, ...}
    _stats.disable()

Not thread-safe and process-local: the counters are plain module globals, so a multi-threaded
or forked worker records only what ran in its own process/thread interleaving. It is a
diagnostic, not an accounting ledger.
"""
from __future__ import annotations

_enabled = False
_calls = 0
_graphed = 0
_steps_total = 0
_steps_min = 0
_steps_max = 0
_early = 0
_configured: set = set()
_configured_total = 0        # sum of the budget over CALLS, for a call-weighted mean


def reset() -> None:
    """Clear the counters and ENABLE recording."""
    global _enabled, _calls, _graphed, _steps_total, _steps_min, _steps_max, _early
    global _configured, _configured_total
    _enabled = True
    _configured_total = 0
    _calls = 0
    _graphed = 0
    _steps_total = 0
    _steps_min = 0
    _steps_max = 0
    _early = 0
    _configured = set()


def disable() -> None:
    """Stop recording. Counters already collected stay readable via :func:`summary`."""
    global _enabled
    _enabled = False


def record(steps_run: int, steps_configured: int, early_stopped: bool,
           graphed: bool = False) -> None:
    """Register one fine-loop invocation. No-op unless :func:`reset` enabled recording.

    ``steps_run`` is value+grad evaluations actually executed, ``steps_configured`` the budget
    the caller was given (``steps_fine``), ``early_stopped`` whether the loop left before it
    exhausted that budget, and ``graphed`` whether this invocation ran the CUDA-graph replay
    loop rather than the eager one. ``graphed`` defaults False so an eager site records by
    passing nothing, which is the shape every eager call site already had.
    """
    if not _enabled:
        return
    global _calls, _graphed, _steps_total, _steps_min, _steps_max, _early, _configured_total
    steps_run = int(steps_run)
    if _calls == 0 or steps_run < _steps_min:
        _steps_min = steps_run
    if steps_run > _steps_max:
        _steps_max = steps_run
    _calls += 1
    if graphed:
        _graphed += 1
    _steps_total += steps_run
    if early_stopped:
        _early += 1
    _configured.add(int(steps_configured))
    _configured_total += int(steps_configured)


def summary() -> dict:
    """Aggregate of everything recorded since the last :func:`reset`; ``{}`` if nothing was.

    ``graphed`` is how many of ``calls`` ran the CUDA-graph replay loop; the eager count is
    ``calls - graphed``, which is why only the one counter is kept. On CPU it is always 0.

    ``steps_configured`` is the budget itself when every recorded call shared one -- the usual
    single-mode case. When calls with DIFFERENT budgets share one window it is the CALL-weighted
    mean, so an effort fraction taken against it stays meaningful, and ``steps_configured_mixed``
    is True so a consumer can tell that the window spanned more than one budget. Resetting per
    mode is still preferable, because min/max then describe one budget rather than several.
    """
    if not _calls:
        return {}
    mixed = len(_configured) > 1
    if not mixed:
        cfg = next(iter(_configured))
    else:
        # CALL-weighted, not a mean over distinct budgets. Averaging the distinct values made
        # 99 calls at 30 plus one at 50 report 40.0 instead of 30.2, which drove effort_frac to
        # 0.755 and stamped effort_truncated on a cell that ran 100% of its budget.
        cfg = _configured_total / _calls
    return {
        "calls": _calls,
        "graphed": _graphed,
        "steps_min": _steps_min,
        "steps_max": _steps_max,
        "steps_mean": _steps_total / _calls,
        "steps_configured": cfg,
        "steps_configured_mixed": mixed,
        "early_stop_frac": _early / _calls,
    }
