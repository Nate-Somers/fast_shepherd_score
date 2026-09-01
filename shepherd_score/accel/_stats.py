"""Opt-in step-count recorder for the fine-loop optimizers (off by default).

Every fine loop -- the eager torch loop in ``drivers/*.py``, the CUDA-graph replay loop in
``drivers/_graphed.py``, and the fused numba loop in ``kernels/cpu_fused.py`` -- calls
:func:`record` exactly once as it leaves its loop, reporting how many value+grad evaluations
it actually executed against the budget it was configured with. That is what makes an
early-stop visible: a mode configured for 50 steps that reports 11 is stopping early, and a
throughput number measured on it is not a number for the configured search effort.

Off by default and zero-overhead when off: :func:`record` reads one module-global boolean and
returns. Nothing is called per step, only once per fine-loop invocation.

    from shepherd_score.accel import _stats
    _stats.reset()                     # clear counters AND enable recording
    ...                                # run an alignment / a benchmark cell
    _stats.summary()                   # {'calls': 12, 'steps_min': 50, ...}
    _stats.disable()

Not thread-safe and process-local: the counters are plain module globals, so a multi-threaded
or forked worker records only what ran in its own process/thread interleaving. It is a
diagnostic, not an accounting ledger.
"""
from __future__ import annotations

_enabled = False
_calls = 0
_steps_total = 0
_steps_min = 0
_steps_max = 0
_early = 0
_configured: set = set()


def reset() -> None:
    """Clear the counters and ENABLE recording."""
    global _enabled, _calls, _steps_total, _steps_min, _steps_max, _early, _configured
    _enabled = True
    _calls = 0
    _steps_total = 0
    _steps_min = 0
    _steps_max = 0
    _early = 0
    _configured = set()


def disable() -> None:
    """Stop recording. Counters already collected stay readable via :func:`summary`."""
    global _enabled
    _enabled = False


def record(steps_run: int, steps_configured: int, early_stopped: bool) -> None:
    """Register one fine-loop invocation. No-op unless :func:`reset` enabled recording.

    ``steps_run`` is value+grad evaluations actually executed, ``steps_configured`` the budget
    the caller was given (``steps_fine``), ``early_stopped`` whether the loop left before it
    exhausted that budget.
    """
    if not _enabled:
        return
    global _calls, _steps_total, _steps_min, _steps_max, _early
    steps_run = int(steps_run)
    if _calls == 0 or steps_run < _steps_min:
        _steps_min = steps_run
    if steps_run > _steps_max:
        _steps_max = steps_run
    _calls += 1
    _steps_total += steps_run
    if early_stopped:
        _early += 1
    _configured.add(int(steps_configured))


def summary() -> dict:
    """Aggregate of everything recorded since the last :func:`reset`; ``{}`` if nothing was.

    ``steps_configured`` is the budget itself when every recorded call shared one -- the usual
    single-mode case, and the only case in which it means anything. When calls with DIFFERENT
    budgets share one recording window it is the unweighted mean over the DISTINCT budgets seen,
    NOT over calls: 99 calls at 30 plus one at 50 reports 40.0, not 30.2. Any effort fraction
    computed against a mixed window is therefore wrong (here, low enough to look truncated).
    Reset per mode.
    """
    if not _calls:
        return {}
    if len(_configured) == 1:
        cfg = next(iter(_configured))
    else:
        cfg = sum(_configured) / len(_configured)
    return {
        "calls": _calls,
        "steps_min": _steps_min,
        "steps_max": _steps_max,
        "steps_mean": _steps_total / _calls,
        "steps_configured": cfg,
        "early_stop_frac": _early / _calls,
    }
