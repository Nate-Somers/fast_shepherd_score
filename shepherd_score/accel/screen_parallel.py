"""Shard-parallel CPU screening — near-linear core scaling for query-vs-library screens.

The in-process fused CPU path parallelises the fine loop over poses, but each ``align`` call has a
single-threaded prologue (coarse seed-gen + torch->numpy marshal); by Amdahl that caps a screen at
~3-5x on 8 cores no matter how many numba threads it gets. This driver moves the parallelism ABOVE
that prologue: it splits the library into one contiguous shard per worker process, and each worker
runs whole, independent aligns pinned to a single thread. There is no shared serial section, so C
workers give ~Cx — the clean way to turn 8 cores into ~8x.

Zero-copy on Linux: the featurized library is stashed as a module global in the parent BEFORE the
pool forks, so workers inherit it copy-on-write; only small index ranges are sent over the pipe.
Each worker composes with the rest of the CPU stack (fused loop + SoA/SVML kernels).

fork-safety: ALL numba work happens in the forked workers — the parent never runs a numba prange,
so libgomp is never active in it at fork time (forking a process with a live GNU-OpenMP pool aborts
the child). So do not run an in-process numba align before calling this; featurize, then screen.

    from shepherd_score.container import Molecule
    scores = screen_parallel(query_mol, library_mols, "surf", n_workers=8, alpha=0.81)
"""
from __future__ import annotations

import os
from multiprocessing import get_context

# Parent-set globals inherited by forked workers (never pickled).
_QUERY = None
_LIBRARY: list = []
_MODE = ""
_KW: dict = {}

# mode -> (align method name, aligned-similarity attribute on MoleculePair). Derived from
# the single-source-of-truth mode registry (_modes.MODE_ATTRS) instead of a third hand-kept
# copy: the batch method is always ``align_with_<mode>`` and the score attr is the registry's
# score_attr. (_modes is pure data / no torch, so importing it here stays fork-safe.)
from ._modes import MODE_ATTRS as _MODE_ATTRS
_ALIGN_ATTR = {m: (f"align_with_{m}", score_attr) for m, (_tf, score_attr) in _MODE_ATTRS.items()}


def _shard(index_range):
    """Align the query against library[index_range] in a single-threaded worker. Runs in a
    forked child, so _QUERY/_LIBRARY/_MODE/_KW are the parent's objects (copy-on-write)."""
    import torch
    import numba
    torch.set_num_threads(1)          # each worker is one core; no torch tail contention
    numba.set_num_threads(1)          # active-count mask (pool size is capped by the parent env)
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    method, attr = _ALIGN_ATTR[_MODE]
    pairs = [MoleculePair(_QUERY, _LIBRARY[i], do_center=True) for i in index_range]
    getattr(MoleculePairBatch(pairs), method)(backend="numba", **_KW)
    return [(i, float(getattr(p, attr))) for i, p in zip(index_range, pairs)]


def _chunks(n, k):
    """k contiguous index ranges covering range(n), balanced to +/-1."""
    return [range(i * n // k, (i + 1) * n // k) for i in range(k) if i * n // k < (i + 1) * n // k]


def screen_parallel(query, library, mode, n_workers=None, **align_kwargs):
    """Screen ``query`` against ``library`` (lists of pre-featurized ``Molecule``s) with the
    numba CPU backend, sharded across ``n_workers`` processes. Returns aligned similarity scores
    in library order. ``align_kwargs`` are the mode's required kwargs (e.g. ``alpha=0.81`` for
    surf, ``lam=0.3`` for vol_esp). ALWAYS forks (even for n_workers==1) so this parent never runs
    numba in-process and stays libgomp-safe for the fork."""
    global _QUERY, _LIBRARY, _MODE, _KW
    if mode not in _ALIGN_ATTR:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_ALIGN_ATTR)}")
    n = len(library)
    n_workers = max(1, min(n_workers or os.cpu_count() or 1, n))
    _QUERY, _LIBRARY, _MODE, _KW = query, library, mode, align_kwargs
    chunks = _chunks(n, n_workers)

    # Cap each worker's thread pool to ONE thread BEFORE forking. numba fixes its pool size from
    # NUMBA_NUM_THREADS at IMPORT time; a forked child inherits that value and its own
    # numba.set_num_threads(1) only masks it, leaving cpu_count-1 idle threads that spin-wait -- so C
    # workers oversubscribe C x cpu_count threads and aggregate throughput REGRESSES past ~16 workers
    # (measured 64<16). Setting the env here works only if this process has not yet imported numba
    # (screen_parallel's numba-clean contract holds for that); for a GUARANTEED cap, export
    # NUMBA_NUM_THREADS=1 (+ OMP_NUM_THREADS=1) before starting the process. Restored in finally.
    _cap = {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"}
    _saved = {k: os.environ.get(k) for k in _cap}
    os.environ.update(_cap)
    try:
        # Always fork (even for 1 worker): keeps the parent numba-clean so the fork is libgomp-safe.
        with get_context("fork").Pool(len(chunks)) as pool:  # fork -> COW-inherit the library
            shards = pool.map(_shard, chunks)
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    scores = [None] * n
    for shard in shards:
        for i, s in shard:
            scores[i] = s
    return scores
