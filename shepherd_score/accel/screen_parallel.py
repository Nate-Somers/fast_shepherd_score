"""Shard-parallel CPU screening — near-linear core scaling for query-vs-library screens.

The in-process fused CPU path parallelises the fine loop over poses, but each ``align`` call has a
single-threaded prologue (coarse seed-gen + torch->numpy marshal), so in-call thread parallelism is
Amdahl-capped no matter how many numba threads it gets. This driver moves the parallelism ABOVE
that prologue: it splits the library into one contiguous shard per worker process, and each worker
runs whole, independent aligns pinned to a single thread. There is no shared serial section, so
throughput scales with the worker count.

Zero-copy on Linux: the featurized library is stashed as a module global in the parent BEFORE the
pool forks, so workers inherit it copy-on-write; only the query, the mode and small index ranges
travel over the pipe. Each worker composes with the rest of the CPU stack (fused loop + SoA/SVML
kernels).

THE POOL PERSISTS. Forking is what made the per-call cost grow with the worker count -- about
0.13 s per worker from a parent holding a 10^5-molecule library, two thirds of a 64-worker screen
at that size (Shepherd-Score-Paper, SI) -- so the pool is forked on the first call for a library
and reused by every later call against the same library. It is keyed by the library object's
identity and length: the workers hold the library AS IT WAS AT FORK TIME, so a different list, or
the same list resized, forks a new pool, while molecules mutated in place after the first call are
not seen by the workers. Call :func:`screen_parallel_close` to release the workers early (it is
also registered with :mod:`atexit`).

WORKERS ARE PINNED, SHARDS ARE STRIDED (Linux). Measured on a 96-core, 192-thread node
(Shepherd-Score-Paper fig2_speed/p6_cpu_place_probe.py, 2026-09-14): unpinned, the scheduler put
8 of 64 workers on the hyperthread sibling of an already-busy core, the slowest chunk ran at half
speed and the screen took 1.83x its ideal time; pinned to one physical core each it took 1.20x.
So each worker pins itself to its own physical core (one CPU per sibling group of the process's
affinity mask; a no-op where /sys has no topology) at pool start. The remaining 20% was chunk
imbalance: a CONTIGUOUS range of a library that holds whole-compound conformer ensembles gives
one worker the largest compounds, so the library is dealt out strided instead (worker w gets
w, w+k, w+2k, ...), which spreads every compound across the workers. Scores are returned in
library order either way; a different split moves a score at the 1e-4 level (the padded batch
composition feeds the seed frame), as any change of worker count already did.

fork-safety: ALL numba work happens in the forked workers — the parent never runs a numba prange,
so libgomp is never active in it at fork time (forking a process with a live GNU-OpenMP pool aborts
the child). So do not run an in-process numba align before the FIRST call for a library; featurize,
then screen.

    from shepherd_score.container import Molecule
    scores = screen_parallel(query_mol, library_mols, "surf", n_workers=8, alpha=0.81)
"""
from __future__ import annotations

import atexit
import os
from multiprocessing import get_context

# _modes is pure data (no torch), so importing it here stays fork-safe.
from ._modes import MODE_ATTRS as _MODE_ATTRS

# Parent-set global inherited by forked workers (never pickled).
_LIBRARY: list = []
_POOL = None                       # {"key": (id(library), len(library), n_workers), "pool": Pool}

_ALIGN_ATTR = {m: (f"align_with_{m}", score_attr) for m, (_tf, score_attr) in _MODE_ATTRS.items()}


def _shard(task):
    """Align ``query`` against library[index_range] in a single-threaded forked worker.
    ``_LIBRARY`` is the parent's list (copy-on-write); the query, mode and kwargs come with the
    task, so a pool forked for one library serves every screen against it."""
    query, index_range, mode, kw = task
    import torch
    import numba
    torch.set_num_threads(1)          # each worker is one core; no torch tail contention
    numba.set_num_threads(1)          # active-count mask (pool size is capped by the parent env)
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    method, attr = _ALIGN_ATTR[mode]
    pairs = [MoleculePair(query, _LIBRARY[i], do_center=True) for i in index_range]
    getattr(MoleculePairBatch(pairs), method)(backend="numba", **kw)
    return [(i, float(getattr(p, attr))) for i, p in zip(index_range, pairs)]


def _chunks(n, k):
    """k strided index ranges covering range(n): worker w gets w, w+k, w+2k, ... (balanced to
    +/-1 in count, and in molecule size when the library is stored as compound ensembles)."""
    return [range(w, n, k) for w in range(min(k, n))]


def _physical_cores(allowed):
    """One CPU per physical core among ``allowed`` (the lowest hardware-thread sibling), in CPU
    order; ``allowed`` itself where the topology is not exposed."""
    try:
        seen, cores = set(), []
        for c in sorted(allowed):
            if c in seen:
                continue
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                spec = f.read().strip()
            sib = set()
            for part in spec.split(","):
                a, _, b = part.partition("-")
                sib.update(range(int(a), int(b or a) + 1))
            sib &= set(allowed)
            seen |= sib
            cores.append(min(sib))
        return cores
    except (OSError, ValueError):
        return sorted(allowed)


def _pin_worker(cores):
    """Pool initializer: pin this worker to ``cores[worker number]`` (wrapping when there are
    more workers than cores). Worker numbers are contiguous within one pool."""
    if not cores:
        return
    from multiprocessing import current_process
    ident = getattr(current_process(), "_identity", None) or (1,)
    try:
        os.sched_setaffinity(0, {cores[(ident[0] - 1) % len(cores)]})
    except (AttributeError, OSError):      # no sched_setaffinity, or a CPU we may not use
        pass


def screen_parallel_close():
    """Shut down the persistent worker pool, if any, and drop the library reference."""
    global _POOL, _LIBRARY
    held, _POOL = _POOL, None
    if held is not None:
        held["pool"].close()
        held["pool"].join()
    _LIBRARY = []


atexit.register(screen_parallel_close)


def _pool_for(library, n_workers):
    """The pool for this library and worker count: reused when it exists, forked otherwise."""
    global _LIBRARY, _POOL
    key = (id(library), len(library), n_workers)
    if _POOL is not None and _POOL["key"] == key:
        return _POOL["pool"]
    screen_parallel_close()
    _LIBRARY = library
    # Cap each worker's thread pool to ONE thread BEFORE forking. numba fixes its pool size from
    # NUMBA_NUM_THREADS at IMPORT time; a forked child inherits that value and its own
    # numba.set_num_threads(1) only masks it, leaving cpu_count-1 idle threads that spin-wait -- so C
    # workers oversubscribe C x cpu_count threads and aggregate throughput regresses. Setting the env
    # here works only if this process has not yet imported numba (screen_parallel's numba-clean
    # contract holds for that); for a GUARANTEED cap, export NUMBA_NUM_THREADS=1 (+ OMP_NUM_THREADS=1)
    # before starting the process. Restored in finally.
    _cap = {"NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"}
    _saved = {k: os.environ.get(k) for k in _cap}
    os.environ.update(_cap)
    try:
        # Always fork (even for 1 worker): keeps the parent numba-clean so the fork is libgomp-safe.
        # Each worker pins itself to its own physical core (see the module docstring).
        try:
            cores = _physical_cores(os.sched_getaffinity(0))
        except AttributeError:                           # no affinity API on this platform
            cores = []
        pool = get_context("fork").Pool(n_workers, initializer=_pin_worker, initargs=(cores,))
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _POOL = {"key": key, "pool": pool}
    return pool


def screen_parallel(query, library, mode, n_workers=None, **align_kwargs):
    """Screen ``query`` against ``library`` (lists of pre-featurized ``Molecule``s) with the
    numba CPU backend, sharded across ``n_workers`` processes. Returns aligned similarity scores
    in library order. ``align_kwargs`` are the mode's required kwargs (e.g. ``alpha=0.81`` for
    surf, ``lam=0.3`` for vol_esp). ALWAYS forks (even for n_workers==1) so this parent never runs
    numba in-process and stays libgomp-safe for the fork; the forked pool is kept for later calls
    against the same library (see the module docstring)."""
    if mode not in _ALIGN_ATTR:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_ALIGN_ATTR)}")
    n = len(library)
    n_workers = max(1, min(n_workers or os.cpu_count() or 1, n))
    chunks = _chunks(n, n_workers)
    pool = _pool_for(library, len(chunks))
    shards = pool.map(_shard, [(query, rng, mode, align_kwargs) for rng in chunks], chunksize=1)
    scores = [None] * n
    for shard in shards:
        for i, s in shard:
            scores[i] = s
    return scores
