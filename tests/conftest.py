"""Cap the numba thread pool before any test imports numba.

The test batches are a handful of pairs, so a ``parallel=True`` kernel spends its time waking
threads rather than computing; on a busy many-core node a call over every core is ~250x slower
than over a few. Export ``NUMBA_NUM_THREADS`` to override.
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
