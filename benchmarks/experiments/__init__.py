"""Tuning / experiment tooling (see ../../SPEED_EXPERIMENTS.md).

These are NOT the headline benchmark (that is ``benchmarks/benchmark.py``); they
are the A/B and micro-bench tools used while optimizing the Triton kernels:

* ``speedlab.py``      — in-process paired A/B lab (noise-robust on a jittery GPU).
* ``kernelbench.py``   — kernel-level microbench for the overlap value+grad kernel.
* ``parity_scores.py`` — deterministic bit-exact score gate (stash/diff).
"""
