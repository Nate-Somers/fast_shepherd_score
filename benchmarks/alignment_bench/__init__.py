"""
Speed + accuracy benchmark suite for shepherd_score alignment modes.

The suite compares the different alignment *modes* (volume, surface, ESP,
pharmacophore) across the different compute *configurations*:

    device x granularity
    ----------------------
    CPU  x single   (one pair per call, torch reference optimiser)
    CPU  x multi    (many pairs in parallel on CPU)
    GPU  x single   (one pair per call, torch + Triton fast paths)
    GPU  x multi    (many pairs per call, Triton batch + size bucketing)
    GPU  x multi-device (JAX shard_map across GPUs -- only runs if >1 device)

See ``benchmarks/README.md`` for the full methodology and the fairness ledger
(what each timing includes/excludes, how the bucketing bias is handled, etc.).
"""
from benchmarks.alignment_bench.workloads import (
    Cohort,
    SyntheticMolecule,
    make_cohort,
    MODES,
)

__all__ = ["Cohort", "SyntheticMolecule", "make_cohort", "MODES"]
