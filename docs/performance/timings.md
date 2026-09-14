Alignment throughput
====================

Measured by the benchmark harness of the accompanying paper (Shepherd-Score-Paper, `paper/fig2_speed`) at library commit `6993bef`, on MIT ORCD Engaging: one NVIDIA L40S (node3615, exclusively allocated) for the GPU rows and one CPU core (numba with Intel SVML) for the CPU rows. Every number is the median of repeated timed passes of the complete default alignment call -- the same seeds and step budgets a user gets -- on a library drawn from a 10,497,450-conformer Enamine REAL pool (50 ETKDG conformers per compound, MMFF94-optimised, MMFF94 charges, 200 surface points). Preparation of the library is not included and is reported separately below. The harness, its results files and the exact commands are in that repository; the tables here are read from its `results/*.json`.

## Screening, GPU (`screen`, one query streamed against an on-disk store)

Alignments per second. `vol` uses a canonical-frame store, the library default.

| mode | N = 10^5 | N = 10^6 |
|---|---:|---:|
| `vol` | 1,024,885 | 1,055,873 |
| `vol_esp` | 275,056 | 272,177 |
| `vol_lipo` | 174,780 | 176,236 |
| `vol_color` | 259,915 | 262,351 |
| `vol_and_surf_esp` | 60,919 | 61,041 |
| `surf_esp` | 36,368 | 36,379 |
| `surf` | 50,745 | 50,266 |
| `pharm` | 125,743 | 126,017 |

## Pairwise, GPU (`MoleculePairBatch`, 100,000 pairs held in memory)

| mode | alignments / s |
|---|---:|
| `vol` | 117,589 |
| `vol_esp` | 59,647 |
| `vol_lipo` | 50,419 |
| `vol_color` | 57,497 |
| `vol_and_surf_esp` | 24,805 |
| `surf_esp` | 28,120 |
| `surf` | 38,546 |
| `pharm` | 52,220 |

## Screening, one CPU core (`screen(..., backend="numba")`, SVML build)

Alignments per second at the largest library size each mode's sweep reached.

| mode | N | alignments / s |
|---|---:|---:|
| `vol` | 100,000 | 2,457 |
| `vol_esp` | 100,000 | 816 |
| `vol_lipo` | 10,000 | 87 |
| `vol_color` | 100,000 | 770 |
| `vol_and_surf_esp` | 10,000 | 101 |
| `surf_esp` | 1,000 | 8 |
| `surf` | 1,000 | 30 |
| `pharm` | 100,000 | 573 |

Without the SVML build (numba <= 0.59 + `icc_rt`, see the README) these kernels run unvectorised, several times slower, and warn once.

## Preparing a library (one CPU core, per conformer)

| step | ms per conformer |
|---|---:|
| embed only (ETKDG + MMFF94, 50-conformer ensembles) | 53 |
| `Molecule` for `vol` (embed + volumetric profile) | 55 |
| `Molecule` for `surf` (+ 200-point surface) | 178 |
| `Molecule` for `vol_esp` (+ surface + xTB charges) | 259 |

Writing a prepared molecule into a `vol` profile store costs about 29 us (canonical store; about 11 us without the principal-frame rotation), so a library is prepared once, in CPU-hours, and indexed and screened in seconds.

## Multi-device scaling

Sharding a 10^6-conformer `vol` screen across L40S devices with `screen(ndev=k)`, one worker process per device kept for the life of the calling process, gave 1,071,914 alignments/s on one device, 1,105,253 on two and 1,808,852 on four (1.7x): 0.93, 0.90 and 0.55 s per screen after the first. Starting the workers costs 2.9 s (two devices) or 4.0 s (four), once per process, and each screen then carries about 0.02 s of fixed cost. On CPU the fork-sharded driver (`shepherd_score.accel.screen_parallel`), with its pool kept across calls, gave 1,891 alignments/s on one worker and 66,541 on 64 (35x; 7.9x on 8, 15.6x on 16, 18.9x on 32) for N = 10^4 read from the same store on an exclusively allocated 96-core node; its one-worker rate is 0.80x the in-process screen of that store (2,378 alignments/s). Before f43a1bd, when the pool was forked on every call, 64 workers reached 4.4x at N = 10^5. Both are documented in the paper repository's results.

The earlier version of this page reported the Jax/PyTorch batch paths at 10^2-10^3 alignments per second; those paths are superseded by the numba and triton back ends measured above, which `backend=None` selects.
