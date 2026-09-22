Alignment throughput
====================

Measured with the benchmark harness of the paper repository (Shepherd-Score-Paper, `paper/fig2_speed`). GPU rows: one NVIDIA L40S. CPU rows: one core of a 96-core node, numba with Intel SVML. Each value is the number of alignments divided by the median time of repeated timed passes of the default alignment call, after one untimed warm-up. The library is a 10,497,450-conformer Enamine REAL sample (50 conformers per compound, MMFF94 charges, 200 surface points); the 10^8 column uses a 104,978,250-conformer sample. Library preparation is not included in these rates; it is listed separately below.

## Screening on one GPU (`screen`, one query against an on-disk store)

Alignments per second. `vol` and the other atom-cloud-seeded modes use a canonical-frame store, the default.

| mode | N = 10^5 | N = 10^6 | N = 10^7 | N = 10^8 |
|---|---:|---:|---:|---:|
| `vol` | 965,481 | 1,042,576 | 1,079,056 | 1,078,354 |
| `vol_esp` | 292,908 | 297,171 | 282,347 | 277,431 |
| `surf` | 51,469 | 51,222 | 51,491 |  |
| `surf_esp` | 37,202 | 37,284 | 38,066 |  |
| `vol_and_surf_esp` | 62,076 | 62,585 | 62,574 |  |
| `pharm` | 147,219 | 148,423 | 149,880 | 148,528 |
| `vol_color` | 266,988 | 272,058 | 273,317 | 270,934 |
| `vol_lipo` | 184,458 | 186,758 | 188,171 | 184,011 |

## Pairwise on one GPU (`MoleculePairBatch`, 100,000 pairs in memory)

| mode | alignments / s |
|---|---:|
| `vol` | 98,174 |
| `vol_esp` | 73,898 |
| `surf` | 38,337 |
| `surf_esp` | 27,004 |
| `vol_and_surf_esp` | 22,091 |
| `pharm` | 52,080 |
| `vol_color` | 52,474 |
| `vol_lipo` | 48,342 |

## Screening on one CPU core (`screen(..., backend="numba")`, SVML build)

Alignments per second at N = 1,000 and at the largest library size each mode was run at.

| mode | N = 1,000 | largest N | alignments / s at largest N |
|---|---:|---:|---:|
| `vol` | 2,540 | 100,000 | 2,501 |
| `vol_esp` | 832 | 100,000 | 828 |
| `surf` | 30 | 1,000 | 30 |
| `surf_esp` | 28 | 1,000 | 28 |
| `vol_and_surf_esp` | 122 | 10,000 | 118 |
| `pharm` | 563 | 100,000 | 567 |
| `vol_color` | 814 | 100,000 | 773 |
| `vol_lipo` | 460 | 100,000 | 452 |

Without the SVML build (numba <= 0.59 + `icc_rt`, see the README) these kernels run unvectorised, several times slower, and warn once.

## Additional modes

Same protocol: CPU screen at N = 1,000, GPU screen at N = 100,000, GPU pairwise at N = 100,000.

| mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol_pharm` | 264 | 90,549 | 34,864 |
| `vol_atomtype` | 258 | 185,938 | 52,589 |
| `vol_mr` | 321 | 184,111 | 49,814 |
| `surf_tversky` | 20 | 48,430 | 37,064 |
| `surf_esp_tversky` | 19 | 37,421 | 27,537 |
| `vol_and_surf_esp_tversky` | 120 | 62,078 | 23,715 |
| `vol_color_tversky` | 772 | 266,032 | 52,867 |
| `vol_lipo_tversky` | 445 | 183,799 | 48,002 |
| `pharm_tversky` | 555 | 147,520 | 54,818 |
| `vol_fukui` | 449 | not measured | 46,236 |
| `vol_tversky` | 1,902 | 454,401 | 95,470 |
| `vol_esp_tversky` | 823 | 275,017 | 70,671 |

## Preparing a library (one CPU core, per conformer)

| step | ms per conformer |
|---|---:|
| embed only (ETKDG + MMFF94, 50-conformer ensembles) | 53 |
| `Molecule` for `vol` / `vol_lipo` (embed + heavy-atom profile) | 53 |
| `Molecule` for `pharm` / `vol_color` (+ pharmacophores) | 55 |
| `Molecule` for `vol_esp` (+ xTB charges) | 134 |
| `Molecule` for `surf` (+ 200-point surface) | 178 |
| `Molecule` for `surf_esp` / `vol_and_surf_esp` (+ surface + xTB charges) | 259 |

Writing a prepared molecule into a `vol` profile store costs about 51 us with the canonical-frame layout (209 bytes per conformer) and about 27 us without it (172 bytes per conformer). A library is prepared once, in CPU-hours, and indexed and screened in seconds.

## Multi-device scaling

`screen(ndev=k)` on a 10^8-conformer `vol` store, one worker process per L40S, workers kept for later calls:

| GPUs | alignments / s | seconds per screen | speed-up |
|---:|---:|---:|---:|
| 1 | 1,087,828 | 91.9 | 1.00x |
| 2 | 2,076,182 | 48.2 | 1.91x |
| 4 | 4,038,920 | 24.8 | 3.71x |

Starting the workers costs 3.5 s for two devices and 4.4 s for four, once per process.

`shepherd_score.accel.screen_parallel` on a 10^4-conformer `vol` store, one forked worker per physical core, 96-core node:

| workers | alignments / s | speed-up |
|---:|---:|---:|
| 1 | 1,854 | 1.0x |
| 8 | 14,886 | 8.0x |
| 16 | 29,003 | 15.6x |
| 32 | 56,285 | 30.4x |
| 64 | 100,019 | 54.0x |

