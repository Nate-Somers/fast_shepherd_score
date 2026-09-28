# Alignment throughput

These measurements come from the explicit selections in
[`paper_results.py`](https://github.com/Nate-Somers/Shepherd-Score-Paper/blob/main/paper/fig2_speed/paper_results.py).
They describe the recorded commits, not a new benchmark of this checkout.
GPU: NVIDIA L40S. CPU: one AMD EPYC 9474F core with Numba/SVML.
Rates exclude molecular preparation and index construction and include an
untimed warm-up before repeated measurements. See the manuscript SI for repeat
counts, library composition, and CPU/GPU stopping differences.

## Main modes

Alignments per second. CPU screening uses 1,000 conformers, GPU screening ten
million conformers, and GPU pairwise alignment 100,000 pairs.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol` | 2,567.3 | 1,079,056 | 98,174 |
| `vol_esp` | 832.5 | 282,347 | 73,898 |
| `surf` | 30.2 | 51,491 | 38,337 |
| `surf_esp` | 28.1 | 38,066 | 27,004 |
| `vol_and_surf_esp` | 121.6 | 62,574 | 22,091 |
| `pharm` | 578.9 | 149,880 | 52,080 |
| `vol_color` | 830.4 | 273,317 | 52,474 |
| `vol_lipo` | 381.9 | 188,171 | 48,342 |

## Additional modes and existing Tversky variants

CPU screening uses 1,000 conformers; GPU screening and pairwise alignment each
use 100,000 conformer comparisons.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol_pharm` | 275.2 | 90,549 | 34,864 |
| `vol_atomtype` | 258.2 | 185,938 | 52,589 |
| `vol_mr` | 329.0 | 184,111 | 49,814 |
| `surf_tversky` | 20.1 | 48,430 | 37,064 |
| `surf_esp_tversky` | 19.1 | 37,421 | 27,537 |
| `vol_and_surf_esp_tversky` | 119.9 | 62,078 | 23,715 |
| `vol_color_tversky` | 813.5 | 266,032 | 52,867 |
| `vol_lipo_tversky` | 458.4 | 183,799 | 48,002 |
| `pharm_tversky` | 562.4 | 147,520 | 54,818 |
| `vol_fukui` | 461.3 | 185,423 | 46,236 |
| `vol_tversky` | 1,901.9 | 454,401 | 95,470 |
| `vol_esp_tversky` | 823.4 | 275,017 | 70,671 |

The SI reports preparation cost and multi-worker scaling separately. Multi-GPU
scaling uses the fastest of three repeated screens, whereas these tables use
median timings; those estimators should not be conflated.
