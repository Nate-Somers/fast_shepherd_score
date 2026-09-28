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
million conformers, and GPU pairwise alignment 100,000 pairs. Pairwise results
use the fixed 250-query by 400-library selection in `PAIRWISE_100K.json`,
with disjoint compound sets and the median of three timed passes.
`vol_avoid` is excluded because this workload supplies no avoid-point cloud.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol` | 2,567.3 | 1,079,056 | 111,681 |
| `vol_esp` | 832.5 | 282,347 | 80,302 |
| `surf` | 30.2 | 51,491 | 38,361 |
| `surf_esp` | 28.1 | 38,066 | 27,519 |
| `vol_and_surf_esp` | 121.6 | 62,574 | 23,484 |
| `pharm` | 578.9 | 149,880 | 52,441 |
| `vol_color` | 830.4 | 273,317 | 59,586 |
| `vol_lipo` | 381.9 | 188,171 | 54,240 |

## Additional modes and existing Tversky variants

CPU screening uses 1,000 conformers; GPU screening and pairwise alignment each
use 100,000 conformer comparisons.

| Mode | CPU screen | GPU screen | GPU pairwise |
|---|---:|---:|---:|
| `vol_pharm` | 275.2 | 90,549 | 36,956 |
| `vol_atomtype` | 258.2 | 185,938 | 55,282 |
| `vol_mr` | 329.0 | 184,111 | 54,171 |
| `surf_tversky` | 20.1 | 48,430 | 37,335 |
| `surf_esp_tversky` | 19.1 | 37,421 | 27,717 |
| `vol_and_surf_esp_tversky` | 119.9 | 62,078 | 23,417 |
| `vol_color_tversky` | 813.5 | 266,032 | 58,354 |
| `vol_lipo_tversky` | 458.4 | 183,799 | 53,290 |
| `pharm_tversky` | 562.4 | 147,520 | 52,312 |
| `vol_fukui` | 461.3 | 185,423 | 54,809 |
| `vol_tversky` | 1,901.9 | 454,401 | 111,457 |
| `vol_esp_tversky` | 823.4 | 275,017 | 78,064 |

The SI reports preparation cost and multi-worker scaling separately. Multi-GPU
scaling uses the fastest of three repeated screens, whereas these tables use
median timings; those estimators should not be conflated.
