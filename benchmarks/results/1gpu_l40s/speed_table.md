# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: 1gpu_l40s   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.5.1 / CUDA 12.4   ·   node3615   ·   2026-06-19 00:30_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 208 | 1917 | 15260 | 47612 | 55282 | 58373 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 206 | 690 | 6017 | 27906 | 43411 | 53723 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 203 | 1827 | 10804 | 44722 | 45456 | 48368 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 193 | 415 | 3478 | 14797 | 36140 | 42281 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 96 | 1005 | 7845 | 21783 | 23041 | 23608 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 98 | 204 | 2005 | 11910 | 19094 | 20277 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 32 | 309 | 2723 | 13799 | 20061 | 21475 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 31 | 310 | 1492 | 9632 | 18847 | 20898 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | — | — | — | — | — | — |
| vol | cross | — | — | — | — | — | — |
| surf | same | — | — | — | — | — | — |
| surf | cross | — | — | — | — | — | — |
| esp | same | — | — | — | — | — | — |
| esp | cross | — | — | — | — | — | — |
| pharm | same | — | — | — | — | — | — |
| pharm | cross | — | — | — | — | — | — |
