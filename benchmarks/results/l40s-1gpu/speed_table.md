# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: l40s-1gpu   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 15:43_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 205 | 1933 | 14340 | 48619 | 58944 | 57843 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 206 | 699 | 6046 | 28591 | 53816 | 48265 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 201 | 1836 | 11223 | 39801 | 43062 | 41645 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 195 | 416 | 3507 | 15183 | 36284 | 36983 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 94 | 987 | 7721 | 18459 | 17898 | 16209 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 99 | 209 | 1976 | 11614 | 16036 | 15365 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 33 | 308 | 2777 | 13454 | 19645 | 17583 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 31 | 314 | 1493 | 9279 | 18033 | 16391 |

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
