# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: l40s-4gpu   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 16:57_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 204 | 1935 | 14457 | 48361 | 61353 | 76289 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 202 | 686 | 6081 | 28675 | 54660 | 79831 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 200 | 1844 | 11163 | 39492 | 44980 | 70694 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 191 | 418 | 3499 | 15258 | 38048 | 67620 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 97 | 988 | 7921 | 18476 | 18212 | 21606 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 95 | 205 | 1994 | 11839 | 16858 | 24090 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 33 | 309 | 2755 | 13709 | 19161 | 17104 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 31 | 311 | 1509 | 9493 | 18668 | 15881 |

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
