# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: l40s_1gpu   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 18:11_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 194 | 1868 | 15919 | 92591 | 160497 | 141671 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 191 | 668 | 6015 | 38571 | 123879 | 121823 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 189 | 1757 | 11901 | 64526 | 81695 | 74368 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 183 | 391 | 3409 | 17758 | 60600 | 61836 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 89 | 944 | 8769 | 32133 | 32124 | 30140 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 91 | 189 | 1976 | 15754 | 27451 | 27346 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 32 | 311 | 2948 | 24666 | 57316 | 52198 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 30 | 299 | 1534 | 13904 | 49568 | 51232 |

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
