# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: rtx4050   ·   GPU: NVIDIA GeForce RTX 4050 Laptop GPU   ·   CPU: Intel(R) Core(TM) Ultra 9 185H   ·   torch 2.5.1 / CUDA 12.4   ·   LAPTOP-U787TIJJ   ·   2026-06-19 18:21_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 108 | 980 | 7067 | 36960 | 54231 | 31853 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 44 | 150 | 3370 | 17508 | 41605 | 42625 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 69 | 814 | 5806 | 24563 | 28817 | 23320 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 91 | 189 | 1317 | 4554 | 13318 | 3616 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 42 | 461 | 4025 | 8455 | 8369 | 2275 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 41 | 84 | 609 | 1158 | 1316 | 1296 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 14 | 132 | 1357 | 11695 | 22174 | 23279 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 13 | 131 | 774 | 6742 | 18490 | 10078 |

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
