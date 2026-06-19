# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_GPU: NVIDIA GeForce RTX 4050 Laptop GPU   ·   CPU: Intel(R) Core(TM) Ultra 9 185H   ·   torch 2.5.1 / CUDA 12.4   ·   LAPTOP-U787TIJJ   ·   2026-06-18 23:48_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | 0 | 15 | 153 | 155 | 157 | — |
| vol | same | fork | 68 | 494 | 5205 | 21606 | 23170 | 21688 |
| vol | cross | original | 2 | 12 | 57 | 60 | — | — |
| vol | cross | fork | 117 | 328 | 2741 | 13434 | 22333 | 10520 |
| surf | same | original | 2 | 4 | 53 | 51 | — | — |
| surf | same | fork | 90 | 654 | 2118 | 6152 | 16594 | 12902 |
| surf | cross | original | 2 | 3 | 24 | 36 | — | — |
| surf | cross | fork | 85 | 181 | 1490 | 4034 | 10223 | 5673 |
| esp | same | original | 2 | 5 | 30 | 31 | — | — |
| esp | same | fork | 48 | 523 | 3223 | 6139 | 5854 | 1938 |
| esp | cross | original | 2 | 3 | 17 | 20 | — | — |
| esp | cross | fork | 48 | 106 | 861 | 3314 | 3550 | 1154 |
| pharm | same | original | 1 | 12 | — | — | — | — |
| pharm | same | fork | 12 | 129 | 1162 | 6093 | 9012 | 7759 |
| pharm | cross | original | 2 | 10 | — | — | — | — |
| pharm | cross | fork | 14 | 138 | 632 | 4597 | 8677 | 4480 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | 153.0× | 32.6× | 33.9× | 139.8× | 147.2× | — |
| vol | cross | 72.3× | 28.4× | 48.4× | 223.6× | — | — |
| surf | same | 58.4× | 178.9× | 40.1× | 120.8× | — | — |
| surf | cross | 46.5× | 57.7× | 62.7× | 112.5× | — | — |
| esp | same | 21.2× | 102.5× | 108.4× | 197.9× | — | — |
| esp | cross | 21.6× | 30.5× | 50.6× | 169.9× | — | — |
| pharm | same | 10.1× | 10.7× | — | — | — | — |
| pharm | cross | 9.2× | 13.9× | — | — | — | — |
