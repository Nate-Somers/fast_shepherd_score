# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: h200   ·   GPU: NVIDIA H200   ·   CPU: INTEL(R) XEON(R) PLATINUM 8580   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node4100   ·   2026-06-19 18:16_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 153 | 1522 | 13442 | 86018 | 165737 | 156487 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 161 | 546 | 5054 | 33136 | 115376 | 133663 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 159 | 1487 | 10047 | 52845 | 70557 | 69953 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 151 | 322 | 2815 | 15934 | 57380 | 62356 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 73 | 807 | 7612 | 26110 | 31776 | 30435 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 73 | 162 | 1593 | 13772 | 27046 | 27698 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 27 | 258 | 2483 | 21352 | 74836 | 67534 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 26 | 270 | 1285 | 11516 | 57242 | 66514 |

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
