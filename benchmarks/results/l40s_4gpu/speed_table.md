# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: l40s_4gpu   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 18:23_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 192 | 1836 | 15822 | 93516 | 160224 | 144813 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 188 | 648 | 6034 | 38865 | 123162 | 155837 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 189 | 1763 | 11961 | 64936 | 81495 | 125839 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 183 | 386 | 3396 | 17609 | 61086 | 114336 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 91 | 940 | 8935 | 32072 | 32370 | 70501 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 89 | 188 | 1911 | 15527 | 27791 | 72431 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 32 | 303 | 2908 | 24961 | 59156 | 83893 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 30 | 305 | 1539 | 13598 | 50058 | 73336 |

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
