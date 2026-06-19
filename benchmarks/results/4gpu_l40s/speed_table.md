# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: 4gpu_l40s   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.5.1 / CUDA 12.4   ·   node3615   ·   2026-06-18 23:58_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | — | — | — | — | — | — |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | — | — | — | — | — | — |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | — | — | — | — | — | — |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | — | — | — | — | — | — |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | — | — | — | — | — | — |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | — | — | — | — | — | — |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | — | — | — | — | — | — |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | — | — | — | — | — | — |

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
