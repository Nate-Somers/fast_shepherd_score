# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: 4gpu_l40s_smoke   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 10:57_


## pairs / s

| mode | bucket | engine | 1 | 10 |
|---|---|---|--:|--:|
| vol | same | original | — | — |
| vol | same | fork | 156 | 527 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 10 |
|---|---|--:|--:|
| vol | same | — | — |
