# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: l40s_1gpu_smoke   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 18:00_


## pairs / s

| mode | bucket | engine | 1 | 100 |
|---|---|---|--:|--:|
| vol | same | original | — | — |
| vol | same | fork | 153 | 11761 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 100 |
|---|---|--:|--:|
| vol | same | — | — |
