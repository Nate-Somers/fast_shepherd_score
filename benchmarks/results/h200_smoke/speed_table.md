# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: h200_smoke   ·   GPU: NVIDIA H200   ·   CPU: INTEL(R) XEON(R) PLATINUM 8580   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node4100   ·   2026-06-19 18:00_


## pairs / s

| mode | bucket | engine | 1 | 100 |
|---|---|---|--:|--:|
| vol | same | original | — | — |
| vol | same | fork | 119 | 10310 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 100 |
|---|---|--:|--:|
| vol | same | — | — |
