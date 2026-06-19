# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: h100_smoke   ·   GPU: NVIDIA H100 NVL   ·   CPU: AMD EPYC 9474F 48-Core Processor   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node1709   ·   2026-06-19 18:12_


## pairs / s

| mode | bucket | engine | 1 | 100 |
|---|---|---|--:|--:|
| vol | same | original | — | — |
| vol | same | fork | 150 | 11997 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 100 |
|---|---|--:|--:|
| vol | same | — | — |
