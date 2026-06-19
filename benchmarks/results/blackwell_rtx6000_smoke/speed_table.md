# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: blackwell_rtx6000_smoke   ·   GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition   ·   CPU: AMD EPYC 9135 16-Core Processor   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node5106   ·   2026-06-19 18:00_


## pairs / s

| mode | bucket | engine | 1 | 100 |
|---|---|---|--:|--:|
| vol | same | original | — | — |
| vol | same | fork | 170 | 14146 |

## fork speedup over original (×, matched size)

| mode | bucket | 1 | 100 |
|---|---|--:|--:|
| vol | same | — | — |
