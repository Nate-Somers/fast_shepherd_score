# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: h100   ·   GPU: NVIDIA H100 NVL   ·   CPU: AMD EPYC 9474F 48-Core Processor   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node1709   ·   2026-06-19 18:23_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 182 | 1721 | 14960 | 103634 | 177623 | 166570 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 179 | 608 | 5550 | 36117 | 102215 | 132244 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 177 | 1631 | 10241 | 55398 | 61779 | 65685 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 168 | 360 | 3053 | 16029 | 51190 | 60179 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 91 | 942 | 8989 | 25858 | 28405 | 28467 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 90 | 189 | 1906 | 14670 | 24950 | 26565 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 32 | 297 | 2927 | 25018 | 65383 | 67756 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 30 | 301 | 1509 | 13447 | 54077 | 64525 |

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
