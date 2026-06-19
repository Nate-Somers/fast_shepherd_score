# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: blackwell_rtx6000   ·   GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition   ·   CPU: AMD EPYC 9135 16-Core Processor   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node5106   ·   2026-06-19 18:11_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 224 | 2087 | 18702 | 120627 | 174350 | 162668 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 225 | 754 | 6958 | 50442 | 138362 | 151339 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 220 | 2041 | 15262 | 91420 | 108845 | 98300 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 208 | 453 | 4073 | 24415 | 79777 | 85625 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 129 | 1337 | 12531 | 48911 | 48798 | 44112 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 127 | 266 | 2742 | 21818 | 39681 | 40747 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 44 | 426 | 4148 | 33846 | 64524 | 58834 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 43 | 429 | 2140 | 18903 | 55666 | 60221 |

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
