# Alignment throughput — real drug pairs (pair-alignments / s)

Each cell stops at the first size over the wall-clock cap. `—` = not run.


_run: 1gpu_l40s_fss   ·   GPU: NVIDIA L40S   ·   CPU: INTEL(R) XEON(R) GOLD 6542Y   ·   torch 2.11.0+cu128 / CUDA 12.8   ·   node3615   ·   2026-06-19 11:18_


## pairs / s

| mode | bucket | engine | 1 | 10 | 100 | 1k | 10k | 100k |
|---|---|---|--:|--:|--:|--:|--:|--:|
| vol | same | original | — | — | — | — | — | — |
| vol | same | fork | 207 | 1980 | 14449 | 49782 | 62060 | 58684 |
| vol | cross | original | — | — | — | — | — | — |
| vol | cross | fork | 208 | 703 | 6070 | 28629 | 55079 | 54090 |
| surf | same | original | — | — | — | — | — | — |
| surf | same | fork | 205 | 1855 | 11186 | 40754 | 45183 | 41562 |
| surf | cross | original | — | — | — | — | — | — |
| surf | cross | fork | 191 | 417 | 3453 | 15438 | 38255 | 37971 |
| esp | same | original | — | — | — | — | — | — |
| esp | same | fork | 98 | 1025 | 7836 | 18470 | 18388 | 17597 |
| esp | cross | original | — | — | — | — | — | — |
| esp | cross | fork | 96 | 199 | 2001 | 11960 | 16753 | 15846 |
| pharm | same | original | — | — | — | — | — | — |
| pharm | same | fork | 33 | 315 | 2756 | 13579 | 19519 | 18803 |
| pharm | cross | original | — | — | — | — | — | — |
| pharm | cross | fork | 32 | 310 | 1509 | 9582 | 18447 | 18511 |

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
