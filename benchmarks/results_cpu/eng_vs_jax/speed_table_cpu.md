# CPU alignment throughput — original JAX vs fork numba (pair-alignments / s)

Real drug self-SE(3)-copy pairs (optimum 1.0). `—` = over the wall-clock cap / not run.


_run: eng_vs_jax   ·   GPU: n/a   ·   CPU: AMD EPYC 9474F 48-Core Processor   ·   torch 2.5.1   ·   node3312   ·   2026-06-21 23:07_


## 1 CPU core — pairs / s

| mode | bucket | engine | 10 | 100 | 1k |
|---|---|---|--:|--:|--:|
| vol | cross | jax | 41 | 48 | 40 |
| vol | cross | numba | 352 | 461 | 473 |
| surf | cross | jax | 29 | 24 | 25 |
| surf | cross | numba | 60 | 60 | 60 |
| esp | cross | jax | 11 | 7 | 8 |
| esp | cross | numba | 25 | 24 | 24 |
| pharm | cross | jax | 92 | — | — |
| pharm | cross | numba | 447 | 1047 | 1527 |

### 1-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k |
|---|---|--:|--:|--:|
| vol | cross | 8.7× | 9.7× | 11.7× |
| surf | cross | 2.1× | 2.5× | 2.4× |
| esp | cross | 2.3× | 3.2× | 2.9× |
| pharm | cross | 4.9× | — | — |

## 48 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k |
|---|---|---|--:|--:|--:|
| vol | cross | jax | 110 | 272 | 250 |
| vol | cross | numba | 422 | 2130 | 5650 |
| surf | cross | jax | 1 | — | — |
| surf | cross | numba | 225 | 888 | 1360 |
| esp | cross | jax | 1 | 2 | — |
| esp | cross | numba | 107 | 458 | 679 |
| pharm | cross | jax | 204 | — | — |
| pharm | cross | numba | 329 | 1075 | 3316 |

### 48-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k |
|---|---|--:|--:|--:|
| vol | cross | 3.8× | 7.8× | 22.6× |
| surf | cross | 353.7× | — | — |
| esp | cross | 148.5× | 255.4× | — |
| pharm | cross | 1.6× | — | — |
