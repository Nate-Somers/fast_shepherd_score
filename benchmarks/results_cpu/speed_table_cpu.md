# CPU alignment throughput — original JAX vs fork numba (pair-alignments / s)

Real drug self-SE(3)-copy pairs (optimum 1.0). `—` = over the wall-clock cap / not run.


_GPU: n/a   ·   CPU: Intel(R) Core(TM) Ultra 9 185H   ·   torch 2.5.1   ·   LAPTOP-U787TIJJ   ·   2026-06-21 19:34_


## 1 CPU core — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | 64 | 63 | 63 | 60 |
| vol | cross | numba | 572 | 821 | 838 | 792 |
| surf | cross | jax | 49 | 44 | 45 | — |
| surf | cross | numba | 102 | 101 | 95 | 93 |
| esp | cross | jax | 34 | 25 | 26 | — |
| esp | cross | numba | 38 | 37 | 38 | — |
| pharm | cross | jax | 68 | — | — | — |
| pharm | cross | numba | 538 | 1306 | 1879 | 1916 |

### 1-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | 8.9× | 13.1× | 13.4× | 13.2× |
| surf | cross | 2.1× | 2.3× | 2.1× | — |
| esp | cross | 1.1× | 1.5× | 1.5× | — |
| pharm | cross | 8.0× | — | — | — |

## 8 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | 151 | 276 | 232 | 227 |
| vol | cross | numba | 553 | 1623 | 2917 | 2950 |
| surf | cross | jax | 1 | 6 | 34 | — |
| surf | cross | numba | 178 | 344 | 439 | 434 |
| esp | cross | jax | 1 | 5 | 27 | — |
| esp | cross | numba | 68 | 140 | 159 | 151 |
| pharm | cross | jax | 201 | — | — | — |
| pharm | cross | numba | 316 | 1169 | 3184 | 4142 |

### 8-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | 3.7× | 5.9× | 12.6× | 13.0× |
| surf | cross | 119.2× | 56.9× | 13.0× | — |
| esp | cross | 47.9× | 26.2× | 6.0× | — |
| pharm | cross | 1.6× | — | — | — |
