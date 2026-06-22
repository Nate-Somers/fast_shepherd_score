# CPU alignment throughput — original JAX vs fork numba (pair-alignments / s)

Real drug self-SE(3)-copy pairs (optimum 1.0). `—` = over the wall-clock cap / not run.


_run: eng_threads   ·   GPU: n/a   ·   CPU: AMD EPYC 9474F 48-Core Processor   ·   torch 2.5.1   ·   node3312   ·   2026-06-21 21:38_


## 1 CPU core — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 354 | 459 | 476 | 460 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 60 | 60 | 60 | 57 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 25 | 24 | 24 | — |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 441 | 1058 | 1513 | 1575 |

### 1-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 4 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 513 | 1304 | 1613 | 1625 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 161 | 210 | 228 | 223 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 70 | 86 | 93 | 90 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 404 | 1258 | 2360 | 2924 |

### 4-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 8 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 568 | 1895 | 2702 | 2521 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 218 | 364 | 429 | 434 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 101 | 152 | 176 | 176 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 391 | 1212 | 2766 | 3844 |

### 8-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 16 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 544 | 2299 | 3894 | 4466 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 229 | 459 | 783 | 824 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 112 | 259 | 328 | 340 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 377 | 1187 | 2884 | 4558 |

### 16-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 32 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 472 | 2445 | 4632 | 6491 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 242 | 880 | 1342 | 1470 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 119 | 399 | 577 | 549 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 354 | 1178 | 3108 | 5086 |

### 32-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 48 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 411 | 2278 | 5294 | 7131 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 217 | 837 | 1480 | 2063 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 105 | 474 | 783 | 883 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 334 | 1130 | 3022 | 4627 |

### 48-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 64 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 402 | 2161 | 5628 | 7618 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 214 | 945 | 2124 | 2267 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 104 | 474 | 949 | 1091 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 253 | 989 | 2859 | 4310 |

### 64-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |

## 96 CPU cores — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 347 | 2218 | 6152 | 8232 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 194 | 1037 | 2025 | 2477 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 99 | 473 | 897 | 1034 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 238 | 926 | 3212 | 4483 |

### 96-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |
