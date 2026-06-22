# CPU alignment throughput — original JAX vs fork numba (pair-alignments / s)

Real drug self-SE(3)-copy pairs (optimum 1.0). `—` = over the wall-clock cap / not run.


_run: eng_pool   ·   GPU: n/a   ·   CPU: AMD EPYC 9474F 48-Core Processor   ·   torch 2.5.1   ·   node3312   ·   2026-06-21 22:51_


## 1 CPU core — pairs / s

| mode | bucket | engine | 10 | 100 | 1k | 10k |
|---|---|---|--:|--:|--:|--:|
| vol | cross | jax | — | — | — | — |
| vol | cross | numba | 351 | 460 | 474 | 459 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 60 | 60 | 59 | 57 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 25 | 24 | 24 | — |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 434 | 1058 | 1524 | 1571 |

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
| vol | cross | numba | 720 | 1460 | 1774 | 1745 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 158 | 125 | 228 | 225 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 66 | 90 | 92 | 90 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 535 | 1920 | 4495 | 5042 |

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
| vol | cross | numba | 900 | 2322 | 3061 | 3416 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 175 | 363 | 399 | 444 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 73 | 148 | 161 | 179 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 197 | 2264 | 4996 | 8598 |

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
| vol | cross | numba | 955 | 3224 | 5397 | 6284 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 154 | 422 | 756 | 846 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 73 | 220 | 305 | 339 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 195 | 2464 | 9257 | 13001 |

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
| vol | cross | numba | 937 | 3817 | 5765 | 11014 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 175 | 778 | 1266 | 1579 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 72 | 245 | 518 | 636 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 199 | 1414 | 11172 | 16962 |

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
| vol | cross | numba | 914 | 3788 | 8486 | 14477 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 173 | 792 | 1329 | 2314 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 72 | 274 | 673 | 922 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 198 | 1812 | 9008 | 19394 |

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
| vol | cross | numba | 901 | 3944 | 10348 | 16452 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 172 | 909 | 1605 | 3018 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 72 | 304 | 899 | 1227 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 196 | 1287 | 9163 | 20309 |

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
| vol | cross | numba | 817 | 4552 | 11179 | 16153 |
| surf | cross | jax | — | — | — | — |
| surf | cross | numba | 172 | 1174 | 1864 | 3374 |
| esp | cross | jax | — | — | — | — |
| esp | cross | numba | 72 | 480 | 910 | 1651 |
| pharm | cross | jax | — | — | — | — |
| pharm | cross | numba | 196 | 1375 | 9362 | 21572 |

### 96-core numba speedup over JAX (×, matched size)

| mode | bucket | 10 | 100 | 1k | 10k |
|---|---|--:|--:|--:|--:|
| vol | cross | — | — | — | — |
| surf | cross | — | — | — | — |
| esp | cross | — | — | — | — |
| pharm | cross | — | — | — | — |
