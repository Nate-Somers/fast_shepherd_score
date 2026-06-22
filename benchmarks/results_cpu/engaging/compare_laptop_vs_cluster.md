# numba CPU throughput — laptop vs cluster, threads vs pool (peak pairs/s over batch sizes)


## vol

| series | 1c | 4c | 8c | 16c | 32c | 48c | 64c | 96c |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Intel 185H (laptop) · threads | 838 | — | 2950 | — | — | — | — | — |
| EPYC 9474F (cluster) · threads | 476 | 1625 | 2702 | 4466 | 6491 | 7131 | 7618 | 8232 |
| EPYC 9474F (cluster) · pool | 474 | 1774 | 3416 | 6284 | 11014 | 14477 | 16452 | 16153 |

## surf

| series | 1c | 4c | 8c | 16c | 32c | 48c | 64c | 96c |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Intel 185H (laptop) · threads | 102 | — | 439 | — | — | — | — | — |
| EPYC 9474F (cluster) · threads | 60 | 228 | 434 | 824 | 1470 | 2063 | 2267 | 2477 |
| EPYC 9474F (cluster) · pool | 60 | 228 | 444 | 846 | 1579 | 2314 | 3018 | 3374 |

## esp

| series | 1c | 4c | 8c | 16c | 32c | 48c | 64c | 96c |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Intel 185H (laptop) · threads | 38 | — | 159 | — | — | — | — | — |
| EPYC 9474F (cluster) · threads | 25 | 93 | 176 | 340 | 577 | 883 | 1091 | 1034 |
| EPYC 9474F (cluster) · pool | 25 | 92 | 179 | 339 | 636 | 922 | 1227 | 1651 |

## pharm

| series | 1c | 4c | 8c | 16c | 32c | 48c | 64c | 96c |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Intel 185H (laptop) · threads | 1916 | — | 4142 | — | — | — | — | — |
| EPYC 9474F (cluster) · threads | 1575 | 2924 | 3844 | 4558 | 5086 | 4627 | 4310 | 4483 |
| EPYC 9474F (cluster) · pool | 1571 | 5042 | 8598 | 13001 | 16962 | 19394 | 20309 | 21572 |
