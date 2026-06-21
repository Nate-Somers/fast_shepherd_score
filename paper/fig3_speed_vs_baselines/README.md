# Figure 3 — speed & capability vs open-source 3D-similarity tools

## Caption
**Several open tools each do part of 3D molecular similarity; we show that
`fast_shepherd_score` is the only open tool that does the full correspondence-free shape +
electrostatic + pharmacophore overlay with GPU acceleration, and that on one machine its GPU
electrostatic throughput far exceeds the only other open ESP tool.** Every tool runs in a
single process on one node (NVIDIA L40S + Intel Xeon Gold 6542Y), so the CPU baselines and the
fss GPU path are a genuine same-machine comparison rather than a cross-hardware extrapolation.
Molecules and descriptors are built once (untimed); each tool's native per-pair operation is
then timed (mean ± SD) over a diverse 300-molecule DUD-E library, with the GPU batch comprising
4,096 *unique* pairs. **(Left)** throughput on a log axis for USRCAT (alignment-free moment
descriptor), RDKit O3A (atom-mapping alignment), ESP-Sim (O3A alignment then Gaussian shape +
ESP, CPU), and fss `vol`/`surf`/`pharm`/`esp` (GPU). The fss shape modes reach 10k–32k pairs/s,
and even its slowest mode (`esp`, shape + ESP) runs at ≈4k pairs/s — about **56× the
same-machine ESP-Sim** (70 pairs/s) and 140–440× the CPU aligners; USRCAT is faster (≈350k/s)
but performs a weaker, alignment-free task with no pose and no electrostatics. **(Right)** a
capability matrix over seven features makes the structural point: fss is the only row that is
simultaneously shape + ESP + pharmacophore, correspondence-free, GPU-accelerated, and open
source. (ROSHAMBO2's Gaussian *shape* is compared head-to-head in Fig 5; it carries no ESP
overlay.)

**Claim defended:** among open-source tools, `fast_shepherd_score` is the only one that
combines **correspondence-free aligned shape + ESP + pharmacophore** overlay **with GPU
acceleration** — and on the **same machine** its GPU ESP throughput far exceeds the only
other open ESP tool (ESP-Sim), which is CPU-bound.

## Design — every tool on ONE machine
All molecules/descriptors are built once (untimed); we then time each tool's native per-pair
op. Crucially, **the CPU baselines and fss-GPU run in the same process on the same node**, so
the ESP speedup is a real on-machine number, not the old cross-hardware footnote
(fss-on-L40S vs ESP-Sim-on-laptop). Library = a **diverse** set (300 unique DUD-E molecules);
GPU batch = 4096 **unique** pairs (not a handful tiled). Throughput is mean ± SD over reps.

## Result (NVIDIA L40S + Intel Xeon Gold 6542Y, 300 molecules)
| Tool / fss mode | pairs/s (mean) | ESP overlay? |
|---|--:|:--:|
| USRCAT (descriptor) | 352,907 | no |
| **fss · vol** (GPU, shape atoms) | **32,334** | — |
| **fss · surf** (GPU, shape surface) | **10,408** | — |
| **fss · pharm** (GPU) | **9,456** | — |
| **fss · esp** (GPU, shape+ESP) | **3,954** | **yes** |
| RDKit O3A (CPU) | 74 | no |
| ESP-Sim (CPU) | 70 | yes |
| fss · esp (CPU) | 2 | yes |

**Same-machine: fss esp (GPU) is ~56× ESP-Sim (CPU)** (3,954 vs 70 pairs/s), and fss's shape
modes are ~140–440× the CPU aligners. USRCAT is faster but does a fundamentally weaker job
(alignment-free moments — no pose, no electrostatics).

## Honest interpretation
- **USRCAT/O3A are faster but weaker:** USRCAT has no pose/ESP; O3A aligns by atom
  correspondence (no Gaussian shape, no ESP) and fails for dissimilar molecules — the
  scaffold-hopping case these methods exist for.
- **ESP-Sim is the only other aligned shape+ESP tool and is CPU-bound** — on this node fss
  runs the same ESP task ~56× faster. (The CPU baselines are slower here than on a laptop
  because the DUD-E library is larger/more flexible than the curated drug set — a more
  realistic screen.)
- fss does *more* work per pair (16-restart multi-start SE(3) optimization) than O3A's single
  atom-mapping align, and is still far faster — the comparison is unfavorable to fss by design.
- **Capability matrix (right) is the headline:** fss is the only tool that is
  shape+ESP+pharmacophore, correspondence-free, GPU, and open. ROSHAMBO2's Gaussian shape is
  benchmarked head-to-head in Fig 5 (it has no ESP overlay).

## Reproduce
```bash
sbatch paper/_engaging/figs134_refresh.sbatch     # runs fig1, fig3, fig4 on one GPU node
PYTHONPATH=. python paper/fig3_speed_vs_baselines/plot.py
```
