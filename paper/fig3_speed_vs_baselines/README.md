# Figure 3 — speed & capability vs open-source 3D-similarity tools

**Claim defended:** among open-source tools, `fast_shepherd_score` is the only one
that combines **correspondence-free aligned shape + ESP + pharmacophore** overlay
**with GPU acceleration** — and its GPU throughput on the same ESP task far exceeds
the only other open ESP tool (ESP-Sim), which is CPU-bound.

## Design (fair, simple)
All molecules / conformers / descriptors are built **once** (setup, untimed); we
then time each tool's **native per-pair operation** — the realistic amortized cost
in a screen — on the same machine (RTX 4050 laptop + Intel Core Ultra 9 185H):

| Tool | timed op | does ESP overlay? |
|---|---|---|
| USRCAT (RDKit) | `GetUSRScore(d_i,d_j)` (alignment-free moments) | no |
| RDKit O3A | `GetO3A(prb,ref).Align().Score()` (atom-mapping) | no |
| ESP-Sim | O3A-align then `GetShapeSim`+`GetEspSim` (MMFF) | yes (CPU) |
| fss (CPU) | `align_with_esp` (JAX backend) | yes |
| fss (GPU) | `align_with_esp` (Triton backend, batched) | yes |

ESP mode throughout. fss uses its real config (`num_repeats=16, steps=100`), i.e.
a thorough multi-start gradient SE(3) optimization — strictly more work per pair
than O3A's single atom-mapping alignment.

## Result (measured, RTX 4050 + Intel CPU)
| Tool | pairs/s |
|---|--:|
| USRCAT | 557,735 |
| RDKit O3A | 1,610 |
| ESP-Sim | 1,134 |
| fss (CPU) | 2.2 |
| fss (GPU), batched | 2,147 |
| fss (GPU), L40S esp peak [Fig 2] | 32,133 |

## Honest interpretation
- **USRCAT/O3A are faster but do a different, weaker job** — USRCAT is an
  alignment-free moment descriptor (no pose, no electrostatics); O3A aligns by
  atom correspondence (no Gaussian shape, no ESP) and *fails for dissimilar
  molecules* — exactly the scaffold-hopping case shape/ESP methods exist for.
- **ESP-Sim is the only other aligned shape+ESP tool**, and it is competitive on a
  6 GB laptop only because it aligns with cheap atom-mapping (O3A) rather than
  optimizing the overlay. It is **CPU-bound**: on a datacenter GPU the *same* fss
  ESP task runs ~**28×** faster (L40S 32k vs ESP-Sim 1.1k pairs/s).
- fss on this laptop is shown at a **mixed-size batch** (the realistic screening
  case); same-size batches reach ~8.5k pairs/s here (Fig 2).
- **Capability matrix** (right panel) is the real headline: fss is the only tool
  that is shape+ESP+pharmacophore, correspondence-free, GPU, open. ROSHAMBO2 is
  included from the literature (GPL; CUDA build not run here — see fig5).

## Reproduce
```bash
PYTHONPATH=. python paper/fig3_speed_vs_baselines/run.py    # GPU env; writes results.json
PYTHONPATH=. python paper/fig3_speed_vs_baselines/plot.py   # writes fig3_speed_vs_baselines.{png,pdf}
```
