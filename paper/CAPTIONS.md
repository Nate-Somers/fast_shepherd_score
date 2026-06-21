# Figure captions (manuscript-ready)

Each caption opens at a high level (what the figure shows and why it matters), then moves into
methods and a panel-by-panel reading with the measured numbers. All figures were produced on
MIT Engaging datacenter GPUs in a single software environment; molecules throughout are the
same curated real-drug / DUD-E sets built with RDKit ETKDG+MMFF conformers, Open3D molecular
surfaces, MMFF or xTB partial charges, and RDKit pharmacophores.

---

## Figure 1 — A fast GPU implementation that computes the *same* answer as the reference

**A fast similarity engine is only trustworthy if it reproduces the established reference
math; here we show it does, so every speed and accuracy result that follows rests on identical
calculations rather than an approximation.** `fast_shepherd_score` exposes one alignment API
(`MoleculePairBatch.align_with_*`) with two interchangeable back ends — a reference
JAX/XLA implementation and the fork's fast Triton/GPU kernels — selected only by a `backend=`
flag. We compare them on all 105 distinct pairs of 15 marketed drugs, under one identical
configuration (16 deterministic SE(3) starts, 100 optimizer steps, α = 0.81, λ = 0.3), for the
four scoring modes the package supports: `vol` (atom-centred Gaussian volume), `surf` (sampled
molecular-surface overlap), `esp` (surface shape + electrostatic potential) and `pharm`
(pharmacophore). Two complementary experiments separate the scoring kernel from the optimizer.
**(Top row)** end-to-end aligned Tanimoto, JAX back end (x) vs Triton back end (y): points lie
on y = x, with a small residual (mean |Δ| ≈ 7×10⁻⁴ for `vol`, 4×10⁻³ for `surf`) that is
*directional* — a systematic bias, not zero-mean scatter (e.g. JAX > Triton on 85/105 `vol`
pairs, sign-test p ≪ 10⁻³). **(Bottom row)** the same molecule pairs scored at a single fixed
(identity) pose by three implementations — NumPy fp64 (reference), JAX, and PyTorch fp32 (the
precision the Triton kernels use): the fp32 GPU kernels agree with the fp64 reference to
≈10⁻⁷, i.e. 3–4 orders of magnitude tighter than the aligned residual. Because the multi-start
seeds are deterministic and identical across back ends (no random restarts), the larger aligned
residual is not stochastic restart noise; it is the optimizer trajectory diverging under fp32
versus fp64 arithmetic, while each scoring step itself agrees to near machine precision.
Measured on an NVIDIA L40S.

## Figure 2 — High GPU throughput that scales with batch size, then saturates

**Virtual screening aligns millions of molecule pairs, so the figure of merit is sustained
throughput at scale; we show the batched GPU path reaches ~10⁵ alignments/second on a single
datacenter card and characterise honestly how that throughput is governed by batch fill rather
than raw GPU horsepower.** Each "pair" is a real drug aligned to a rigid SE(3) copy of itself
(known optimum = 1.0), so this isolates kernel + launch throughput from convergence
difficulty. For every (GPU, mode, batch-size) cell the batched Triton alignment is timed
best-of-N over 7 repetitions and reported as mean ± SD; all four GPUs (L40S, A100-80GB,
H100-80GB, H200) run the **same conda environment** (identical torch/CUDA), removing the
mixed-software confound of stitching together runs from different machines. **(Left)** surface-
mode throughput versus batch size rises as larger batches amortise fixed per-call host
overhead, then flattens at a saturation knee and the GPU curves converge. **(Right)** peak
throughput per mode per GPU, with SD error bars, reaches ≈190k pairs/s (H200, `vol`) and ≈107k
(L40S, `surf`). Critically, there is **no clean generational ordering** — the L40S exceeds the
H100 on both `vol` and `surf`, and the ranking reshuffles by mode — which is the expected
signature of a launch/host-bound workload at large batch. We therefore claim high absolute
throughput and batch-scaling, not "newer GPU = faster"; multi-GPU is sub-linear and not
featured.

## Figure 3 — The only open-source tool combining aligned shape + ESP + pharmacophore on a GPU

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

## Figure 4 — The electrostatic term encodes information shape cannot, at a quantified weight

**ESP is the package's scientific differentiator, so it must measure something beyond shape;
using molecules that are deliberately the same shape but differ in polarity, we show the ESP
term genuinely encodes electrostatics — and state candidly that at the shipped default weight
it is nearly inert, becoming discriminative only when up-weighted.** A benzene-ring analog
series (benzene → nitrobenzene, spanning nonpolar to strongly polar at essentially constant
shape) is aligned to benzene **by shape**, after which ESP similarity is scored at that fixed
shape pose across a sweep of the ESP weight λ; charges are xTB (the intended physical model),
and every value is the mean ± SD over **12 replicates** (independent conformer seeds and
nondeterministic surface resampling) so the effect is not a single draw. **(A)** ESP similarity
versus λ, one line per analog: the lines are bunched at the package default λ = 0.3 (ESP ≈
shape) and fan apart only at smaller λ, with polar analogs dropping and nonpolar ones
essentially unchanged. **(B)** discrimination, defined as the spread (SD) across analogs:
shape-only is 0.031 ± 0.005, while ESP rises to 0.079 ± 0.008 at λ = 0.003 — a significant
separation (non-overlapping uncertainty bands) that emerges only once λ is increased. **(C)**
the ESP signal (shape − ESP at λ = 0.003) versus molecular dipole magnitude (from the xTB
charges) is essentially linear (Pearson r = 0.99), confirming the separation tracks
electrostatics, not residual shape. The practical implication — expose λ and report the sweep —
is validated by retrieval in Fig 6.

## Figure 5 — The first apples-to-apples benchmark against ROSHAMBO2

**The closest open competitor, ROSHAMBO2 (GPU Gaussian-shape overlay), has only ever published
speed-ups against its own predecessor — no head-to-head against an external tool exists — so
running that controlled comparison is itself a contribution; we show `fast_shepherd_score` is
faster on the matched shape task while reaching the same answer.** ROSHAMBO2 was built from
source into its own environment; both tools read the **same conformers** (embedded once and
written to SDF) and run on the **same GPU** (NVIDIA L40S) over a real 3,986-molecule library,
averaged across 8 query molecules. The representation-matched comparison is fss `vol`
(atom-centred Gaussian volume) versus ROSHAMBO2 shape (also atom-centred Gaussian); one-time
featurisation (prep) is separated from the repeated alignment (compute), each timed with
warm-up, best-of-N and CUDA synchronisation. Because the two optimisers differ (fss: 50 SE(3)
seeds × 100 steps; ROSHAMBO2: start_mode = 2 × 100 steps), effort is pinned and reported, and
the fairness anchor is recovered self-overlap on rigid SE(3) self-copies (known optimum 1.0).
**(Left)** fss `vol` sustains 67.9k pairs/s versus ROSHAMBO2's 17.7k compute-only — **3.8×
faster** — and 1.5× end-to-end, despite fss performing more search per pair (50 vs 10 starts);
ROSHAMBO2's typical combo (shape+color) mode is ~6× slower still (3.0k pairs/s), so the matched
3.8× is not an artifact of benchmarking only ROSHAMBO2's fastest path. **(Right)** all three
recover a self-overlap Tanimoto of 1.000 on self-copies, confirming the throughput is measured
at matched, fully-solved alignment quality rather than by one tool cutting corners. ROSHAMBO2's
"color" is a pharmacophore-feature overlay (fss's analog is `pharm`); fss additionally offers an
electrostatic (ESP) channel ROSHAMBO2 has no equivalent for — the differentiator developed in
Figs 3, 4 and 6.

## Figure 6 — Adding aligned ESP to shape improves active retrieval where electrostatics matter

**The decisive test of the ESP differentiator is not a score on toy analogs but whether it
recovers real actives better in a virtual screen; across eight protein targets we show that
adding the aligned electrostatic overlay to shape improves enrichment on electrostatically-
driven binding sites — and, honestly, not on a hydrophobic one — with the benefit growing
exactly as the ESP weight increases, tying retrieval back to the score-level finding of
Fig 4.** Each target is a retrospective DUD-E screen (all known actives plus ~3,000
property-matched decoys); for each, K = 8 query actives are independently aligned and scored
against the remaining library by shape-only (`surf`) and shape + ESP (`esp`), and enrichment
(ROC-AUC, EF1%, EF5%, BEDROC α = 20) is averaged over queries with bootstrap 95% confidence
intervals. The ablation is run at **equal optimisation budget** for both modes (50 SE(3) seeds,
200 steps — correcting a silent inequality in which shape-only otherwise received more
restarts), with xTB charges, tie-aware AUC, and a sweep of the ESP weight λ. Targets span
charged/polar pockets (ACES, FA10, HMDH, AMPC, ADA, and FABP4's fatty-acid carboxylate site),
an aminergic GPCR (ADRB2), and a hydrophobic control (the androgen receptor, ANDR). **(Left)**
ΔAUC = AUC(shape + ESP) − AUC(shape) versus λ, one line per target coloured by pocket
chemistry, with the package default λ = 0.3 marked: the benefit is near zero at the default and
grows as λ decreases — the same λ-dependence seen at the score level in Fig 4, now expressed as
retrieval power. **(Right)** per-target shape → shape + ESP ROC-AUC at λ = 0.003 with
confidence intervals: ESP improves retrieval on 6 of 8 targets (ΔAUC up to +0.27, positive
across all 8/8 query actives; e.g. ADRB2 +0.27, ACES +0.25, FABP4 +0.18, ADA +0.17), shows no
benefit on the hydrophobic androgen-receptor pocket (ANDR, +0.02, CI spanning zero), and one
charged exception (AMPC). In short, the electrostatic overlay helps precisely where the binding
chemistry predicts it should.
