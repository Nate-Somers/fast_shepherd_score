# Figure 4 — the electrostatic term carries information orthogonal to shape

## Caption
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

**Claim under test:** ESP is the package's differentiator, so it must measure something
**shape does not** — and we quantify *when* (which ESP weight λ) it becomes discriminative.

## Design (the right experiment, now with uncertainty)
Random drug pairs are the wrong test: shape and ESP similarity correlate across diverse
molecules (Spearman≈1.0 on 105 random pairs), so ESP looks redundant. ESP's value is in
separating **shape-matched, electrostatically-different** molecules. So we use a
**benzene-ring analog series** (benzene→nitrobenzene, nonpolar→strongly polar, all ~same
shape): align each to benzene by shape, then score ESP at that fixed pose across a λ sweep.
Charges are **xTB**. **Every value is mean ± SD over 12 replicates** (different conformer
seed + nondeterministic surface resampling) — so the discrimination is not a single draw.

## Result (xTB charges, 12 replicates)
- **A · ESP(λ) per analog:** at default **λ=0.3 the analog lines are bunched** (ESP ~inert);
  they fan out only at small λ, with polar analogs (nitrobenzene, benzaldehyde) dropping
  sharply and nonpolar ones (toluene) barely moving.
- **B · discrimination (SD across analogs) vs λ:** shape-only baseline = **0.031 ± 0.005**;
  ESP = 0.032 ± 0.005 at λ=0.3 (≈ shape, inert) → **0.079 ± 0.008 at λ=0.003** — a clear,
  significant separation (non-overlapping error bands), i.e. ESP out-discriminates shape only
  once λ is turned up.
- **C · the separation tracks electrostatics:** ESP signal (shape − ESP@λ=0.003) vs molecular
  dipole magnitude (from xTB charges) has **Pearson r = 0.99** — the discrimination is driven
  by polarity, not residual shape.

## Honest interpretation
1. **ESP genuinely carries orthogonal electrostatic information** (panel C, r=0.99) — it
   separates electrostatic character shape cannot see.
2. **But the effect is weight-dependent and ~inert at the default λ=0.3.** To use ESP as a
   real discriminator one must lower λ (and use physical xTB charges). The paper should
   **expose λ and report the sweep** rather than rely on the default.
3. The decisive *utility* test is retrieval enrichment — **Fig 6 shows ESP improves active
   retrieval, and the gain grows as λ decreases, exactly mirroring panel B.**

## Reproduce
```bash
sbatch paper/_engaging/figs134_refresh.sbatch     # or: PYTHONPATH=. python paper/fig4_esp_value/run.py
PYTHONPATH=. python paper/fig4_esp_value/plot.py
```
