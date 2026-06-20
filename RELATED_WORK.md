# Related Work & Competitive Landscape — `fast_shepherd_score`

> **Purpose.** Working document to position `fast_shepherd_score` (a fast, open-source,
> GPU-accelerated **shape + electrostatic-potential (ESP) + pharmacophore** molecular
> alignment package) against prior art, for an eventual paper "related work" / positioning
> section.
>
> **Status:** Living document.
> - ✅ **Pass 1 (complete):** ROCS, FastROCS, PAPER, ROSHAMBO/ROSHAMBO2 — adversarially fact-checked (3-vote, 25/25 claims survived).
> - 🔎 **Pass 2 (running):** Silicos tools (Shape-it / Align-it / Pharao), SHAFTS, USR / USRCAT / ElectroShape, RDKit O3A, Pharmer, LS-align, WEGA/gWEGA, ESP-Sim, ShaEP, PheSA, ShEPhERD scoring formulation. Sections marked **[PASS 2 PENDING]** below.
> - 📌 **Source-grounded:** This package's own identity/formulation taken directly from local source (`docs/theory.md`, `shepherd_score/score/`), not web research.
>
> **Confidence tags:** `[verified]` = survived 3-vote adversarial check in pass 1; `[source]` = read directly from this repo's code/docs; `[pending]` = awaiting pass-2 verification.

---

## 0. What this package is (source-grounded)

`fast_shepherd_score` is a fast/GPU-accelerated fork of **`shepherd_score`** (Coley lab), the
scoring + alignment library behind **ShEPhERD** (*Shape, Electrostatics, and Pharmacophores
Explicit Representation Diffusion*).

| Attribute | Value | Source |
|---|---|---|
| License | **MIT** (permissive) | `LICENSE` `[source]` |
| Scoring objectives | **Shape** (Gaussian volume/surface overlap) + **ESP** (continuous surface electrostatics) + **Pharmacophore** (typed, directional) | `shepherd_score/score/` `[source]` |
| Alignment | Continuous **SE(3) optimization** (max Tanimoto over rotation+translation), multi-start (`num_repeats`) | `docs/theory.md`, `alignment/` `[source]` |
| Gradients | **Analytical** PyTorch gradients (since v1.3.1, ~2–2.5× over autograd) + autograd + JAX | `docs/theory.md`, `score/analytical_gradients/` `[source]` |
| Backends | PyTorch, JAX (incl. `shard_map` multi-CPU/parallel), NumPy, **Triton GPU kernels** (shape, ESP, pharmacophore), multi-GPU | `score/*_triton.py`, `container/_batch*.py` `[source]` |
| Canonical citation | Adams, Abeywardane, Fromer & Coley (2025), *ShEPhERD*, ICLR 2025; arXiv:2411.04130 | `README.md` `[source]` |

**Key point for positioning:** the ESP term is **genuine electrostatic-potential overlap**, not
pharmacophore "color." Two ESP scoring variants exist in the code:

1. **ShEPhERD ESP** (`get_overlap_esp` / `VAB_2nd_order_esp`, `electrostatic_scoring.py`) `[source]`
   — Gaussian surface-point overlap **weighted by the Coulombic potential difference** at each
   point pair. With surface points $\boldsymbol{S}_3$ and per-point potential $\boldsymbol{v}$:

   $$O^{\text{ESP}}_{A,B} = \sum_{a}\sum_{b}\Big(\tfrac{\pi}{2\alpha}\Big)^{3/2}
   \exp\!\Big(-\tfrac{\alpha}{2}\lVert \boldsymbol{r}_a-\boldsymbol{r}_b\rVert^2\Big)
   \exp\!\Big(-\tfrac{\lVert \boldsymbol{v}_A[a]-\boldsymbol{v}_B[b]\rVert^2}{\lambda}\Big)$$

   scored as a Tanimoto $O_{AB}/(O_{AA}+O_{BB}-O_{AB})$. The ESP is rigid-motion-invariant, so the
   charge-difference factor is a constant during alignment (enables analytical gradients).

2. **ShaEP-style combo** (`esp_combo_score`, `electrostatic_scoring.py`) `[source]` — explicitly
   "a similarity score defined by ShaEP": a volume-masked surface-ESP comparison +
   volumetric shape overlap, blended by `esp_weight` (default 0.5). This is a direct nod to the
   **ShaEP** tool (see §3).

Foundational refs the package itself cites `[source]`: Grant & Pickup (1995, 1996, 1997) Gaussian
shape; Taminau et al. (2008) **Pharao**; Wahl (2024) **PheSA**.

---

## 1. The competitive landscape at a glance

| Tool | Open / Proprietary | License | GPU? | Scoring | Speed (reported) | Canonical cite |
|---|---|---|---|---|---|---|
| **ROCS** | Proprietary (OpenEye/Cadence) | Commercial | No (CPU) | Shape + pharmacophore "color"; TanimotoCombo | ~thousands conf/s `[verified]` | Hawkins/Skillman/Nicholls *JMC* 2007, [10.1021/jm0603365](https://pubs.acs.org/doi/10.1021/jm0603365) |
| **FastROCS** | Proprietary (OpenEye/Cadence) | Commercial | **Yes (CUDA)** | Same as ROCS, full GPU rewrite | "millions → hundreds of millions" conf/s (vendor) `[verified]` | [eyesopen.com/fastrocs](https://www.eyesopen.com/fastrocs) |
| **PAPER** | **Open** | **GPL** | **Yes (CUDA)** | Gaussian **shape only** | ~20–100× vs 1-thread CPU; 5–35× vs oeROCS (GTX 280) `[verified]` | Haque & Pande *JCC* 2010, [10.1002/jcc.21307](https://doi.org/10.1002/jcc.21307) |
| **ROSHAMBO** | **Open** | **GPL-3.0** | **Yes (wraps PAPER)** | Shape **+ color** (RDKit pharmacophores) | GPU-accelerated; DUDE-Z benchmarks `[verified]` | Atwi et al. *JCIM* 2024, [10.1021/acs.jcim.4c01225](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01225) |
| **ROSHAMBO2** | **Open** | **GPL-3.0** | **Yes (multi-GPU)** | Shape + pharmacophore | **>200×** over ROSHAMBO v1 (self-reported; dual RTX 4090) `[verified]` | *JCIM* 2025, [10.1021/acs.jcim.5c01322](https://doi.org/10.1021/acs.jcim.5c01322) |
| **ShaEP** | Open (academic binary) | [pending] | No (CPU) `[pending]` | **Shape + ESP field** overlay | [PASS 2 PENDING] | Vainio/Puranen/Johnson *JCIM* 2009 [pending] |
| **PheSA** | **Open** | [pending] | No (CPU/Java) `[pending]` | Pharmacophore-enhanced shape | [PASS 2 PENDING] | Wahl *JCIM* 2024, [10.1021/acs.jcim.4c00516](https://doi.org/10.1021/acs.jcim.4c00516) |
| **Shape-it** | **Open** | [pending] | No (CPU) | Gaussian shape (RDKit-adjacent) | [PASS 2 PENDING] | [silicos-it/shape-it](https://github.com/silicos-it/shape-it) |
| **Align-it / Pharao** | **Open** | [pending] | No (CPU) | Pharmacophore alignment | [PASS 2 PENDING] | Taminau et al. *JMGM* 2008, [10.1016/j.jmgm.2008.04.003](https://doi.org/10.1016/j.jmgm.2008.04.003) |
| **SHAFTS** | [pending] | [pending] | [pending] | Hybrid shape + feature triplets | [PASS 2 PENDING] | [pending] |
| **USR / USRCAT** | **Open** | [pending] | No (CPU; moments) | Alignment-free shape descriptor | very fast (moment-based) `[pending]` | [PASS 2 PENDING] |
| **ElectroShape** | **Open** | [pending] | No (CPU; moments) | USR + **partial-charge electrostatics** | [PASS 2 PENDING] | [pending] |
| **RDKit O3A** | **Open** | **BSD** | No (CPU) | Atom-mapping (MMFF/Crippen) alignment | [PASS 2 PENDING] | Tosco et al. *JCAMD* 2011 [pending] |
| **ESP-Sim** | **Open** | [pending] | No (CPU/RDKit) | **Shape + ESP** similarity | [PASS 2 PENDING] | [hesther/espsim](https://github.com/hesther/espsim) |
| **`fast_shepherd_score`** (this) | **Open** | **MIT** | **Yes (Triton/JAX, multi-GPU)** | **Shape + ESP + pharmacophore** | [own benchmarks → cite SPEED_EXPERIMENTS.md] | ShEPhERD ICLR 2025, arXiv:2411.04130 |

---

## 2. Verified detail — the ROCS / FastROCS / PAPER / ROSHAMBO spine (pass 1)

All claims below survived a 3-vote adversarial verification (need 2/3 refutes to kill; 25/25 survived).

### ROCS (the foundational method) `[verified]`
Ligand-centric Gaussian overlay. Shape via "fuzzy" Gaussian atom functions (Grant, Gallardo &
Pickett 1996 fused-sphere volume); chemistry via six "hard" Gaussian **color** features (H-bond
donor/acceptor, hydrophobe, anion, cation, ring). Scores **Shape Tanimoto + Color Tanimoto =
TanimotoCombo** (0–2). CPU, ~thousands conformers/s. The "ROCS more consistent than/superior to
docking" result (Hawkins/Skillman/Nicholls, *JMC* 2007) is **vendor-authored** (all three authors
OpenEye) — cite as such.
Sources: [10.1021/jm0603365](https://pubs.acs.org/doi/10.1021/jm0603365), [docs.eyesopen.com/applications/rocs](https://docs.eyesopen.com/applications/rocs)

### FastROCS (commercial GPU gold standard) `[verified]`
A **complete rewrite of the ROCS algorithm for NVIDIA GPUs** (OpenEye's own words). Same shape +
color scoring. Throughput tiered by OpenEye as *ROCS: thousands/s → FastROCS: millions/s*; product
page claims "millions to hundreds of millions of conformations/sec." 2011 benchmark: 5M cpds ×10
conf vs 1 query in 30–40 s on 4 GPUs.
**Caveat:** all throughput numbers are **vendor marketing**, conflate GPU generations, and the
"hundreds of millions/s" figure has no peer-reviewed verification. Treat as order-of-magnitude.
Proprietary (OpenEye, now **Cadence Molecular Sciences**); commercial license; **no open-source fork
exists** (confirmed by targeted adversarial search).
Sources: [eyesopen.com/fastrocs](https://www.eyesopen.com/fastrocs), [FastROCS TK theory](https://docs.eyesopen.com/toolkits/cpp/fastrocstk/theory.html)

### PAPER (the open-source GPU root) `[verified]`
Haque & Pande (Stanford, 2010): **GPL** CUDA implementation of ROCS-style Gaussian **shape** overlay
(Grant & Pickett analytic volume), "an open platform for further development." Speedups (GTX 280 era):
~20× vs icc/Xeon, >100× vs gcc/AMD; **30–35× vs commercial oeROCS** on medium/large molecules (5–10×
small). Crucially, **DUD enrichment (ROC AUC) was statistically indistinguishable from oeROCS** on
most systems → an open GPU Gaussian-overlay engine *can* match commercial enrichment (oeROCS more
reliably found the global-max orientation).
Sources: [10.1002/jcc.21307](https://doi.org/10.1002/jcc.21307), [simtk.org/home/paper](https://simtk.org/home/paper/)

### ROSHAMBO / ROSHAMBO2 (closest open-source analogue) `[verified]`
**ROSHAMBO** (Atwi et al., *JCIM* 2024; **GPL-3.0**): Python package using **PAPER as its GPU
alignment backend**, adding **shape + color** (RDKit pharmacophore features: donor/acceptor/
cation/anion/ring) → ComboTanimoto, with SDF output. Closest current open analogue to FastROCS's
shape+color. **ROSHAMBO2** (*JCIM* 2025): re-engineered GPU stack, **>200× over v1** (self-reported,
dual RTX 4090), multi-GPU/server mode.
**Caveat:** the 200× is vs the authors' *own* prior version, **not** vs FastROCS — no clean
open-vs-commercial GPU head-to-head exists.
Sources: [10.1021/acs.jcim.4c01225](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01225), [github.com/molecularinformatics/roshambo](https://github.com/molecularinformatics/roshambo), [10.1021/acs.jcim.5c01322](https://doi.org/10.1021/acs.jcim.5c01322)

---

## 3. CPU shape/pharmacophore/ESP tools  **[PASS 2 PENDING]**

_To be filled from pass 2. Targets: Shape-it, Align-it/Pharao (Silicos), SHAFTS, ShaEP, PheSA,
RDKit O3A, Pharmer, LS-align, WEGA/gWEGA._

- **ShaEP** — directly relevant: this package's `esp_combo_score` implements a ShaEP-style
  shape+ESP blend. Shape + electrostatic-field overlay, CPU. [confirm license, citation, speed]
- **PheSA** — Wahl 2024, open-source pharmacophore-enhanced shape alignment (cited by this repo).
- **ESP-Sim** — open, RDKit-based **shape + ESP** similarity; confirm CPU-only (the closest "open
  ESP" prior art → sharpens our gap to *GPU* ESP). [confirm ESP definition: partial-charge ESP on
  grid vs Gaussian charge density]

## 4. Ultrafast descriptor methods  **[PASS 2 PENDING]**

_USR (Ballester & Richards), USRCAT (Schreyer & Blundell), ElectroShape (Armstrong et al.)._
These are **alignment-free** moment descriptors (very fast, but do not produce an overlay/pose).
ElectroShape folds partial charges into the descriptor — note as the descriptor-side analogue of
"shape+electrostatics," but distinct from optimization-based overlay. [confirm citations, licenses]

---

## 5. The gap `fast_shepherd_score` fills (positioning argument)

Synthesizing pass 1 (`[verified]`) + source-grounded package facts:

1. **Open-source GPU shape overlay is solved** (PAPER → ROSHAMBO → ROSHAMBO2), but it is
   **shape + pharmacophore-"color"** — `[verified]` none of the confirmed open GPU baselines score
   **electrostatic-potential overlap** as a co-equal objective.
2. **FastROCS** sets the commercial GPU throughput bar but is **proprietary** `[verified]`.
3. The only confirmed open **ESP** tools (ESP-Sim; ShaEP) are **CPU**-based [pending confirmation].
4. **Licensing:** the open GPU lineage is **GPL/GPL-3.0**; this package is **MIT** `[source]` — a
   more permissive license for downstream/commercial reuse.

> **White space:** *open-source + GPU-accelerated + genuine 3-way shape / **ESP** / pharmacophore
> overlay with differentiable SE(3) alignment.* The **electrostatic-potential** term (true surface
> ESP, not pharmacophore color) is the primary differentiator, **MIT license** and the
> **Triton/JAX multi-GPU + analytical-gradient** alignment stack are secondary differentiators.

**Confidence:** medium — the "no open GPU ESP tool exists" claim is an inference from *absence* in
the verified set; pass 2 (ESP-Sim, ShaEP, ElectroShape, gWEGA) must confirm none is GPU + ESP-overlay
to make this airtight.

---

## 6. Caveats (carry into the paper)

1. FastROCS throughput = **vendor marketing**, mixed GPU generations, no peer review of the headline
   figure → "commercial baseline, order of magnitude."
2. "ROCS > docking" (2007) is **self-authored** (OpenEye employees).
3. Cross-tool GPU throughput is **apples-to-oranges**: no published FastROCS-vs-ROSHAMBO2 benchmark
   on identical hardware/libraries → we should run our own controlled benchmark.
4. PAPER speed numbers are **2010-era** (GTX 280) — historical prior art, not today's ratios.
5. The ESP-gap finding proves a negative from a non-exhaustive set — needs pass-2 confirmation.

---

## 7. Open questions / TODO

- [ ] **Pass 2:** fill §3–§4 (Silicos, SHAFTS, USR family, O3A, Pharmer, LS-align, WEGA/gWEGA, ESP-Sim, ShaEP, PheSA).
- [ ] Confirm **ESP-Sim** is CPU-only and document its exact ESP-similarity definition.
- [ ] Confirm whether **any** open-source GPU tool scores ESP overlap co-equally with shape (gWEGA? — check).
- [ ] Pin the canonical **ShaEP** citation (Vainio, Puranen, Johnson, *JCIM* 2009) and license.
- [ ] Run a controlled **`fast_shepherd_score` vs ROSHAMBO2** GPU benchmark (same hardware/library) for a fair speed claim; cross-ref `SPEED_EXPERIMENTS.md`.
- [ ] Decide framing: emphasize **ESP** (capability gap) vs **speed** (FastROCS/ROSHAMBO2 parity) vs **MIT license** (reuse) — likely all three, ESP-first.

---

## 8. Sources (pass 1, primary)

- OpenEye ROCS/FastROCS: [eyesopen.com/fastrocs](https://www.eyesopen.com/fastrocs) · [FastROCS TK theory](https://docs.eyesopen.com/toolkits/cpp/fastrocstk/theory.html) · [docs.eyesopen.com/applications/rocs](https://docs.eyesopen.com/applications/rocs)
- Hawkins/Skillman/Nicholls *JMC* 2007: [10.1021/jm0603365](https://pubs.acs.org/doi/10.1021/jm0603365)
- Grant, Gallardo & Pickett 1996 (Gaussian shape): [10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K)
- PAPER — Haque & Pande *JCC* 2010: [10.1002/jcc.21307](https://doi.org/10.1002/jcc.21307) · [simtk.org/home/paper](https://simtk.org/home/paper/)
- ROSHAMBO — Atwi et al. *JCIM* 2024: [10.1021/acs.jcim.4c01225](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01225) · [github.com/molecularinformatics/roshambo](https://github.com/molecularinformatics/roshambo)
- ROSHAMBO2 *JCIM* 2025: [10.1021/acs.jcim.5c01322](https://doi.org/10.1021/acs.jcim.5c01322)
- ShEPhERD — Adams et al. ICLR 2025: [arXiv:2411.04130](https://arxiv.org/abs/2411.04130) · [openreview KSLkFYHlYg](https://openreview.net/forum?id=KSLkFYHlYg)
- ESP-Sim: [github.com/hesther/espsim](https://github.com/hesther/espsim)
