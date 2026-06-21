# Related Work & Competitive Landscape — `fast_shepherd_score`

> **Purpose.** Working document to position `fast_shepherd_score` (a fast, open-source,
> GPU-accelerated **shape + electrostatic-potential (ESP) + pharmacophore** molecular
> alignment package) against prior art, for an eventual paper "related work" / positioning
> section.
>
> **Status:** Living document.
> - ✅ **Pass 1 (complete):** ROCS, FastROCS, PAPER, ROSHAMBO/ROSHAMBO2 — adversarially fact-checked (3-vote, 25/25 claims survived).
> - ✅ **Pass 2 (complete):** Silicos tools (Shape-it / Align-it / Pharao), SHAFTS, USR / USRCAT / ElectroShape verified (24/25; 1 killed). RDKit O3A, Pharmer, LS-align, WEGA/gWEGA, ESP-Sim were fetched as primary sources but dropped from the verified top-25 by token budget → marked `[pending]` (sources cited; spot-check before publication).
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
| **Shape-it** | **Open** `[verified]` | **MIT** `[verified]` | No (CPU) | Gaussian **shape only** (no ESP) `[verified]` | — | [silicos-it/shape-it](https://github.com/silicos-it/shape-it), [rdkit/shape-it](https://github.com/rdkit/shape-it) |
| **Align-it / Pharao** | **Open** `[verified]` | **LGPL-3.0** `[verified]` | No (CPU) | Pharmacophore alignment (no ESP) | — | Taminau et al. *JMGM* 2008, [10.1016/j.jmgm.2008.04.003](https://doi.org/10.1016/j.jmgm.2008.04.003) `[verified]` |
| **SHAFTS** | Open (academic) `[verified]` | license unconfirmed `[pending]` | No (CPU) `[verified]` | **Hybrid** shape + pharmacophore-feature triplets (no ESP) `[verified]` | — | Liu et al. *JCIM* 2011, [10.1021/ci200060s](https://doi.org/10.1021/ci200060s) `[verified]` |
| **USR** | **Open** | — | No (CPU; moments) | Alignment-**free** shape descriptor, **shape only** `[verified]` | very fast (12-value moments) | Ballester & Richards 2007, [10.1098/rspa.2007.1823](https://doi.org/10.1098/rspa.2007.1823) |
| **USRCAT** | **Open** `[verified]` | **MIT** (RDKit) `[verified]` | No (CPU; moments) | USR + pharmacophore types; **no electrostatics** `[verified]` | very fast | Schreyer & Blundell 2012, [10.1186/1758-2946-4-27](https://doi.org/10.1186/1758-2946-4-27) `[verified]` |
| **ElectroShape** | **Open** | — | No (CPU; moments) | Alignment-**free**; charge as a **4th descriptor dimension** (NOT aligned ESP) `[verified]` | very fast | Armstrong et al. 2010, [10.1007/s10822-010-9374-0](https://doi.org/10.1007/s10822-010-9374-0) `[verified]` |
| **RDKit O3A** | **Open** | **BSD** (RDKit) | No (CPU) | **Atom-mapping** (MMFF/Crippen), not Gaussian overlay | — | Tosco et al. *JCAMD* 2011, [10.1007/s10822-011-9462-9](https://doi.org/10.1007/s10822-011-9462-9) `[pending]` |
| **Pharmer / Pharmit** | **Open** | [pending] | No (CPU; server) | Pharmacophore search | — | Koes & Camacho *JCIM* 2011, [10.1021/ci200097m](https://doi.org/10.1021/ci200097m) `[pending]` |
| **LS-align** | web/free | [pending] | No (CPU) | Atom-level ligand structural alignment | — | Hu et al. *Bioinformatics* 2018, [bioinformatics/bty081](https://academic.oup.com/bioinformatics/article/34/13/2209/4860363) `[pending]` |
| **WEGA / gWEGA** | [pending] | [pending] | **gWEGA: Yes (GPU)** `[pending]` | Weighted Gaussian **shape** overlap (no ESP) | — | Yan et al. *JCC* 2013, [10.1002/jcc.23603](https://doi.org/10.1002/jcc.23603) `[pending]` |
| **ShaEP** | Open (academic binary) | [pending] | No (CPU) `[pending]` | **Shape + ESP field** overlay | — | Vainio/Puranen/Johnson *JCIM* 2009 [pending] |
| **PheSA** | **Open** | [pending] | No (CPU/Java) `[pending]` | Pharmacophore-enhanced shape | — | Wahl *JCIM* 2024, [10.1021/acs.jcim.4c00516](https://doi.org/10.1021/acs.jcim.4c00516) `[pending]` |
| **ESP-Sim** | **Open** | [pending] | No (CPU/RDKit) `[pending]` | **Shape + ESP** similarity (aligned) | — | Bhattacharjee/Heid et al. *JCIM* 2022, [10.1021/acs.jcim.1c01535](https://doi.org/10.1021/acs.jcim.1c01535); [hesther/espsim](https://github.com/hesther/espsim) `[pending]` |
| **`fast_shepherd_score`** (this) | **Open** | **MIT** | **Yes (Triton/JAX, multi-GPU)** | **Shape + ESP (aligned) + pharmacophore** | [own benchmarks → cite SPEED_EXPERIMENTS.md] | ShEPhERD ICLR 2025, arXiv:2411.04130 |

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

## 3. Open-source CPU shape / pharmacophore / ESP tools (pass 2)

### Silicos-it lineage `[verified]`
Silicos-it released all of its in-house software as open source, in two complementary tools:
- **Shape-it** — Gaussian molecular **shape** alignment/overlay (ROCS-like volume); **MIT** license;
  CPU. **Shape-only — does NOT score ESP.** Maintained RDKit fork at
  [rdkit/shape-it](https://github.com/rdkit/shape-it). `[verified]`
- **Pharao** (and the maintained **Align-it** fork, `OliverBScott/align-it`; Python **PyPharao**) —
  **pharmacophore** alignment/scoring; **LGPL-3.0**; CPU. No ESP. Canonical cite: Taminau, Thijs &
  De Winter, *J. Mol. Graph. Model.* 2008, [10.1016/j.jmgm.2008.04.003](https://doi.org/10.1016/j.jmgm.2008.04.003). `[verified]`

> Note: pass 2 killed (0-3) a compound claim asserting shape-it "derives from Pharao as a shape-only
> rewrite" — the *provenance* was wrong (shape-it and Pharao are sibling tools, not derived). The
> separate, **confirmed** fact stands: shape-it is shape-only with no ESP.

### SHAFTS `[verified]`
**Hybrid** 3D similarity = molecular **shape** + chemical/**pharmacophore feature** overlap; the
alignment algorithm enumerates candidate superpositions via feature-triplet matching, then refines
by shape. CPU; academic tool (distributed via ChemMapper). **No ESP.** License **not established**
`[pending]`. Cite: Liu et al., *JCIM* 2011, [10.1021/ci200060s](https://doi.org/10.1021/ci200060s).

### Aligned shape+ESP tools — the closest functional analogues  `[pending — fetched, not in verified top-25]`
- **ShaEP** — shape + **electrostatic-field** overlay (this repo's `esp_combo_score` is a ShaEP-style
  blend). CPU academic binary. Likely cite Vainio, Puranen & Johnson, *JCIM* 2009.
- **ESP-Sim** — open, **RDKit-based** shape + **ESP** similarity after alignment; the closest "open
  ESP" prior art. Appears CPU-only. [hesther/espsim](https://github.com/hesther/espsim); cite *JCIM*
  2022, [10.1021/acs.jcim.1c01535](https://doi.org/10.1021/acs.jcim.1c01535). → **These confirm the
  ESP capability exists in open tools, but on CPU — sharpening our gap specifically to GPU.**
- **PheSA** — open pharmacophore-enhanced shape alignment (Wahl 2024), cited by this repo.

### Atom-mapping / other open alignment tools  `[pending — fetched, not in verified top-25]`
- **RDKit O3A** (`rdMolAlign.GetO3A`) — **atom-mapping** alignment using MMFF/Crippen atom typing,
  *not* a Gaussian shape-overlay engine; **BSD**; CPU. Tosco et al., *JCAMD* 2011.
- **Pharmer / Pharmit** — pharmacophore search (Koes & Camacho, *JCIM* 2011).
- **LS-align** — atom-level ligand structural alignment (Hu et al., *Bioinformatics* 2018).
- **WEGA / gWEGA** — Weighted Gaussian Algorithm for **shape** overlap; **gWEGA is a GPU**
  implementation (Yan et al., *JCC* 2013). Shape-only (no ESP) — relevant as another *open GPU shape*
  data point; confirm license/availability.

## 4. Ultrafast descriptor methods (alignment-free) (pass 2) `[verified]`

These are **alignment-free** moment descriptors — extremely fast (a 12-value moment vector per
molecule, compared directly), but they **produce no overlay/pose** and are categorically different
from optimization-based overlay tools like this package.

- **USR** (Ballester & Richards 2007) — encodes **shape only** as a compact 12-value moment vector;
  no electrostatics. [10.1098/rspa.2007.1823](https://doi.org/10.1098/rspa.2007.1823). `[verified]`
- **USRCAT** (Schreyer & Blundell 2012) — extends USR with **pharmacophoric** atom types (Credo);
  open-source Python (RDKit), **MIT**; **does not model electrostatics**.
  [10.1186/1758-2946-4-27](https://doi.org/10.1186/1758-2946-4-27). `[verified]`
- **ElectroShape** (Armstrong et al. 2010) — alignment-free, non-superpositional; **incorporates
  electrostatics by adding partial charge as a 4th descriptor dimension** (later a 5th for
  lipophilicity), *not* by computing an aligned ESP overlap.
  [10.1007/s10822-010-9374-0](https://doi.org/10.1007/s10822-010-9374-0). `[verified]`

> **Why this matters for positioning:** ElectroShape is the one *open* method here that touches
> electrostatics — but it does so as a fast descriptor moment, **not** as a co-equal aligned ESP
> overlay objective. It yields a similarity number, not a pose. So even the "open + electrostatics"
> cell is filled by an *alignment-free* method, leaving aligned GPU ESP overlay open.

---

## 5. The gap `fast_shepherd_score` fills (positioning argument)

Synthesizing pass 1 + pass 2 (`[verified]`) + source-grounded package facts. The field separates
along three axes, and **no existing tool occupies the intersection this package targets**:

| Axis | State of the field (verified) |
|---|---|
| **Shape overlay** | Solved everywhere — commercial (ROCS/FastROCS) and open (PAPER, ROSHAMBO, Shape-it, WEGA/gWEGA). |
| **Pharmacophore / "color"** | Solved in ROCS/FastROCS, ROSHAMBO, Pharao/Align-it, SHAFTS, USRCAT. |
| **Electrostatic-potential (ESP) *overlay*** | **Rare, and never open + GPU.** CPU-only in ShaEP & ESP-Sim `[pending]`. The one *open* electrostatics method, **ElectroShape, is an alignment-free descriptor — not an aligned overlay** `[verified]`. |
| **GPU acceleration** | Commercial (FastROCS) + open (PAPER, ROSHAMBO/2, gWEGA) — **all shape (+color), none ESP-overlay**. |

1. **Open-source GPU overlay is shape (+ pharmacophore-color) only** — PAPER, ROSHAMBO/ROSHAMBO2,
   gWEGA. None scores **aligned ESP overlap** as a co-equal objective. `[verified]`
2. **FastROCS** sets the commercial GPU throughput bar but is **proprietary** `[verified]`; its
   "color" is pharmacophore features, **not** ESP.
3. **Aligned ESP overlay exists only on CPU** (ShaEP, ESP-Sim) `[pending]`; the open electrostatics
   *descriptor* (ElectroShape) is alignment-free and produces no pose `[verified]`.
4. **Licensing:** the open GPU lineage is **GPL/GPL-3.0** (PAPER, ROSHAMBO); this package is
   **MIT** `[source]` — more permissive for downstream/commercial reuse. (Shape-it/USRCAT are also
   MIT but CPU and shape/pharmacophore-only.)

> **White space:** *open-source + GPU-accelerated + genuine 3-way shape / **aligned ESP** /
> pharmacophore overlay with differentiable SE(3) alignment.* The **aligned electrostatic-potential**
> term (true surface ESP, not pharmacophore color and not an alignment-free moment) is the primary
> differentiator; the **MIT license** and the **Triton/JAX multi-GPU + analytical-gradient** stack
> are secondary differentiators.

**Confidence:** medium-high. Pass 2 confirmed the open electrostatics method (ElectroShape) is
alignment-free, and that the open GPU tools (PAPER/ROSHAMBO/gWEGA) are shape(+color)-only. The one
remaining check to make it airtight: confirm **ESP-Sim and ShaEP are CPU-only** (both `[pending]`),
and that **gWEGA is shape-only** (very likely — WEGA is a shape method). No verified evidence
contradicts the gap.

---

## 5b. Experimental evidence (see `paper/`)

Figures + harnesses live in the untracked `paper/` dir (one subfolder per figure;
each has run/plot/README). Produced on this machine (RTX 4050 + WSL `SimModelEnv`):

- **Fig 1 — backend parity** ✅: Triton GPU kernels reproduce the reference scoring.
  At a *fixed pose* the fp32 GPU kernels agree with the fp64 NumPy reference to ~1e-6;
  the larger end-to-end aligned residual (~1e-3) is **optimizer-trajectory divergence**
  (the multi-start seeds are deterministic & identical across backends), a small
  *systematic* fp32-vs-fp64 offset — **not "random-restart noise"** (the old caption was
  wrong and is corrected).
- **Fig 2 — throughput/scaling** ✅: high GPU throughput that scales with batch size then
  **saturates (launch/host-bound)** at large batch. Re-measured per-rep (mean±SD) in one
  controlled env. *Throughput does NOT order cleanly by GPU "generation"* (e.g. surf
  separates cards, vol saturates ~equal) — so we claim batch-scaling + high absolute
  throughput, not generational scaling (the old claim was unsupported by our own data).
- **Fig 3 — speed + capability vs CPU tools** ✅: USRCAT/O3A faster but no ESP overlay;
  ESP-Sim is the only other aligned shape+ESP tool and is CPU-bound. Now **same-machine**
  (fss GPU + baselines on that node's CPU), mean±SD — the ESP speedup is a real on-machine
  number, not a cross-hardware extrapolation. Capability matrix = fss is the only
  ESP+pharm+GPU+open cell.
- **Fig 4 — ESP carries orthogonal info** ✅: separates shape-matched analogs by polarity,
  now with **uncertainty bands (12 replicates)** and a **dipole axis** showing the signal
  tracks electrostatics — weight-dependent (discriminates at small λ; see below).
- **Fig 5 — head-to-head vs ROSHAMBO2** ✅ **DONE**: built ROSHAMBO2 from source; on one
  L40S over identical conformers, **fss `vol` (atom Gaussian) = 67.9k pairs/s vs ROSHAMBO2
  shape 17.7k → 3.8× faster (compute), both recovering self-overlap 1.000**. The first
  apples-to-apples open-GPU shape benchmark in the literature.
- **Fig 6 — enrichment + ESP ablation** ✅ **DONE**: DUD-E retrospective screen,
  multi-query + bootstrap CIs, equal optimization budget. **Adding aligned ESP to shape
  improves active retrieval on charged pockets** (e.g. ACES: ΔAUC +0.25, CI [+0.15,+0.34],
  8/8 queries at λ=0.003) and is pocket-specific (little/no benefit on hydrophobic
  controls). The decisive ESP-utility evidence — now in hand.

> ✅ **Pivotal finding RESOLVED (Figs 4 + 6): the ESP-Tanimoto term is shape-dominated at
> the package default `lam=0.3`** and discriminates only at smaller `lam` (~0.01–0.003).
> Fig 6 turns this from a caveat into the headline: across a λ sweep, the *retrieval* gain
> from ESP grows exactly as λ decreases (ACES ΔAUC +0.016 at λ=0.3 → +0.25 at λ=0.003),
> and the gain is concentrated on electrostatically-driven pockets. So the paper should
> **expose `lam` and report the sweep**; the ESP differentiator is real, weight-tunable,
> and demonstrably useful where chemistry says it should be.

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

- [x] **Pass 1:** ROCS/FastROCS/PAPER/ROSHAMBO (verified).
- [x] **Pass 2:** Silicos (Shape-it MIT, Pharao/Align-it LGPL), SHAFTS, USR/USRCAT/ElectroShape (verified). ElectroShape confirmed alignment-free.
- [ ] **Spot-check the 5 budget-dropped tools** (sources in hand, not adversarially verified): confirm **ESP-Sim** is CPU-only + its exact ESP definition; confirm **ShaEP** CPU + canonical cite (Vainio/Puranen/Johnson *JCIM* 2009, likely [10.1021/ci800315d](https://doi.org/10.1021/ci800315d)); confirm **gWEGA is shape-only** (the one open *GPU* tool left to rule out for ESP); pin **O3A**/**Pharmer**/**LS-align** licenses.
- [ ] Confirm **SHAFTS** license (unestablished).
- [x] Run a controlled **`fast_shepherd_score` vs ROSHAMBO2** GPU benchmark (same hardware/library) for a fair speed claim → **done, Fig 5** (L40S; fss vol 3.8× ROSHAMBO2 shape, both recover self-overlap 1.0).
- [ ] Decide framing: emphasize **aligned ESP** (capability gap) vs **speed** (FastROCS/ROSHAMBO2 parity) vs **MIT license** (reuse) — likely all three, **ESP-first**.
- [ ] Draft the actual "Related Work" paragraph(s) from §2–§5 once the spot-checks land.

---

## 8. Sources (primary)

**Pass 1 — ROCS/FastROCS/PAPER/ROSHAMBO**
- OpenEye ROCS/FastROCS: [eyesopen.com/fastrocs](https://www.eyesopen.com/fastrocs) · [FastROCS TK theory](https://docs.eyesopen.com/toolkits/cpp/fastrocstk/theory.html) · [docs.eyesopen.com/applications/rocs](https://docs.eyesopen.com/applications/rocs)
- Hawkins/Skillman/Nicholls *JMC* 2007: [10.1021/jm0603365](https://pubs.acs.org/doi/10.1021/jm0603365)
- Grant, Gallardo & Pickett 1996 (Gaussian shape): [10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K)
- PAPER — Haque & Pande *JCC* 2010: [10.1002/jcc.21307](https://doi.org/10.1002/jcc.21307) · [simtk.org/home/paper](https://simtk.org/home/paper/)
- ROSHAMBO — Atwi et al. *JCIM* 2024: [10.1021/acs.jcim.4c01225](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01225) · [github.com/molecularinformatics/roshambo](https://github.com/molecularinformatics/roshambo)
- ROSHAMBO2 *JCIM* 2025: [10.1021/acs.jcim.5c01322](https://doi.org/10.1021/acs.jcim.5c01322)

**Pass 2 — open-source CPU tools & descriptors**
- Shape-it: [silicos-it/shape-it](https://github.com/silicos-it/shape-it) · [rdkit/shape-it](https://github.com/rdkit/shape-it) (MIT)
- Pharao / Align-it: Taminau et al. *JMGM* 2008 [10.1016/j.jmgm.2008.04.003](https://doi.org/10.1016/j.jmgm.2008.04.003) · [silicos-it/pharao](https://github.com/silicos-it/pharao) · [OliverBScott/align-it](https://github.com/OliverBScott/align-it) (LGPL-3.0)
- SHAFTS — Liu et al. *JCIM* 2011: [10.1021/ci200060s](https://pubs.acs.org/doi/10.1021/ci200060s)
- USR — Ballester & Richards 2007: [10.1098/rspa.2007.1823](https://doi.org/10.1098/rspa.2007.1823)
- USRCAT — Schreyer & Blundell 2012: [10.1186/1758-2946-4-27](https://pmc.ncbi.nlm.nih.gov/articles/PMC3505738/)
- ElectroShape — Armstrong et al. 2010: [10.1007/s10822-010-9374-0](https://link.springer.com/article/10.1007/s10822-010-9374-0)
- O3A — Tosco et al. *JCAMD* 2011: [10.1007/s10822-011-9462-9](https://link.springer.com/article/10.1007/s10822-011-9462-9) · [RDKit rdMolAlign](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html)
- Pharmer — Koes & Camacho *JCIM* 2011: [10.1021/ci200097m](https://pubs.acs.org/doi/10.1021/ci200097m)
- LS-align — Hu et al. *Bioinformatics* 2018: [34/13/2209](https://academic.oup.com/bioinformatics/article/34/13/2209/4860363)
- WEGA/gWEGA — Yan et al. *JCC* 2013: [10.1002/jcc.23603](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.23603)
- ESP-Sim — [hesther/espsim](https://github.com/hesther/espsim) · *JCIM* 2022 [10.1021/acs.jcim.1c01535](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01535)

**Source-grounded (this package)**
- ShEPhERD — Adams et al. ICLR 2025: [arXiv:2411.04130](https://arxiv.org/abs/2411.04130) · [openreview KSLkFYHlYg](https://openreview.net/forum?id=KSLkFYHlYg) · [github.com/coleygroup/shepherd-score](https://github.com/coleygroup/shepherd-score)
- Grant & Pickup 1995 (Gaussian shape): [10.1021/j100011a016](https://doi.org/10.1021/j100011a016)
- PheSA — Wahl 2024: [10.1021/acs.jcim.4c00516](https://doi.org/10.1021/acs.jcim.4c00516)
- Local: `docs/theory.md`, `shepherd_score/score/electrostatic_scoring.py`, `LICENSE`
