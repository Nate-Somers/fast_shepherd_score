# Figure 4 — the electrostatic term carries information orthogonal to shape

**Claim under test:** ESP is the package's differentiator, so it must measure
something **shape does not**. This figure tests that directly — and reports an
important, honest nuance about *when* the ESP term is discriminative.

## Design (the right experiment)
Random drug pairs are the **wrong** test: across diverse molecules, shape and ESP
similarity are naturally correlated (a pair that is shape-similar is usually
ESP-similar too), so ESP looks redundant (we verified: Spearman(shape,esp)≈1.0 on
105 random drug pairs). ESP's value is specifically in separating **shape-matched,
electrostatically-different** molecules.

So we use a **benzene-ring analog series** (benzene, toluene, halobenzenes, phenol,
aniline, pyridine, benzaldehyde, nitrobenzene) — all ~the same shape, spanning
nonpolar→strongly polar. Each analog is aligned to benzene **by shape**, then ESP
similarity is scored at that shape pose across a sweep of the ESP weight `lam`
(smaller `lam` = sharper electrostatic weighting). Charges are **xTB** partial
charges (the package's intended physical model; MMFF gives the same qualitative
picture, slightly weaker).

## Result (xTB charges, measured)
- **At the default `lam=0.3`, ESP ≈ shape** — discrimination (std across analogs)
  is 0.027, essentially equal to shape's 0.026. The ESP term is nearly inert at
  the default weight.
- **As `lam` decreases, ESP discrimination emerges and is electrostatically
  correct:** std grows 0.027 → 0.077 (lam 0.3 → 0.003, ~3×); polar analogs are
  pushed down (nitrobenzene 0.606 → **0.404**, benzaldehyde 0.621 → 0.526) while
  nonpolar ones barely move (toluene 0.640 → 0.636; benzene-self ≈ unchanged).
- At a discriminating weight, shape ranks every analog ~0.6–0.7 (all benzene-shaped)
  but ESP separates them by polarity (right panel: polar analogs fall below the
  shape diagonal).

## Honest interpretation (important for the paper)
1. **ESP genuinely carries orthogonal electrostatic information** — it cleanly
   separates electrostatic character that shape cannot see.
2. **But its effect on the Tanimoto score is weight-dependent and small at the
   default `lam=0.3`.** The default makes ESP a gentle tie-breaker, not a dominant
   term. To use ESP as a real discriminator one must (a) use a smaller `lam` and
   (b) use physical (xTB) charges. **This is a finding the paper should address
   head-on** — either justify a default `lam`, expose it as a tunable, or report
   the ShaEP-style additive `esp_combo` (which weights ESP at a fixed fraction).
3. The decisive utility test is not the score magnitude on analogs but **retrieval
   enrichment** — does ESP improve actives recovery in a virtual screen (fig5).

## Reproduce
```bash
PYTHONPATH=. python paper/fig4_esp_value/run.py    # GPU env + xTB on PATH; writes analog_esp.json
PYTHONPATH=. python paper/fig4_esp_value/plot.py   # writes fig4_esp_value.{png,pdf}
```
