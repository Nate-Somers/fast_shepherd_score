# Scoring-math pitfalls

Recurring traps when writing a new reference mode. These are general properties of the math, not
war stories about any one mode.

## Zeroing a direction vector is not "directionless"
A pharmacophore feature with an orientation vector contributes both a positional Gaussian and a
cosine term on the vectors. Setting the vector to zero does **not** make the feature isotropic:
the cosine similarity of two zero vectors is not 0 in the weighting used, so a zero-vector
feature still perturbs the score. A genuinely directionless feature must **skip the cosine path
entirely** and score on position only. If your objective has an "ignore orientation" mode, route
it through the point-only branch — do not fake it by nulling the vectors.

## Tanimoto self-overlap must be exactly 1.000
For a Tanimoto objective, a molecule aligned to a copy of itself must score 1.000. If it does
not, the self-overlap normalization is inconsistent — usually because the numerator and the two
self-overlap terms in the denominator are computed with different widths, masks, or precision.
This is the fastest correctness check you have; make it pass before anything else. If your mode
precomputes self-overlaps for speed, ensure the precomputed value is byte-identical to what the
live path would produce, or the self-check silently drifts.

## Flag polarity: name a boolean for what `True` means
If a flag reads `directionless`, then `directionless=True` must mean isotropic. Do not ship a
flag whose `True` value means the opposite of its name, and do not introduce two flags with
opposite polarity for the same idea (one at extraction time, one at scoring time) — that is a
guaranteed footgun. Pick one name and one polarity and use it everywhere the concept appears.

## Keep the `_np` (and `_jax`) mirror in sync — or skip it deliberately
The `score/*_np.py` mirrors exist so the objective can be checked without Torch. If you add a
Torch overlap, either mirror it in `_np` or leave the mirror untouched and note that the mode is
Torch-only. What you must not do is edit the mirror halfway, so the two diverge silently.

## Determinism: fix the seed set
The multi-start optimizer must produce the same result across runs given the same inputs. Derive
the SO(3) seeds deterministically (a fixed Fibonacci/structured set, not `torch.rand` without a
seed). The accel skill will compare its kernels against your optimizer numerically; a
non-deterministic reference makes that comparison meaningless.

## Gaussian width conventions
Overlap widths (`alpha`) follow the library's existing conventions — heavy-atom shape uses
`alpha=0.81`. If your channel needs a different width, make it an explicit argument with a
documented default rather than a magic number buried in the objective. Mismatched widths between
the numerator and the self-overlap terms is the usual cause of a self-overlap that is close to
but not exactly 1.000.

## Only `fit` moves
The reference molecule is fixed; only the fit molecule is transformed by the SE(3) pose. Keep the
gradient flowing through the transform applied to the fit inputs only. Accidentally transforming
both, or detaching the fit transform, produces a plausible-looking but wrong optimizer.
