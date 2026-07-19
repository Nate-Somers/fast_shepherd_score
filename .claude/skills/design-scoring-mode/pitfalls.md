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

## Finite-difference gradient checks: float32, and not at the identity pose
Two gotchas make a naive autograd-vs-finite-difference test either crash or pass vacuously.

- **Use float32.** The SE(3) helpers (`get_SE3_transform`) build the rotation matrix in float32, so
  feeding float64 inputs raises `Input dtypes must be the same, got: input float, batch1: double`
  inside the transform. Run the check in the library-native float32, with a step `eps ≈ 1e-3` and a
  loose tolerance (`atol ≈ 2e-3`): float32 finite differences are noisy, and a tight float64
  tolerance reports a false failure even when the gradient is correct.
- **Evaluate at a non-identity pose.** Do the check on a fit molecule that has been rotated /
  translated away from the reference, not at the identity SE(3) the template starts from. A
  self-overlap objective at the identity is at its *optimum*, where the gradient is ≈0 in every
  direction — comparing autograd-≈0 to finite-difference-≈0 passes `allclose` trivially and never
  exercises the derivative. Plant a real pose first so the gradient is genuinely nonzero.

Both are test-harness gotchas, not mode bugs — do not "fix" them by changing the objective.

## Per-atom data must stay aligned with atom positions — and mind the retained-H basis
Two distinct traps live here.

**(a) Atom order.** A per-atom scalar (charge, lipophilicity, any new field) is a full array in
RDKit-mol (with-H) order, exactly like `partial_charges`. Build it in that order and slice heavy
atoms with the **same `_nonH_atoms_idx`** the charges use. An array built in a different order
silently pairs each scalar with the wrong atom — the overlap is wrong though nothing errors. With
`ref == fit` a *consistently* wrong mapping still gives self-overlap 1.000, so the self-check
passes; only a **planted-pose** test (rotate the fit, confirm recovery to ≈1.000) catches it.
Include one.

**(b) `self.atom_pos` is NOT the heavy set `_nonH_atoms_idx` selects.** A `Molecule` has *three*
atom bases and they are not interchangeable:
- **with-H** — `self.mol`, `partial_charges`, `mol.GetConformer().GetPositions()`: length `N_full`.
- **true-heavy** — `X[self._nonH_atoms_idx]` (atomic number != 1): the atoms that carry
  `partial_charges[self._nonH_atoms_idx]`.
- **RemoveHs** — `self.atom_pos`, the shape channel's coordinates. Usually equals the true-heavy
  set, **but `Chem.RemoveHs` retains isotope-labelled H (deuterium)**, so `atom_pos` can be *one
  (or more) longer* than the true-heavy set.

To pair positions with the heavy charges, use `mol.GetConformer().GetPositions()[self._nonH_atoms_idx]`
(true-heavy) — **never** `self.atom_pos`. Reaching for `atom_pos` because "it is already the heavy
coordinates" is the trap: on a deuterated molecule `atom_pos` has `N` rows while
`partial_charges[self._nonH_atoms_idx]` has `N-1`, and any elementwise pairing (a Coulomb sum, an
overlap, a distance matrix) broadcasts `(…, N)` against `(…, N-1)` and crashes — or silently
mis-pairs if the counts happen to divide. This is exactly the retained-H case `vol_esp` handles by
keeping a separate heavy-centre array (`atom_pos_noH`); a new charge-like or field channel must do
the same. **Always add a retained-H test** with a SMILES whose deuterium survives RemoveHs, e.g.
`[2H]OC(=O)c1ccccc1`, and assert `atom_pos.shape[0] != len(_nonH_atoms_idx)` so the test premise
can't silently rot — a plain heavy-atom molecule exercises none of this.
