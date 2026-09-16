# Scoring-math pitfalls

Recurring traps when writing a new reference mode. These are properties of the math and the
library's conventions, not war stories about one mode.

## Per-atom data must stay aligned with atom positions

Two distinct traps live here, and the second is the one that keeps recurring.

**(a) Atom order.** A per-atom scalar is a full array in RDKit-mol (with-H) order, exactly like
`partial_charges`. Build it in that order and slice heavy atoms with the **same
`_nonH_atoms_idx`** the charges use. An array built in a different order silently pairs each
scalar with the wrong atom: the overlap is wrong and nothing raises. With `ref == fit` a
*consistently* wrong mapping still gives self-overlap 1.000, so the self-check passes. Only a
**planted-pose** test catches it.

**(b) `self.atom_pos` is NOT the strict-heavy set.** A `Molecule` has three atom bases:

| Basis | Where | Length |
|---|---|---|
| with-H | `self.mol`, `partial_charges`, `mol.GetConformer().GetPositions()` | `N_full` |
| strict-heavy | `X[self._nonH_atoms_idx]` (atomic number ≠ 1) | the atoms carrying every per-atom field |
| RemoveHs | `self.atom_pos`, the shape channel's coordinates | usually heavy, but **retains isotope-labelled H** |

`Chem.RemoveHs` keeps deuterium, so `atom_pos` can be one or more rows longer than the strict-heavy
set. Pairing a heavy field with `atom_pos` broadcasts `(…, N)` against `(…, N−1)` and crashes — or
mispairs silently if the counts happen to divide.

**The library now solves this for you**: every per-atom channel ships a `get_<feature>_positions()`
accessor returning `mol.GetConformer().GetPositions()[self._nonH_atoms_idx]`. Use it, and add one
for your own channel. Do not open-code the indexing at the call site, and do not reach for
`atom_pos` because "it is already the heavy coordinates" — that is the trap.

There is **no `atom_pos_noH` attribute on `Molecule`.** That name exists only on `MoleculeProfile`
in `screen.py`, the RDKit-free screening stand-in. Older notes that tell you `vol_esp` keeps an
`atom_pos_noH` array on the molecule are wrong.

**Always add a retained-H test** with a SMILES whose deuterium survives RemoveHs, e.g.
`[2H]OC(=O)c1ccccc1`, and assert `atom_pos.shape[0] != len(_nonH_atoms_idx)` so the premise cannot
silently rot. A plain heavy-atom molecule exercises none of this.

## `lam` is a width, and its scaling convention splits by point type

`LAM_SCALING = COULOMB_SCALING**2 ≈ 207`, from `score/constants.py`. Two conventions coexist:

- **Surface-ESP modes** (`surf_esp`, `surf_esp_tversky`, `vol_and_surf_esp`, `score_with_esp`) take
  `lam` raw from the caller and multiply by `LAM_SCALING` internally.
- **Atom-centred modes** (`vol_esp`, `vol_esp_tversky`, `vol_lipo`, `vol_mr`, `vol_fukui`) take
  `lam` raw and do **not** scale it.

`get_overlap_esp`'s signature default is `0.3 * LAM_SCALING`, which is surface-tuned. Self-overlap
is `lam`-invariant, so the self-check passes at any value and will not warn you. For an
atom-centred field pass `lam=0.1` explicitly. Shipped defaults: `vol_esp` 0.1, `surf_esp` 0.3,
`vol_and_surf_esp` 0.001.

## `get_overlap_esp` is shape-weighted, not an independent field

It folds the shape Gaussian into the field overlap, giving a shape-weighted field similarity. That
is the intended ESP-style behaviour and what the whole `vol_lipo` family relies on. If your
description calls for an **independent** Gaussian field channel, this function cannot express it —
write a new one in `score/`.

It is also batch-capable and reshapes a `(N,)` scalar to `(N,1)` internally, so pass the array
straight in without reshaping.

## Tanimoto self-overlap must be exactly 1.000

A molecule aligned to a copy of itself must score 1.000. If it does not, the normalization is
inconsistent — usually the numerator and the two self-overlap terms computed with different widths,
masks, or precision. Make this pass before anything else. If your mode precomputes self-overlaps
for speed, ensure the precomputed value is byte-identical to the live path.

Two modes legitimately do not score 1.000 on a self-pair: `vol_and_surf_esp` (its ESP term is
evaluated on a stride and enters the score but not the gradient) and any Tversky mode with
`ta ≠ tb` on a non-identical pair. Know which case you are in before you call a 0.98 a bug.

## Tversky is a reduction, not a channel

`T = AB / (AB + ta·(AA − AB) + tb·(BB − AB))`. `AA` and `BB` are SE(3)-invariant, so only `AB`
moves and the optimizer sees the parent's landscape. That is why the eight shipped Tversky modes reuse
their parent's kernel and only change the host-side reduction. Two consequences for the reference:

- The objective is **asymmetric**; add a gate asserting `O(a,b) != O(b,a)` on a distinct pair.
- The score is **not clamped to [0, 1]** — a small dense query inside a larger molecule can
  legitimately exceed 1.0. Do not assert a ceiling.

## Finite-difference gradient checks: float32, and not at the identity pose

- **Use float32.** `get_SE3_transform` builds the rotation matrix in float32, so float64 inputs
  raise `Input dtypes must be the same, got: input float, batch1: double`. Run in the library-native
  float32 with `eps=1e-3` and `atol=2e-3`; float32 finite differences are noisy and a tight
  tolerance reports a false failure on a correct gradient.
- **Evaluate at a non-identity pose.** A self-overlap objective at the identity sits at its
  optimum, where the gradient is ≈0 in every direction. Comparing autograd-≈0 to
  finite-difference-≈0 passes `allclose` trivially and never exercises the derivative. The shipped
  tests all plant `se3 = [0.966, 0.259, 0, 0, 0.3, -0.2, 0.1]` first.

Both are harness gotchas, not mode bugs. Do not "fix" them by changing the objective.

## Zeroing a direction vector is not "directionless"

A pharmacophore feature with an orientation vector contributes both a positional Gaussian and a
cosine term. Setting the vector to zero does not make the feature isotropic: the cosine similarity
of two zero vectors is not 0 in the weighting used, so a zero-vector feature still perturbs the
score. A genuinely directionless feature must **skip the cosine path entirely** and score on
position only.

Note also that `directionless=True` changes the pharmacophore **count**, not just the vectors: the
donor/acceptor multi-vector branches are skipped, so a donor that expanded into one anchor per
hydrogen now emits a single anchor at the feature position.

## Flag polarity: name a boolean for what `True` means

`directionless=True` means isotropic. The library previously carried a fork-only `directional=`
flag with the opposite polarity, and the two were a standing footgun until it was renamed with
inverted meaning and no alias. Do not reintroduce that shape: one name, one polarity, everywhere
the concept appears.

## Determinism: fix the seed set

The multi-start optimizer must produce the same result across runs given the same inputs. Derive
the SO(3) seeds deterministically — the library uses a fixed Fibonacci set plus structured
rotations, not unseeded `torch.rand`. A non-deterministic reference makes the accel skill's parity
comparison meaningless.

Be aware that the reference seeder and the accelerated seeder are **different sets by design**: the
accelerated path adds up to six structured ±90° rotations about each reference principal axis
before falling back to the Fibonacci fill. Scores are therefore not comparable across backends,
and the accel skill compares against your optimizer at matched settings rather than expecting
equality with a JAX run.

## Gaussian width conventions

Heavy-atom shape uses `alpha=0.81`. If your channel needs a different width, make it an explicit
argument with a documented default rather than a magic number in the objective. Mismatched widths
between the numerator and the self-overlap terms are the usual cause of a self-overlap that is
close to but not exactly 1.000.

## Only `fit` moves

The reference molecule is fixed; only the fit molecule is transformed. Keep the gradient flowing
through the transform applied to the fit inputs only. Transforming both, or detaching the fit
transform, produces a plausible-looking but wrong optimizer.

## Charges are lazy, and the default model is xTB

`Molecule.partial_charges` is a property that computes on first read using gfn2-xTB, falling back
to MMFF94 with a `RuntimeWarning` if xTB is missing or fails. `get_partial_charges()` is the MMFF
path, not the general accessor.

Laziness is defeated by a surface: constructing a `Molecule` with `num_surf_points` builds its
surface ESP eagerly and forces charge generation. Only a pure-volumetric molecule skips the xTB
subprocess. If your tests would otherwise need xTB on PATH, inject a deterministic synthetic field
through the constructor instead — that is what `tests/test_vol_fukui.py` does, and it keeps the
overlap math under test without a binary dependency.
