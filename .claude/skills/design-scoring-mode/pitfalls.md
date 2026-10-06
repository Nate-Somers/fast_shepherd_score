# Scoring and preparation pitfalls

## Coordinate bases and transforms

`Molecule.atom_pos` uses `Chem.RemoveHs`, which can retain isotope-labelled
hydrogen. Atomic fields use strict-heavy indices (`atomic number != 1`). A
field must use its corresponding `get_<feature>_positions()` accessor. Include
a molecule such as `[2H]OC(=O)c1ccccc1` and verify the different counts explicitly.
Neither a self-score nor rigid recovery alone proves chemical assignment:
also verify field values against their intended atom indices.

Only the fit moves. Positions rotate and translate; vectors rotate but do not
translate. Re-score returned transforms, including canonical-store transforms,
instead of checking only the optimizer's reported scalar.

## Reductions

`get_overlap` and `get_overlap_esp` return normalized scores. To introduce a
new reduction, use their underlying raw cross- and self-overlaps. The ESP
overlap includes the positional Gaussian; it is not a spatially independent field.

For Gaussian Tversky, `A` is reference and `B` is fit:
`AB / (AB + ta*(AA-AB) + tb*(BB-AB))`. The default weights are 0.95/0.05.
These ratios can exceed one. Pharmacophore Tversky instead clamps at one and
has its own guarded derivative. The combined ESP Tversky mode changes only
the shape reduction, not the electrostatic agreement term.

Test a normalized overlap at identical fixed inputs for a unit self-score.
An optimized self-alignment can stop short of that value. Surface-potential
agreement and excluded-volume penalties need tests of their own definitions;
do not impose a universal unit-self-score gate on composite objectives.

## Charge and width conventions

`LAM_SCALING = COULOMB_SCALING**2`. Surface ESP modes scale the user's `lam`
internally; atom-centred ESP/descriptor modes use it without that scaling.
Read both the public method and `ModeSpec.lam_scaling` to avoid scaling twice.
Typical public defaults are 0.1 for atom ESP/descriptor modes, 0.3 for surface
ESP, and 0.001 for combined volume-and-surface ESP.

Charges are lazy, defaulting to xTB with an MMFF94 fallback on failure. The
default xTB charge call uses total charge zero; MMFF follows the supplied
molecule's formal charge. Fukui calculations use the formal charge and adjacent
charge states, and have no MMFF substitute. Select `charge_model="mmff"` or
inject fixed fields when testing scoring without an xTB executable.

Use `num_surf_points=None` (and no density or supplied surface) to omit surfaces
from `Molecule`; integer zero does not mean disabled. Preparing a surface also
computes its electrostatic potential and therefore accesses charges. Freeze
surface positions and features when comparing scoring implementations.

## Pharmacophores

Directional and directionless scoring can consume the same stored anchors and
types. Selecting `vol_color` does not regenerate features with different SMARTS.
Feature-generation options are separate: requesting directionless extraction
can change anchor multiplicity. Zero vectors are not a substitute for bypassing
the directional weighting. Direction weights clamp the dot product to [0, 1];
aromatic normals use its absolute value before clamping.

## Gradient and optimizer checks

Use nontrivial poses and distinct molecules. Test finite differences away from
hard exclusion-mask boundaries and other nondifferentiable points; also test
the boundary convention separately. `tests/test_esp_agreement_grad.py` is a
current example for the combined ESP gradient.

The SE(3) helpers used by the reference commonly work in float32. Check their
dtypes before choosing finite-difference precision. Existing float32 tests use
steps around 1e-3; choose tolerances based on the actual function and scale.

Reference and accelerated seed generators and Adam implementations differ.
Fixed-pose score/gradient agreement is a stronger isolation test than demanding
identical optimized scores. Freeze all inputs, seeds, feature arrays, backend,
and numerical settings for repeatability tests; report any remaining differences.
