---
name: design-scoring-mode
description: >-
  Implement a new molecular similarity objective and a tested PyTorch reference
  alignment method in shepherd_score from a mathematical or plain-language description.
  Use for adding scoring modes; use accelerate-scoring-mode for their Numba/Triton backend.
---

# Design a scoring mode

Work in the requested FSS checkout. Read the implementation before choosing an
extension point; the instructions below describe the current architecture, not a
substitute for inspecting it. Paths are relative to the repository root, with
implementation paths below `shepherd_score/` unless stated otherwise.

## Establish the objective

Read `accel/_modes.py::SPECS` and the nearest `MoleculePair.align_with_*` method
in `container/_core.py`. The registry currently has 21 modes, including eight
Tversky variants. If an existing mode already implements the request, explain
how to use it instead of adding a duplicate.

Write the score as a function of a fixed reference/query and a transformed fit
molecule. Specify inputs, units, normalization, weights, and which terms steer
optimization. Combined volume-and-surface ESP currently differentiates both
terms; do not copy the old shape-only optimization with ESP reranking.
State whether the objective is symmetric and what its self-score should be.
Penalties and sampled electrostatic agreement do not necessarily have the
normalization properties of a Gaussian Tanimoto overlap.

A third input need not make a mode pairwise-only: `vol_avoid` accepts a fixed
avoid cloud, and screening carries it as a query-side keyword. Read its actual
public method signature and the screen `_with_avoid` helper.

## Implement the reference

Read [seams.md](seams.md) for extension points and [pitfalls.md](pitfalls.md)
before adding a channel or reduction.

1. Reuse an existing optimizer if only its inputs or reduction options differ.
   `vol_mr` and `vol_fukui` reuse `optimize_vol_lipo_overlay`; `vol_pharm`
   reuses `optimize_vol_color_overlay` with directional weighting.
2. If new mathematics is needed, implement the score in `score/` and a
   differentiable objective/optimizer in `alignment/_torch.py`. Build a new
   reduction from raw overlaps, not an already-normalized Tanimoto score.
   Only the fit moves. Transform positions and direction vectors appropriately.
3. Add any new molecular property in with-H RDKit atom order, with a matching
   strict-heavy accessor and position accessor. Avoid unnecessary xTB or
   surface generation in tests; inject fixed fields where appropriate.
4. Add `MoleculePair.align_with_<mode>` and its `_ALIGN_KEYS` result entry.
   Store the score and transform using the existing result properties. Export
   new optimizer functions through `alignment/__init__.py` as appropriate.
5. Keep a reference-only mode out of `accel/_modes.py` until the accelerated
   terms and channels exist. Use explicit reference defaults meanwhile.
   On promotion, compare both public method signatures with the registry:
   existing eager methods do not all share its seed/step defaults.

Do not change existing defaults or unrelated mode behavior as part of adding
a reference. JAX/NumPy mirrors are separate implementations; either support
and test the requested backend or document that the new reference is PyTorch-only.

## Validate

Adapt [template_test.py](template_test.py); its placeholders deliberately fail
until implemented. Tests should establish:

- the expected fixed-pose self-score, separately from optimizer convergence;
- autograd versus finite differences at nontrivial poses and on distinct inputs;
- recovery of a planted rigid transform and direct rescoring of the returned pose;
- repeatability with frozen inputs, preparation, seeds, and settings;
- correct field/position correspondence on retained-isotope-H molecules;
- reference/fit asymmetry and clamping behavior if using Tversky.

Use the nearest existing tests to choose and justify numerical tolerances.
A passing self-score alone cannot establish correct features or gradients.
The reference is an oracle only after validation; investigate a discrepancy
in either implementation rather than assuming the reference cannot be wrong.

Run the mode's tests and `tests/test_mode_registry.py`. GPU tests need an
explicit CUDA availability check; a marker alone does not skip them. Report
unrun backend checks accurately. Preserve the user's requested scope and
authorization for experiments and external actions.

## Handoff

Provide the accelerated implementation with the objective, reference entry
point, terms and reductions, feature arrays and their coordinate bases,
required third inputs, defaults, tests, and numerical tolerances. Use
`accelerate-scoring-mode` to register the canonical batched/screening mode.
