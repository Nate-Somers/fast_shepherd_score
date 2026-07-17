---
name: design-scoring-mode
description: >-
  Turn a plain-English description of a molecular similarity objective into a correct,
  autograd-validated Python alignment mode in shepherd_score. Use when someone wants a new
  scoring / overlap function (a new way to compare two molecules and optimize their relative
  pose) and has described it in words rather than code. Produces the reference implementation
  only; hand the result to `accelerate-scoring-mode` to add the fast backend.
---

# Design a scoring mode

You are given a description, in words, of how two molecules should be scored against each
other and aligned. Your job is to turn it into a **correct, readable Python alignment mode**
that plugs into `shepherd_score` — the *reference* implementation. Correctness first; speed is
a separate skill (`accelerate-scoring-mode`).

## The two-layer picture

`shepherd_score` has two layers for every mode:

1. **Reference layer** (this skill) — pure PyTorch/NumPy math and an eager autograd optimizer.
   Slow, obviously-correct, easy to read. This is the ground truth.
2. **Accel layer** (`accelerate-scoring-mode`) — hand-written Triton (GPU) + numba (CPU) kernels
   that reproduce the reference at screening throughput.

The single most important thing to understand: **the eager optimizer you write here is the
parity oracle the accel skill validates against.** Everything the fast kernels do later must
reproduce your `optimize_<mode>_overlay` to floating-point tolerance. So your reference does
not need to be fast, but it must be *right* and *deterministic*.

## The contract you must deliver

By the end of this skill the mode must satisfy all of:

- `MoleculePair.align_with_<mode>(...)` runs end-to-end on a pair of real molecules and writes
  `self.transform_<mode>` / `self.sim_aligned_<mode>`.
- A self-comparison (a molecule against a copy of itself) scores **exactly 1.000** for a
  Tanimoto objective (the mode's built-in sanity check).
- `optimize_<mode>_overlay(...)` is a self-contained eager function: given reference/fit inputs
  and a seed count it returns `(aligned_fit_points, se3_transform, score)`, is deterministic
  given the seeds, and uses only autograd — **no custom kernels**. This is the oracle.
- The mode's result slots are registered via `_ALIGN_KEYS` in `container/_core.py` (**not**
  `accel/_modes.py` — see step 6) and its public functions are exported.
- A test file exists and passes (see `template_test.py`).

## Steps

### 1. Pin the objective in math before touching code
Write down `O(ref, T·fit)` — the scalar overlap after the fit molecule is moved by an SE(3)
transform `T`. Decide explicitly:
- **Channels**: shape (atom-centred Gaussians), electrostatics (ESP), pharmacophore ("color"),
  or a weighted combination. Reuse an existing channel wherever the description allows.
- **Similarity**: Tanimoto (self-overlap = 1) vs Tversky (asymmetric). Tanimoto is the default.
- **Symmetry**: is `O(a,b) == O(b,a)`? If not, say so — it changes the tests.
- **What moves**: only `fit` is transformed; `ref` is fixed. The optimization variable is the
  SE(3) pose, parameterized as a unit quaternion + translation.
- **Inputs**: what per-atom / per-point data does each channel consume? If it is something the
  `Molecule` does not already carry (positions, charges, surface, pharmacophores), you add it in
  step 2.

### 2. Add any new per-atom `Molecule` data (only if your mode needs it)
Most modes reuse data the `Molecule` already carries — atom positions, partial charges, surface
points/ESP, pharmacophores. But if a channel needs a **per-atom property the `Molecule` does not
compute yet**, add it to `Molecule` in `container/_core.py` **before** wiring the channel,
following the established `partial_charges` "trio":
- an **attribute** set in `Molecule.__init__` — a full `(N,)` array over *all* atoms in RDKit mol
  order (e.g. `self.<feature> = self.get_<feature>_contribs()`, right after the `partial_charges`
  block);
- a **compute method** that derives it from the `Molecule`'s RDKit mol and returns a float32 array
  (mirror `get_partial_charges`);
- a **heavy-atom slicer** `get_<feature>(no_H=True)` returning `self.<feature>[self._nonH_atoms_idx]`
  (mirror `get_charges`).

**The invariant that matters:** the per-atom array must be in the same atom order as `atom_pos`,
and the heavy-atom slice must reuse the **same `_nonH_atoms_idx`** that positions and charges use.
If it does not, the scalar desyncs from the geometry and the overlap is silently wrong — and a
self-overlap (`ref == fit`) still reads 1.000 under a *consistently* wrong mapping, so it will not
catch the bug; only a planted-pose test (step 8) will. Read `get_partial_charges` / `get_charges`
and copy their structure exactly.

Compute the attribute at construction (it only needs the RDKit mol), but call the
`get_<feature>(no_H)` **slicer only at align time**: `_nonH_atoms_idx` is defined *later* in
`Molecule.__init__`, so slicing during `__init__` would raise `AttributeError` (this is why
`get_charges` is likewise only ever called at align time, not in the constructor). (Most modes skip
this step entirely.)

### 3. Write the pure overlap in `score/`
Put the channel math in the matching module — `score/gaussian_overlap.py` (shape),
`score/electrostatic_scoring.py` (ESP), `score/pharmacophore_scoring.py` (color) — or add a new
`score/<family>_scoring.py` if it is genuinely a new family. Provide the Torch version and keep
the `_np` mirror in sync (add a `_jax` mirror only if you want the jax path). The function is
pure: positions/charges/types in, scalar overlap out, no optimization. Verify by hand that a
molecule scored against itself gives 1.000 under Tanimoto.

**If the mode fully reuses existing channels there is nothing to add here and nothing to export** —
skip straight to the objective. (E.g. a shape + scalar-field combo can reuse `get_overlap` and
`get_overlap_esp`, feeding a new per-atom scalar as `get_overlap_esp`'s "charges" argument — pass
the `(N,)` array straight in, `get_overlap_esp` reshapes it to `(N,1)` internally and is
batch-capable, so no manual reshaping. Note `get_overlap_esp` folds the shape Gaussian into its
field overlap, so you get a *shape-weighted* field similarity — the intended ESP-style behaviour,
not an independent scalar field. Its `lam` sets the field's influence; self-overlap is
`lam`-invariant so the self-check passes for any value, but choose `lam` for your point type rather
than the function's signature default: that default (`0.3*LAM_SCALING`) is tuned for **surface**
point clouds, while the docstring recommends **`lam=0.1` for volumetric / atom-centred** overlap —
pass `lam=0.1` for an atom-centred field.)

### 4. Write the eager objective + optimizer in `alignment/_torch.py`
Two functions, following the shape of the existing `optimize_*_overlay` functions:
- `objective_<mode>_overlay(se3_params, ...)` — applies the SE(3) transform to the fit inputs
  and returns the (negative) overlap for a single pose. Autograd differentiates this.
- `optimize_<mode>_overlay(ref_..., fit_..., num_repeats, lr, max_num_steps, ...)` — generates
  `num_repeats` SO(3) seeds, runs Adam on each, and returns the best `(aligned_points,
  se3_transform, score)`.

**Naming**: name both functions after the *canonical mode id* (e.g. `optimize_vol_color_overlay`),
not after the physics. The oldest modes use physics names (`optimize_ROCS_overlay` is the `vol`
mode) for historical reasons — do not follow that; the id-based name is the modern convention
and is what the accel skill expects to find.

### 5. Expose the per-pair API in `container/_core.py`
Add `MoleculePair.align_with_<mode>(...)`. Mirror an existing `align_with_*` method:
- Give `num_repeats` and `max_num_steps` sensible **literal** defaults (the optimizer's own
  defaults). Do **not** resolve them via `_default_seeds` / `_default_steps` here — those read
  `MODE_SEEDS` / `MODE_STEPS`, the *canonical* registry a reference-only mode is deliberately not
  in yet (step 6). The accel skill moves the defaults there when it promotes the mode.
- Pull inputs off `self.ref_molec` / `self.fit_molec` (use the existing cached `_ref_xyz_t` /
  `_fit_xyz_t` tensors where they fit; read any new per-atom feature via its `get_<feature>(no_H)`
  slicer from step 2).
- Call your `optimize_<mode>_overlay`, write `self.transform_<mode>` and
  `self.sim_aligned_<mode>`, and return the aligned fit coordinates as a NumPy array.
- Validate inputs the mode requires (e.g. raise a clear `ValueError` if pharmacophores are
  missing), matching the tone of the surrounding methods.

### 6. Register the mode's result slots in `container/_core.py`
Add your mode id to the `_ALIGN_KEYS` tuple in `container/_core.py`. That is what generates the
`transform_<mode>` / `sim_aligned_<mode>` accessors (backed by an `AlignmentResult`), giving your
`align_with_<mode>` method somewhere to write. This edit is purely additive and safe.

**Do not add the mode to `accel/_modes.py` (`MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`).** That
registry is for *canonical* (screening) modes, and adding to it here **breaks the build**:
`MODE_ATTRS` feeds `CANONICAL_MODES`, which the `@_bind_batch_aligners` decorator on `MoleculePair`
walks *at import time*, calling `getattr(accel.batch, "_align_batch_<mode>")` — an aligner that does
not exist until the accel skill builds it, so `import shepherd_score.container` raises
`AttributeError`. `tests/test_mode_registry.py` also pins `len(CANONICAL_MODES) == 7`,
`set(MODE_SEEDS) == set(CANONICAL_MODES)`, and a batch-bind for every canonical mode — all of which
fail the instant you add a mode with no aligner. Promoting your mode to canonical is the accel
skill's job, done once the aligner exists. Leave `accel/_modes.py`, `PROCESS_MODES`, and
`_MODE_SPEC` untouched; `screen.py`, `accel/multi_gpu.py`, and `accel/cpu_pool.py` derive from that
registry and so pick the mode up only after promotion — correct, since a reference-only mode has no
batched screening path yet.

### 7. Export the public functions
Add your `optimize_<mode>_overlay` / scoring functions to the relevant package `__init__.py`
(`score/`, `alignment/`, `container/`) following the existing export style. (If the mode reused
existing channels and added no new `score/` function, there is nothing to export from `score/`.)

### 8. Validate
Copy `template_test.py` to `tests/test_<mode>.py`, replace the `YOURMODE` token, and make it
pass. The required checks: self-overlap = 1.000, autograd gradient (evaluated at a **non-identity**
pose — see `pitfalls.md`) agrees with a finite-difference gradient, the optimizer recovers a planted
rotation on a self-pair, and results are deterministic given a fixed seed. Run
`tests/test_mode_registry.py` too — it must still pass **unchanged**, which confirms you kept the
mode out of the canonical registry (step 6). Some suite tests need open3d or a GPU and may error for
reasons unrelated to your change; run your own test and `test_mode_registry.py` specifically rather
than the whole suite.

## Handoff to `accelerate-scoring-mode`
State three things for the next skill: the exact name of your `optimize_<mode>_overlay` oracle,
the path of your test file, and the gradient structure of the objective (which channels
contribute, and how the SE(3) gradient decomposes). That is all it needs. It will promote the mode
to canonical (`accel/_modes.py`) and move the seed/step defaults into `MODE_SEEDS` / `MODE_STEPS`
as part of building the batched path. If you added a new per-atom `Molecule` feature (step 2), flag
that too — the batched path must pad that new per-atom scalar into its input tensors.

## Constraints
- **Additive and minimal.** Add functions; do not rewrite shared code. Keep the diff small.
- **Match the surrounding code** — naming, docstring format, argument order, error style.
- **Do not break existing modes.** Reuse channels rather than forking them; run the full test
  suite before declaring done.
- **One clear name per concept.** Do not add back-compat aliases for a name only this mode uses.

See `seams.md` for the exact file map and name-map, and `pitfalls.md` for the recurring traps.
