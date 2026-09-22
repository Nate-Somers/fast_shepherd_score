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

You are given a description, in words, of how two molecules should be scored against each other
and aligned. Turn it into a **correct, readable Python alignment mode** that plugs into
`shepherd_score` — the *reference* implementation. Correctness first; speed is a separate skill
(`accelerate-scoring-mode`).

## Before you write anything: check whether the mode already exists

The library ships **21 canonical modes**. Read `shepherd_score/accel/_modes.py` first — `MODE_ATTRS`
is the complete list, in public order:

```
vol  vol_esp  surf  surf_esp  vol_and_surf_esp  pharm  vol_color  vol_tversky  vol_lipo
vol_esp_tversky  vol_mr  surf_tversky  surf_esp_tversky  vol_lipo_tversky  vol_color_tversky
vol_atomtype  vol_pharm  pharm_tversky  vol_and_surf_esp_tversky  vol_fukui  vol_avoid
```

Eight of those are Tversky reductions of a parent, six reuse another mode's optimizer outright.
If the request restates one of these, say so and stop. `esp_field` was built and then removed
entirely — do not use it as a model; nothing by that name exists.

## The two-layer picture

Every mode has two layers:

1. **Reference layer** (this skill) — pure PyTorch math and an eager Adam optimizer over autograd.
   Slow, obviously correct, easy to read.
2. **Accel layer** (`accelerate-scoring-mode`) — a `ModeSpec` in the mode registry, over
   hand-written Triton (GPU) and numba (CPU) value+gradient kernels, reproducing the reference at
   screening throughput. One generic engine runs every mode, so that layer is mostly data.

The single most important thing to understand: **the eager optimizer you write here is the oracle
the accel skill validates against.** Your reference does not need to be fast, but it must be right
and deterministic.

## Progress checklist

```
Mode design progress:
- [ ] 0. Confirm the mode is not already one of the 21
- [ ] 1. Pin the objective in math (channels, similarity, symmetry, what moves, inputs)
- [ ] 2. Decide the reuse level: existing optimizer / existing channel / new channel
- [ ] 3. Add new per-atom Molecule data — ONLY if a channel needs it (the accessor QUARTET)
- [ ] 4. Write the pure overlap in score/ — only if no existing channel fits
- [ ] 5. Write objective_<mode>_overlay + optimize_<mode>_overlay — only if no optimizer fits
- [ ] 6. Add MoleculePair.align_with_<mode>
- [ ] 7. Register result slots in _ALIGN_KEYS — NOT accel/_modes.py
- [ ] 8. Export from alignment/__init__.py (NOT score/__init__.py — it is empty)
- [ ] 9. Validate: self-overlap 1.000, grad vs finite-diff, planted pose, determinism, retained-H
```

## The contract you must deliver

- `MoleculePair.align_with_<mode>(...)` runs end to end on a pair of real molecules and writes
  `self.transform_<mode>` / `self.sim_aligned_<mode>`.
- A self-comparison scores **1.000** for a Tanimoto objective, at a generous search budget.
  Pass an explicit `num_repeats` / `max_num_steps` (20 x 200 is what the shipped tests use) —
  this gate is a statement about the objective, not about the per-mode defaults. At its own
  registry defaults even `vol` self-scores 0.9996, which would fail an `atol=1e-4` assertion.
- The optimizer entry point is self-contained and deterministic, returns
  `(aligned_fit_points, se3_transform, score)`, and uses only autograd — no custom kernels.
- The mode id is in `_ALIGN_KEYS` in `container/_core.py` (**not** `accel/_modes.py`).
- A test file exists and passes, modeled on `template_test.py`.

## Steps

### 1. Pin the objective in math before touching code

Write down `O(ref, T·fit)` — the scalar overlap after the fit molecule is moved by an SE(3)
transform `T`. Decide explicitly:

- **Channels**: shape (atom-centred Gaussians), electrostatics, pharmacophore ("color"), a
  per-atom scalar field, or a weighted blend. Reuse wherever the description allows.
- **Similarity**: Tanimoto (self-overlap 1.000) or Tversky (asymmetric,
  `AB / (AB + ta·(AA−AB) + tb·(BB−AB))`). Tanimoto is the default.
- **Symmetry**: is `O(a,b) == O(b,a)`? A Tversky mode is not, and its test needs an asymmetry gate.
- **What moves**: only `fit` is transformed; `ref` is fixed. The variable is the SE(3) pose, a unit
  quaternion plus a translation.
- **Inputs**: what per-atom or per-point data each channel consumes. A mode may also take a
  **third, non-molecule input** — `vol_avoid` takes a fixed `avoid_points` cloud off
  `MoleculePair.avoid_points`. Such a mode is pairwise-only by construction: the per-molecule
  screening store has nowhere to put a global constant.

### 2. Decide the reuse level — most modes write no new math

Work down this list and stop at the first that fits. Reuse is the norm, not a shortcut.

**(a) Reuse an existing optimizer outright.** Six of the 21 modes add *no* `objective_*` /
`optimize_*` function at all — they call another mode's optimizer with different inputs:

| Mode | Calls | Difference |
|---|---|---|
| `vol_mr` | `optimize_vol_lipo_overlay` | molar refractivity fed as the field |
| `vol_fukui` | `optimize_vol_lipo_overlay` | Fukui dual descriptor fed as the field |
| `vol_pharm` | `optimize_vol_color_overlay` | `directionless=False` |
| `surf_tversky` | `optimize_vol_tversky_overlay` | surface points instead of atom centres |
| `surf_esp_tversky` | `optimize_vol_esp_tversky_overlay` | surface points and surface ESP |
| `pharm_tversky` | `optimize_pharm_overlay` | `similarity='tversky'` |

If your objective is an existing one with a different field, different point set, or a different
similarity flag, this is your path. Skip steps 4 and 5 entirely.

**(b) Reuse an existing channel, new blend.** Write only the objective and optimizer, built from
`get_overlap` (shape) and `get_overlap_esp` (a signed scalar field). No `score/` addition.

**(c) Reuse an existing channel, new *reduction*.** A different way of combining the same overlaps —
Tversky, Dice, anything that is not Tanimoto. Write only the objective and optimizer. **Build it
from `VAB_2nd_order`, not from `get_overlap`.**

**(d) Genuinely new channel math.** Write the pure overlap in `score/` too.

> **`get_overlap` and `get_overlap_esp` return the Tanimoto, not the raw overlap.** They call
> `shape_tanimoto` internally, so `AB`, `AA` and `BB` are already consumed and cannot be recovered
> from the result. Any mode that changes the reduction must go one level lower and call
> `VAB_2nd_order(centers_1, centers_2, alpha)` three times — `AB`, `AA`, `BB` — then combine them
> itself. That is exactly what `objective_vol_tversky_overlay` does; read it before writing a
> reduction. Reaching for `get_overlap` here produces a reduction *of a Tanimoto*, which is not
> what anyone asked for and still scores 1.000 on a self-pair, so the mode's own sanity check will
> not catch it.

> **`get_overlap_esp` folds the shape Gaussian into its field overlap**, so what you get is a
> *shape-weighted* field similarity, not an independent field. That is the intended ESP-style
> behaviour and is what `vol_lipo` / `vol_mr` / `vol_fukui` all rely on. If you need an
> **independent** field channel, `get_overlap_esp` is the wrong tool — write a new one.

> **`lam` is a width, and the scaling convention differs by point type.** `get_overlap_esp`'s
> signature default is `0.3 * LAM_SCALING` (`LAM_SCALING ≈ 207`, from `score/constants.py`), which
> is tuned for **surface** clouds. Surface modes take `lam` raw from the user and multiply by
> `LAM_SCALING` internally; **atom-centred modes take `lam` raw and do not scale it**. For an
> atom-centred field pass `lam=0.1`, matching `vol_lipo`, `vol_mr`, `vol_fukui` and `vol_esp`.

### 3. Add new per-atom `Molecule` data — the accessor quartet

Most modes skip this. Only if a channel needs a per-atom property `Molecule` does not carry, add it
in `container/_core.py` following the **four-part pattern** the existing per-atom channels use.
Read `get_lipophilicity_contribs` / `get_lipophilicity` / `get_lipo_positions` together before
starting; they are the reference implementation of this pattern.

1. **Storage** — a full `(N,)` array in RDKit-mol (with-H) order. Eager for a cheap descriptor
   (`self.lipophilicity` at `__init__`); a **lazy property** backed by `self._<feature>` for
   anything expensive (`partial_charges` and `fukui` both shell out to xTB and are computed on
   first read).
2. **Compute** — `get_<feature>_contribs()` derives the full array from the RDKit mol.
3. **Slice** — `get_<feature>(no_H=True)` returns `self.<feature>[self._nonH_atoms_idx]`.
4. **Centres** — `get_<feature>_positions()` returns
   `self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]`, the strict-heavy coordinates the
   field is 1:1 with.

**Step 4 is not optional and is the whole point.** `self.atom_pos` is the `Chem.RemoveHs`
coordinate set, which *retains isotope-labelled hydrogen*, so it is longer than the strict-heavy
set on a deuterated molecule. Pairing a heavy field with `atom_pos` broadcasts `(N)` against
`(N−1)` and crashes, or silently mispairs. Providing `get_<feature>_positions()` also lets the
RDKit-free `MoleculeProfile` in `screen.py` duck-type into the same aligner by exposing the same
method name — the accel skill depends on that. There is **no `atom_pos_noH` attribute on
`Molecule`**; that name exists only on `MoleculeProfile`.

Call the slicer at align time, never in `__init__` — `_nonH_atoms_idx` is set late in the
constructor.

### 4. Write the pure overlap in `score/`

Only for genuinely new channel math. Put it in the matching module —
`score/gaussian_overlap.py`, `score/electrostatic_scoring.py`, `score/pharmacophore_scoring.py` —
or add `score/<family>_scoring.py`. The function is pure: positions and per-atom data in, scalar
out, no optimization. Verify by hand that a molecule against itself gives 1.000 under Tanimoto.

Mirrors are a judgement call with precedent both ways: `gaussian_overlap`,
`electrostatic_scoring` and `pharmacophore_scoring` each keep `_np` and `_jax` twins, while
`atomtype_scoring.py` is deliberately torch-only. Either mirror fully or not at all, and say which
you chose. Do not edit a mirror halfway.

### 5. Write the eager objective and optimizer in `alignment/_torch.py`

- `objective_<mode>_overlay(se3_params, ...)` — applies the SE(3) transform to the fit inputs and
  returns the negative overlap for a single pose. Autograd differentiates this.
- `optimize_<mode>_overlay(...)` — generates `num_repeats` SO(3) seeds, runs Adam on each, returns
  the best `(aligned_points, se3_transform, score)`.

**Name after the canonical mode id**, not the physics. Three functions carry legacy physics names
for historical reasons — `optimize_ROCS_overlay` is `vol`, `optimize_ROCS_esp_overlay` is the
`vol_esp`/`surf_esp` pair, `optimize_esp_combo_score_overlay` is `vol_and_surf_esp`. Everything
added since uses the id, and the accel skill expects the id.

Keep the eager defaults `num_repeats=50, lr=0.1, max_num_steps=200`, matching every other
optimizer in the file. The per-mode registry values are applied by the caller, not here.

### 6. Expose the per-pair API in `container/_core.py`

Add `MoleculePair.align_with_<mode>(...)`, mirroring an existing method:

- Pull inputs off `self.ref_molec` / `self.fit_molec`, converting with `self._to_tensor(...)`.
  Read shape centres through `get_positions(no_H)`, a per-atom field through its
  `get_<feature>(no_H)` slicer, and its centres through `get_<feature>_positions()`.
  **Do not reach for the cached `_ref_xyz_t` / `_fit_xyz_t` tensors if your mode takes a `no_H`
  flag.** Those are built once in `MoleculePair.__init__` from `atom_pos` alone, so a mode that
  reads them ignores `no_H=False` silently. `align_with_vol` and `align_with_vol_tversky` both
  go through `get_positions(no_H)` for exactly this reason.
- Call your optimizer, write `self.transform_<mode>` and `self.sim_aligned_<mode>`, return the
  aligned fit coordinates as a NumPy array.
- Validate what the mode requires, raising a clear `ValueError` in the tone of the surrounding
  methods.

**Seed and step defaults.** `_default_seeds(mode)` / `_default_steps(mode)` read `MODE_SEEDS` /
`MODE_STEPS` from the canonical registry, which a reference-only mode is deliberately not in yet
(step 7). So a reference-only mode takes literal defaults, and the accel skill switches it over
when it promotes the mode.

> Eleven shipped modes were promoted to the registry but never had their eager defaults switched
> over, so their per-pair and batched defaults diverge: eager `vol_tversky` runs 50×200 where
> `MoleculePairBatch` runs 10×40. Do not add a twelfth. Once the accel skill adds the registry
> rows, switch to `_default_seeds` / `_default_steps` so both paths read one source.

### 7. Register the result slots in `_ALIGN_KEYS`

Add your mode id to the `_ALIGN_KEYS` tuple in `container/_core.py`. A class-body loop turns each
key into `transform_<mode>` / `sim_aligned_<mode>` properties backed by an `AlignmentResult`
dataclass in `MoleculePair._alignments`, so this one line is the whole registration. The tuple has
23 entries for 21 modes because `vol` and `vol_esp` each keep a legacy `_noH` twin.

**Do not add the mode to `accel/_modes.py`.** That registry is for *canonical* (screening) modes,
and a `ModeSpec` there is the accel skill's deliverable, not yours. Adding one here would be
premature in the strong sense: the spec names the kernels its terms run on and the channels its
data comes from, and until those exist the spec describes a mode that cannot execute. It would
also immediately generate `_align_batch_<mode>`, promise a screen path, and trip
`tests/test_mode_registry.py`, which pins `len(CANONICAL_MODES) == 21` on purpose so that
promoting a mode is a visible act. Leave `accel/channels.py`, `_MODE_SPEC` and `screen.py` alone
too; they are accel-skill territory.

### 8. Export the public functions

Add your `objective_<mode>_overlay` / `optimize_<mode>_overlay` to `alignment/__init__.py`'s
`__all__`, following the existing style.

**`score/__init__.py` is an empty file.** Nothing is exported from it and nothing should be —
score functions are imported by full path (`from shepherd_score.score.gaussian_overlap import
get_overlap`). Do not add exports there.

### 9. Validate

Copy `template_test.py` to `tests/test_<mode>.py`, replace the `YOURMODE` token, and make it pass.

| Gate | What it proves |
|---|---|
| 1. Self-overlap = 1.000 (explicit budget) | normalization is consistent |
| 2. Autograd vs central finite difference | the objective is analytically correct |
| 3. Planted-pose recovery | the multi-start optimizer works, and atom order is right |
| 4. Determinism | the reference can serve as a parity oracle |
| 5. Retained-H molecule | only if the mode reads a per-atom field |
| 6. Asymmetry | only if the mode is Tversky |

Gates 2 and 3 are the ones that catch real bugs; gate 1 passes under a consistently-wrong atom
mapping. Run `tests/test_mode_registry.py` as well — it must pass **unchanged**, which is what
confirms you kept the mode out of the canonical registry.

Register any new pytest marker in `pytest.ini`; `--strict-markers` is on. There is no `conftest.py`,
so a test needing CUDA carries both `@pytest.mark.cuda` and its own `skipif` guard. Keep the code
Python 3.9-compatible and `ruff check shepherd_score/ tests/` clean — CI runs both.

Some suite tests need Open3D or a GPU and may error for unrelated reasons. Run your own file and
`test_mode_registry.py` specifically rather than the whole suite.

## Handoff to `accelerate-scoring-mode`

That skill turns your mode into a `ModeSpec`: a list of TERMS (one kernel launch each, with a
reduction and a blend weight) over named CHANNELS (per-molecule arrays). Everything it needs from
you maps onto that, so state it in those words:

1. **The optimizer entry point** you wrote, or which existing optimizer you reused.
2. **The terms**: for each channel pair the objective scores, which kernel family computes it
   (shape / signed scalar field / pharmacophore / element identity / surface-ESP agreement /
   hard-sphere penalty), how it reduces (Tanimoto, Tversky, raw, agreement) and its blend weight.
   A blend is two terms; a Tversky variant is a reduction; a penalty is a negative weight.
3. **The channels**: which per-molecule arrays each term reads, and whether any is data the
   library does not already carry — i.e. whether you added an accessor quartet in step 3. That is
   the one thing that still costs the accel skill real wiring, because the screening store has to
   learn to persist and rotate it.
4. **Anything a third input**: a non-molecule input like `vol_avoid`'s `avoid_points`. It is a
   pair-level channel now, not a reason the mode cannot screen.
5. **Your test file's path**, which is the oracle.

If your mode reuses existing kernels and existing per-molecule data — the common case — the accel
work is one spec and one API method, and it inherits the batched aligner, the CUDA-graph and fused
CPU fine loops, the array-native screen, the multi-GPU path and constant seeds automatically.

## Constraints

- **Additive and minimal.** Add functions; do not rewrite shared code.
- **Match the surrounding code** — naming, docstring format, argument order, error style.
- **Do not break existing modes.** Reuse channels rather than forking them.
- **One clear name per concept.** No back-compat aliases for a name only this mode uses.

See `seams.md` for the file map and the reuse tables, and `pitfalls.md` for the recurring traps.
