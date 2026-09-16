# Reference-layer seams

The files a new mode touches, bottom-up. Everything is additive: you add functions, one string to
`_ALIGN_KEYS`, and — only if a channel needs data the `Molecule` does not carry — a per-atom
accessor quartet. You do **not** touch `accel/_modes.py` or the front end (`screen.py`,
`accel/multi_gpu.py`, `accel/cpu_pool.py`); those are the canonical registry and its consumers,
which `accelerate-scoring-mode` wires once a batched path exists.

| Layer | File | What you add |
|---|---|---|
| Molecule data (if needed) | `container/_core.py` | The accessor quartet: storage (eager attribute or lazy property), `get_<f>_contribs()`, `get_<f>(no_H)`, `get_<f>_positions()`. Most modes skip this. |
| Channel math (if needed) | `score/<family>_scoring.py` (+ `_np`, `_jax`) | The pure overlap: inputs in, scalar out, no optimization. Skipped when an existing channel fits. |
| Eager optimizer (if needed) | `alignment/_torch.py` | `objective_<mode>_overlay` and `optimize_<mode>_overlay`. **The oracle.** Skipped when an existing optimizer fits. |
| Per-pair API | `container/_core.py` | `MoleculePair.align_with_<mode>(...)`; writes `transform_<mode>` / `sim_aligned_<mode>`. |
| Result slots | `container/_core.py` | One string in `_ALIGN_KEYS`. |
| Exports | `alignment/__init__.py` | Public function names in `__all__`. **`score/__init__.py` is empty — add nothing there.** |
| Tests | `tests/test_<mode>.py` | From `template_test.py`. `tests/test_mode_registry.py` must still pass unchanged. |

## What a real mode actually cost

Two recent additions, by file:

```
vol_fukui  (reuses vol_lipo's optimizer, adds a new per-atom field)
  container/_core.py        118 +    accel/...                (accel skill)
  conformer_generation.py    61 +    tests/test_vol_fukui.py  191 +

vol_avoid  (reuses optimize_ROCS_overlay, adds a third non-molecule input)
  container/_core.py         67 +    tests/test_vol_avoid_accel.py 145 +
```

Neither touched `alignment/_torch.py`. That is the normal outcome, not an unusual one.

## The reuse ladder

Stop at the first rung that fits.

**Rung 1 — an existing optimizer, different inputs.** Six shipped modes do this and add no
alignment code at all:

| Mode | Reuses | What differs |
|---|---|---|
| `vol_mr` | `optimize_vol_lipo_overlay` | molar refractivity as the field, `lipo_weight=mr_weight` |
| `vol_fukui` | `optimize_vol_lipo_overlay` | Fukui dual descriptor as the field |
| `vol_pharm` | `optimize_vol_color_overlay` | `directionless=False` (the directional twin of `vol_color`) |
| `surf_tversky` | `optimize_vol_tversky_overlay` | `surf_pos` instead of atom centres |
| `surf_esp_tversky` | `optimize_vol_esp_tversky_overlay` | `surf_pos` + `surf_esp`, `lam × LAM_SCALING` |
| `pharm_tversky` | `optimize_pharm_overlay` | `similarity='tversky'` |

**Rung 2 — existing channels, new blend.** Write the objective and optimizer from `get_overlap`
and `get_overlap_esp`; add nothing to `score/`. `vol_lipo` is the model.

**Rung 2b — existing channel, new reduction.** Tversky, Dice, anything that is not Tanimoto.
Build it from **`VAB_2nd_order`**, which returns the raw overlap; `get_overlap` has already
reduced to a Tanimoto and cannot be un-reduced. `objective_vol_tversky_overlay` is the model.

**Rung 3 — new channel math.** Add a `score/` function too. `vol_atomtype` is the only recent
example, and it brought `score/atomtype_scoring.py` with no `_np` or `_jax` mirror.

## Which existing mode to read

| Your target | Read |
|---|---|
| shape only | `align_with_vol` → `optimize_ROCS_overlay` |
| shape + per-atom scalar field | `align_with_vol_lipo` → `optimize_vol_lipo_overlay` |
| ESP on surfaces | `align_with_surf_esp` / `align_with_vol_and_surf_esp` |
| pharmacophore, or shape + color | `align_with_pharm` / `align_with_vol_color` |
| a Tversky reduction of any parent | `align_with_vol_tversky` → `optimize_vol_tversky_overlay` |
| a third non-molecule input | `align_with_vol_avoid` (reads `MoleculePair.avoid_points`) |

## Adding a per-atom `Molecule` feature

Four parts, all in `container/_core.py`. Read `get_lipophilicity_contribs` (`:574`),
`get_lipophilicity` (`:593`) and `get_lipo_positions` (`:616`) together first.

1. **Storage.** A full `(N,)` float32 array in RDKit-mol (with-H) order, the same order as
   `partial_charges`. Cheap descriptors are computed eagerly in `__init__`
   (`self.lipophilicity`, `self.molar_refractivity`). Anything expensive is a **lazy property**
   backed by `self._<feature>`, generated on first read — `partial_charges` and `fukui` both shell
   out to xTB and follow this pattern. `partial_charges` falls back to MMFF94 with a
   `RuntimeWarning` if xTB fails; `fukui` deliberately does not, and propagates.
2. **Compute.** `get_<feature>_contribs()` derives the full array from `self.mol`.
3. **Slice.** `get_<feature>(no_H=True)` returns `self.<feature>[self._nonH_atoms_idx]`.
4. **Centres.** `get_<feature>_positions()` returns
   `self.mol.GetConformer().GetPositions()[self._nonH_atoms_idx]`.

The four existing centre accessors — `get_lipo_positions`, `get_mr_positions`,
`get_fukui_positions`, `get_atomtype_positions` — have *identical bodies*. They are kept separate
on purpose: `MoleculeProfile` in `screen.py` exposes the same method names, so a profile and a
`Molecule` feed the same aligner without a branch. Adding a fifth identical accessor is correct,
not duplication to be refactored away.

**Invariants.** The array is full `(N_full,)` in with-H order; its heavy slice uses the **same
`_nonH_atoms_idx`** the charges use; the centres come from the with-H conformer, never from
`atom_pos`. Compute at construction (or lazily), but call the slicer only at align time —
`_nonH_atoms_idx` is set late in `__init__`.

## Two registries: result slots vs canonical modes

Telling these apart is the one non-obvious part of this skill.

- **`_ALIGN_KEYS`** (`container/_core.py`) — the result slots. A class-body loop turns each key
  into `transform_<key>` / `sim_aligned_<key>` **data descriptors** over an `AlignmentResult`
  dataclass stored in `MoleculePair._alignments`. Adding your mode id here is the only
  registration a reference-only mode does. Purely additive and safe.
- **`accel/_modes.py`** (`MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`) — the canonical screening
  registry. **Do not touch it in this skill.** `MODE_ATTRS` feeds `CANONICAL_MODES`, which
  `@_bind_batch_aligners` walks at import time calling
  `getattr(accel.batch, "_align_batch_<mode>")`; that aligner does not exist yet, so adding your
  mode here makes `import shepherd_score.container` raise. `tests/test_mode_registry.py` pins
  `len(CANONICAL_MODES) == 21` and several set equalities that all fail at the same moment.

Because your mode is not in `MODE_SEEDS` / `MODE_STEPS`, you cannot use `_default_seeds` /
`_default_steps` for its defaults — they read those tables. Use literals, and let the accel skill
move them over on promotion.

Since the descriptors are data descriptors, they win over the instance dict. That is what broke
old pickles and why both `Molecule.__setstate__` and `MoleculePair.__setstate__` exist. If you
change how a result is stored, those two methods are the back-compat surface.

## Return signature

`optimize_<mode>_overlay(...) -> (aligned_fit_points, se3_transform, score)`, all torch tensors on
the input device. The pharmacophore optimizers are the exception and return a 4-tuple.
`align_with_<mode>` converts to NumPy, stores transform and score, and returns the aligned points.

## Upstream structures you will meet

`Molecule` has been refactored onto three dataclasses from upstream, with back-compat properties
so old attribute names still work:

- `Surface` — `Molecule._surface`; `surf_pos`, `surf_esp`, `probe_radius` are properties over it.
- `Pharmacophore` — `Molecule._pharmacophore`; `pharm_types`, `pharm_ancs`, `pharm_vecs` are
  properties over it. It unpacks as the old 3-tuple.
- `AlignmentResult` — one per `_ALIGN_KEYS` entry in `MoleculePair._alignments`.

Write against the properties; they are the stable surface.
