# What the screening store still needs from you

Most of what this file used to describe is gone. `accel/_modes.py` + `accel/channels.py` now teach
the out-of-core store what a mode reads, so registering a `ModeSpec` whose channels already exist
makes the mode screen — store schema, profile extraction, pre-centring, the canonical rotation,
shard I/O, the query tensors, the fit tensors, the array aligner and the multi-GPU worker all derive
from it. **Zero edits in `screen.py`.**

What remains is the case where your mode reads per-molecule data the library does not yet carry.

## Three paths, not two

A screen picks one of three engines, in this order:

1. **Array-native** — `fast and _use_arrays(mode)`. The shard's contiguous arrays go straight to
   `accel/batch/_arrays.py::align_arrays`. No per-molecule Python object is ever built. This is what
   `screen`, `screen(ndev>1)` and every published benchmark run.
2. **Object-fast (`_FastPair`)** — `fast and not _use_arrays(mode)`. Per-molecule `_FastPair` stubs
   with pre-set tensors, no `Molecule` / `MoleculePair`.
3. **Object** — `not fast`. Real `MoleculeProfile` → `MoleculePair` → `MoleculePairBatch`.

```python
fast = (mode in _FAST_MODES and store.pre_centered
        and not align_kwargs.get("trans_init") and backend != "jax")
def _use_arrays(mode): return mode in _ARRAY_MODES and _arrays.ENABLED
```

`_FAST_MODES` and `_ARRAY_MODES` are both `tuple(SPECS)`, so in a normal screen path 2 never runs.
Path 3 is the fallback for a non-pre-centred store, `trans_init=True`, or `backend="jax"`.

> **Path 2 is not dead code.** Gate 5's reference leg is built by flipping
> `_arrays.ENABLED = False`, which sends the *same* mode down path 2. Both paths are generic now, so
> a new mode reaches both automatically — but if you ever special-case one, you must special-case
> the other or you have no gate 5.

What actually gates a mode is `_store_supports(schema, mode)`, which is `all(schema[flag] for flag
in the mode's channels' flags)`. `_VALID_MODES` holds all 21 canonical modes; the store decides.

## Tier A — reuses data the store already holds

**Nothing to do.** Register the spec; the schema, the support predicate and every builder follow.

## Tier B — needs new per-molecule data

Four places, each once:

1. **`accel/channels.py`** — a `Channel` row per array (positions and their per-point scalar are
   two rows sharing one basis). State its `kind` (`points` / `vectors` / `scalar` / `labels`), its
   `basis`, the pair tensor stem, the reader, the store key, the schema flag and the pad value.
   If the basis is new, add its offset-table key to `BASES`; add the flag to `SCHEMA_FLAGS`.

2. **`screen.py::_BASIS_TABLE`** — one row: `(basis, flag, offset key, ((profile attr, store key),
   ...))`. This is what `_profile_from_schema`, `_concat` and `_reconstruct` all read.

3. **`screen.py::MoleculeProfile`** — the `__slots__` entries, the keyword-only `__init__` args
   (stored through `_f32`), and a `get_<data>()` accessor **named exactly like the `Molecule`
   method the reader calls**, so a profile duck-types into the aligner identically:

   ```python
   def get_lipo_positions(self): return self.lipo_pos
   def get_lipophilicity(self, no_H: bool = True): return self.lipophilicity   # no_H for parity
   ```

   `center_to` shifts every position channel; add yours there too.

4. **`container/_core.py`** — the `Molecule` accessor quartet, if `design-scoring-mode` has not
   already added it.

**Give a per-atom field its OWN offset table; do not reuse `atom_off`.** `atom_pos` is the
`Chem.RemoveHs` set and is *longer* than the true-heavy set whenever RemoveHs retains an
isotope-labelled H, so one shared table desyncs the field from its positions on exactly the
molecules that are hardest to notice. The positions and the per-point scalar share one table
because they are 1:1 with each other.

Use `schema.get("<flag>", False)`, never `schema["<flag>"]`, for any flag you add — a store built
before your flag existed has no such key.

## The canonical frame — now derived, and that is the point

`ProfileStore.create(canonical=None)` resolves to **`True`** whenever the store serves any
constant-seed mode and `pre_centered` is on, so most stores are canonical: every coordinate channel
is rotated into that molecule's principal frame, with the rotation kept per molecule as `rot`
(float32 regardless of the store dtype — a float16 rotation carries ~3.9e-4 of axis error, enough
to move a pose).

`_profile_from_schema` rotates **every channel whose `kind` is `points` or `vectors`**, read off the
channel table. You no longer write that rotation by hand, which removes the single most dangerous
manual step in this whole procedure.

Why it was dangerous, and why the derivation matters: **getting it wrong is invisible to a score
test.** Leave a channel unrotated and the library and the query both sit in the unrotated frame, so
they score consistently and every parity test passes. What breaks is `Hit.transform` —
`_compose_rot` multiplies every survivor's rotation by `rot` on any canonical store,
unconditionally and without consulting the mode, so the pose is composed with a rotation its
coordinates never received. Right scores, wrong poses. That happened: `atom_pos_noH` was shifted in
the pre-centre block but missed in the canonical block, and the retained-H molecule's returned pose
re-scored **0.619 below** its reported score while every other molecule sat at 1e-7. The existing
retained-H screen test passed throughout, because it compares scores.

So: **verify a mode's poses by re-scoring them**, not by comparing scores. Still true.

Constant seeds are the payoff, and their gate is derived too: `_canonical_batch_kw` supplies
`const_seeds` when the store is canonical, there is a single query, the array path is active, and
the mode's `seed_channel` resolves to `atoms` or `heavy`. `CONST_SEED_MODES` is computed from
exactly that predicate, so a mode seeding from the atom cloud inherits the speedup the day it lands.

## Modes with a non-molecule input

`vol_avoid` scores shape Tanimoto MINUS an excluded-volume penalty against a cloud that belongs to
the QUERY, not to any library molecule. The store has nowhere to put it, so it travels as a
`screen(..., avoid_points=...)` keyword: uploaded once in `_with_avoid`, broadcast over every
bucket exactly as the query's own channels are, and declared in the registry as a `basis="pair"`
channel. If your mode takes a third input of that shape, follow it — a pair-level channel is a
first-class thing now, not a reason to be pairwise-only.

## Front-end special cases

All derived; listed so you can check they do the right thing for your mode:

| Site | Derivation |
|---|---|
| `_SURF_ALPHA_MODES` | modes whose SHAPE term runs on the `surf` channel: `alpha` auto-fills from `ALPHA(num_surf_points)` |
| `_resolve_screen` required kwargs | any spec parameter whose default is `None` |
| `screen_many` num_surf_points check | any mode with a `surf`-basis channel |
| `screen_many` query precondition | every channel's reader is called once up front, so a missing field is named before the stream starts |

If your mode needs `alpha` to mean "which cloud is the shape channel" (as the combo modes do), give
it a `channel_switch` and leave its `alpha` default `None` so the caller must choose.

## Verify (gate 5)

Build a `ProfileStore`, `screen()` a query, and assert the scores match
`MoleculePairBatch.align_with_<mode>` on the same centred molecules (deep-copy and `center_to` each
library molecule to its own heavy-atom COM to match a `pre_centered` store).

- **Read shards through the store, not by globbing a filename.** The default `shard_format` is
  `"npy"` — one memory-mapped `.npy` per array, named `shard_00000__lipo_pos.npy` — so a test that
  opens `shard_*.npz` finds nothing. Use `store.read_shard(0)[1]`, which handles both formats.
- **Assert the data reached disk** for a Tier-B mode: check the new arrays *and* the offset table
  are among those keys. A mode that silently stores nothing still passes a score comparison when
  every molecule's data is empty.
- **Assert the pose, not only the score**, on a canonical store. Apply the returned transform to the
  molecule's own centred coordinates and recompute the similarity directly.
- **Tolerance.** Against the object path on the same store the array path is **bit-identical** — use
  `abs=1e-4` and expect exact. Across *different* stores, or against the pairwise path, a canonical
  store differs by ~1e-3 on average and a non-canonical store matches to ~1e-4. Do not loosen past
  `1e-2`; a real wiring bug moves scores by far more.
- **Drive the test off `_ARRAY_MODES` / `CANONICAL_MODES`, never a hardcoded list.** The store
  fixture, the parametrize lists, the required-kwarg table and the self-copy exclusion in
  `tests/test_screen_arrays.py` are all derived now; keep them that way.

Test files: `tests/test_screen.py` (store round-trip), `tests/test_screen_arrays.py` (array parity
and the canonical re-score), `tests/test_screen_ndev_dispatch.py` (the one-selector rule),
`tests/test_screen_pipeline.py` (shard-format equivalence), `tests/test_screen_const_seeds.py`
(which seed generator ran).

## Edit inventory

| Step | edits outside `accel/_modes.py` |
|---|---|
| Tier A (reuses stored data) | **0** |
| Tier B (new per-molecule data) | 1 `channels.py` block + 1 `_BASIS_TABLE` row + `MoleculeProfile` slots/accessors |
| A non-molecule third input | 1 `Channel(basis="pair")` row |

Compare that with what it cost before the registry: roughly nineteen edit sites in `screen.py`, an
array aligner, an object-path builder and three registry rows — which is why this file used to be
the longest reference in the skill.
