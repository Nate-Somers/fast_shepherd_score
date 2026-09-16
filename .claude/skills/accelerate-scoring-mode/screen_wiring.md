# Wiring a mode into `screen` (the out-of-core store)

The full procedure for step 11 of `SKILL.md`. Read it once the mode has a batched path (steps 1–9
done) and must screen a library through the on-disk store.

`accel/_modes.py` makes `screen` / `multi_gpu` / `cpu_pool` **dispatch** your mode: resolve its
name, result attributes and seed/step defaults. It does **not** teach the on-disk store what
per-molecule arrays your mode *reads*, and it does not give you a fast path. `screen` streams a
library through `MoleculeProfile` (an RDKit-free, arrays-only stand-in for `Molecule`) and
`ProfileStore` (sharded `.npy`). This is the step that is easy to miss: the in-memory path works,
the mode imports, tests pass, and `screen(..., mode="<yours>")` still raises or feeds zeros.

All line numbers below are `shepherd_score/screen.py` unless stated.

## Contents
- [Three paths, not two](#three-paths-not-two)
- [Decide the tier](#decide-the-tier)
- [Tier A — reuses stored data](#tier-a--reuses-data-the-store-already-holds)
- [Tier B — needs new per-molecule data](#tier-b--needs-new-per-molecule-data)
- [The canonical frame — do not skip this](#the-canonical-frame--do-not-skip-this)
- [Array-native registration](#array-native-registration--the-production-path)
- [The `_FastPair` object-fast path](#the-_fastpair-object-fast-path--required-for-gate-5)
- [Front-end special cases](#front-end-special-cases)
- [Verify (gate 5)](#verify-gate-5)
- [Edit inventory](#edit-inventory)

## Three paths, not two

A screen picks one of three engines, in this order:

1. **Array-native** — `fast and _use_arrays(mode)`. The shard's contiguous arrays go straight to an
   array aligner. No per-molecule Python object is ever built. This is what `screen`,
   `screen(ndev>1)` and every published benchmark run.
2. **Object-fast (`_FastPair`)** — `fast and not _use_arrays(mode)`. Per-molecule `_FastPair` stubs
   with pre-set tensors, no `Molecule` / `MoleculePair`.
3. **Object** — `not fast`. Real `MoleculeProfile` → `MoleculePair` → `MoleculePairBatch`.

```python
# L2250 in screen_many
fast = (mode in _FAST_MODES and store.pre_centered
        and not align_kwargs.get("trans_init") and backend != "jax")
# L1210
def _use_arrays(mode): return mode in _ARRAY_MODES and _arrays.ENABLED
```

`_FAST_MODES` (L877) and `_ARRAY_MODES` (L1205) currently hold **the same 11 modes**, so in a normal
screen path 2 never runs. Path 3 is the fallback for a non-pre-centered store, `trans_init=True`, or
`backend="jax"`.

> **Path 2 is not optional, and this is the single easiest way to ship a broken mode.** Gate 5
> builds its reference leg by flipping `_arrays.ENABLED = False` (`accel/batch/_arrays.py:47`),
> which sends the *same* mode down path 2. So every mode you add to `_ARRAY_MODES` **must** also be
> wired into `_build_fit_fast_pairs` and `_fast_batch_kwargs`, or the parity test that proves your
> array path correct dies with `ValueError: <your mode>` from `screen.py:1161`. Wire both paths, or
> you have no gate 5.

**Consequence for you:** a mode that lands only on path 3 is correct but is *not* comparable on a
throughput axis with the eleven. Reporting its aligns/s next to theirs understates it by 1.3–5×
(measured, commit `698e074`). Wire the array path, or do not plot the number.

`_VALID_MODES` is derived from `MODE_ATTRS`, so it holds all 21 canonical modes while only 11
screen. The gate that actually rejects an unwired mode is `_store_supports`, not `_VALID_MODES`.

## Decide the tier

- **Tier A** — reuses per-molecule data the store already holds (`atom_pos` is unconditional).
- **Tier B** — needs a new per-molecule array the store does not carry.

## Tier A — reuses data the store already holds

Edit **one** function: `_store_supports(schema, mode)` (L271). It is an `if mode == ...` chain
returning a predicate over the schema flags, with `return False` as the fallthrough at L300.

```python
    if mode == "vol_tversky":
        return True                                     # asymmetric shape overlay; atom_pos always stored
```

**Declare an unsupported mode explicitly rather than letting it fall through.** `vol_avoid` does
this, and the comment is the point — a reader otherwise cannot tell "deliberately pairwise-only"
from "someone forgot":

```python
    if mode == "vol_avoid":
        return False                                    # PAIRWISE-ONLY (deliberate): the avoid
        # cloud is a fixed non-molecule input (a query/global constant), which the per-molecule
        # ProfileStore does not model -- use MoleculePairBatch.align_with_vol_avoid, not screen().
```

Use `schema.get("<flag>", False)`, not `schema["<flag>"]`, for any flag you add — a store built
before your flag existed has no such key and `[]` raises `KeyError` on open. The two newest
channels (`lipophilicity`, `fukui`) already follow this; the older ones do not.

## Tier B — needs new per-molecule data

Wire it end to end, mirroring the **lipo / fukui** plumbing (the `pharm` block is an older model
that shares `atom_off`; do not copy it for a per-atom field — see the offset-table note below).

1. **`_schema_from_modes` (L253)** — add one boolean flag, set when your mode is requested. Nine
   flags exist: `surf`, `surf_esp`, `charges`, `with_H`, `radii`, `centers_w_H`, `pharm`,
   `lipophilicity`, `fukui`. There is no flag for atoms; `atom_pos` is unconditional.

2. **`_store_supports` (L271)** — `return schema.get("<flag>", False)`.

3. **`MoleculeProfile` (L137)** — add the arrays to `__slots__` (L154); accept them in the
   keyword-only `__init__` (L160), storing through `_f32`; shift the **positional** ones in
   `center_to` (L202); and add a `get_<data>()` accessor **named exactly like the `Molecule`
   method the batched driver calls**, so a profile duck-types into the aligner identically:

   ```python
   def get_lipo_positions(self): return self.lipo_pos
   def get_lipophilicity(self, no_H: bool = True): return self.lipophilicity   # no_H for parity, ignored
   ```

   `center_to` shifts `atom_pos`, `atom_pos_noH`, `surf_pos`, `pharm_ancs`, `lipo_pos`,
   `fukui_pos` and the with-H `mol` shim. It deliberately does **not** shift `pharm_vecs`
   (directions carry no translation) and does not touch `rot`.

4. **`_profile_from_schema` (L303)** — **three** sub-edits, and missing any one is a silent
   wrong answer:
   - **extract** the data off the `Molecule` behind your schema flag, raising a clear `ValueError`
     if the molecule cannot provide it;
   - **shift** the positional channel in the `pre_center` block (L368) — by the **`atom_pos` COM**,
     never its own COM, so it matches the in-memory conformer transform;
   - **rotate** it in the `canonical` block (L386) — see the next section.

5. **`ProfileStore._concat` (L625)** — pack to the shard. A fixed-width array uses `np.stack`; a
   **variable-length** set gets its **own offset table**:

   ```python
   if sch.get("lipophilicity"):
       lens = [len(r.lipo_pos) for r in recs]
       out["lipo_off"] = offsets(lens)
       out["lipo_pos"] = np.concatenate([...]).astype(dt)
       out["lipophilicity"] = np.concatenate([...]).astype(dt)
   ```

   **Give a per-atom field its own table; do not reuse `atom_off`.** `atom_pos` is the
   `Chem.RemoveHs` set and is *longer* than the true-heavy set whenever RemoveHs retains an
   isotope-labelled H, so one shared table desyncs the field from its positions on exactly the
   molecules that are hardest to notice. The positions and the per-point scalar share one table
   because they are 1:1 with each other.

6. **`ProfileStore._reconstruct` (L787)** — hoist the offset table out of the per-molecule loop and
   slice each segment back out. Note `_reconstruct` does **not** restore `rot`: a profile read back
   from a canonical store is in the canonical frame with no record of it.

7. If the data is **cached on the `Molecule`**, mirror the `center_to` shift in
   `container/_core.py` too, so a query centered by `screen` never carries a stale copy.

## The canonical frame — do not skip this

`ProfileStore.create(canonical=None)` resolves to **`True`** whenever the store's modes include
`vol` and `pre_centered` is on (L527). Most stores are therefore canonical, and a canonical store
holds every coordinate channel rotated into that molecule's principal-axis frame, with the
rotation kept per molecule as `rot` (float32 regardless of the store dtype — a float16 rotation
carries ~3.9e-4 of axis error, enough to move a pose).

**Every coordinate channel you add must be rotated in the `canonical` block of
`_profile_from_schema` (L386).** Positions rotate; so do direction vectors (`pharm_vecs` rotates,
without a translation). A per-point scalar does not.

```python
        rot = np.ascontiguousarray(_v.T, dtype=np.float32)   # maps original -> canonical
        atom_pos = (atom_pos @ rot.T).astype(np.float32)
        ...
        if lipo_pos is not None and len(lipo_pos):
            lipo_pos = (lipo_pos @ rot.T).astype(np.float32)
```

**Getting this wrong is invisible to a score test**, which is what makes it worth a paragraph.
Leave a channel unrotated and the library and the query both sit in the unrotated frame, so they
score consistently and every parity test passes. What breaks is `Hit.transform`: `_compose_rot`
(L1767) multiplies every survivor's rotation by `rot` on any canonical store, **unconditionally
and without consulting the mode**, so the pose is composed with a rotation its coordinates never
received. Right scores, wrong poses.

That is not hypothetical. `atom_pos_noH` — the strict-heavy centre set `vol_esp` reads as
`xyz_noH` — was shifted in the `pre_center` block but missed in the `canonical` block. Because
`_concat` fills every *other* molecule's `xyz_noH` row from its already-rotated `atom_pos`, a
single row of that array sat in the raw centred frame while its neighbours were canonical. It
surfaced only on molecules whose `Chem.RemoveHs` retains a hydrogen, since that is the only case
where `atom_pos_noH` is materialised at all. Measured: the retained-H molecule's returned pose
re-scored **0.619 below** its reported score while every other molecule sat at 1e-7. The existing
retained-H screen test passed throughout, because it compares scores.

So: rotate every coordinate channel, and verify a mode's poses by **re-scoring them**, not by
comparing scores.

Constant seeds are the payoff, and they are narrower than the rotation: `_canonical_batch_kw`
(L2004) supplies `const_seeds` only when the store is canonical **and** `mode == "vol"` **and**
there is a single query **and** the array path is active. `const_seeds` is a parameter of
`align_batch_vol_arrays` alone; the object path's `_align_batch_vol` raises `TypeError` on it.
Every other mode pays the canonical store's score shift and gains no speed from it, which is why
a store without `vol` stays non-canonical.

## Array-native registration — the production path

Five edits in `screen.py` plus one aligner in `accel/batch/_arrays.py`.

- **`_ARRAY_MODES` (L1205)** — add the mode id.
- **`_build_fit_arrays_<mode>(arrs, device)` → `(ids, *tensors)`** — split the shard's contiguous
  arrays into the device tensors the aligner takes. Upload through `_to_device` (L1264), which
  keeps persistent pinned staging buffers.
- **`_align_fast_arrays_<mode>(ref, fit, batch_kw)` → `(scores, SE3)`** — a thin call into
  `accel.batch._arrays.align_batch_<mode>_arrays`.
- **`_ARRAY_BUILDERS` / `_ARRAY_ALIGNERS` (L1713/L1724)** — one row each.

Those five cover the **fit** side only. The **ref** (query) side is shared with the object-fast
path: both branches call `_query_ref_arrays(q, mode)` and `_ref_tensors_from_arrays(ra, mode,
device)` per query before dispatching. Add your mode's branch to each of those two once, and both
paths pick it up. Every one of them ends in `raise ValueError(mode)`, so a missing branch fails
loudly rather than feeding zeros.

Then `_array_dispatch(mode)` (L1737) resolves the pair. **Use that one selector; never add a
second table.** A second dispatch table is precisely what made `screen(ndev>1)` run `vol`'s builder
and aligner under every mode's name, silently returning `vol` answers for `vol_color`, `vol_esp`
and the rest (behaviour change B10). The in-process driver and the multi-GPU worker both read
`_array_dispatch`, so one table keeps them honest.

**Reuse the shared scalar-field builder** rather than writing a near-copy. A shape channel plus one
per-atom field is fully served by:

```python
_build_fit_arrays_scalar_field(arrs, device, *, pos_key, val_key, off_key)   # L1556
```

`vol_lipo` and `vol_fukui` are each one line over it, parameterised only by store key. A third such
mode should be a third line, not a third function.

**Bit-identity is the gate here, not a tolerance.** The array path is a front-end refactor with no
arithmetic in it, so the correct result is `0` of `N` scores moved and `max|Δ|` exactly `0.000e+00`
against the object path, with identical ids and identical 4×4 transform elements. If your array
builder produces a tolerance-level difference, it is feeding different inputs — find that, do not
widen the gate.

## The `_FastPair` object-fast path — required for gate 5

Six edits: `_FAST_MODES`, `_FastPair.__slots__` (L904), `_query_ref_arrays` (L936),
`_ref_tensors_from_arrays` (L984), `_build_fit_fast_pairs` (L1033), `_fast_batch_kwargs` (L1148).
Of these, `_query_ref_arrays` and `_ref_tensors_from_arrays` are shared with the array path and
`_fast_batch_kwargs` is used by both, so only `_FastPair.__slots__` and `_build_fit_fast_pairs` are
specific to this path — and gate 5 needs them anyway, because it flips `_arrays.ENABLED` off to
build its reference leg. `_build_fit_fast_pairs` and `_fast_batch_kwargs` both end in
`raise ValueError(mode)`, so a missing branch fails loudly in the parity test rather than
silently degrading. The path pre-sets every tensor the driver's `_batch_upload` would
otherwise fill, so `_batch_upload` skips its lambdas (it only fills cold `None` attributes).

`_FastPair.__slots__` is hand-mirrored from `MODE_ATTRS`, not derived — add both the result pair
(`transform_<mode>` / `sim_aligned_<mode>`, using the historical `_noH` spelling for `vol` and
`vol_esp`) and any new `_ref_*_t` / `_fit_*_t` tensor slots.

Because `_FAST_MODES` also gates `fast` itself, **a mode must be in `_FAST_MODES` to reach the
array path at all** — `fast` is computed from `_FAST_MODES` before `_use_arrays` is consulted. So
add the mode to both tuples. `_fast_batch_kwargs` is likewise still on the live path: it translates
screen kwargs into the exact `_align_batch_<mode>` signature, and some drivers take their seed count
internally, so pass only the kwargs the driver accepts.

## Front-end special cases

Add these only if they apply:

| Site | When |
|---|---|
| `_SURF_ALPHA_MODES` (L79) | the mode reads surface points and should auto-fill `alpha=ALPHA(num_surf_points)` |
| `_resolve_screen` (L1939) | the mode has a **required** kwarg (`vol_and_surf_esp` needs `alpha`, `vol_esp` needs `lam`) |
| `screen_many` L2217 | the mode needs the store's `num_surf_points` to match the query's |
| `screen_many` L2228/L2236 | the query itself must carry a field, checked before streaming |
| `ProfileStore.create` L534 | only if your mode should change the canonical default (it should not) |

Routing you get for free, with zero edits: `_TRANSFORM_ATTR` / `_SCORE_ATTR` / `_VALID_MODES`
(from `MODE_ATTRS`), `_align_fast`'s `getattr(aligners, "_align_batch_" + mode)`, the object path's
`getattr(MoleculePairBatch(pairs), "align_with_" + mode)`, `_steps_for` / `_seeds_for`, and
`accel/screen_parallel.py`'s `_ALIGN_ATTR`.

## Verify (gate 5)

Build a `ProfileStore`, `screen()` a query, and assert the scores match
`MoleculePairBatch.align_with_<mode>` on the same centered molecules (deep-copy and `center_to`
each library molecule to its own heavy-atom COM to match a `pre_centered` store).

- **Read shards through the store, not by globbing a filename.** The default `shard_format` is
  `"npy"` — one memory-mapped `.npy` per array, named `shard_00000__lipo_pos.npy` — so a test that
  opens `shard_*.npz` finds nothing. Use `store.read_shard(0)[1]`, which handles both formats.
- **Assert the data reached disk** for a Tier-B mode: check the new arrays *and* the offset table
  are among those keys. A mode that silently stores nothing still passes a score comparison when
  every molecule's data is empty, and that is the check that survives a loose score tolerance.
- **Assert the pose, not only the score**, on a canonical store. A missing rotation leaves the
  score at 1.000000 while the returned transform re-scores to ~0.907. Apply the returned transform
  to the molecule's own centered coordinates and recompute the similarity directly.
- **Tolerance.** Against the object path on the same store the array path is **bit-identical** —
  use `abs=1e-4` and expect exact. Across *different* stores, or against the pairwise path, a
  canonical store differs by ~1e-3 on average (a few molecules land in another optimizer basin, max
  ~0.1), and a non-canonical store matches to ~1e-4. Do not loosen past `1e-2`; a real wiring bug
  moves scores by far more than that.
- **Drive the test off `_ARRAY_MODES`, not a hardcoded list.** `tests/test_screen_arrays.py` does
  this, and its negative control is computed as `CANONICAL_MODES` minus `_ARRAY_MODES` rather than
  as a second hardcoded list. Both were hardcoded once; six modes shipped with zero coverage
  because of it.

Test files: `tests/test_screen.py` (store round-trip), `tests/test_screen_arrays.py` (array parity
and the canonical re-score), `tests/test_screen_ndev_dispatch.py` (guards the one-selector rule),
`tests/test_screen_pipeline.py` (shard-format equivalence).

## Edit inventory

What a new mode actually costs today, cumulative:

| Step | `screen.py` edits |
|---|---|
| Registry (`accel/_modes.py`) — always | 0 |
| Tier A, object path only | 1 (`_store_supports`) |
| Tier B, new per-molecule data | +7 (`_schema_from_modes`, 4 × `MoleculeProfile`, `_profile_from_schema` ×3 sub-edits, `_concat`, `_reconstruct`) |
| `_FastPair` path (required: it is gate 5's reference leg) | +6 |
| Array-native path | +5, plus `align_batch_<mode>_arrays` in `accel/batch/_arrays.py` |
| Conditional front-end | up to +5 |

A fully wired, benchmark-honest Tier-B mode is roughly **19 edit sites in `screen.py`**, one
aligner in `_arrays.py`, and three registry rows. That is larger than it looks from the registry
alone, and it is the reason `SKILL.md` calls this step out rather than folding it into "wire the
driver". Per-molecule *data* is inherently mode-specific and cannot be derived; mode *routing*
can be and already is.
