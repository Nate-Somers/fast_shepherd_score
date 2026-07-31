# Wiring a mode into `screen` (the out-of-core store)

This is the full procedure for step 10 of `SKILL.md`. Read it once the mode has a batched path
(steps 1–9 done) and must screen a library through the on-disk store.

The registry (`accel/_modes.py`) makes `screen` / `multi_gpu` / `cpu_pool` **dispatch** your mode:
resolve its name, result attributes, and seed/step defaults. It does **not** teach the on-disk store
what per-molecule arrays your mode *reads*. `screen` streams a library through `MoleculeProfile` (an
RDKit-free, arrays-only stand-in for `Molecule`) and `ProfileStore` (sharded `npz`). If your mode
needs data the store does not persist, `screen` cannot serve it — the pairwise `MoleculePairBatch`
API is not enough. This is the step that is easy to miss: the in-memory path works, the mode imports,
tests pass, and yet `screen(..., mode="<yours>")` fails or feeds zeros.

## Contents
- [Decide the tier](#decide-the-tier)
- [Tier A — reuses stored data (one edit)](#tier-a--reuses-data-the-store-already-holds)
- [Tier B — needs new per-molecule data (seven edits)](#tier-b--needs-new-per-molecule-data-the-store-doesnt-carry)
- [Fast-engine registration (six edits)](#fast-engine-registration--optional-for-correctness-required-before-you-report-throughput)
- [Verify (gate 5)](#verify-gate-5)

## Decide the tier

- **Tier A** — your mode reuses per-molecule data the store already holds.
- **Tier B** — your mode needs new per-molecule data the store does not carry.

## Tier A — reuses data the store already holds

A reduction over an existing channel (e.g. `vol_tversky` is a different reduction of the same
heavy-atom shape overlap; `atom_pos` is *always* stored). Edit **one** function:
`_store_supports(schema, mode)` in `screen.py`, returning the right predicate (`return True` for a
shape-only mode). Nothing else — it routes through the `fast=False` `MoleculePairBatch` path
automatically.

## Tier B — needs new per-molecule DATA the store doesn't carry

E.g. `vol_lipo`'s variable-length per-heavy-atom lipophilicity centres and per-atom Crippen logP.
Wire it end to end in `screen.py`, mirroring the `pharm` / `lipo` plumbing already there:

1. **`MoleculeProfile`** — add the array(s) to `__slots__`; accept them in `__init__` (store via
   `_f32`); shift the positional ones in `center_to` (they move rigidly with the molecule); add a
   `get_<data>()` accessor **named exactly like the `Molecule` method the batched driver calls**
   (`_batch_upload(pairs, ..., lambda p: p.<side>_molec.get_<data>()[k], ...)`), so a profile
   duck-types into the aligner identically to a `Molecule`.
2. **`_schema_from_modes`** — add a boolean schema flag set when your mode is requested.
3. **`_store_supports`** — return `schema.get("<flag>", False)`.
4. **`_profile_from_schema`** — extract the data off the `Molecule`, pre-center it alongside
   `atom_pos` when `pre_center`, and pass it into the returned `MoleculeProfile(...)`.
5. **`ProfileStore._concat`** (pack to `npz`) — fixed-width array → `np.stack`; a **variable-length**
   set (per-molecule count differs) → write an **offset table** (`<name>_off = offsets(lens)`)
   plus the concatenated points, mirroring the `pharm_off` block exactly.
6. **`ProfileStore._reconstruct`** (unpack) — read the offset table and slice each molecule's
   segment back out, mirroring the pharm unpack.
7. If the data is **cached on the `Molecule`** (e.g. a lazy field), mirror the `center_to` shift in
   `container/_core.py` too, so a query centered by `screen` never carries a stale copy.

## Fast-engine registration — optional for correctness, REQUIRED before you report throughput

`_FAST_MODES` in `screen.py` lists the modes that get the direct array→kernel fast path: fit tensors
built once per shard and reused across the query panel, bypassing per-shard `MoleculePair`
construction. A mode left out still screens *correctly* through the `fast=False` object path — but
that path rebuilds `MoleculePair` objects every shard, pure host-side Python overhead the fast modes
do not pay. **The GPU kernel is identical; only the harness differs.** So if you benchmark a
non-fast mode's screening throughput against the fast modes (e.g. a fig2 A/B/E screening panel), you
measure your mode's object-path overhead against their resident-tensor path and **understate your
mode's true throughput**. Two honest options, in order of preference:

1. **Add the mode to the fast engine** (do this for anything that will appear in a screening-speed
   figure). It is mechanical once the driver exists — the fast path just pre-sets the tensors the
   driver's `_batch_upload` would otherwise fill, so `_batch_upload` skips its lambdas (it only
   fills cold `None` attrs). Six edits in `screen.py`:
   - `_FAST_MODES` — add the mode id.
   - `_FastPair.__slots__` — add the mode's result slots (`transform_<mode>`/`sim_aligned_<mode>`)
     and any new fit/ref tensor slots the driver reads (e.g. `_ref_lipo_pos_t`/`_fit_lipo_pos_t`).
   - `_query_ref_arrays(q, mode)` — the query's numpy arrays for the ref side.
   - `_ref_tensors_from_arrays(ra, mode, device)` — those arrays → the pre-set `_ref_*_t` tensors
     (no `_ArrView`/`.ref_molec` needed if you pre-set every tensor the driver uploads).
   - `_build_fit_fast_pairs(arrs, mode, device)` — split the shard's contiguous arrays into
     per-molecule fit tensor views (`splitT` on the offset table for a variable-length set).
   - `_fast_batch_kwargs(mode, ak)` — translate screen kwargs to the exact `_align_batch_<mode>`
     signature (mirror the pairwise `align_with_<mode>(backend="triton")` dispatch — pass ONLY the
     kwargs the driver accepts; some drivers take their seed count internally).
2. **If you deliberately keep it on the object path** (a mode you are not speed-benchmarking):
   `log`/note in the figure that this mode uses the object path, and do NOT plot its screening
   throughput on the same axis as the fast modes. Reporting an object-path aligns/s next to
   resident-tensor aligns/s is the silent-cap failure from `Workflow`'s discipline, one level up.

The `vol_tversky` fast path is a drop-in copy of `vol`'s (shape-only, same fit tensors); `vol_lipo`
adds its four lipophilicity tensors alongside the shape tensors. A shape-only or reduction mode is
almost free to fast-path; a mode with new per-molecule data pre-sets that data's tensors too.

## Verify (gate 5)

The screen analog of parity gate 3, one level further out. Build a `ProfileStore`, `screen()` a
query, and assert the scores match `MoleculePairBatch.align_with_<mode>` on the same centered
molecules. Once the mode is in `_FAST_MODES` and the store is `pre_centered`, this test exercises the
FAST path (that is the point — it proves the resident-tensor path is score-faithful).

Tolerance: a single-channel / reduction mode is usually bit-identical (`abs=1e-4`, like
`vol_tversky`); a basin-sensitive multi-channel mode can differ by ~1e-3 when fp-noise-level input
differences flip a multi-start seed near a tie, so use the same `abs=1e-2` the accel test uses
(`vol_lipo`), and lean on the "assert the data reached disk" check to catch a real wiring bug (which
moves scores by `>>1e-2` or changes the point counts). **Also assert the data reached disk** for a
Tier-B mode: load a `shard_*.npz` and check the new arrays (and the offset table for a
variable-length set) are present — a mode that silently stores nothing still "passes" a score
comparison when every molecule's data is empty.

Models: `test_vol_tversky_stream_matches_object` (Tier A, bit-identical) and
`test_vol_lipo_stream_matches_object` (Tier B, basin tolerance + disk check), in
`tests/test_screen.py`. Full gate-5 detail is in `parity_gates.md`.
