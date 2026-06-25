# Design: stream-from-disk screening mode for billion-scale libraries

> **Status: IMPLEMENTED** — shipped in [`shepherd_score/screen.py`](../shepherd_score/screen.py)
> (tests: [`tests/test_screen.py`](../tests/test_screen.py); streaming scores are
> bit-equivalent to the in-memory `MoleculePairBatch` path). This document is the
> design + rationale; the shipped API matches it, with the on-disk format refined to
> one `.npz` per shard (§3.2) and `vol_and_surf_esp` supported via a conformer shim (§3.1).
> An opt-in, out-of-core screening path that aligns a query (or a handful of queries)
> against a library far larger than host RAM, without changing any existing class,
> kernel, or default. Companion to
> [`MOLECULE_CONSTRUCTION_PROBLEM.md`](./MOLECULE_CONSTRUCTION_PROBLEM.md), which
> covers the *one-time* cost of populating the on-disk store.

## 1. The problem this solves

At ~10.7 KB per `Molecule` with a 200-point surface (the RDKit `Mol` is ~66% of
that), a library does not fit in RAM long before a billion molecules:

| dataset | object RAM (with surface) |
|---|--:|
| 25 M molecules | ~256 GB (fills a typical node) |
| 100 M molecules | ~1 TB |
| **1 B molecules** | **~10 TB** (volumetric-only ~7 TB) |

So the entire library cannot be materialized as `Molecule`/`MoleculePair`
objects on a normal node. Two facts make this *easy* to fix rather than hard:

1. **Alignment compute is cheap and already chunked.** On one H100, 1e9 *cross*
   (distinct) pairs is ~2.7 h (vol) to ~11 h (surf_esp); a few GPUs bring it under an
   hour. Inside the GPU path, `_subbatched_align()` already streams each
   size-band bucket through the device under a 70%-free-memory budget with
   OOM-retry — **no global `K×N×M` tensor is ever built**
   ([`aligners.py`](../shepherd_score/accel/batch/aligners.py),
   [`accel/batch/_pad.py`](../shepherd_score/accel/batch/_pad.py)). GPU memory is
   a solved problem.
2. **Alignment never needs the RDKit `Mol`.** The batched aligners read only
   numpy arrays off `ref_molec`/`fit_molec` (`atom_pos`, `surf_pos`, `surf_esp`,
   `partial_charges`, `_nonH_atoms_idx`, `pharm_types/ancs/vecs`, `radii`). Only
   `vol_and_surf_esp` touches `.mol` (for with-H centers), and even there the fast path
   prefers a pre-cached tensor.

Therefore the **only** missing piece is a *host-side* loop that (a) persists
compact, RDKit-free profiles to disk once, and (b) streams shards back into the
**existing** `MoleculePairBatch.align_with_*` / `align_multi_gpu` API, reducing
scores on the fly. No kernel, no GPU-memory, and no existing-class changes.

> **Scope note.** Node-sharding (split 1B across ~40 nodes, each holding ~25M in
> RAM) is the alternative when you *have* a large cluster. Disk-streaming is the
> right tool for (a) single nodes / modest GPU boxes, (b) re-screening the same
> library against many query sets over time, and (c) the fact that at this scale
> you *must* persist precomputed profiles to disk **anyway** (see the companion
> doc) — so streaming them back is free. The two compose: each node streams its
> own disk shards.

## 2. Design principles

- **Zero edits to `Molecule`, `MoleculePair`, `MoleculePairBatch`, the aligners,
  the kernels, or `multi_gpu.py`.** Everything is reused via duck typing + the
  existing precomputed-array constructor path + `do_center=False`.
- **One new module**, `shepherd_score/screen.py`, holding three additive pieces.
- **Opt-in, default-off**, mirroring the repo's existing `backend="triton"` /
  `FSS_*` opt-in pattern. Nothing about the current API changes.
- **The store is the unit of parallelism for the expensive part** (construction):
  shards are written independently by many workers/nodes, with no locking.

## 3. Three new pieces (all in `shepherd_score/screen.py`)

### 3.1 `MoleculeProfile` — RDKit-free, duck-typed stand-in for `Molecule`

A `__slots__` container exposing exactly the attributes the aligners read, and
nothing else. This is the linchpin that keeps shard reconstruction at
array-copy speed: reconstructing a real RDKit `Mol` per molecule (`Chem.Mol`
deserialize + `RemoveHs` + `GetAtoms` + radii) costs ~50–150 µs each → 5–15 s per
100k shard, which would *dwarf* the ~1 s GPU alignment of that shard. Array
slicing is ~memcpy (<<1 s/shard), so the streaming layer stays alignment-bound.

```python
class MoleculeProfile:
    """Numeric-only stand-in for Molecule. Holds just the arrays the batched
    aligners consume; .mol is None. Duck-types into MoleculePair unchanged."""
    __slots__ = ("atom_pos", "surf_pos", "surf_esp", "partial_charges",
                 "radii", "_nonH_atoms_idx", "pharm_types", "pharm_ancs",
                 "pharm_vecs", "num_surf_points", "centers_w_H", "mol")

    def __init__(self, *, atom_pos, surf_pos=None, surf_esp=None,
                 partial_charges=None, radii=None, pharm_types=None,
                 pharm_ancs=None, pharm_vecs=None, centers_w_H=None):
        self.atom_pos = atom_pos
        self.surf_pos = surf_pos
        self.surf_esp = surf_esp
        self.partial_charges = partial_charges
        self.radii = radii
        # heavy-atom charges are stored already-indexed, so identity index works
        self._nonH_atoms_idx = (None if partial_charges is None
                                else np.arange(len(partial_charges)))
        self.pharm_types = pharm_types
        self.pharm_ancs = pharm_ancs
        self.pharm_vecs = pharm_vecs
        self.num_surf_points = None if surf_pos is None else len(surf_pos)
        self.centers_w_H = centers_w_H   # only for vol_and_surf_esp
        self.mol = None

    def center_to(self, xyz_means):      # RDKit-free; only the arrays move
        self.atom_pos = self.atom_pos - xyz_means
        if self.surf_pos is not None:  self.surf_pos  = self.surf_pos  - xyz_means
        if self.pharm_ancs is not None: self.pharm_ancs = self.pharm_ancs - xyz_means
        if self.centers_w_H is not None: self.centers_w_H = self.centers_w_H - xyz_means

    @classmethod
    def from_molecule(cls, m, *, modes=_VALID_MODES, id=None):   # _VALID_MODES = all 6
        """Extract exactly the arrays the requested modes need; drop the Mol."""
        need_surf = bool({"surf", "surf_esp", "vol_and_surf_esp"} & set(modes))
        need_chg  = bool({"surf_esp", "vol_esp", "vol_and_surf_esp"} & set(modes))
        need_ph   = "pharm" in modes
        return cls(
            atom_pos=m.atom_pos.astype("float32"),
            surf_pos=(m.surf_pos.astype("float32") if need_surf else None),
            surf_esp=(m.surf_esp.astype("float32") if need_surf and "surf_esp" in modes else None),
            partial_charges=(m.partial_charges[m._nonH_atoms_idx].astype("float32")
                             if need_chg else None),
            pharm_types=(m.pharm_types if need_ph else None),
            pharm_ancs =(m.pharm_ancs  if need_ph else None),
            pharm_vecs =(m.pharm_vecs  if need_ph else None),
        )
```

**Compatibility check (verified against the code):**
- `MoleculePair.__init__` does `self.ref_molec = ref_mol` for any non-`Chem.Mol`
  and then `torch.as_tensor(self.ref_molec.atom_pos, ...)`. ✅ works.
- With `do_center=False` (the screening default) `center_to` is never called;
  if `do_center=True`, the array-only `center_to` above covers it.
- `vol_esp` reads `partial_charges[_nonH_atoms_idx]`; storing heavy-atom charges
  with an identity index returns them unchanged. ✅
- `vol_and_surf_esp` is the **only** mode that reads `.mol.GetConformer().GetPositions()`
  (for with-H centers). `MoleculeProfile` supplies a tiny conformer **shim** so that
  call returns the stored `centers_w_H` — no real RDKit `Mol` needed. (Pre-seeding the
  aligner's `_ref_centers_w_H_t` tensor does **not** suffice: the aligner evaluates
  `p.ref_molec.mol.GetConformer().GetPositions()` *eagerly* as a call argument, so
  `.mol` must respond to it.) All six modes — `vol/vol_esp/surf/surf_esp/pharm/vol_and_surf_esp` —
  are supported RDKit-free.

### 3.2 `ProfileStore` — sharded, memory-mappable, on-disk profile store

A directory of independently-written shards plus a small JSON manifest. Each shard is
one `.npz` bundling that shard's molecules: fixed-size arrays (surface) as dense
`(n, S, 3)` / `(n, S)` blocks, ragged arrays (atoms, pharmacophores) as flat buffers +
CSR offsets. The shard is the resident unit (a whole shard is aligned, then discarded),
so a per-shard `.npz` is the natural granularity — no within-shard mmap needed.

```
library.fss/
├── manifest.json          # {version, n_total, num_surf_points, modes, schema,
│                          #  dtype, pre_centered, shard_size, shards:[{name,n,start}]}
├── shard_00000.npz        # one shard's molecules, concatenated:
│                          #   atom_pos (Σnᵢ,3) + atom_off (n+1,)     ragged (CSR)
│                          #   surf_pos (n,S,3), surf_esp (n,S)       fixed S
│                          #   charges (+ all_off/nonH when with-H), radii, cwh  (vol_and_surf_esp)
│                          #   pharm_types/ancs/vecs + pharm_off      ragged (CSR)
│                          #   ids (n,)
└── shard_00001.npz ...
```

```python
class ProfileStore:
    # ---- WRITE (once; embarrassingly parallel across workers/nodes) ----
    @classmethod
    def create(cls, path, *, num_surf_points, modes, dtype="float16",
               shard_size=100_000, pre_centered=True, overwrite=False): ...
    def add(self, molecule, id=None): ...      # accepts Molecule OR MoleculeProfile
    def add_profile(self, profile, id=None): ...
    def add(self, molecule_or_profile): ...   # extract arrays, buffer, auto-flush shard
    def close(self): ...                       # flush tail shard + manifest
    def __enter__(self); __exit__(self, *exc)  # context-managed writer

    # ---- READ (streamed) ----
    @classmethod
    def open(cls, path): ...
    def __len__(self): ...
    def iter_shards(self):                      # -> yields list[MoleculeProfile]
        ...
    def iter_pair_shards(self, query, *, do_center=False, device="cpu"):
        # -> yields list[MoleculePair], ready for MoleculePairBatch
        for profiles in self.iter_shards():
            yield [MoleculePair(query, p, do_center=do_center, device=device)
                   for p in profiles]
```

Notes:
- **`dtype="float16"` halves disk + IO** (200×3 surface: 2.4 KB → 1.2 KB).
  fp16 coordinates over a ±20 Å box carry ~0.01 Å error — at the surface
  resampling noise floor (`POTENTIAL_FUTURE_WORK.md` §4), i.e. invisible to
  screening rankings. Reconstruction upcasts to fp32 for the kernels. Make it a
  flag; default fp16 for the store, fp32 in RAM.
  → 1B @ ~2 KB/mol (fp16, surface) = **~2 TB on disk**; ~4 TB fp32; ~0.5–1 TB
  volumetric-only. All trivially manageable as cluster storage.
- **`pre_centered=True`**: store each profile already centered to its own COM, so
  screening runs `do_center=False` and stays fully RDKit-free. (The store builder
  centers once at write time.)
- **Free padding win:** because `S` is constant, a surface shard slice
  `surf_pos[a:b]` is *already* the `(K, S, 3)` shape the surf/surf_esp kernels want —
  surface/surf_esp screening can feed the kernel with zero re-padding. Vol/pharm stay
  ragged and pad per shard as today.

### 3.3 `screen()` — the high-level streaming driver

Ties it together: stream shards → align each with the existing batch API →
reduce to a running top-K → discard. Optionally persists the full score vector to
a memmapped file (1B float32 = 4 GB, fine on disk; transforms only kept for the
top-K, since 1B×4×4 fp32 = 64 GB is too much to retain).

```python
def screen(query, store, mode="surf_esp", *, backend=None, do_center=None,
           top_k=1000, ndev=None, scores_out=None, alpha=None,
           progress=False, **align_kwargs):
    """Stream `store` past `query`, aligning each shard with `mode`, returning the
    top_k Hit(score, id, transform). backend=None -> auto ("triton" on CUDA, else
    "numba"). For a pre-centered store the query is centered once and profiles run
    do_center=False; alpha auto-fills ALPHA(num_surf_points) for surf/surf_esp. The rest
    of align_kwargs (lam, num_repeats, similarity, ...) pass straight to
    align_with_<mode>."""
    heap = TopK(top_k)                                  # (score, id, transform)
    base = 0
    for profiles in store.iter_shards():
        pairs = [MoleculePair(q, p, do_center=False) for p in profiles]   # q = centered query
        if ndev and ndev > 1:
            scores, transforms = align_multi_gpu(pairs, mode, ndev=ndev,
                                                 backend=backend, do_center=False, **align_kwargs)
        else:
            batch = MoleculePairBatch(pairs)
            result = getattr(batch, "align_with_" + mode)(backend=backend, **align_kwargs)
            scores = result[0]                          # [0] is scores for 2- and 3-tuple returns
            transforms = [getattr(p, transform_attr(mode)) for p in pairs]
        if scores_out is not None:
            scores_out[base:base+len(scores)] = scores  # memmapped full vector
        for i, s in enumerate(scores):
            heap.offer(s, profiles[i].id, transforms[i])
        base += len(scores)
    return heap.sorted()
```

End-to-end usage:

```python
from shepherd_score.container import Molecule
from shepherd_score.screen import ProfileStore, screen
from shepherd_score.score.constants import ALPHA

# 1) BUILD THE STORE ONCE (parallel across the cluster; the expensive step —
#    see MOLECULE_CONSTRUCTION_PROBLEM.md). Each worker writes its own shards.
with ProfileStore.create("library.fss", num_surf_points=200,
                         modes=("surf", "surf_esp", "pharm"), dtype="float16") as store:
    for mol in library_rdkit_mols:                 # this worker's shard of the library
        m = Molecule(mol, num_surf_points=200, pharm_multi_vector=False)
        store.add(m)                               # arrays kept, RDKit Mol dropped

# 2) SCREEN (streamed; never holds the library in RAM)
query = Molecule(query_mol, num_surf_points=200, pharm_multi_vector=False)
store = ProfileStore.open("library.fss")            # one shard read at a time
hits  = screen(query, store, mode="surf_esp", alpha=ALPHA(200), lam=0.3,
               num_repeats=50, backend="triton", ndev=4, top_k=1000)
# hits: [Hit(score, id, transform), ...] sorted desc  (ndev>1 -> persistent shard pool)
```

Low-level escape hatch (reuse the public batch API directly):

```python
for pairs in store.iter_pair_shards(query, do_center=False):
    scores, _ = MoleculePairBatch(pairs).align_with_surf(alpha=ALPHA(200),
                                                         backend="triton")
    sink.update(scores)
```

## 4. Why this is minimally invasive

| Component | Change |
|---|---|
| `Molecule`, `MoleculePair`, `MoleculePairBatch` | **none** (duck-typed reuse) |
| aligners / Triton+numba kernels / `_subbatched_align` | **none** (GPU chunking already exists) |
| `multi_gpu.py` (`align_multi_gpu`, `MultiGPUAligner`) | **none** (consumes the pair list as-is) |
| `shepherd_score/screen.py` | **new file**: `MoleculeProfile`, `ProfileStore`, `screen()` |
| `container/__init__.py` | *optional* one-line re-export for discoverability |
| `Molecule.to_profile(modes=...)` | *optional* additive convenience method |

The streaming layer is pure host-side I/O + a reduce loop. It inherits multi-GPU
scaling, band-bucketing, sub-batch memory safety, and every backend (`triton` /
`numba` / `jax`) for free, because it feeds the **unchanged** public align API.

## 5. Reduction, queries, and resumability

> **Update (shipped):** the three performance bottlenecks below are now implemented in
> `screen.py`, beyond the original v1 loop. (1) A **fast direct path** for
> `vol/surf/surf_esp/pharm` on a pre-centered store loads each shard's fit arrays once as
> device tensors and feeds the batched aligner lightweight `_FastPair`s (fit = views,
> ref = shared query) — no per-molecule `MoleculeProfile`/`MoleculePair`. (2)
> **`screen_many(queries, ...)`** reads each shard once and aligns the whole query panel
> (shard outer loop, query inner). (3) **`ndev>1`** uses a persistent shard-parallel pool
> (one process per GPU, spawned once, shards pulled off a queue) instead of re-spawning
> per shard. The fast direct path is score-equivalent to the per-pair object path.

- **Top-K sink**: a bounded heap keyed on score; O(N log K) host work, negligible
  vs alignment. Keep the SE(3) transform only for retained hits.
- **Full scores (optional)**: memmap a `(N,) float32` file (`scores_out`), written
  per shard at its index range — 4 GB for 1B, restart-friendly.
- **Multiple queries** (e.g. hundreds of actives): `screen_many` makes the **shard** the
  outer loop and queries the inner loop, so each shard is read/built once and aligned
  against every query (amortizes IO + shard build over the panel).
- **Resumability**: shards are independent and indexed; record the last completed
  shard id so a killed screen resumes mid-library.

## 6. Phased implementation plan

*All steps below are implemented in [`shepherd_score/screen.py`](../shepherd_score/screen.py);
this records the build/validation order. `vol_and_surf_esp` (step 6) shipped via the conformer
shim of §3.1.*

1. **`MoleculeProfile` + duck-type test.** Build one from a real `Molecule`, wrap
   in `MoleculePair(query, profile, do_center=False)`, and assert
   `MoleculePairBatch([...]).align_with_{vol,surf,surf_esp,pharm}(backend="triton")`
   gives **bit-identical** scores to the all-`Molecule` path. This is the
   correctness gate; everything else is plumbing.
2. **`ProfileStore` writer/reader** (sharded `.npy`, CSR offsets, fp16 flag,
   `pre_centered`, mmap). Round-trip test: store→load→profile equals source arrays.
3. **`screen()` driver** with top-K sink + optional memmapped scores, single-GPU.
4. **Wire `ndev>1`** through `align_multi_gpu` per shard; confirm scaling.
5. **fp16 accuracy gate** (rank ρ / top-K overlap vs an fp32 store on a few-k set).
6. **vol_and_surf_esp extension**: store `centers_w_H` + pre-seed `_*_centers_w_H_t`.
7. Docs + a `scripts/stream_screen.py` example mirroring `scripts/docking_screen.py`.

## 7. Constraints (hard requirements)

- **Default behavior unchanged / bit-identical.** New module is opt-in; importing
  or not importing `screen` changes nothing about existing flows.
- **`ALPHA` calibration.** Keep `num_surf_points` in `[50, 400]` (the store records
  it in the manifest; the screen reuses `ALPHA(num_surf_points)`), since
  `score/constants.py::ALPHA` is `bounds_error=True`.
- **Pre-centering must be exact.** If `pre_centered=True`, center each profile to
  its own heavy-atom COM at build time (matching `MoleculePair(do_center=True)`
  semantics) so screened scores match the in-memory path.
- **The store is precompute-once.** Populating it is the real cost and a separate
  workstream — see [`MOLECULE_CONSTRUCTION_PROBLEM.md`](./MOLECULE_CONSTRUCTION_PROBLEM.md).
