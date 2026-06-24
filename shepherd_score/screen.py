"""Out-of-core *streaming* screening for libraries too large to hold in RAM.

A query (or a handful of queries) can be aligned against a library of effectively
unbounded size by (1) precomputing each library molecule's interaction profile
**once**, persisting it to a sharded on-disk store, and (2) streaming shards back
through the **existing** :class:`~shepherd_score.container.MoleculePairBatch` /
:func:`~shepherd_score.accel.multi_gpu.align_multi_gpu` API, reducing scores on
the fly. Only one shard is resident at a time, so host RAM never holds the whole
library.

Three pieces, all additive — nothing here changes ``Molecule`` / ``MoleculePair``
/ ``MoleculePairBatch`` / the kernels:

* :class:`MoleculeProfile` -- an RDKit-free, duck-typed stand-in for ``Molecule``
  holding only the numeric arrays the batched aligners read. Dropping the RDKit
  ``Mol`` (~66% of a ``Molecule``'s bytes) is what keeps shard reload at
  array-copy speed instead of reconstruction-bound.
* :class:`ProfileStore` -- a sharded, on-disk store of profiles. Shards are
  independent ``.npz`` files, so the (expensive) build parallelises across
  workers/nodes with no locking, and a screen streams them back one at a time.
* :func:`screen` -- the driver: stream shards past a query, align each with the
  unchanged batch API, reduce to a running top-K.

Why this works (verified against the code): the batched aligners read only numpy
arrays off ``ref_molec``/``fit_molec`` (``atom_pos``, ``surf_pos``, ``surf_esp``,
``partial_charges`` + ``_nonH_atoms_idx``, ``pharm_*``, ``radii``); the only mode
that touches ``.mol`` is ``esp_combo`` (with-H centers), handled here by a tiny
conformer shim. ``MoleculePair`` accepts any non-``Chem.Mol`` object verbatim, so
a :class:`MoleculeProfile` duck-types straight in. And the GPU path is *already*
internally streaming (band-bucketing + ``_subbatched_align``), so this layer is
pure host-side I/O + a reduce loop — no kernel or GPU-memory changes.

Example
-------
>>> from shepherd_score.container import Molecule
>>> from shepherd_score.screen import ProfileStore, screen
>>> # BUILD ONCE (parallel across the cluster; each worker writes its own shards)
>>> with ProfileStore.create("lib.fss", num_surf_points=200,
...                           modes=("surf", "esp", "pharm")) as store:
...     for mol in library_rdkit_mols:
...         store.add(Molecule(mol, num_surf_points=200, pharm_multi_vector=False))
>>> # SCREEN (streamed; never holds the library in RAM)
>>> query = Molecule(query_mol, num_surf_points=200, pharm_multi_vector=False)
>>> hits = screen(query, ProfileStore.open("lib.fss"), mode="esp",
...               lam=0.3, num_repeats=50, top_k=1000)   # backend auto: triton on GPU, numba on CPU
>>> hits[0].score, hits[0].id            # best match
"""
from __future__ import annotations

import copy
import heapq
import itertools
import json
import os
from collections import namedtuple
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

__all__ = ["MoleculeProfile", "ProfileStore", "screen", "screen_many", "Hit"]


# Per-mode result attributes written in-place by ``MoleculePairBatch.align_with_*``.
_SCORE_ATTR = {
    "vol": "sim_aligned_vol_noH", "vol_esp": "sim_aligned_vol_esp_noH",
    "surf": "sim_aligned_surf", "esp": "sim_aligned_esp",
    "esp_combo": "sim_aligned_esp_combo", "pharm": "sim_aligned_pharm",
    "vol_color": "sim_aligned_vol_color",
}
_TRANSFORM_ATTR = {
    "vol": "transform_vol_noH", "vol_esp": "transform_vol_esp_noH",
    "surf": "transform_surf", "esp": "transform_esp",
    "esp_combo": "transform_esp_combo", "pharm": "transform_pharm",
    "vol_color": "transform_vol_color",
}
_VALID_MODES = tuple(_SCORE_ATTR)
# Modes whose surface ``alpha`` should auto-default to ALPHA(num_surf_points).
_SURF_ALPHA_MODES = {"surf", "esp"}


Hit = namedtuple("Hit", ["score", "id", "transform"])


# --------------------------------------------------------------------------- #
# RDKit-free conformer shim (only the esp_combo aligner reads ``.mol``).
# --------------------------------------------------------------------------- #
class _ConformerShim:
    __slots__ = ("_p",)

    def __init__(self, p):
        self._p = p

    def GetPositions(self):
        return self._p


class _MolShim:
    """Minimal stand-in so ``profile.mol.GetConformer().GetPositions()`` returns
    the stored with-H atom centers -- the only RDKit access made by the batched
    ``esp_combo`` aligner. ``None`` for every other mode."""
    __slots__ = ("_c",)

    def __init__(self, centers_w_H):
        self._c = _ConformerShim(centers_w_H)

    def GetConformer(self, *args, **kwargs):
        return self._c


def _f32(a):
    # Always an independent C-contiguous float32 copy. `np.array(copy=True)` matters
    # on the reload path: an ascontiguousarray of an fp32 npz slice would be a *view*
    # into the shard's array, pinning the whole shard's npz buffers alive for as long
    # as any profile survives. A copy keeps each profile self-contained.
    return None if a is None else np.array(a, dtype=np.float32, order="C")


def _heavy_positions(m):
    """Strict-heavy (Z != 1) atom coordinates, ordered to match
    ``partial_charges[_nonH_atoms_idx]`` -- exactly the Gaussian centers the vol_esp
    aligner uses. Read from the with-H conformer when present (a ``Molecule``, or an
    esp_combo ``MoleculeProfile`` via its ``_MolShim``); falls back to ``atom_pos`` for a
    heavy-only profile (self-consistent only when ``Chem.RemoveHs`` kept no H). This is
    what lets vol_esp stream correctly when RemoveHs retains an H -- ``atom_pos`` is then
    the RemoveHs set, longer than and misaligned with the heavy charges."""
    mol = getattr(m, "mol", None)
    idx = getattr(m, "_nonH_atoms_idx", None)
    if mol is not None and idx is not None:
        return np.asarray(mol.GetConformer().GetPositions())[np.asarray(idx)]
    return np.asarray(m.atom_pos)


# --------------------------------------------------------------------------- #
# MoleculeProfile
# --------------------------------------------------------------------------- #
class MoleculeProfile:
    """Numeric-only, RDKit-free stand-in for :class:`~shepherd_score.container.Molecule`.

    Holds exactly the arrays the batched aligners read; ``.mol`` is ``None``
    (or a lightweight shim for ``esp_combo``). Duck-types into ``MoleculePair``
    unchanged, so it feeds the existing ``align_with_*`` API with no edits.

    Heavy-atom convention mirrors ``Molecule``: ``atom_pos`` is heavy atoms;
    ``partial_charges`` may be stored heavy (with an identity ``_nonH_atoms_idx``)
    or with-H (with a real index), matching whatever the store was built for.

    Scope: this is a duck-type for the **batched aligners** and :func:`screen` only —
    it deliberately omits ``Molecule``'s build-time fields (``density``,
    ``probe_radius``, ``pharm_multi_vector``, the RDKit ``Mol``) and is not a general
    ``Molecule`` replacement.
    """

    __slots__ = ("atom_pos", "atom_pos_noH", "surf_pos", "surf_esp", "partial_charges", "radii",
                 "_nonH_atoms_idx", "pharm_types", "pharm_ancs", "pharm_vecs",
                 "num_surf_points", "mol", "id")

    def __init__(self, *, atom_pos, surf_pos=None, surf_esp=None,
                 partial_charges=None, radii=None, nonH_atoms_idx=None,
                 pharm_types=None, pharm_ancs=None, pharm_vecs=None,
                 centers_w_H=None, atom_pos_noH=None, id=None):
        self.atom_pos = _f32(atom_pos)
        # Strict-heavy vol_esp centers (1:1 with the heavy charges); None when identical to
        # atom_pos (RemoveHs kept no H) -- callers then use atom_pos.
        self.atom_pos_noH = _f32(atom_pos_noH)
        self.surf_pos = _f32(surf_pos)
        self.surf_esp = _f32(surf_esp)
        self.partial_charges = _f32(partial_charges)
        self.radii = _f32(radii)
        if nonH_atoms_idx is not None:
            self._nonH_atoms_idx = np.asarray(nonH_atoms_idx, dtype=np.int64)
        elif self.partial_charges is not None:
            # heavy-atom charges stored already-indexed -> identity selection so
            # the aligner's ``partial_charges[_nonH_atoms_idx]`` is a no-op.
            self._nonH_atoms_idx = np.arange(len(self.partial_charges), dtype=np.int64)
        else:
            self._nonH_atoms_idx = None
        self.pharm_types = None if pharm_types is None else np.asarray(pharm_types)
        self.pharm_ancs = _f32(pharm_ancs)
        self.pharm_vecs = _f32(pharm_vecs)
        self.num_surf_points = None if self.surf_pos is None else len(self.surf_pos)
        self.mol = _MolShim(_f32(centers_w_H)) if centers_w_H is not None else None
        self.id = id

    def center_to(self, xyz_means) -> None:
        """RDKit-free counterpart of ``Molecule.center_to``: shift the equivariant
        arrays only (no conformer transform)."""
        mu = np.asarray(xyz_means, dtype=np.float32)
        self.atom_pos = self.atom_pos - mu
        if self.atom_pos_noH is not None:
            self.atom_pos_noH = self.atom_pos_noH - mu
        if self.surf_pos is not None:
            self.surf_pos = self.surf_pos - mu
        if self.pharm_ancs is not None:
            self.pharm_ancs = self.pharm_ancs - mu
        if self.mol is not None:
            self.mol = _MolShim(self.mol.GetConformer().GetPositions() - mu)

    @classmethod
    def from_molecule(cls, m, *, modes=_VALID_MODES, id=None) -> "MoleculeProfile":
        """Extract the arrays the requested ``modes`` need from a ``Molecule``
        (or another ``MoleculeProfile``); the RDKit ``Mol`` is dropped."""
        sch = _schema_from_modes(modes)
        return _profile_from_schema(m, sch, id=id, pre_center=False)


# --------------------------------------------------------------------------- #
# storage schema helpers
# --------------------------------------------------------------------------- #
def _schema_from_modes(modes) -> dict:
    modes = set(modes)
    unknown = modes - set(_VALID_MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}; valid: {list(_VALID_MODES)}")
    return dict(
        surf=bool({"surf", "esp", "esp_combo"} & modes),
        surf_esp=bool({"esp", "esp_combo"} & modes),
        charges=bool({"vol_esp", "esp_combo"} & modes),
        with_H=("esp_combo" in modes),
        radii=("esp_combo" in modes),
        centers_w_H=("esp_combo" in modes),
        pharm=bool({"pharm", "vol_color"} & modes),   # vol_color = atoms + directionless pharm
    )


def _store_supports(schema: dict, mode: str) -> bool:
    if mode == "vol":
        return True                                   # atom_pos is always stored
    if mode == "vol_esp":
        return schema["charges"]
    if mode == "surf":
        return schema["surf"]
    if mode == "esp":
        return schema["surf"] and schema["surf_esp"]
    if mode == "pharm":
        return schema["pharm"]
    if mode == "vol_color":
        return schema["pharm"]                          # atoms (always) + pharm types/anchors
    if mode == "esp_combo":
        return (schema["surf"] and schema["surf_esp"] and schema["centers_w_H"]
                and schema["radii"] and schema["charges"] and schema["with_H"])
    return False


def _profile_from_schema(m, sch: dict, *, id, pre_center: bool) -> "MoleculeProfile":
    """Pull the schema's arrays off a ``Molecule``/``MoleculeProfile`` ``m``,
    optionally centering to the heavy-atom COM. Returns a ``MoleculeProfile``."""
    atom_pos = _f32(m.atom_pos)
    surf = surf_esp = charges = nonH = radii = cwh = None
    atom_pos_noH = None
    ph_t = ph_a = ph_v = None

    if sch["surf"]:
        if m.surf_pos is None:
            raise ValueError("store needs surfaces but molecule has none "
                             "(build Molecule with num_surf_points / surface_points)")
        surf = _f32(m.surf_pos)
    if sch["surf_esp"]:
        if m.surf_esp is None:
            raise ValueError("store needs surface ESP but molecule has none")
        surf_esp = _f32(m.surf_esp)
    if sch["charges"]:
        if m.partial_charges is None:
            raise ValueError("store needs partial charges but molecule has none")
        if sch["with_H"]:
            charges = _f32(m.partial_charges)
            nonH = np.asarray(m._nonH_atoms_idx, dtype=np.int64)
        else:
            # heavy charges. Index by _nonH_atoms_idx universally: it is the real
            # heavy index for a Molecule (full charges) and the identity for a
            # heavy MoleculeProfile, so both -- and a with-H profile -- reduce correctly.
            charges = _f32(np.asarray(m.partial_charges)[m._nonH_atoms_idx])
        # Heavy Gaussian centers for vol_esp, 1:1 with the heavy charges. Kept only when it
        # actually differs from atom_pos (i.e. RemoveHs retained an H); else atom_pos serves
        # and nothing extra is stored.
        hp = np.asarray(_heavy_positions(m), dtype=np.float32)
        if hp.shape != atom_pos.shape or not np.array_equal(hp, atom_pos):
            atom_pos_noH = hp
    if sch["radii"]:
        if m.radii is None:
            raise ValueError("store needs vdW radii but molecule has none")
        radii = _f32(m.radii)
    if sch["centers_w_H"]:
        cwh = _f32(m.mol.GetConformer().GetPositions())
    if sch["pharm"]:
        if m.pharm_types is None:
            raise ValueError("store needs pharmacophores but molecule has none "
                             "(build Molecule with pharm_multi_vector set)")
        ph_t = np.asarray(m.pharm_types, dtype=np.int32)
        ph_a = _f32(m.pharm_ancs)
        ph_v = _f32(m.pharm_vecs)

    if pre_center:
        mu = atom_pos.mean(0)
        atom_pos = atom_pos - mu
        if surf is not None:
            surf = surf - mu
        if ph_a is not None:
            ph_a = ph_a - mu
        if cwh is not None:
            cwh = cwh - mu
        if atom_pos_noH is not None:
            atom_pos_noH = atom_pos_noH - mu   # shift by the atom_pos COM (matches the
                                               # in-memory conformer transform, not its own COM)

    return MoleculeProfile(
        atom_pos=atom_pos, surf_pos=surf, surf_esp=surf_esp, partial_charges=charges,
        radii=radii, nonH_atoms_idx=nonH, pharm_types=ph_t, pharm_ancs=ph_a,
        pharm_vecs=ph_v, centers_w_H=cwh, atom_pos_noH=atom_pos_noH, id=id,
    )


def _id_to_py(x):
    """Numpy scalar -> python int/str for clean Hit ids."""
    if isinstance(x, np.generic):
        return x.item()
    return x


# --------------------------------------------------------------------------- #
# ProfileStore
# --------------------------------------------------------------------------- #
class ProfileStore:
    """Sharded, on-disk store of :class:`MoleculeProfile` arrays.

    Layout::

        <path>/
        ├── manifest.json          # schema, num_surf_points, dtype, shard list
        ├── shard_00000.npz        # concatenated arrays + CSR offsets for one shard
        └── shard_00001.npz ...

    Write once (``create`` -> ``add`` -> ``close``), then ``open`` and stream.
    Shards are independent, so writers can run in parallel (each producing its own
    shards) and a partially-written store is resumable.
    """

    MANIFEST = "manifest.json"
    VERSION = 1

    def __init__(self, path, manifest, rw):
        self.path = path
        self.manifest = manifest
        self._rw = rw
        # writer state
        self._buf: List[MoleculeProfile] = []
        self._n = int(manifest.get("n_total", 0))
        self._shard_id = len(manifest.get("shards", []))

    # ---- writer ---------------------------------------------------------- #
    @classmethod
    def create(cls, path, *, num_surf_points: int, modes: Sequence[str],
               dtype: str = "float16", shard_size: int = 100_000,
               pre_centered: bool = True, overwrite: bool = False) -> "ProfileStore":
        """Open a store for writing.

        Parameters
        ----------
        num_surf_points : int
            Surface point count every profile uses (must match the query at screen
            time; kept in ``[50, 400]`` for ``ALPHA`` calibration).
        modes : sequence of str
            Which alignment modes this store must support. Only the arrays those
            modes need are stored (``"vol"`` works from any store -- ``atom_pos``
            is always kept). Valid: ``vol vol_esp surf esp pharm esp_combo``.
        dtype : {"float16", "float32"}
            On-disk dtype for coordinate/charge arrays. ``float16`` halves disk +
            IO at ~0.01 A error (the surface resampling noise floor); reconstructed
            to float32 in RAM. Default ``"float16"``.
        shard_size : int
            Molecules per shard file. Default 100k.
        pre_centered : bool
            If True (default) each profile is centered to its own heavy-atom COM at
            write time, so a screen runs ``do_center=False`` and stays RDKit-free
            while matching ``MoleculePair(do_center=True)`` global-alignment semantics.
        overwrite : bool
            If True, delete any existing shards + manifest in ``path`` first.

        Notes
        -----
        A store directory is **single-writer**: ``create`` owns ``manifest.json`` and
        the ``shard_*.npz`` sequence. For a parallel cluster build, give each worker its
        **own** store directory (``lib.part0.fss``, ``lib.part1.fss``, ...) and screen
        across them (shards are independent — iterate the parts and merge the per-part
        ``Hit`` lists). Do not point multiple concurrent writers at one directory.
        """
        if dtype not in ("float16", "float32"):
            raise ValueError("dtype must be 'float16' or 'float32'")
        modes = tuple(modes)
        schema = _schema_from_modes(modes)
        os.makedirs(path, exist_ok=True)
        manifest_path = os.path.join(path, cls.MANIFEST)
        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists; pass overwrite=True to replace it")
        if overwrite:
            for f in os.listdir(path):
                if f.endswith(".npz") or f == cls.MANIFEST:
                    os.remove(os.path.join(path, f))
        manifest = dict(version=cls.VERSION, num_surf_points=int(num_surf_points),
                        modes=list(modes), schema=schema, dtype=dtype,
                        shard_size=int(shard_size), pre_centered=bool(pre_centered),
                        n_total=0, shards=[])
        store = cls(path, manifest, "w")
        store._write_manifest()
        return store

    def add(self, molecule, id=None) -> None:
        """Buffer one ``Molecule`` (or ``MoleculeProfile``); flushes a shard
        automatically every ``shard_size`` molecules. ``id`` defaults to the
        molecule's global position in the store."""
        if self._rw != "w":
            raise RuntimeError("store is open for reading")
        if id is None:
            id = self._n + len(self._buf)
        prof = _profile_from_schema(molecule, self.manifest["schema"], id=id,
                                    pre_center=self.manifest["pre_centered"])
        self._buf.append(prof)
        if len(self._buf) >= self.manifest["shard_size"]:
            self._flush()

    def add_profile(self, profile: "MoleculeProfile", id=None) -> None:
        """Add a pre-built :class:`MoleculeProfile` directly (no extraction)."""
        if not isinstance(profile, MoleculeProfile):
            raise TypeError("add_profile expects a MoleculeProfile")
        self.add(profile, id=id if id is not None else profile.id)

    def close(self) -> None:
        if self._rw == "w":
            self._flush()
            self._write_manifest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write_manifest(self) -> None:
        with open(os.path.join(self.path, self.MANIFEST), "w") as fh:
            json.dump(self.manifest, fh, indent=2)

    def _flush(self) -> None:
        if not self._buf:
            return
        name = f"shard_{self._shard_id:05d}.npz"
        arrs = self._concat(self._buf)
        np.savez(os.path.join(self.path, name), **arrs)
        self.manifest["shards"].append(dict(name=name, n=len(self._buf), start=self._n))
        self._n += len(self._buf)
        self.manifest["n_total"] = self._n
        self._buf = []
        self._shard_id += 1
        self._write_manifest()

    def _concat(self, recs: List["MoleculeProfile"]) -> dict:
        sch = self.manifest["schema"]
        dt = np.float16 if self.manifest["dtype"] == "float16" else np.float32

        def offsets(lengths):
            return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

        out = {"ids": np.array([r.id for r in recs])}
        atom_lens = [len(r.atom_pos) for r in recs]
        out["atom_off"] = offsets(atom_lens)
        out["atom_pos"] = np.concatenate([r.atom_pos for r in recs]).astype(dt)

        if sch["surf"]:
            out["surf_pos"] = np.stack([r.surf_pos for r in recs]).astype(dt)
        if sch["surf_esp"]:
            out["surf_esp"] = np.stack([r.surf_esp for r in recs]).astype(dt)
        if sch["charges"]:
            if sch["with_H"]:
                all_lens = [len(r.partial_charges) for r in recs]
                out["all_off"] = offsets(all_lens)
                out["charges"] = np.concatenate([r.partial_charges for r in recs]).astype(dt)
                out["nonH"] = np.concatenate([r._nonH_atoms_idx for r in recs]).astype(np.int64)
                if sch["radii"]:
                    out["radii"] = np.concatenate([r.radii for r in recs]).astype(dt)
                if sch["centers_w_H"]:
                    out["cwh"] = np.concatenate(
                        [r.mol.GetConformer().GetPositions() for r in recs]).astype(dt)
                heavy_lens = [len(r._nonH_atoms_idx) for r in recs]
            else:
                out["charges"] = np.concatenate([r.partial_charges for r in recs]).astype(dt)
                heavy_lens = [len(r.partial_charges) for r in recs]
            # vol_esp needs heavy centers 1:1 with the heavy charges. When a molecule's
            # RemoveHs kept an H, atom_pos/atom_off no longer match the heavy set, so the heavy
            # charges (and with-H nonH index) can't be split by atom_off -- persist explicit
            # heavy offsets + the strict-heavy centers. Emitted only when some atom_pos_noH
            # exists, so non-retained-H stores are byte-for-byte unchanged (and legacy stores,
            # which lack these keys, fall back to atom_off in the reader).
            if any(r.atom_pos_noH is not None for r in recs):
                out["heavy_off"] = offsets(heavy_lens)
                out["xyz_noH"] = np.concatenate(
                    [(r.atom_pos_noH if r.atom_pos_noH is not None else r.atom_pos)
                     for r in recs]).astype(dt)
        if sch["pharm"]:
            ph_lens = [len(r.pharm_types) for r in recs]
            out["pharm_off"] = offsets(ph_lens)
            out["pharm_types"] = np.concatenate([r.pharm_types for r in recs]).astype(np.int32)
            out["pharm_ancs"] = np.concatenate([r.pharm_ancs for r in recs]).astype(dt)
            out["pharm_vecs"] = np.concatenate([r.pharm_vecs for r in recs]).astype(dt)
        return out

    # ---- reader ---------------------------------------------------------- #
    @classmethod
    def open(cls, path) -> "ProfileStore":
        with open(os.path.join(path, cls.MANIFEST)) as fh:
            manifest = json.load(fh)
        return cls(path, manifest, "r")

    def __len__(self) -> int:
        return int(self.manifest["n_total"])

    @property
    def num_surf_points(self) -> int:
        return int(self.manifest["num_surf_points"])

    @property
    def modes(self) -> tuple:
        return tuple(self.manifest["modes"])

    @property
    def schema(self) -> dict:
        return self.manifest["schema"]

    @property
    def pre_centered(self) -> bool:
        return bool(self.manifest["pre_centered"])

    def supports(self, mode: str) -> bool:
        return _store_supports(self.manifest["schema"], mode)

    @property
    def num_shards(self) -> int:
        return len(self.manifest["shards"])

    def _load_raw(self, sh) -> dict:
        """Materialize one shard's arrays into a plain ``{name: np.ndarray}`` dict
        (npz closed before return). Cheaper than :meth:`iter_shards` -- it skips the
        per-molecule ``MoleculeProfile`` split, which the fast screen path does on
        the GPU/device side instead."""
        with np.load(os.path.join(self.path, sh["name"])) as data:
            return {k: data[k] for k in data.files}

    def iter_shard_raw(self) -> Iterator[tuple]:
        """Yield ``(shard_meta, arrays_dict)`` per shard -- raw contiguous arrays for
        the direct device path (no per-molecule objects)."""
        for sh in self.manifest["shards"]:
            yield sh, self._load_raw(sh)

    def read_shard(self, idx: int) -> tuple:
        """Return ``(shard_meta, arrays_dict)`` for shard ``idx`` (random access; used
        by the multi-GPU shard pool so each worker reads only its assigned shards)."""
        sh = self.manifest["shards"][idx]
        return sh, self._load_raw(sh)

    def iter_shards(self) -> Iterator[List["MoleculeProfile"]]:
        """Yield one shard at a time as a ``list[MoleculeProfile]``."""
        for sh in self.manifest["shards"]:
            with np.load(os.path.join(self.path, sh["name"])) as data:
                yield self._reconstruct(data, sh)

    def iter_pair_shards(self, query, *, do_center=False, device="cpu"):
        """Yield one shard at a time as a ``list[MoleculePair]`` (query vs each
        profile), ready for :class:`MoleculePairBatch`."""
        from shepherd_score.container import MoleculePair
        for profiles in self.iter_shards():
            yield [MoleculePair(query, p, do_center=do_center, device=device)
                   for p in profiles]

    def read_profiles(self, idx: int) -> List["MoleculeProfile"]:
        """Reconstruct shard ``idx`` as ``list[MoleculeProfile]`` (random access)."""
        sh, arrs = self.read_shard(idx)
        return self._reconstruct(arrs, sh)

    def _reconstruct(self, data, sh) -> List["MoleculeProfile"]:
        sch = self.manifest["schema"]
        n = sh["n"]
        files = set(data.files) if hasattr(data, "files") else set(data.keys())
        atom_off = data["atom_off"]
        atom_pos = data["atom_pos"]
        ids = data["ids"]
        surf_pos = data["surf_pos"] if sch["surf"] else None
        surf_esp = data["surf_esp"] if sch["surf_esp"] else None
        all_off = data["all_off"] if (sch["charges"] and sch["with_H"]) else None
        pharm_off = data["pharm_off"] if sch["pharm"] else None

        out = []
        for i in range(n):
            a0, a1 = int(atom_off[i]), int(atom_off[i + 1])
            kw = dict(atom_pos=atom_pos[a0:a1], id=_id_to_py(ids[i]))
            if sch["surf"]:
                kw["surf_pos"] = surf_pos[i]
            if sch["surf_esp"]:
                kw["surf_esp"] = surf_esp[i]
            if sch["charges"]:
                if sch["with_H"]:
                    c0, c1 = int(all_off[i]), int(all_off[i + 1])
                    kw["partial_charges"] = data["charges"][c0:c1]
                    kw["nonH_atoms_idx"] = data["nonH"][a0:a1]
                    if sch["radii"] and "radii" in files:
                        kw["radii"] = data["radii"][c0:c1]
                    if sch["centers_w_H"] and "cwh" in files:
                        kw["centers_w_H"] = data["cwh"][c0:c1]
                else:
                    kw["partial_charges"] = data["charges"][a0:a1]
            if sch["pharm"]:
                p0, p1 = int(pharm_off[i]), int(pharm_off[i + 1])
                kw["pharm_types"] = data["pharm_types"][p0:p1]
                kw["pharm_ancs"] = data["pharm_ancs"][p0:p1]
                kw["pharm_vecs"] = data["pharm_vecs"][p0:p1]
            out.append(MoleculeProfile(**kw))
        return out


# --------------------------------------------------------------------------- #
# screen()
# --------------------------------------------------------------------------- #
def _default_backend() -> str:
    try:
        import torch
        return "triton" if torch.cuda.is_available() else "numba"
    except Exception:
        return "numba"


def _transform_of(pair, tf_attr):
    t = getattr(pair, tf_attr, None)
    if t is None:
        return None
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(t)


def _centered_copy(query):
    q = copy.deepcopy(query)
    q.center_to(q.atom_pos.mean(0))
    return q


# --------------------------------------------------------------------------- #
# Fast engine: contiguous store arrays -> device tensors -> batched aligner,
# bypassing the per-molecule MoleculeProfile/MoleculePair objects. The batched
# ``_align_batch_{vol,surf,esp,pharm}`` only read cached ``_*_t`` tensors (never
# ``.ref_molec``) when those are pre-set and trans_init=False, so a lightweight
# ``_FastPair`` whose FIT tensors are *views* into one device-resident shard tensor
# and whose REF tensors are the *shared* query feeds them directly. The query (ref)
# is swapped across a panel while the fit views stay resident -> one shard build
# serves every query.
# --------------------------------------------------------------------------- #
_FAST_MODES = ("vol", "surf", "esp", "pharm", "vol_color", "vol_esp", "esp_combo")
# Modes whose batched aligner reads ``p.ref_molec.<attr>`` *eagerly* (vol_color probes
# pharm; vol_esp/esp_combo do a None-guard on partial_charges / surf_pos+surf_esp), so the
# fast path attaches a tiny ``_ArrView`` to satisfy those reads (tensors are still pre-cached).
_NEEDS_ARRVIEW = ("vol_color", "vol_esp", "esp_combo")


class _ArrView:
    """Minimal numpy-array holder so the eager ``p.ref_molec.<attr>`` reads in the
    vol_color (and esp-family) aligners succeed without a full ``Molecule``. The
    ``_*_t`` tensors are pre-set, so these arrays are only read, never re-converted."""
    __slots__ = ("atom_pos", "pharm_types", "pharm_ancs",
                 "partial_charges", "surf_pos", "surf_esp", "radii", "mol")

    def __init__(self, atom_pos=None, pharm_types=None, pharm_ancs=None,
                 partial_charges=None, surf_pos=None, surf_esp=None,
                 radii=None, mol=None):
        self.atom_pos = atom_pos
        self.pharm_types = pharm_types
        self.pharm_ancs = pharm_ancs
        self.partial_charges = partial_charges   # vol_esp/esp_combo None-guard
        self.surf_pos = surf_pos                 # esp_combo None-guard
        self.surf_esp = surf_esp                 # esp_combo None-guard
        self.radii = radii                       # esp_combo _ensure source (eagerly read)
        self.mol = mol                           # esp_combo: _MolShim for centers_w_H


class _FastPair:
    """Cached-tensor-only stand-in a batched aligner can consume. ``ref_molec`` /
    ``fit_molec`` are set only for modes whose aligner reads them eagerly (vol_color)."""
    __slots__ = ("device", "ref_molec", "fit_molec",
                 "_ref_xyz_t", "_fit_xyz_t",
                 "_ref_surf_t", "_fit_surf_t",
                 "_ref_surf_esp_t", "_fit_surf_esp_t",
                 "_ref_pharm_types_t", "_fit_pharm_types_t",
                 "_ref_pharm_ancs_t", "_fit_pharm_ancs_t",
                 "_ref_pharm_vecs_t", "_fit_pharm_vecs_t",
                 "_ref_xyz_esp_t", "_fit_xyz_esp_t",                 # vol_esp heavy charges
                 "_ref_xyz_noH_t", "_fit_xyz_noH_t",                 # vol_esp heavy-atom centers (array-sourced; no .mol here)
                 "_ref_centers_w_H_t", "_fit_centers_w_H_t",         # esp_combo with-H centers
                 "_ref_partial_t", "_fit_partial_t",                 # esp_combo with-H charges
                 "_ref_radii_t", "_fit_radii_t",                     # esp_combo with-H radii
                 "transform_vol_noH", "sim_aligned_vol_noH",
                 "transform_surf", "sim_aligned_surf",
                 "transform_esp", "sim_aligned_esp",
                 "transform_pharm", "sim_aligned_pharm",
                 "transform_vol_color", "sim_aligned_vol_color",
                 "transform_vol_esp_noH", "sim_aligned_vol_esp_noH",
                 "transform_esp_combo", "sim_aligned_esp_combo")

    def __init__(self, device):
        self.device = device


def _query_ref_arrays(q, mode: str) -> dict:
    """The (centered) query's numpy arrays needed as the reference for ``mode``.
    Plain numpy so it is cheap to ship to multi-GPU workers."""
    if mode == "vol":
        return {"xyz": np.asarray(q.atom_pos, np.float32)}
    if mode == "surf":
        return {"surf": np.asarray(q.surf_pos, np.float32)}
    if mode == "esp":
        return {"surf": np.asarray(q.surf_pos, np.float32),
                "surf_esp": np.asarray(q.surf_esp, np.float32)}
    if mode == "pharm":
        return {"ptypes": np.asarray(q.pharm_types), "pancs": np.asarray(q.pharm_ancs, np.float32),
                "pvecs": np.asarray(q.pharm_vecs, np.float32)}
    if mode == "vol_color":
        return {"xyz": np.asarray(q.atom_pos, np.float32),
                "ptypes": np.asarray(q.pharm_types), "pancs": np.asarray(q.pharm_ancs, np.float32)}
    if mode == "vol_esp":
        # Strict-heavy centers (from the with-H conformer) 1:1 with the heavy charges -- NOT
        # atom_pos, which is the RemoveHs set and longer when an H was retained.
        return {"xyz": np.asarray(_heavy_positions(q), np.float32),
                "charges": np.asarray(np.asarray(q.partial_charges)[q._nonH_atoms_idx], np.float32)}
    if mode == "esp_combo":
        return {"surf": np.asarray(q.surf_pos, np.float32),
                "surf_esp": np.asarray(q.surf_esp, np.float32),
                "cwh": np.asarray(q.mol.GetConformer().GetPositions(), np.float32),  # with-H centers
                "partial": np.asarray(q.partial_charges, np.float32),                # with-H charges
                "radii": np.asarray(q.radii, np.float32),                            # with-H radii
                "xyz": np.asarray(q.atom_pos, np.float32)}                           # heavy (alpha=0.81 shape)
    raise ValueError(mode)


def _ref_tensors_from_arrays(ra: dict, mode: str, device) -> dict:
    import torch
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    if mode == "vol":
        return {"_ref_xyz_t": f(ra["xyz"])}
    if mode == "surf":
        return {"_ref_surf_t": f(ra["surf"])}
    if mode == "esp":
        return {"_ref_surf_t": f(ra["surf"]), "_ref_surf_esp_t": f(ra["surf_esp"])}
    if mode == "pharm":
        return {"_ref_pharm_types_t": torch.as_tensor(ra["ptypes"], dtype=torch.int64, device=device),
                "_ref_pharm_ancs_t": f(ra["pancs"]), "_ref_pharm_vecs_t": f(ra["pvecs"])}
    if mode == "vol_color":
        return {"_ref_xyz_t": f(ra["xyz"]),
                "_ref_pharm_types_t": torch.as_tensor(ra["ptypes"], dtype=torch.int64, device=device),
                "_ref_pharm_ancs_t": f(ra["pancs"]),
                "ref_molec": _ArrView(ra["xyz"], ra["ptypes"], ra["pancs"])}
    if mode == "vol_esp":
        # ra["xyz"] is the strict-heavy centers (1:1 with the heavy charges; see
        # _query_ref_arrays). Pre-set _ref_xyz_noH_t so the aligner reads it instead of
        # dereferencing .mol (absent on this array-only path).
        xyz = f(ra["xyz"])
        return {"_ref_xyz_t": xyz, "_ref_xyz_noH_t": xyz, "_ref_xyz_esp_t": f(ra["charges"]),
                "ref_molec": _ArrView(atom_pos=ra["xyz"], partial_charges=ra["charges"])}
    if mode == "esp_combo":
        return {"_ref_surf_t": f(ra["surf"]), "_ref_surf_esp_t": f(ra["surf_esp"]),
                "_ref_centers_w_H_t": f(ra["cwh"]), "_ref_partial_t": f(ra["partial"]),
                "_ref_radii_t": f(ra["radii"]), "_ref_xyz_t": f(ra["xyz"]),
                "ref_molec": _ArrView(surf_pos=ra["surf"], surf_esp=ra["surf_esp"],
                                      partial_charges=ra["partial"], radii=ra["radii"],
                                      mol=_MolShim(ra["cwh"]))}
    raise ValueError(mode)


def _build_fit_fast_pairs(arrs: dict, mode: str, device):
    """Load one shard's FIT arrays as device tensors once; return (ids, [_FastPair])
    whose fit tensors are views into them."""
    import torch
    ids = arrs["ids"]
    K = len(ids)
    pairs = [_FastPair(device) for _ in range(K)]
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    if mode == "vol":
        big = f(arrs["atom_pos"]); off = arrs["atom_off"]
        for i, p in enumerate(pairs):
            p._fit_xyz_t = big[int(off[i]):int(off[i + 1])]
    elif mode == "surf":
        big = f(arrs["surf_pos"])                       # (K, S, 3)
        for i, p in enumerate(pairs):
            p._fit_surf_t = big[i]
    elif mode == "esp":
        bs = f(arrs["surf_pos"]); be = f(arrs["surf_esp"])
        for i, p in enumerate(pairs):
            p._fit_surf_t = bs[i]; p._fit_surf_esp_t = be[i]
    elif mode == "pharm":
        bt = torch.as_tensor(arrs["pharm_types"], dtype=torch.int64, device=device)
        ba = f(arrs["pharm_ancs"]); bv = f(arrs["pharm_vecs"]); off = arrs["pharm_off"]
        for i, p in enumerate(pairs):
            a, b = int(off[i]), int(off[i + 1])
            p._fit_pharm_types_t = bt[a:b]; p._fit_pharm_ancs_t = ba[a:b]; p._fit_pharm_vecs_t = bv[a:b]
    elif mode == "vol_color":
        big = f(arrs["atom_pos"]); aoff = arrs["atom_off"]
        bt = torch.as_tensor(arrs["pharm_types"], dtype=torch.int64, device=device)
        ba = f(arrs["pharm_ancs"]); poff = arrs["pharm_off"]
        np_atom, np_pt, np_pa = arrs["atom_pos"], arrs["pharm_types"], arrs["pharm_ancs"]
        for i, p in enumerate(pairs):
            a0, a1 = int(aoff[i]), int(aoff[i + 1])
            p0, p1 = int(poff[i]), int(poff[i + 1])
            p._fit_xyz_t = big[a0:a1]
            p._fit_pharm_types_t = bt[p0:p1]; p._fit_pharm_ancs_t = ba[p0:p1]
            # numpy holder for the aligner's eager p.fit_molec.<attr> reads
            p.fit_molec = _ArrView(np_atom[a0:a1], np_pt[p0:p1], np_pa[p0:p1])
    elif mode == "vol_esp":
        big = f(arrs["atom_pos"]); aoff = arrs["atom_off"]
        # heavy_off + xyz_noH exist only when some molecule's RemoveHs retained an H; else
        # atom_off/atom_pos already are the heavy set (so this is a no-op slice for the common
        # case and back-compat for legacy stores that predate these keys).
        hoff = arrs["heavy_off"] if "heavy_off" in arrs else aoff
        xnoH = f(arrs["xyz_noH"]) if "xyz_noH" in arrs else big
        if "all_off" in arrs:                       # with-H store: heavy charges = charges[all_off][nonH]
            whc = arrs["charges"]; alloff = arrs["all_off"]; nonH = arrs["nonH"]
            for i, p in enumerate(pairs):
                a0, a1 = int(aoff[i]), int(aoff[i + 1])
                h0, h1 = int(hoff[i]), int(hoff[i + 1])
                heavy = whc[int(alloff[i]):int(alloff[i + 1])][nonH[h0:h1]]
                p._fit_xyz_esp_t = f(heavy); p.fit_molec = _ArrView(partial_charges=heavy)
                p._fit_xyz_noH_t = xnoH[h0:h1]          # heavy centers, 1:1 with heavy charges
                p._fit_xyz_t = big[a0:a1]               # RemoveHs atom_pos (trans-init centers only)
        else:                                        # heavy charges stored directly
            chg = arrs["charges"]
            for i, p in enumerate(pairs):
                a0, a1 = int(aoff[i]), int(aoff[i + 1])
                h0, h1 = int(hoff[i]), int(hoff[i + 1])
                p._fit_xyz_esp_t = f(chg[h0:h1]); p.fit_molec = _ArrView(partial_charges=chg[h0:h1])
                p._fit_xyz_noH_t = xnoH[h0:h1]          # heavy centers, 1:1 with heavy charges
                p._fit_xyz_t = big[a0:a1]               # RemoveHs atom_pos (trans-init centers only)
    elif mode == "esp_combo":
        bs = f(arrs["surf_pos"]); be = f(arrs["surf_esp"])     # (K, S, 3) / (K, S)
        cwh = f(arrs["cwh"]); part = f(arrs["charges"]); rad = f(arrs["radii"]); alloff = arrs["all_off"]
        big = f(arrs["atom_pos"]); aoff = arrs["atom_off"]
        np_surf, np_se, np_part = arrs["surf_pos"], arrs["surf_esp"], arrs["charges"]
        np_cwh, np_rad = arrs["cwh"], arrs["radii"]
        for i, p in enumerate(pairs):
            c0, c1 = int(alloff[i]), int(alloff[i + 1])
            a0, a1 = int(aoff[i]), int(aoff[i + 1])
            p._fit_surf_t = bs[i]; p._fit_surf_esp_t = be[i]
            p._fit_centers_w_H_t = cwh[c0:c1]; p._fit_partial_t = part[c0:c1]; p._fit_radii_t = rad[c0:c1]
            p._fit_xyz_t = big[a0:a1]            # heavy atoms (alpha=0.81 shape channel)
            p.fit_molec = _ArrView(surf_pos=np_surf[i], surf_esp=np_se[i], partial_charges=np_part[c0:c1],
                                   radii=np_rad[c0:c1], mol=_MolShim(np_cwh[c0:c1]))
    else:
        raise ValueError(mode)
    return ids, pairs


def _fast_batch_kwargs(mode: str, ak: dict) -> dict:
    """Translate screen()'s align_kwargs to the ``_align_batch_<mode>`` kwargs,
    mirroring the defaults the ``MoleculePairBatch.align_with_*`` triton path uses."""
    steps = ak.get("max_num_steps", 200)
    if mode in ("vol", "surf"):
        return dict(alpha=ak.get("alpha", 0.81), steps_fine=steps)
    if mode == "esp":
        return dict(alpha=ak.get("alpha", 0.81), lam=ak.get("lam", 0.3),
                    num_repeats=ak.get("num_repeats", 50), trans_init=False,
                    lr=ak.get("lr", 0.1), steps_fine=steps)
    if mode == "pharm":
        return dict(similarity=ak.get("similarity", "tanimoto"),
                    extended_points=ak.get("extended_points", False),
                    only_extended=ak.get("only_extended", False), trans_init=False,
                    num_repeats=ak.get("num_repeats", 50), steps_fine=steps, lr=ak.get("lr", 0.1))
    if mode == "vol_color":
        return dict(alpha=ak.get("alpha", 0.81), color_weight=ak.get("color_weight", 0.5),
                    trans_init=False, num_repeats=ak.get("num_repeats", 50),
                    steps_fine=steps, lr=ak.get("lr", 0.1))
    if mode == "vol_esp":   # mirrors align_with_vol_esp(backend="triton") dispatch
        return dict(alpha=ak.get("alpha", 0.81), lam=ak["lam"],
                    num_repeats=ak.get("num_repeats", 50), trans_init=False,
                    lr=ak.get("lr", 0.1), steps_fine=steps)
    if mode == "esp_combo":  # mirrors align_with_esp_combo(backend="triton") dispatch
        return dict(alpha=ak["alpha"], lam=ak.get("lam", 0.001),
                    probe_radius=ak.get("probe_radius", 1.0), esp_weight=ak.get("esp_weight", 0.5),
                    num_repeats=ak.get("num_repeats", 50), trans_init=False,
                    lr=ak.get("lr", 0.1), steps_fine=steps)
    raise ValueError(mode)


def _align_fast(pairs, ref_tensors: dict, mode: str, batch_kw: dict):
    """Set the shared query ref tensors on the resident fit-pairs and run the batched
    aligner; return the per-pair scores (np). Transforms are NOT built here -- they are
    materialized lazily for top-K survivors only (``_TopK.offer_pair``), since building
    all K per shard is the dominant overhead and a screen keeps only ~top_k."""
    from shepherd_score.accel.batch import aligners
    for p in pairs:
        for k, v in ref_tensors.items():
            setattr(p, k, v)
    getattr(aligners, "_align_batch_" + mode)(pairs, **batch_kw)
    return np.array([float(getattr(p, _SCORE_ATTR[mode])) for p in pairs], dtype=float)


class _TopK:
    """Bounded max-list keyed on score (min-heap of size k). Tie-break by a counter
    so transforms are never compared."""
    __slots__ = ("k", "heap", "_c")

    def __init__(self, k):
        self.k = k; self.heap = []; self._c = 0

    def _push(self, score, id_, transform):
        self._c += 1
        item = (score, self._c, id_, transform)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def offer_pair(self, score, id_, pair, tf_attr):
        """Offer a candidate, materializing its transform from ``pair`` ONLY if the
        score makes the top-K. A screen keeps ~k of K, so this builds ~k transforms
        instead of K (the dominant per-shard overhead). Must be called while ``pair``
        still holds this query's pose (before the next query/shard re-aligns it)."""
        if len(self.heap) < self.k or score > self.heap[0][0]:
            self._push(score, id_, _transform_of(pair, tf_attr))

    def merge_raw(self, raw):
        for (s, i, t) in raw:
            self._push(s, i, t)

    def raw(self):
        return [(s, i, t) for (s, _, i, t) in self.heap]

    def sorted(self):
        return [Hit(score=s, id=i, transform=t)
                for (s, _, i, t) in sorted(self.heap, key=lambda x: x[0], reverse=True)]


def _resolve_screen(store, mode, alpha, align_kwargs):
    """Shared validation + alpha resolution for screen()/screen_many()."""
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; valid: {list(_VALID_MODES)}")
    if not store.supports(mode):
        raise ValueError(f"store at {store.path!r} was built for modes {store.modes} "
                         f"and does not support {mode!r}")
    if align_kwargs.get("no_H") is False:
        raise ValueError("screen() aligns heavy atoms only; no_H=False is not supported")
    if alpha is None and mode in _SURF_ALPHA_MODES:
        from shepherd_score.score.constants import ALPHA
        alpha = float(ALPHA(store.num_surf_points))
    if alpha is None and mode == "esp_combo":
        raise ValueError("esp_combo requires an explicit alpha (it selects volumetric "
                         "shape at alpha=0.81, otherwise surface shape); pass alpha=...")
    if mode == "vol_esp" and "lam" not in align_kwargs:
        # align_with_vol_esp makes lam a required positional (no default), so don't invent
        # one here -- raise the same clear error the fast and slow paths should both give
        # (the fast path would otherwise KeyError deep inside _fast_batch_kwargs).
        raise ValueError("vol_esp requires an explicit lam=... (the ESP/partial-charge "
                         "weight); pass lam=...")
    if alpha is not None:
        align_kwargs["alpha"] = alpha
    return align_kwargs


def _run_shards_inproc(store, shard_idxs, qs_ref, mode, device, top_k, batch_kw,
                       align_kwargs, backend, fast, center_profiles, scores_out, progress,
                       n_total):
    """Process ``shard_idxs`` against the query panel, one shard load per shard,
    aligning every query against it. Returns a ``_TopK`` per query."""
    heaps = [_TopK(top_k) for _ in qs_ref]
    tf_attr = _TRANSFORM_ATTR[mode]
    done = 0
    if not fast:
        from shepherd_score.container import MoleculePair, MoleculePairBatch
    for idx in shard_idxs:
        if fast:
            sh, arrs = store.read_shard(idx)
            ids, pairs = _build_fit_fast_pairs(arrs, mode, device)
            ids = [_id_to_py(x) for x in ids]
            start = sh["start"]
            for qi, ra in enumerate(qs_ref):
                ref = _ref_tensors_from_arrays(ra, mode, device)
                scores = _align_fast(pairs, ref, mode, batch_kw)
                _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi, start)
        else:
            sh = store.manifest["shards"][idx]
            profiles = store.read_profiles(idx)
            if center_profiles:
                for p in profiles:
                    p.center_to(p.atom_pos.mean(0))
            ids = [_id_to_py(p.id) for p in profiles]
            start = sh["start"]
            for qi, q in enumerate(qs_ref):
                pairs = [MoleculePair(q, p, do_center=False) for p in profiles]
                result = getattr(MoleculePairBatch(pairs), "align_with_" + mode)(
                    backend=backend, **align_kwargs)
                scores = np.asarray(result[0], dtype=float)
                _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi, start)
        done += sh["n"]
        if progress:
            print(f"[screen] {done}/{n_total} library molecules aligned "
                  f"x {len(qs_ref)} queries", flush=True)
    return heaps


def _accumulate(heap, ids, scores, pairs, tf_attr, scores_out, qi, start):
    for i in range(len(ids)):
        heap.offer_pair(float(scores[i]), ids[i], pairs[i], tf_attr)
    if scores_out is not None and scores_out[qi] is not None:
        scores_out[qi][start:start + len(ids)] = scores


def _normalize_scores_out(scores_out, n_queries):
    if scores_out is None:
        return [None] * n_queries
    if isinstance(scores_out, np.ndarray) and n_queries == 1:
        return [scores_out]
    if isinstance(scores_out, (list, tuple)) and len(scores_out) == n_queries:
        return list(scores_out)
    raise ValueError("scores_out must be None, a single array (1 query), or a list "
                     "of one array per query")


def screen_many(queries: Sequence, store: "ProfileStore", mode: str = "esp", *,
                backend: Optional[str] = None, do_center: Optional[bool] = None,
                top_k: int = 1000, ndev: Optional[int] = None,
                scores_out=None, alpha: Optional[float] = None,
                progress: bool = False, **align_kwargs) -> List[List["Hit"]]:
    """Screen a **panel** of queries against ``store`` in a single streaming pass.

    Each shard is read from disk **once** and aligned against *every* query (so the
    library is streamed once for the whole panel, not once per query). For the fast
    modes (all of ``vol/vol_esp/surf/esp/pharm/vol_color/esp_combo``) on a pre-centered
    store, the shard's fit tensors are built once on-device and reused across the panel
    via the direct array->kernel path (no per-molecule ``MoleculeProfile``/``MoleculePair``).

    Returns a list aligned with ``queries``: ``out[j]`` is query ``j``'s ``top_k``
    ``Hit``s (sorted, descending).

    See :func:`screen` for the per-query parameters. ``scores_out`` may be a list of
    one preallocated array per query (single-process only). ``ndev>1`` streams shards
    across a persistent one-process-per-GPU pool (fast modes only).
    """
    import torch
    queries = list(queries)
    align_kwargs = _resolve_screen(store, mode, alpha, align_kwargs)
    backend = backend or _default_backend()

    if mode in ("surf", "esp", "esp_combo"):
        for q in queries:
            qn = getattr(q, "num_surf_points", None)
            if qn is not None and qn != store.num_surf_points:
                raise ValueError(f"query num_surf_points ({qn}) != store "
                                 f"({store.num_surf_points}); ALPHA is calibrated to it")

    # Fast-path query preconditions: name the missing field up front instead of crashing
    # opaquely inside _query_ref_arrays (mirrors _profile_from_schema's clear-error convention).
    # A bare RDKit Molecule always has these; a MoleculeProfile reconstructed without the mode's
    # arrays does not.
    if mode == "esp_combo":
        for q in queries:
            miss = [a for a in ("surf_pos", "surf_esp", "mol", "partial_charges", "radii")
                    if getattr(q, a, None) is None]
            if miss:
                raise ValueError(f"esp_combo query is missing {miss}; build the query "
                                 f"Molecule/MoleculeProfile with esp_combo arrays (surface, "
                                 f"with-H centers, partial charges, radii)")
    elif mode == "vol_esp":
        for q in queries:
            if getattr(q, "partial_charges", None) is None or getattr(q, "_nonH_atoms_idx", None) is None:
                raise ValueError("vol_esp query is missing partial_charges/_nonH_atoms_idx; "
                                 "build the query Molecule/MoleculeProfile with partial charges")

    center = (not store.pre_centered) if do_center is None else bool(do_center)
    if store.pre_centered or center:
        qs = [_centered_copy(q) for q in queries]
        center_profiles = (not store.pre_centered) and center
    else:
        qs = list(queries)
        center_profiles = False

    fast = (mode in _FAST_MODES and store.pre_centered
            and not align_kwargs.get("trans_init") and backend != "jax")
    device = (torch.device("cpu") if backend in ("numba", "cpu")
              else torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    if ndev and ndev > 1:
        if not fast:
            raise ValueError("ndev>1 requires the fast path (a pre-centered store, a "
                             f"{sorted(_FAST_MODES)} mode, trans_init=False, GPU backend)")
        heaps = _screen_many_multigpu(qs, store.path, mode, ndev,
                                      _fast_batch_kwargs(mode, align_kwargs), top_k, progress)
        return [h.sorted() for h in heaps]

    so = _normalize_scores_out(scores_out, len(queries))
    if fast:
        qs_ref = [_query_ref_arrays(q, mode) for q in qs]
        batch_kw = _fast_batch_kwargs(mode, align_kwargs)
    else:
        qs_ref = qs
        batch_kw = None
    heaps = _run_shards_inproc(store, range(store.num_shards), qs_ref, mode, device, top_k,
                               batch_kw, align_kwargs, backend, fast, center_profiles, so,
                               progress, len(store))
    return [h.sorted() for h in heaps]


def screen(query, store: "ProfileStore", mode: str = "esp", *,
           backend: Optional[str] = None, do_center: Optional[bool] = None,
           top_k: int = 1000, ndev: Optional[int] = None,
           scores_out: Optional[np.ndarray] = None, alpha: Optional[float] = None,
           progress: bool = False, **align_kwargs) -> List["Hit"]:
    """Stream ``store`` past a single ``query`` and return the ``top_k`` hits.

    Thin wrapper over :func:`screen_many` (a one-query panel). The library is never
    materialized in RAM: shards stream through the batched aligner, reduce into a
    running top-K, and are discarded.

    Parameters
    ----------
    query : Molecule or MoleculeProfile
        The reference. Built once; reused across every shard.
    store : ProfileStore
        Opened for reading.
    mode : str
        One of ``vol vol_esp surf esp pharm esp_combo vol_color`` (must be supported by
        the store). **All seven** take the fast direct array->kernel path on a pre-centered
        store (``trans_init=False``, non-jax backend). ``vol_color`` (ROCS/ROSHAMBO-style
        shape + directionless pharmacophore color) needs only a pharm store (atoms +
        anchors), no surfaces.
    backend : str, optional
        Default auto: ``"triton"`` on CUDA, else ``"numba"``.
    do_center : bool, optional
        COM centering. Default: pre-centered store -> query auto-centered, profiles
        as-stored; else both centered. The caller's query is never mutated.
    top_k : int
        Number of best hits to retain. Default 1000.
    ndev : int, optional
        Stream shards across this many GPUs via a persistent pool (fast modes only).
    scores_out : np.ndarray, optional
        Preallocated ``(len(store),)`` array (e.g. an ``np.memmap``) written with every
        score in library order. Single-process only.
    alpha : float, optional
        Shape Gaussian width; auto-fills ``ALPHA(num_surf_points)`` for ``surf``/``esp``,
        required for ``esp_combo`` (``alpha=0.81`` selects volumetric shape, else surface),
        defaults to ``0.81`` for ``vol``/``vol_esp``, ignored only for ``pharm``.
    **align_kwargs
        Passed to the aligner (``lam``, ``num_repeats``, ``max_num_steps``, ``lr``,
        ``similarity``, ...). ``trans_init=True`` falls back off the fast path.

    Returns
    -------
    list[Hit]
        ``Hit(score, id, transform)`` sorted by score, descending (length ``<= top_k``).
    """
    return screen_many([query], store, mode, backend=backend, do_center=do_center,
                       top_k=top_k, ndev=ndev, scores_out=scores_out, alpha=alpha,
                       progress=progress, **align_kwargs)[0]


# --------------------------------------------------------------------------- #
# Multi-GPU: persistent one-process-per-GPU pool, spawned ONCE, that streams shards
# (NOT respawned per shard like align_multi_gpu). Each worker pins to a GPU, holds
# its query panel resident, pulls shard indices off a queue, reads + aligns them with
# the fast path, and returns its per-query top-K. Mirrors accel.multi_gpu's spawn +
# thread-cap pattern (the host-bound align needs one process per GPU to parallelise).
# --------------------------------------------------------------------------- #
def _screen_worker(rank, threads, store_path, ref_arrays_list, mode, batch_kw, top_k,
                   shard_q, out_q):
    try:
        import torch
        from shepherd_score.accel.multi_gpu import _cap_threads
        try:
            from shepherd_score.accel.batch import _DISPATCH_LOCAL
        except Exception:
            from shepherd_score.container._core import _DISPATCH_LOCAL
        _cap_threads(threads)
        torch.cuda.set_device(rank)
        _DISPATCH_LOCAL.active = True
        dev = torch.device("cuda", rank)
        store = ProfileStore.open(store_path)
        ref_tensors = [_ref_tensors_from_arrays(ra, mode, dev) for ra in ref_arrays_list]
        heaps = [_TopK(top_k) for _ in ref_arrays_list]
        tf_attr = _TRANSFORM_ATTR[mode]
        while True:
            idx = shard_q.get()
            if idx is None:
                break
            _sh, arrs = store.read_shard(idx)
            ids, pairs = _build_fit_fast_pairs(arrs, mode, dev)
            ids = [_id_to_py(x) for x in ids]
            for qi, ref in enumerate(ref_tensors):
                scores = _align_fast(pairs, ref, mode, batch_kw)
                for i in range(len(ids)):
                    heaps[qi].offer_pair(float(scores[i]), ids[i], pairs[i], tf_attr)
            torch.cuda.synchronize()
        out_q.put((rank, [h.raw() for h in heaps]))
    except Exception:                            # noqa: BLE001 - relayed to parent
        import traceback
        out_q.put((rank, "__ERR__", traceback.format_exc()))


def _screen_many_multigpu(qs, store_path, mode, ndev, batch_kw, top_k, progress):
    import os as _os
    import torch
    import torch.multiprocessing as mp

    ndev = max(1, min(ndev, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    try:
        cores = len(_os.sched_getaffinity(0))
    except AttributeError:
        cores = _os.cpu_count() or ndev
    threads = max(1, cores // ndev)
    ref_arrays_list = [_query_ref_arrays(q, mode) for q in qs]

    store = ProfileStore.open(store_path)
    n_shards = store.num_shards

    _saved = {k: _os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in _saved:
        _os.environ[k] = str(threads)
    try:
        ctx = mp.get_context("spawn")
        shard_q = ctx.Queue()
        out_q = ctx.Queue()
        for idx in range(n_shards):
            shard_q.put(idx)
        for _ in range(ndev):
            shard_q.put(None)                    # one sentinel per worker
        procs = []
        for r in range(ndev):
            p = ctx.Process(target=_screen_worker,
                            args=(r, threads, store_path, ref_arrays_list, mode,
                                  batch_kw, top_k, shard_q, out_q))
            p.start(); procs.append(p)
        results, errs = {}, []
        for _ in range(ndev):
            msg = out_q.get()
            if len(msg) == 3 and msg[1] == "__ERR__":
                errs.append((msg[0], msg[2]))
            else:
                results[msg[0]] = msg[1]
        for p in procs:
            p.join()
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    if errs:
        raise RuntimeError("multi-GPU screen failed on ranks "
                           f"{[r for r, _ in errs]}:\n" +
                           "\n".join(f"[rank {r}]\n{tb}" for r, tb in errs))

    heaps = [_TopK(top_k) for _ in qs]
    for rank in results:
        per_query = results[rank]
        for qi, raw in enumerate(per_query):
            heaps[qi].merge_raw(raw)
    return heaps
