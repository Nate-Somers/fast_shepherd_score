"""Out-of-core *streaming* screening for libraries too large to hold in RAM.

A query (or a handful of queries) can be aligned against a library of effectively
unbounded size by (1) precomputing each library molecule's interaction profile
**once**, persisting it to a sharded on-disk store, and (2) streaming shards back
through the **existing** :class:`~shepherd_score.container.MoleculePairBatch` /
:func:`~shepherd_score.accel.multi_gpu.align_multi_gpu` API, reducing scores on
the fly. Host RAM never holds the whole library: the screen keeps the shard it is
aligning plus (by default) one shard being read ahead on a background thread, so
residency is bounded at two shards. See ``_iter_shards_prefetched`` for the
read-ahead and hold exactly one.

Three pieces:

* :class:`MoleculeProfile` -- an RDKit-free, duck-typed stand-in for ``Molecule``
  holding only the numeric arrays the batched aligners read. The RDKit ``Mol`` is
  not stored, so reloading a shard is an array copy rather than a molecule
  reconstruction.
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
...                           modes=("surf", "surf_esp", "pharm")) as store:
...     for mol in library_rdkit_mols:
...         store.add(Molecule(mol, num_surf_points=200, pharm_multi_vector=False))
>>> # SCREEN (streamed; never holds the library in RAM)
>>> query = Molecule(query_mol, num_surf_points=200, pharm_multi_vector=False)
>>> hits = screen(query, ProfileStore.open("lib.fss"), mode="surf_esp",
...               lam=0.3, top_k=1000)   # seeds/steps default per-mode; backend auto (triton GPU / numba CPU)
>>> hits[0].score, hits[0].id            # best match
"""
from __future__ import annotations

import atexit
import copy
import heapq
import json
import os
import re
from collections import namedtuple
from operator import attrgetter
from typing import Iterator, List, Optional, Sequence

import numpy as np

__all__ = ["MoleculeProfile", "ProfileStore", "screen", "screen_many", "Hit"]


# Per-mode result attributes written in-place by ``MoleculePairBatch.align_with_*``, sourced from
# the mode registry (``accel/_modes.py``) so the screen front-end and the accel layers can never
# disagree on attribute names or valid modes. Legacy mode names resolve via ``canonical()``.
from shepherd_score.accel._modes import (
    MODE_ATTRS as _MODE_ATTRS, canonical as _canon_mode,
    CONST_SEED_MODES as _CONST_SEED_MODES, SPECS as _SPECS, spec_of as _spec_of,
)
from shepherd_score.accel.channels import CHANNELS as _CHANNELS, SCHEMA_FLAGS as _SCHEMA_FLAGS
_TRANSFORM_ATTR = {m: a[0] for m, a in _MODE_ATTRS.items()}
_SCORE_ATTR = {m: a[1] for m, a in _MODE_ATTRS.items()}
# Prebuilt score readers: ``attrgetter`` + ``map`` pulls the per-pair score in C, so the
# K-element score vector costs one iterator pass instead of K Python ``getattr`` calls.
_SCORE_GETTER = {m: attrgetter(a) for m, a in _SCORE_ATTR.items()}
_VALID_MODES = tuple(_SCORE_ATTR)
# Modes whose surface ``alpha`` should auto-default to ALPHA(num_surf_points): the ones whose
# SHAPE channel is the surface cloud, which is calibrated to the point count. A mode that reads
# surfaces for some OTHER channel (the ShaEP combo's ESP agreement) is not one of these -- its
# alpha selects which cloud the shape term scores and has to be given explicitly.
_SURF_ALPHA_MODES = {m for m, sp in _SPECS.items()
                     if any(t.kernel in ("shape", "esp") and t.ref and t.ref[0] == "surf"
                            for t in sp.terms)}


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
                 "lipo_pos", "lipophilicity",
                 "fukui_pos", "fukui",
                 "mr_pos", "molar_refractivity",
                 "atomtype_pos", "atomic_numbers",
                 "num_surf_points", "mol", "id", "rot")

    def __init__(self, *, atom_pos, surf_pos=None, surf_esp=None,
                 partial_charges=None, radii=None, nonH_atoms_idx=None,
                 pharm_types=None, pharm_ancs=None, pharm_vecs=None,
                 lipo_pos=None, lipophilicity=None,
                 fukui_pos=None, fukui=None,
                 mr_pos=None, molar_refractivity=None,
                 atomtype_pos=None, atomic_numbers=None,
                 centers_w_H=None, atom_pos_noH=None, id=None, rot=None):
        self.atom_pos = _f32(atom_pos)
        #: (3,3) float32 principal-axis rotation applied at build time, or None. Present only
        #: on a CANONICAL store: the coords below are already rotated into this frame, so the
        #: screen's seeds are constants instead of a per-molecule eigensolve. Needed to map an
        #: alignment transform back to the molecule's ORIGINAL frame.
        self.rot = None if rot is None else np.asarray(rot, np.float32)
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
        # vol_lipo: the TRUE-heavy atom centres (own count, may differ from atom_pos when
        # RemoveHs retained an H) + the per-atom Crippen logP placed at them (already heavy-sliced).
        self.lipo_pos = _f32(lipo_pos)
        self.lipophilicity = _f32(lipophilicity)
        # vol_fukui: TRUE-heavy atom centres (own count) + the per-atom signed Fukui dual descriptor
        # (f+ - f-) placed at them (already heavy-sliced), exactly like the vol_lipo channel.
        self.fukui_pos = _f32(fukui_pos)
        self.fukui = _f32(fukui)
        # vol_mr / vol_atomtype: the same TRUE-heavy basis, their own centres + per-atom scalar
        # (already heavy-sliced), stored under their own offset tables. ``atomic_numbers`` is a
        # categorical label the colour kernel matches on, kept float32 like the other channels.
        self.mr_pos = _f32(mr_pos)
        self.molar_refractivity = _f32(molar_refractivity)
        self.atomtype_pos = _f32(atomtype_pos)
        self.atomic_numbers = _f32(atomic_numbers)
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
        if self.lipo_pos is not None:
            self.lipo_pos = self.lipo_pos - mu                 # lipo centres move with the molecule
        if self.fukui_pos is not None:
            self.fukui_pos = self.fukui_pos - mu               # fukui centres move with the molecule
        if self.mr_pos is not None:
            self.mr_pos = self.mr_pos - mu
        if self.atomtype_pos is not None:
            self.atomtype_pos = self.atomtype_pos - mu
        if self.mol is not None:
            self.mol = _MolShim(self.mol.GetConformer().GetPositions() - mu)

    def get_lipo_positions(self):
        """TRUE-heavy lipophilicity centres -- mirrors ``Molecule.get_lipo_positions()`` so the
        ``vol_lipo`` aligner reads a profile identically to a full ``Molecule``."""
        return self.lipo_pos

    def get_lipophilicity(self, no_H: bool = True):
        """Per-atom Crippen logP (stored already heavy-sliced) -- mirrors
        ``Molecule.get_lipophilicity()``. ``no_H`` is accepted for signature parity; the profile
        only carries the heavy slice the ``vol_lipo`` aligner reads."""
        return self.lipophilicity

    def get_fukui_positions(self):
        """TRUE-heavy Fukui centres -- mirrors ``Molecule.get_fukui_positions()`` so the
        ``vol_fukui`` aligner reads a profile identically to a full ``Molecule``."""
        return self.fukui_pos

    def get_fukui(self, no_H: bool = True):
        """Per-atom Fukui dual descriptor (stored already heavy-sliced) -- mirrors
        ``Molecule.get_fukui()``. ``no_H`` is accepted for signature parity; the profile only
        carries the heavy slice the ``vol_fukui`` aligner reads."""
        return self.fukui

    def get_mr_positions(self):
        """TRUE-heavy molar-refractivity centres -- mirrors ``Molecule.get_mr_positions()``."""
        return self.mr_pos

    def get_molar_refractivity(self, no_H: bool = True):
        """Per-atom Crippen molar refractivity (stored already heavy-sliced) -- mirrors
        ``Molecule.get_molar_refractivity()``."""
        return self.molar_refractivity

    def get_atomtype_positions(self):
        """TRUE-heavy element-label centres -- mirrors ``Molecule.get_atomtype_positions()``."""
        return self.atomtype_pos

    def get_atomic_numbers(self, no_H: bool = True):
        """Per-atom atomic numbers (stored already heavy-sliced) -- mirrors
        ``Molecule.get_atomic_numbers()``."""
        return self.atomic_numbers

    @classmethod
    def from_molecule(cls, m, *, modes=_VALID_MODES, id=None) -> "MoleculeProfile":
        """Extract the arrays the requested ``modes`` need from a ``Molecule``
        (or another ``MoleculeProfile``); the RDKit ``Mol`` is dropped."""
        sch = _schema_from_modes(modes)
        return _profile_from_schema(m, sch, id=id, pre_center=False)


# --------------------------------------------------------------------------- #
# storage schema helpers
# --------------------------------------------------------------------------- #
def _mode_flags(mode: str) -> set:
    """The schema flags ``mode`` needs, read off its channels. A mode reading every channel of a
    basis needs every flag those channels carry; ``atom_pos`` is unconditional and has none.
    ``vol_and_surf_esp`` reaches ``with_H`` through its ``partial`` channel, which is why a
    combo store keeps the with-H charges plus the ``nonH`` index rather than the heavy slice."""
    spec = _spec_of(mode)
    flags = set()
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue
        if ch.flag:
            flags.add(ch.flag)
        if ch.basis == "withH":
            # A with-H channel needs the store laid out on the with-H basis, which is a property
            # of the BASIS, not of any one channel. ``_flush`` writes ``all_off`` / ``nonH`` /
            # ``radii`` / ``cwh`` only under ``schema["with_H"]``, so a mode reading any of them
            # must demand the flag or it can be handed a store that has none of them.
            flags.add("with_H")
    return flags


def _schema_from_modes(modes) -> dict:
    """The store schema serving ``modes``: one boolean per channel flag, derived from what each
    mode's channels read. Every flag in ``SCHEMA_FLAGS`` is always present (a reader may use
    ``schema[flag]``), so only the VALUES vary with the mode set."""
    modes = {_canon_mode(m) for m in modes}            # accept legacy esp / esp_combo
    unknown = modes - set(_VALID_MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}; valid: {list(_VALID_MODES)}")
    want = set()
    for m in modes:
        want |= _mode_flags(m)
    return {f: (f in want) for f in _SCHEMA_FLAGS}


def _store_supports(schema: dict, mode: str) -> bool:
    """Whether a store with ``schema`` carries every array ``mode`` reads.

    Derived from the mode's channels, so a mode is screenable the moment its data is stored.
    ``vol_avoid`` is the one mode whose objective needs an input the per-molecule store does not
    model -- a fixed avoid cloud that belongs to the QUERY, not to a library molecule -- and it
    screens by carrying that cloud with the query (``screen(..., avoid_points=...)``), so its
    per-molecule requirement is just ``atom_pos`` like ``vol``'s."""
    mode = _canon_mode(mode)                           # accept legacy esp / esp_combo
    if mode not in _SPECS:
        return False
    return all(schema.get(f, False) for f in _mode_flags(mode))


#: How each extra per-molecule BASIS is carried through a shard, beyond the unconditional
#: ``atom_pos``/``atom_off``. ``(schema flag, offset key, [(profile attr, store key), ...])``.
#: ONE table drives extraction, pre-centring, the canonical rotation, ``_concat`` and
#: ``_reconstruct``, so a new per-atom field is a row here plus its channel -- not five edits
#: that must agree. The two bases with irregular layouts keep their own code below: ``surf`` is
#: written at a FIXED width (``np.stack``, no offsets) and ``heavy``/``withH`` share one
#: ``charges`` array whose basis depends on ``with_H``.
_BASIS_TABLE = (
    ("pharm", "pharm", "pharm_off", (("pharm_types", "pharm_types"), ("pharm_ancs", "pharm_ancs"),
                                     ("pharm_vecs", "pharm_vecs"))),
    ("lipo", "lipophilicity", "lipo_off", (("lipo_pos", "lipo_pos"),
                                           ("lipophilicity", "lipophilicity"))),
    ("fukui", "fukui", "fukui_off", (("fukui_pos", "fukui_pos"), ("fukui", "fukui"))),
    ("mr", "mr", "mr_off", (("mr_pos", "mr_pos"), ("molar_refractivity", "mr"))),
    ("atomtype", "atomtype", "atomtype_off", (("atomtype_pos", "atomtype_pos"),
                                              ("atomic_numbers", "atomic_numbers"))),
)

#: The channel that reads each ``_BASIS_TABLE`` profile attribute off a ``Molecule``. Reusing
#: the channel readers is what keeps the store and the aligners on ONE definition of "the
#: molar-refractivity centres" -- the accessor, never ``atom_pos`` (the retained-H trap).
_PROF_READER = {c.prof: c.read for c in _CHANNELS.values() if c.key}


def _profile_from_schema(m, sch: dict, *, id, pre_center: bool, canonical: bool = False) -> "MoleculeProfile":
    """Pull the schema's arrays off a ``Molecule``/``MoleculeProfile`` ``m``, optionally centring
    to the heavy-atom COM and rotating into its principal frame. Returns a ``MoleculeProfile``."""
    atom_pos = _f32(m.atom_pos)
    kw = {"atom_pos": atom_pos}
    if sch["surf"]:
        if m.surf_pos is None:
            raise ValueError("store needs surfaces but molecule has none "
                             "(build Molecule with num_surf_points / surface_points)")
        kw["surf_pos"] = _f32(m.surf_pos)
    if sch["surf_esp"]:
        if m.surf_esp is None:
            raise ValueError("store needs surface ESP but molecule has none")
        kw["surf_esp"] = _f32(m.surf_esp)
    if sch["charges"]:
        if m.partial_charges is None:
            raise ValueError("store needs partial charges but molecule has none")
        if sch["with_H"]:
            kw["partial_charges"] = _f32(m.partial_charges)
            kw["nonH_atoms_idx"] = np.asarray(m._nonH_atoms_idx, dtype=np.int64)
        else:
            # heavy charges. Index by _nonH_atoms_idx universally: it is the real heavy index
            # for a Molecule (full charges) and the identity for a heavy MoleculeProfile.
            kw["partial_charges"] = _f32(np.asarray(m.partial_charges)[m._nonH_atoms_idx])
        # Heavy Gaussian centres for vol_esp, 1:1 with the heavy charges. Kept only when they
        # actually differ from atom_pos (i.e. RemoveHs retained an H).
        hp = np.asarray(_heavy_positions(m), dtype=np.float32)
        if hp.shape != atom_pos.shape or not np.array_equal(hp, atom_pos):
            kw["atom_pos_noH"] = hp
    if sch["radii"]:
        if m.radii is None:
            raise ValueError("store needs vdW radii but molecule has none")
        kw["radii"] = _f32(m.radii)
    if sch["centers_w_H"]:
        kw["centers_w_H"] = _f32(m.mol.GetConformer().GetPositions())
    for _basis, flag, _off, attrs in _BASIS_TABLE:
        if not sch.get(flag, False):
            continue
        for attr, _key in attrs:
            try:
                v = _PROF_READER[attr](m)
            except ValueError as e:
                raise ValueError(f"store needs {flag} but molecule cannot provide it: {e}") from e
            kw[attr] = (np.asarray(v, dtype=np.int32) if attr == "pharm_types" else _f32(v))

    #: profile attributes that hold POSITIONS (translate + rotate) and DIRECTIONS (rotate only).
    pos_attrs = ["atom_pos", "surf_pos", "centers_w_H", "atom_pos_noH"]
    dir_attrs = []
    for _basis, flag, _off, attrs in _BASIS_TABLE:
        for attr, _key in attrs:
            ch = next((c for c in _CHANNELS.values() if c.prof == attr and c.key), None)
            if ch is None:
                continue
            if ch.kind == "points":
                pos_attrs.append(attr)
            elif ch.kind == "vectors":
                dir_attrs.append(attr)

    if pre_center:
        mu = atom_pos.mean(0)
        for a in pos_attrs:
            v = kw.get(a)
            if v is not None and len(v):
                # shifted by the atom_pos COM, NEVER by their own -- this matches the in-memory
                # conformer transform, which moves every channel of a molecule together.
                kw[a] = v - mu

    rot = None
    if canonical:
        # CANONICAL FRAME: rotate every coordinate channel into the molecule's own principal
        # axes. The screen's seeds exist to align the fit molecule's principal axes onto the
        # query's; if the fit is ALREADY in its principal frame, that alignment is one constant
        # for every molecule, so the per-molecule eigensolve disappears (44.1% of a vol screen).
        #
        # Requires pre_center (axes are about the centroid). ``rot`` is kept so a returned pose
        # can be mapped back; without it scores would be right while transforms silently
        # referred to the canonical frame -- which no score-based test can see. EVERY position
        # channel rotates and every direction channel rotates without translating; missing one
        # leaves a single row of that array in the raw frame while its neighbours are canonical
        # (measured on atom_pos_noH: the retained-H molecule's pose re-scored 0.619 low).
        if not pre_center:
            raise ValueError("canonical stores require pre_center=True (axes are centroid-relative)")
        _c = kw["atom_pos"] - kw["atom_pos"].mean(0)
        _cov = _c.T @ _c
        _w, _v = np.linalg.eigh(_cov.astype(np.float64))
        _v = _v[:, ::-1]                                  # descending eigenvalue order
        if np.linalg.det(_v) < 0:                         # keep it a proper rotation
            _v[:, 2] = -_v[:, 2]
        rot = np.ascontiguousarray(_v.T, dtype=np.float32)   # maps original -> canonical
        for a in pos_attrs + dir_attrs:
            v = kw.get(a)
            if v is not None and len(v):
                kw[a] = (v @ rot.T).astype(np.float32)

    return MoleculeProfile(id=id, rot=rot, **kw)


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
               pre_centered: bool = True, overwrite: bool = False,
               canonical: Optional[bool] = None,
               shard_format: str = "npy") -> "ProfileStore":
        """Open a store for writing.

        Parameters
        ----------
        num_surf_points : int
            Surface point count every profile uses (must match the query at screen
            time; kept in ``[50, 400]`` for ``ALPHA`` calibration).
        modes : sequence of str
            Which alignment modes this store must support. Only the arrays those
            modes need are stored (``"vol"`` works from any store -- ``atom_pos``
            is always kept). Valid: ``vol vol_esp surf surf_esp pharm vol_color vol_and_surf_esp``
            (legacy ``esp``/``esp_combo`` also accepted).
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
        canonical : bool, optional
            Store every profile rotated into its own principal-axis frame (the
            centroid-relative eigenframe of its heavy atoms), with the rotation kept so
            that returned transforms are composed back to the original centred frame.
            A screen in any mode that seeds from the heavy-atom cloud
            (``accel._modes.CONST_SEED_MODES``: ``vol``, ``vol_color``, ``vol_esp``,
            ``vol_lipo``, ``vol_fukui``, the volumetric Tversky modes, and
            ``vol_and_surf_esp`` at ``alpha=0.81``) then runs one constant seed set instead
            of a per-molecule eigensolve -- roughly 1.5-2x faster for ``vol`` on GPU, less
            for the heavier modes, whose seed generation is a smaller share. ``surf``,
            ``surf_esp`` and ``pharm`` seed from the surface / anchor clouds and still run
            their own generator on the rotated coordinates. Either way the scores move at
            the 1e-3 level against a non-canonical store (which matches the pairwise path
            to ~1e-4). Requires ``pre_centered``.
            Default (``None``): canonical when the store serves any constant-seed mode
            and is pre-centred, raw centred coordinates otherwise. Pass ``True`` or
            ``False`` to decide explicitly. Screening scores and DUDE-Z enrichment on canonical
            stores were validated against non-canonical stores and the pairwise path
            before this became the default (Shepherd-Score-Paper,
            fig2_speed/validate_canonical.py, results/CANONICAL_validation.json).

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
        if canonical is None:
            # The library default: canonical when the store serves a mode that seeds from the
            # heavy-atom cloud (``_CONST_SEED_MODES``) and is pre-centred, because those are the
            # modes whose screen swaps the per-molecule seed eigensolve for one constant set.
            # ``surf``/``surf_esp``/``pharm`` seed from other clouds and still run their generator
            # on the rotated coordinates, so a store serving only them keeps raw centred
            # coordinates, which match the pairwise path to ~1e-4 (a canonical store moves scores
            # at the 1e-3 level; Shepherd-Score-Paper fig2_speed/validate_canonical.py).
            canonical = bool(pre_centered) and any(_canon_mode(m) in _CONST_SEED_MODES
                                                   for m in modes)
        elif canonical and not pre_centered:
            raise ValueError("canonical stores require pre_centered=True (axes are centroid-relative)")
        schema = _schema_from_modes(modes)
        os.makedirs(path, exist_ok=True)
        manifest_path = os.path.join(path, cls.MANIFEST)
        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already exists; pass overwrite=True to replace it")
        if overwrite:
            # Remove only THIS store's own files (its manifest + its shard sequence, in
            # either format), never every ``.npz`` in the directory -- the target may hold
            # unrelated data the caller did not mean to lose.
            for f in os.listdir(path):
                if (f == cls.MANIFEST or re.fullmatch(r"shard_\d{5}\.npz", f)
                        or re.fullmatch(r"shard_\d{5}__\w+\.npy", f)):
                    os.remove(os.path.join(path, f))
        manifest = dict(version=cls.VERSION, num_surf_points=int(num_surf_points),
                        modes=list(modes), schema=schema, dtype=dtype,
                        shard_size=int(shard_size), pre_centered=bool(pre_centered),
                        canonical=bool(canonical),
                        shard_format=str(shard_format),
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
                                    pre_center=self.manifest["pre_centered"],
                                    canonical=bool(self.manifest.get("canonical", False)))
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
        arrs = self._concat(self._buf)
        if self.manifest.get("shard_format") == "npy":
            # One .npy per array instead of one .npz per shard, so the reader can memory-map
            # them. An .npz is a ZIP, and Python's zipfile CRC-checks every member as it is
            # read: measured on an L40S (job 22596303), a 20.9 MB shard costs 13.30 ms to read
            # with the check and 5.08 ms without, i.e. **62% of the read is CRC** -- and at
            # N=100,000 the store is a single shard, so the read-ahead thread has nothing to
            # overlap it with and the whole 0.133 us/mol sits on the critical path. np.save
            # pads its header so the data begins 64-byte aligned, which is what lets np.load
            # hand back a real mmap rather than a copy.
            name = f"shard_{self._shard_id:05d}"
            for k, v in arrs.items():
                np.save(os.path.join(self.path, f"{name}__{k}.npy"), v)
            entry = dict(name=name, n=len(self._buf), start=self._n, keys=sorted(arrs))
        else:
            name = f"shard_{self._shard_id:05d}.npz"
            np.savez(os.path.join(self.path, name), **arrs)
            entry = dict(name=name, n=len(self._buf), start=self._n)
        self.manifest["shards"].append(entry)
        self._n += len(self._buf)
        self.manifest["n_total"] = self._n
        self._buf = []
        self._shard_id += 1
        self._write_manifest()

    def _concat(self, recs: List["MoleculeProfile"]) -> dict:
        """Pack one shard: every basis the schema carries, as a flat array plus its own CSR
        offset table (or a dense ``np.stack`` block for the fixed-width surface).

        GIVE EACH PER-ATOM FIELD ITS OWN TABLE. ``atom_pos`` is the ``Chem.RemoveHs`` set and is
        LONGER than the true-heavy set whenever RemoveHs retained an isotope-labelled H, so one
        shared table would desync a field from its positions on exactly the molecules that are
        hardest to notice.
        """
        sch = self.manifest["schema"]
        dt = np.float16 if self.manifest["dtype"] == "float16" else np.float32

        def offsets(lengths):
            return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

        out = {"ids": np.array([r.id for r in recs])}
        out["atom_off"] = offsets([len(r.atom_pos) for r in recs])
        out["atom_pos"] = np.concatenate([r.atom_pos for r in recs]).astype(dt)
        if recs[0].rot is not None:
            # float32 REGARDLESS of the store dtype: a float16 rotation carries ~3.9e-04 of axis
            # error, which is enough to move a pose. 36 bytes/molecule (3.6 MB at N=1e5).
            out["rot"] = np.stack([r.rot for r in recs]).astype(np.float32)

        if sch["surf"]:
            out["surf_pos"] = np.stack([r.surf_pos for r in recs]).astype(dt)
        if sch["surf_esp"]:
            out["surf_esp"] = np.stack([r.surf_esp for r in recs]).astype(dt)
        if sch["charges"]:
            if sch["with_H"]:
                out["all_off"] = offsets([len(r.partial_charges) for r in recs])
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
            # vol_esp needs heavy centres 1:1 with the heavy charges. Emitted only when some
            # molecule's RemoveHs retained an H, so ordinary stores stay byte-for-byte unchanged
            # (and legacy stores, which lack these keys, fall back to atom_off in the reader).
            if any(r.atom_pos_noH is not None for r in recs):
                out["heavy_off"] = offsets(heavy_lens)
                out["xyz_noH"] = np.concatenate(
                    [(r.atom_pos_noH if r.atom_pos_noH is not None else r.atom_pos)
                     for r in recs]).astype(dt)
        for _basis, flag, off_key, attrs in _BASIS_TABLE:
            if not sch.get(flag, False):
                continue
            first = attrs[0][0]
            out[off_key] = offsets([len(getattr(r, first)) for r in recs])
            for attr, key in attrs:
                vals = [getattr(r, attr) for r in recs]
                kdt = np.int32 if attr == "pharm_types" else dt
                out[key] = np.concatenate(vals).astype(kdt)
        return out

    # ---- reader ---------------------------------------------------------- #
    @classmethod
    def open(cls, path) -> "ProfileStore":
        with open(os.path.join(path, cls.MANIFEST)) as fh:
            manifest = json.load(fh)
        ver = manifest.get("version")
        if ver != cls.VERSION:
            raise ValueError(
                f"ProfileStore at {path!r} has on-disk format version {ver!r}, but this "
                f"shepherd_score reads version {cls.VERSION}. Rebuild the store.")
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

    @property
    def canonical(self) -> bool:
        """Coordinates were rotated into each molecule's principal frame at build time, and a
        per-molecule ``rot`` is stored. Legacy stores lack the key and report False."""
        return bool(self.manifest.get("canonical", False))

    def supports(self, mode: str) -> bool:
        return _store_supports(self.manifest["schema"], mode)

    @property
    def num_shards(self) -> int:
        return len(self.manifest["shards"])

    def _load_raw(self, sh) -> dict:
        """Materialize one shard's arrays into a plain ``{name: np.ndarray}`` dict
        (npz closed before return). Cheaper than :meth:`iter_shards` -- it skips the
        per-molecule ``MoleculeProfile`` split, which the fast screen path does on
        the GPU/device side instead.

        A ``shard_format="npy"`` store is MEMORY-MAPPED instead of unzipped, which skips both
        the ZIP CRC (62% of a 20.9 MB shard read, measured) and the copy into fresh arrays.
        The returned arrays are copy-on-write mappings of the page cache; every consumer here
        (the array builders' uploads, ``_reconstruct``'s slicing, ``_canonical_rot``) only
        reads, so nothing is ever copied back. Stores written before this format are still ``.npz`` and take the branch below,
        so nothing on disk is invalidated."""
        if sh.get("keys") is not None:
            base = os.path.join(self.path, sh["name"])
            # mmap_mode="c" (copy-on-write), not "r": a read-only mapping makes numpy hand back
            # a non-writable array, and ``torch.from_numpy`` warns on those every call. Nothing
            # here writes, so "c" costs nothing and reads come straight from the page cache.
            return {k: np.load(f"{base}__{k}.npy", mmap_mode="c") for k in sh["keys"]}
        with np.load(os.path.join(self.path, sh["name"])) as data:
            return {k: data[k] for k in data.files}

    def read_shard(self, idx: int) -> tuple:
        """Return ``(shard_meta, arrays_dict)`` for shard ``idx`` (random access; the
        multi-GPU shard pool uses it so each worker reads only its assigned shards, and
        the in-process screen reads every shard through it).

        Must stay free of shared mutable state: :func:`_iter_shards_prefetched` calls this
        on a background thread to read the next shard while the current one aligns. It only
        indexes the (read-only) manifest and opens its own file handle, so it is safe to."""
        sh = self.manifest["shards"][idx]
        return sh, self._load_raw(sh)

    def iter_shards(self) -> Iterator[List["MoleculeProfile"]]:
        """Yield one shard at a time as a ``list[MoleculeProfile]``.

        Goes through :meth:`_load_raw` rather than opening the shard itself: that is the ONE
        place that knows how a shard is stored, and duplicating the path arithmetic here is
        what broke every ``.npy``-format store (a second reader kept opening ``shard_00000``
        with the ``.npz`` reader's assumptions and got FileNotFoundError)."""
        for sh in self.manifest["shards"]:
            yield self._reconstruct(self._load_raw(sh), sh)

    def read_profiles(self, idx: int) -> List["MoleculeProfile"]:
        """Reconstruct shard ``idx`` as ``list[MoleculeProfile]`` (random access)."""
        sh, arrs = self.read_shard(idx)
        return self._reconstruct(arrs, sh)

    def _reconstruct(self, data, sh) -> List["MoleculeProfile"]:
        """Split one shard's arrays back into ``MoleculeProfile``s (the object path). Offset
        tables are hoisted out of the per-molecule loop; ``rot`` is deliberately NOT restored --
        a profile read back from a canonical store is in the canonical frame with no record of
        it, which is why the array path composes poses from the shard's ``rot`` instead."""
        sch = self.manifest["schema"]
        n = sh["n"]
        files = set(data.files) if hasattr(data, "files") else set(data.keys())
        atom_off = data["atom_off"]
        atom_pos = data["atom_pos"]
        ids = data["ids"]
        surf_pos = data["surf_pos"] if sch["surf"] else None
        surf_esp = data["surf_esp"] if sch["surf_esp"] else None
        all_off = data["all_off"] if (sch["charges"] and sch["with_H"]) else None
        offs = {flag: data[off_key] for _b, flag, off_key, _a in _BASIS_TABLE
                if sch.get(flag, False)}

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
            for _b, flag, _off_key, attrs in _BASIS_TABLE:
                if not sch.get(flag, False):
                    continue
                off = offs[flag]
                p0, p1 = int(off[i]), int(off[i + 1])
                for attr, key in attrs:
                    kw[attr] = data[key][p0:p1]
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
#: Modes the fast (object-stand-in or array) screen engines serve. Every registry mode does;
#: what actually decides whether a mode can screen is ``_store_supports`` -- i.e. whether the
#: store carries its channels -- and ``fast`` additionally needs a pre-centred store,
#: ``trans_init=False`` and a non-jax backend.
_FAST_MODES = tuple(_SPECS)


class _FastPair:
    """Cached-tensor-only stand-in a batched aligner can consume.

    ``__slots__`` is DERIVED: one ``_ref_<attr>_t`` / ``_fit_<attr>_t`` per channel plus every
    mode's result pair. It used to be hand-mirrored from ``MODE_ATTRS``, which is how a new
    mode could reach this path and fail with an opaque AttributeError.
    """

    __slots__ = (("device", "ref_molec", "fit_molec")
                 + tuple(dict.fromkeys(
                     [a for c in _CHANNELS.values() for a in (c.ref_attr, c.fit_attr)]))
                 + tuple(a for pair in _MODE_ATTRS.values() for a in pair))

    def __init__(self, device):
        self.device = device


def _query_ref_arrays(q, mode: str) -> dict:
    """The (centred) query's numpy arrays for ``mode``, keyed by channel. Plain numpy, so it is
    cheap to ship to the multi-GPU workers. The readers are the channel table's, so a query
    ``Molecule`` and a ``MoleculeProfile`` are read identically and a missing field raises a
    clear ValueError naming it."""
    spec = _spec_of(mode)
    out = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue                                  # supplied by the caller (avoid_points)
        out[name] = np.ascontiguousarray(
            np.asarray(ch.read(q), dtype=np.int64 if ch.dtype == "int64" else np.float32))
    return out


def _ref_tensors_from_arrays(ra: dict, mode: str, device) -> dict:
    """The query's device tensors, keyed by the pair attribute the aligners read
    (``_ref_<attr>_t``). Uploaded through the same pinned staging as the fit side."""
    import torch
    out = {}
    for name, arr in ra.items():
        ch = _CHANNELS[name]
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        out[ch.ref_attr] = _to_device(arr, device, dtype=dt)
    return out


def _ref_channel_tensors(ref_tensors: dict, mode: str) -> dict:
    """Re-key ``_ref_<attr>_t`` tensors by CHANNEL, which is what the array aligner takes."""
    spec = _spec_of(mode)
    out = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair or ch.ref_attr not in ref_tensors:
            continue
        out[name] = ref_tensors[ch.ref_attr]
    return out


def _build_fit_fast_pairs(arrs: dict, mode: str, device, avoid=None):
    """Load one shard's FIT arrays as device tensors once; return ``(ids, [_FastPair])`` whose
    fit tensors are views into them. The per-molecule views come from ONE ``torch.split`` (or
    ``unbind`` for a dense block) per channel rather than a K-iteration slicing loop.

    This is the OBJECT-fast path -- the reference leg the array-vs-object parity gate compares
    against (it forces it by flipping ``_arrays.ENABLED``), which is why it must stay wired for
    every mode the array path serves."""
    import torch
    from shepherd_score.accel.channels import load_store_channel
    ids = arrs["ids"]
    K = len(ids)
    pairs = [_FastPair(device) for _ in range(K)]
    spec = _spec_of(mode)
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            if avoid is not None:
                t = _to_device(np.ascontiguousarray(np.asarray(avoid, np.float32)), device,
                               dtype=torch.float32)
                for p in pairs:
                    setattr(p, ch.ref_attr, t)
            continue
        flat, off = load_store_channel(arrs, name)
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        big = _to_device(flat, device, dtype=dt)
        views = (torch.unbind(big) if off is None
                 else torch.split(big, np.diff(off).tolist()))
        for p, v in zip(pairs, views):
            setattr(p, ch.fit_attr, v)
    return ids, pairs


def _fast_batch_kwargs(mode: str, ak: dict) -> dict:
    """Translate ``screen()``'s align_kwargs into the ``_align_batch_<mode>`` keywords.

    The mode's own parameters and their defaults come from the registry; the fine-step count
    from ``MODE_STEPS``; and ``lr`` from the spec's ``screen_lr`` -- which is 0.1 for the ESP /
    pharmacophore / colour / field modes and the driver default 0.075 for the shape and Tversky
    ones, preserved per mode because changing it would move those modes' scores.
    """
    from shepherd_score.accel.batch.aligners import _steps_for, _seeds_for
    spec = _spec_of(mode)
    kw = {"steps_fine": ak.get("max_num_steps", _steps_for(mode)), "trans_init": False}
    for name, default in spec.params.items():
        if name == "lr":
            kw["lr"] = ak.get("lr", spec.screen_lr)
        elif default is None:
            if name not in ak:
                raise ValueError(f"{mode} requires an explicit {name}=...; pass {name}=...")
            kw[name] = ak[name]
        else:
            kw[name] = ak.get(name, default)
    if spec.honors_num_repeats:
        kw["num_repeats"] = ak.get("num_repeats", _seeds_for(mode))
    return kw


def _with_avoid(batch_kw: dict, ak: dict, device) -> dict:
    """Attach the query-side avoid cloud, for the one mode whose objective takes a third input.

    ``vol_avoid`` scores shape Tanimoto MINUS an excluded-volume penalty against a cloud that is
    a property of the QUERY (a pocket wall, a ligand to stay clear of), not of any library
    molecule -- which is why the per-molecule store has nowhere to put it and why it arrives as
    a ``screen(..., avoid_points=...)`` keyword instead. It is uploaded ONCE here and broadcast
    over every bucket, exactly as the query's own channels are."""
    cloud = ak.get("avoid_points")
    if cloud is None:
        return batch_kw
    import torch
    batch_kw = dict(batch_kw)
    batch_kw["avoid_points"] = np.ascontiguousarray(np.asarray(cloud, dtype=np.float32))
    batch_kw["avoid_points_t"] = _to_device(batch_kw["avoid_points"], device,
                                            dtype=torch.float32)
    return batch_kw


#: Modes that take the array-native screen path. Every registry mode does: the aligner is one
#: generic body over the mode's channels (``accel/batch/_arrays.py::align_arrays``), so a mode
#: joins by existing rather than by gaining a hand-written builder here. The object path below
#: survives as the parity reference the gates compare against (and for a non-pre-centred /
#: trans_init / jax-backend screen, which reaches neither array branch).
_ARRAY_MODES = tuple(_SPECS)


def _use_arrays(mode: str) -> bool:
    """Whether to take the array-native screen path.

    WHAT THE ARRAY PATH IS WORTH, and the jobs that measured it. Round 1 established that a
    PARTIAL removal of the object model is worth exactly zero -- vectorising the binning alone
    removed the O(K) loop and spent every microsecond back building the per-cell Python lists
    ``Bucket.members`` requires (0.89 vs 0.88 us/mol, reverted) -- so this is all-or-nothing per
    mode rather than a gradual migration.

    THE FIVE INCUMBENTS, measured as the cost of REMOVING the path they already have: 3.87x
    (vol), 7.28x (vol_color), 4.80x (vol_esp), 2.33x (pharm) and 2.02x (vol_and_surf_esp) at
    N=99,984 on an L40S (job 22637592), reproduced at 6.54x (vol_color) / 3.75x (vol) at N=1e5
    on a different store and node (job 22637392).

    SIX MORE were measured afterwards, array leg against object leg at N=99,984 on an L40S,
    each in two independent jobs (22641030 / 22641516, parity from 22640575)::

        vol_esp_tversky  4.97x / 5.12x        vol_lipo  3.70x / 3.56x
        vol_fukui        4.05x / 3.58x        surf_esp  1.34x / 1.29x
        vol_tversky      4.04x / 4.25x        surf      1.33x / 1.29x

    All six were BIT-IDENTICAL to the object path there: 0 of 99,984 scores moved, max|delta|
    exactly 0.000e+00, identical top-1000 ids and identical 4x4 transforms.

    DO NOT QUOTE ONE RANGE ACROSS ALL OF THEM. ``surf`` and ``surf_esp`` gain only ~1.3x because
    they are OPTIMIZER-bound rather than host-bound (0/1 graphed: one eager fine-loop call over
    the 200-point surface clouds for the whole shard), so the host front-end this path deletes
    is a small share of their screen. The modes that joined later, with the generic aligner, are
    UNMEASURED on this axis; measure before quoting a number for them.
    """
    from shepherd_score.accel.batch import _arrays
    return mode in _ARRAY_MODES and _arrays.ENABLED


#: Persistent pinned staging buffers for the shard upload, keyed by (dtype, device index).
#: One per dtype, grown to the largest shard seen; each carries the event that says when the
#: last copy out of it landed, so it can be reused without a blanket stream sync.
_PIN_STAGE: dict = {}


def _to_device(a, device, *, dtype=None):
    """Upload one store array, staged through PINNED host memory, asynchronously.

    Two things, both measured on an L40S vol screen:

    * **stage through pinned memory.** ``torch.as_tensor(numpy_array, device="cuda")`` copies
      from PAGEABLE memory, which the driver must bounce through its own staging buffers while
      the calling thread waits. ``_build_fit_arrays_vol`` measured 0.149 us/mol = 12.1% of the
      wall at N=1,000,000 (job 22594805) doing exactly that, at an effective ~1.2 GB/s. A copy
      out of a pinned buffer is a real asynchronous DMA, so the host hands it to the copy
      engine and goes straight on to enqueue the alignment kernels behind it.
    * **retype on the DEVICE.** A store holds coordinates as float16
      (``ProfileStore.create(dtype="float16")``, the default), so asking for float32 here would
      widen on the host first and push twice the bytes. float16 -> float32 is exact, so doing
      it on the GPU is bit-identical.

    This is deliberately on the CALLING thread. The obvious-looking alternative -- upload the
    next shard from the read-ahead thread -- was built and measured SLOWER twice (0.688x
    pageable, then 0.849x and 0.921x pinned, against 1.02x for the same configuration without
    it), and it cannot be made safe: ``cudaStreamSynchronize`` from a second thread raises
    "operation not permitted when stream is capturing" whenever the main thread is capturing a
    fine-loop CUDA graph, and ``coarse_fine_align_many`` swallows that into a silent fall back
    to the eager loop. See :func:`_iter_shards_prefetched`, which stays I/O-only.
    """
    import torch
    if device.type != "cuda":
        t = torch.as_tensor(a, device=device)
        return t if dtype is None or t.dtype == dtype else t.to(dtype)
    src = torch.from_numpy(np.asarray(a))
    # Key on the RESOLVED device index: torch.device("cuda") carries index None, and two
    # devices must not share one event (pinned host memory is shareable, a CUDA event is not).
    _idx = getattr(device, "index", None)
    key = (src.dtype, torch.cuda.current_device() if _idx is None else int(_idx))
    ent = _PIN_STAGE.get(key)
    if ent is None or ent[0].numel() < src.numel():
        ent = (torch.empty(src.numel(), dtype=src.dtype, pin_memory=True), torch.cuda.Event())
        _PIN_STAGE[key] = ent
    else:
        ent[1].synchronize()          # the previous copy OUT of this buffer has landed
    buf, ev = ent
    flat = buf[:src.numel()]
    flat.copy_(src.reshape(-1))
    out = torch.empty(tuple(src.shape), dtype=src.dtype, device=device)
    out.view(-1).copy_(flat, non_blocking=True)
    ev.record()
    return out if dtype is None or out.dtype == dtype else out.to(dtype)


def _build_fit_arrays(arrs: dict, mode: str, device):
    """Array-native twin of :func:`_build_fit_fast_pairs`: ``(ids, {channel: (flat, off)})``.

    Uploads each of the store's ALREADY-contiguous buffers once, through the pinned staging in
    :func:`_to_device`, and hands the array aligner the buffer plus its CSR offsets -- or the
    dense ``(K, S, ...)`` block plus ``None`` for the fixed-width surface, which needs no
    gather at all. No ``_FastPair``, no ``torch.split``, no per-molecule Python.

    Which arrays a channel comes from is ``channels.load_store_channel``'s business, not this
    function's: the heavy basis in particular is ``heavy_off``/``xyz_noH`` only when some
    molecule's ``Chem.RemoveHs`` retained an H, and the heavy charges on a with-H store are a
    vectorised gather of the with-H array. Reading the keys directly here is how a mode ends up
    silently scoring the wrong field.
    """
    import torch
    from shepherd_score.accel.channels import load_store_channel
    spec = _spec_of(mode)
    fit = {}
    for name in spec.all_channels():
        ch = _CHANNELS[name]
        if ch.is_pair:
            continue
        flat, off = load_store_channel(arrs, name)
        dt = torch.int64 if ch.dtype == "int64" else torch.float32
        fit[name] = (_to_device(flat, device, dtype=dt),
                     None if off is None else _to_device(off, device, dtype=torch.long))
    return arrs["ids"], fit


def _align_fast_arrays(ref: dict, fit: dict, mode: str, batch_kw: dict):
    """Array-native twin of :func:`_align_fast`: ``(scores, SE3)`` for one shard, one query.

    ``batch_kw`` is what :func:`_fast_batch_kwargs` produced, so the mode parameters resolve
    through the SAME ``resolve_params`` the pairwise aligner uses -- which is where the surface
    modes' ``lam`` is scaled by ``LAM_SCALING`` exactly once.
    """
    from shepherd_score.accel.batch._arrays import align_arrays
    from shepherd_score.accel.batch.aligners import resolve_params
    spec = _spec_of(mode)
    params = resolve_params(spec, batch_kw, f"screen({mode})")
    return align_arrays(mode, _ref_channel_tensors(ref, mode), fit, params=params,
                        steps_fine=int(batch_kw["steps_fine"]),
                        num_seeds=batch_kw.get("num_repeats"),
                        const_seeds=batch_kw.get("const_seeds"),
                        avoid=batch_kw.get("avoid_points_t"))


def _make_array_builder(mode: str):
    def build(arrs, device, _m=mode):
        return _build_fit_arrays(arrs, _m, device)
    build.__name__ = build.__qualname__ = f"_build_fit_arrays_{mode}"
    build.__doc__ = (f"Array-native fit builder for ``{mode}``: one name over "
                     ":func:`_build_fit_arrays`, so a parity test can wrap this mode's builder "
                     "alone and assert WHICH one ran.")
    return build


def _make_array_aligner(mode: str):
    def align(ref, fit, batch_kw, _m=mode):
        return _align_fast_arrays(ref, fit, _m, batch_kw)
    align.__name__ = align.__qualname__ = f"_align_fast_arrays_{mode}"
    align.__doc__ = f"Array-native aligner for ``{mode}``; see :func:`_align_fast_arrays`."
    return align


#: mode -> (fit-array builder, array-native aligner). ONE generic body per side, but a DISTINCT
#: named function per mode: the array-vs-object parity gate wraps ``_ARRAY_BUILDERS`` entry by
#: entry to assert which builder a screen actually entered, and a single shared function object
#: could not tell the modes apart. Keys cover ``_ARRAY_MODES`` exactly.
_ARRAY_BUILDERS = {m: _make_array_builder(m) for m in _ARRAY_MODES}
_ARRAY_ALIGNERS = {m: _make_array_aligner(m) for m in _ARRAY_MODES}
for _m, _fn in _ARRAY_BUILDERS.items():
    globals()[_fn.__name__] = _fn
for _m, _fn in _ARRAY_ALIGNERS.items():
    globals()[_fn.__name__] = _fn
del _m, _fn


def _array_dispatch(mode: str):
    """The ``(builder, aligner)`` pair the array-native path uses for ``mode``.

    The SINGLE selector for both drivers -- :func:`_run_shards_inproc` and
    :func:`_screen_worker`. It exists because they used to select independently and the
    worker's copy did not select at all: it called the vol builder and the vol aligner for
    EVERY mode, so ``screen(ndev>1)`` silently returned vol answers under another mode's name
    (``vol_color`` measured max|delta| 3.0654e-01 against a real vol_color screen, with a
    completely different top-10; ``pharm`` raised KeyError instead).

    It reads the TABLES rather than building closures, which is also what lets a test intercept
    one mode's builder by patching ``_ARRAY_BUILDERS``. Keep it one function: a second copy of
    the dispatch is exactly how the worker drifted out of agreement with the driver.
    """
    return _ARRAY_BUILDERS[mode], _ARRAY_ALIGNERS[mode]


def _canonical_rot(store, arrs):
    """The per-molecule rotation a CANONICAL store applies at build time, or ``None``.

    ``None`` for a legacy or ``canonical=False`` store, which is what makes every composition
    below a no-op on those -- the returned pose is already in the molecule's own frame there.
    """
    if not getattr(store, "canonical", False):
        return None
    return arrs.get("rot") if hasattr(arrs, "get") else None


def _compose_rot(T, R):
    """Map one pose out of the CANONICAL frame and back into the molecule's own.

    A canonical store holds ``x_canon = (x_orig - mu) @ R.T`` (``_profile_from_schema`` centres
    first, and refuses ``canonical`` without ``pre_center``), so a transform solved against
    ``x_canon`` satisfies

        x_aligned = x_canon @ T_R.T + T_t = (x_orig - mu) @ (T_R @ R).T + T_t

    -- the rotation composes as ``T_R @ R`` and the translation is unchanged, because both store
    kinds share the same centred origin. Convention is ``points @ R.T + t``, matching
    ``alignment/utils/se3.py::apply_SE3_transform``.

    Without this the SCORES and the RANKING are still right while ``Hit.transform`` silently
    refers to the canonical frame -- a failure no score-based test can see, which is why
    ``tests/test_screen_arrays.py`` re-scores a returned pose instead.

    Called per SURVIVOR, not per molecule: a screen keeps ~k of K, so composing here costs ~1000
    3x3 products instead of one (K,3,3) batched product per shard.
    """
    if R is None or T is None:
        return T
    T = np.array(T, copy=True)
    T[:3, :3] = T[:3, :3] @ np.asarray(R, dtype=T.dtype)
    return T


def _accumulate_arrays(heap, ids, scores, transforms, scores_out, qi, start, rot=None):
    """Array-native twin of :func:`_accumulate`.

    Character-for-character the same reduce -- same block size, same ``threshold()``
    pre-filter, same ascending-index offer order, same ``scores_out`` slice -- except the
    survivor's transform is read from row ``i`` of the (K,4,4) array rather than from a pair
    object. The exactness argument in :func:`_accumulate` carries over unchanged, because it
    rests on ``threshold()`` monotonicity and on a rejected offer mutating nothing, neither of
    which depends on where the transform came from."""
    n = len(ids)
    lo = 0
    while lo < n:
        hi = lo + _ACCUM_BLOCK
        if hi > n:
            hi = n
        thr = heap.threshold()
        if thr == float("-inf"):
            sel = np.arange(lo, hi)
        else:
            sel = np.flatnonzero(scores[lo:hi] > thr) + lo
        # Convert the surviving candidates' scores and ids in ONE vectorised call each,
        # instead of a numpy-scalar access plus ``float()``/``_id_to_py()`` per candidate.
        # ``ndarray.tolist()`` is elementwise ``.item()``, which is exactly what ``_id_to_py``
        # does for a numpy scalar and a no-op for anything else, so the values handed to the
        # heap are unchanged. This is the hot loop: the accumulate is 17.4% of a vol screen's
        # wall at N=100,000 once the canonical composition is deferred (job 22596802), and it
        # runs ~5,200 times for a 1,000-entry heap.
        cs = scores[sel].tolist()
        cid = ids[sel].tolist()
        for j, i in enumerate(sel.tolist()):
            heap.offer_row(cs[j], cid[j], transforms, i, rot)
        lo = hi
    if scores_out is not None and scores_out[qi] is not None:
        scores_out[qi][start:start + n] = scores


def _align_fast(pairs, ref_tensors: dict, mode: str, batch_kw: dict):
    """Set the shared query ref tensors on the resident fit-pairs and run the batched
    aligner; return the per-pair scores (np). Transforms are NOT built here -- they are
    materialized lazily for top-K survivors only (``_TopK.offer_pair``), since building
    all K per shard is the dominant overhead and a screen keeps only ~top_k."""
    from shepherd_score.accel.batch import aligners
    items = tuple(ref_tensors.items())          # materialize the view ONCE, not per pair
    for p in pairs:
        for k, v in items:
            setattr(p, k, v)
    getattr(aligners, "_align_batch_" + mode)(pairs, **batch_kw)
    # The batched aligners write plain Python floats (``scores_cpu.tolist()``), so reading
    # them through ``map(attrgetter(...))`` into a preallocated ``fromiter`` is the same
    # float64 vector as a ``[float(getattr(...)) for p in pairs]`` list comprehension --
    # minus K Python-level ``getattr``/``float`` calls and the intermediate list.
    return np.fromiter(map(_SCORE_GETTER[mode], pairs), dtype=float, count=len(pairs))


class _TopK:
    """Bounded max-list keyed on score (min-heap of size k). Tie-break by a counter
    so transforms are never compared."""
    __slots__ = ("k", "heap", "_c")

    def __init__(self, k):
        self.k = k
        self.heap = []
        self._c = 0

    def _push(self, score, id_, transform):
        self._c += 1
        item = (score, self._c, id_, transform)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def offer_pair(self, score, id_, pair, tf_attr, rot=None):
        """Offer a candidate, materializing its transform from ``pair`` ONLY if the
        score makes the top-K. A screen keeps ~k of K, so this builds ~k transforms
        instead of K (the dominant per-shard overhead). Must be called while ``pair``
        still holds this query's pose (before the next query/shard re-aligns it).

        The MATERIALIZATION has to happen now -- the pair is about to be re-aligned -- but the
        canonical-frame COMPOSITION does not, so it is deferred to :meth:`_materialize` for the
        same reason as :meth:`offer_row`: acceptance into the heap is not survival, and ~4 of
        every 5 accepted candidates are evicted before the screen ends."""
        if len(self.heap) < self.k or score > self.heap[0][0]:
            self._push(score, id_, (_transform_of(pair, tf_attr), rot))

    def offer_row(self, score, id_, transforms, i, rot=None):
        """Array-native twin of :meth:`offer_pair`: the transform comes from row ``i`` of a
        (K,4,4) array instead of an attribute on a pair object.

        Identical acceptance test, identical push, identical ``_c`` tie-break advance -- the
        ONLY difference is where the transform is read from, so the heap state after a shard is
        the same as the object path's down to ties.

        The canonical-frame composition is DEFERRED to :meth:`_materialize`, not done here.
        Acceptance into the heap is not survival: a vol screen at N=100,000 for top_k=1000
        accepts 5,238 candidates, so ~4 of every 5 compositions were being done for a molecule
        evicted before the screen ended. ``_compose_rot`` is a numpy copy plus a 3x3 matmul,
        ~2.7 us of interpreter and allocator time, and it measured at 0.1405 us/mol = **9.7% of
        the whole vol screen's wall clock** (job 22594805, L40S, N=100,000) -- the single
        largest host item in the screen. Storing the pending ``(row, rot_row)`` pair costs two
        numpy views. The rows are views into the shard's arrays, which is exactly what a
        non-canonical store already stores here (``_compose_rot`` returns ``T`` unchanged when
        ``rot`` is None), so this changes nothing about lifetime that was not already true."""
        if len(self.heap) < self.k or score > self.heap[0][0]:
            self._push(score, id_, (transforms[i], None if rot is None else rot[i]))

    @staticmethod
    def _materialize(t):
        """Compose a deferred ``(transform, rot)`` pair; pass anything else through.

        ``offer_pair`` (the object path) pushes a real array, and ``merge_raw`` receives
        already-composed transforms from another worker, so both stay untouched."""
        return _compose_rot(t[0], t[1]) if type(t) is tuple else t

    def threshold(self):
        """Score a candidate must **strictly exceed** to change this heap at all, or
        ``-inf`` while the heap has not yet filled (every offer is accepted then).

        Exactness of the pre-filter in :func:`_accumulate` rests on this being monotone
        non-decreasing once the heap is full: ``_push`` then only ever ``heapreplace``s
        the minimum with a *strictly larger* score, so the minimum never falls. A
        candidate scoring ``<= threshold()`` is therefore guaranteed to be rejected by
        every later ``offer_pair`` in the batch too -- and a rejected ``offer_pair``
        mutates nothing (no push, no ``_c`` increment), so skipping it is a bit-exact
        no-op rather than an approximation.
        """
        # ``self.k`` guard keeps a degenerate k=0 heap failing exactly where it does today
        # (inside offer_pair), instead of raising from here.
        return self.heap[0][0] if (self.k and len(self.heap) >= self.k) else float("-inf")

    def merge_raw(self, raw):
        for (s, i, t) in raw:
            self._push(s, i, t)

    def raw(self):
        # Composed on the way OUT, so what crosses a process boundary (multi_gpu merges heaps
        # through raw()/merge_raw) is a plain (4,4) array, never a view that would drag its
        # whole shard along through pickle.
        return [(s, i, self._materialize(t)) for (s, _, i, t) in self.heap]

    def sorted(self):
        return [Hit(score=s, id=i, transform=self._materialize(t))
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
    if alpha is not None:
        align_kwargs["alpha"] = alpha
    # A parameter the registry declares REQUIRED (spec default None) must be supplied. Raise the
    # same clear error on both the fast and the slow path, rather than letting the fast one fail
    # deep inside _fast_batch_kwargs: ``vol_esp`` needs ``lam`` (the ESP / partial-charge weight,
    # which the per-pair API also makes required), and ``vol_and_surf_esp`` needs ``alpha``,
    # which selects volumetric shape at 0.81 and surface shape otherwise.
    for _name, _default in _spec_of(mode).params.items():
        if _default is None and _name not in align_kwargs:
            raise ValueError(f"{mode} requires an explicit {_name}=...; pass {_name}=...")
    if _spec_of(mode).name == "vol_avoid" and align_kwargs.get("avoid_points") is None:
        raise ValueError("vol_avoid requires an explicit avoid_points=... (an (K,3) cloud in "
                         "the QUERY's frame to keep library molecules out of)")
    return align_kwargs


def _iter_shards_prefetched(store, shard_idxs):
    """Yield ``(shard_meta, arrays)`` for ``shard_idxs`` **in order**, reading the next shard
    on a single background thread so the disk read overlaps the current shard's alignment.

    Pure I/O overlap, and deliberately ONLY that: the obvious extension -- have this thread do
    the host-to-device upload as well -- was built and measured SLOWER twice (0.688x pageable,
    0.849x/0.921x pinned, against 1.02x without it), and it cannot be made safe. A second
    thread calling ``cudaStreamSynchronize`` raises "operation not permitted when stream is
    capturing" whenever the main thread is capturing a fine-loop CUDA graph, and
    ``coarse_fine_align_many`` swallows that into a silent fall back to the eager loop. The
    upload stays on the calling thread; see :func:`_to_device`.

    The worker touches no shared mutable state: :meth:`ProfileStore.read_shard` opens its own
    file handle and returns fresh arrays. The consumer still sees shards strictly in
    ``shard_idxs`` order; a read that raises is re-raised in the caller's thread by
    ``Future.result()`` before the shard is yielded, and the executor is shut down (joining the
    in-flight read) on any exit path, including the generator being closed early.

    Costs one extra resident shard.
    """
    idxs = list(shard_idxs)
    if len(idxs) < 2:
        for i in idxs:
            yield store.read_shard(i)
        return
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fss-screen-prefetch")
    try:
        fut = ex.submit(store.read_shard, idxs[0])
        for nxt in idxs[1:]:
            cur = fut.result()                  # re-raises a failed read here, in order
            fut = ex.submit(store.read_shard, nxt)
            yield cur
            del cur                             # drop before waiting on the next read
        yield fut.result()
    finally:
        ex.shutdown(wait=True)                  # never leave a reader thread behind


def _canonical_batch_kw(store, qs_ref, mode, device, batch_kw, fast=True):
    """``batch_kw`` plus ``const_seeds`` when this screen can use them: a CANONICAL store, the
    array path, ONE query, and a mode that seeds from the heavy-atom cloud
    (``_CONST_SEED_MODES``). Seeds are then one constant set for the whole screen (see
    _common.canonical_seed_quats) instead of a per-molecule eigensolve, which is the 1.5-2x the
    canonical store exists for on ``vol``. Unchanged ``batch_kw`` otherwise.

    One query, because the set depends on the query's frame: :func:`_canonical_batch_kws` is the
    per-panel form, one ``batch_kw`` per query.

    ``vol_and_surf_esp`` qualifies only at ``alpha == 0.81``, where its driver seeds from the atom
    clouds; at any other alpha it seeds from the surfaces, which the store does not canonicalise.

    ONE helper for the in-process shard loop AND the multi-GPU worker, deliberately: the worker
    used to take ``batch_kw`` as handed to it, so on the same canonical store it ran the
    per-molecule seeds and cost 2.1x the in-process screen per shard (measured 2026-09-14,
    Shepherd-Score-Paper fig2_speed/p6_gpu_probe.py: 1.98 s against 0.94 s for 1e6 vol
    conformers on one L40S, whatever the host-thread cap or socket), which is why two devices
    screened no faster than one.

    Gated on ``fast and _use_arrays`` because ``const_seeds`` is a parameter of
    ``align_batch_vol_arrays`` ALONE -- the object path's ``_align_batch_vol`` has no such
    keyword and raises TypeError on it. Those routes keep the per-molecule PCA seeds, which stay
    CORRECT on a canonical store (they are derived from whatever coordinates it holds); they
    just forgo the speedup.
    """
    if not (fast and _use_arrays(mode) and getattr(store, "canonical", False)
            and mode in _CONST_SEED_MODES and len(qs_ref) == 1):
        return batch_kw
    _spec = _spec_of(mode)
    # The seed channel is resolved under THIS call's keywords: the combo modes seed from the
    # atom clouds only at alpha == 0.81 and from the surfaces otherwise, and the store
    # canonicalises the atom frame alone.
    _seed_ch = _spec.resolve_channel(_spec.seed_channel, batch_kw)
    if _seed_ch not in ("atoms", "heavy"):
        return batch_kw
    _rx = qs_ref[0].get(_seed_ch)
    if _rx is None:
        return batch_kw
    from shepherd_score.accel.drivers._common import canonical_seed_quats
    from .accel._modes import MODE_SEEDS
    batch_kw = dict(batch_kw)
    batch_kw["const_seeds"] = canonical_seed_quats(_rx, len(_rx), int(MODE_SEEDS.get(mode, 10)),
                                                   device)
    return batch_kw


def _canonical_batch_kws(store, qs_ref, mode, device, batch_kw, fast=True):
    """One ``batch_kw`` per query of a panel: :func:`_canonical_batch_kw` applied per query.

    The constant seed set is a function of the QUERY's principal frame alone, so a panel simply
    gets one set per query; the single-query gate above is about deriving one set from one
    frame, not a limit of the method. Every entry is the caller's own ``batch_kw`` object when the
    store or mode does not qualify, so the object path (which has no ``const_seeds`` keyword)
    never sees it.
    """
    return [_canonical_batch_kw(store, [ra], mode, device, batch_kw, fast) for ra in qs_ref]


def _run_shards_inproc(store, shard_idxs, qs_ref, mode, device, top_k, batch_kw,
                       align_kwargs, backend, fast, center_profiles, scores_out, progress,
                       n_total):
    """Process ``shard_idxs`` against the query panel, one shard load per shard,
    aligning every query against it. Returns a ``_TopK`` per query."""
    # This driver shards the library itself (one shard per call), so the aligner's
    # transparent multi-GPU dispatch must be suppressed for the duration: the pairs here
    # are lightweight ``_FastPair`` stand-ins that carry only cached tensors -- no
    # ``Molecule`` -- so any dispatch path that re-materializes per-pair Molecule arrays
    # (``p.ref_molec.atom_pos``) would raise AttributeError, and on a multi-GPU host a
    # large shard would otherwise trip the dispatcher's single-GPU warning mid-screen.
    # Restore the previous value afterwards: this runs in the caller's process/thread and
    # may itself be nested inside a per-GPU worker that already set the flag.
    try:
        from shepherd_score.accel.batch import _DISPATCH_LOCAL
    except Exception:
        from shepherd_score.container._core import _DISPATCH_LOCAL
    _prev_active = getattr(_DISPATCH_LOCAL, "active", False)
    _DISPATCH_LOCAL.active = True
    try:
        heaps = [_TopK(top_k) for _ in qs_ref]
        tf_attr = _TRANSFORM_ATTR[mode]
        # CANONICAL store: one constant seed set per QUERY for the whole screen, computed once
        # here instead of per molecule per bucket. The same helper serves the multi-GPU worker.
        batch_kws = _canonical_batch_kws(store, qs_ref, mode, device, batch_kw, fast)
        done = 0
        if not fast:
            from shepherd_score.container import MoleculePair, MoleculePairBatch
        # Shards arrive in order from the read-ahead reader; ``arrs`` is exactly what
        # ``store.read_shard(idx)`` returned, just read one shard earlier.
        for sh, arrs in _iter_shards_prefetched(store, shard_idxs):
            # A canonical store's coordinates are rotated into each molecule's principal frame,
            # so every pose below is solved in THAT frame and has to be composed back. ``None``
            # for every other store, which makes each composition a no-op there. See _compose_rot.
            rot = _canonical_rot(store, arrs)
            if fast:
                # ``ids`` stays the raw store array: _accumulate applies ``_id_to_py`` to
                # top-K survivors only, instead of converting every library molecule here.
                start = sh["start"]
                if _use_arrays(mode):
                    # ARRAY-NATIVE PATH: no per-molecule Python objects
                    # anywhere between the store and the heap. See accel/batch/_arrays.py.
                    # vol used to be an inline branch here so its gate-5 bit-identity stayed
                    # visibly untouched; it is a table entry like the rest now, which moves no
                    # arithmetic (same builder, same tensors, same kwargs into
                    # align_batch_vol_arrays) and leaves _array_dispatch as the one selector
                    # this driver and the multi-GPU worker share.
                    build, align = _array_dispatch(mode)
                    ids, fit = build(arrs, device)
                    for qi, ra in enumerate(qs_ref):
                        ref = _ref_tensors_from_arrays(ra, mode, device)
                        scores, se3 = align(ref, fit, batch_kws[qi])
                        _accumulate_arrays(heaps[qi], ids, scores, se3,
                                           scores_out, qi, start, rot)
                else:
                    ids, pairs = _build_fit_fast_pairs(arrs, mode, device,
                                                       avoid=batch_kw.get("avoid_points"))
                    for qi, ra in enumerate(qs_ref):
                        ref = _ref_tensors_from_arrays(ra, mode, device)
                        scores = _align_fast(pairs, ref, mode, batch_kw)
                        _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi,
                                    start, rot)
            else:
                profiles = store._reconstruct(arrs, sh)   # == store.read_profiles(idx)
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
                    _accumulate(heaps[qi], ids, scores, pairs, tf_attr, scores_out, qi,
                                start, rot)
            done += sh["n"]
            if progress:
                print(f"[screen] {done}/{n_total} library molecules aligned "
                      f"x {len(qs_ref)} queries", flush=True)
        return heaps
    finally:
        _DISPATCH_LOCAL.active = _prev_active


# Candidates are pre-filtered against the heap threshold in blocks of this many, so the
# threshold used is refreshed as the heap tightens instead of being read once per shard
# (a 100k-molecule shard would otherwise pre-filter its whole tail against the stale
# threshold it had before its own first molecule was offered). Block size only trades a
# handful of numpy calls against a few wasted offers; it never changes the result.
_ACCUM_BLOCK = 4096


def _accumulate(heap, ids, scores, pairs, tf_attr, scores_out, qi, start, rot=None):
    """Reduce one shard's scores for one query: full score vector out, top-K heap in.

    ``scores_out`` still receives EVERY score, in library order, via the same single
    vectorised slice assignment as before.

    The heap, by contrast, is only ever changed by a candidate that strictly beats its
    current minimum, so the per-molecule Python offer loop is pre-selected in C: a
    numpy ``> threshold`` comparison plus ``flatnonzero`` picks the candidates that can
    actually enter, and only those pay a ``float()``, an ``_id_to_py()`` and a heap call.
    On a large screen the heap threshold sits near the k-th best score seen so far, so
    this is a handful of survivors per block instead of one Python iteration per library
    molecule.

    **This is exact, not approximate.** ``_TopK.threshold()`` is ``-inf`` until the heap
    fills (nothing is skipped during that phase) and non-decreasing afterwards, so a
    candidate scoring ``<= threshold`` at the start of a block still scores ``<=`` the
    heap minimum when its turn comes and would be rejected by ``offer_pair``. A rejected
    ``offer_pair`` performs no push and does not advance the ``_c`` tie-break counter, so
    it leaves *no* trace: skipping it reproduces the old push sequence, the old counters,
    the old heap array layout and therefore the old hit order down to ties. Survivors are
    still offered in ascending library index, and still inside this query/shard iteration
    so ``offer_pair`` materialises each transform while its ``pair`` holds THIS query's
    pose.
    """
    n = len(ids)
    lo = 0
    while lo < n:
        hi = lo + _ACCUM_BLOCK
        if hi > n:
            hi = n
        thr = heap.threshold()
        if thr == float("-inf"):
            sel = np.arange(lo, hi)                  # heap not full: every offer is taken
        else:
            sel = np.flatnonzero(scores[lo:hi] > thr) + lo
        # One vectorised conversion per BLOCK instead of a numpy-scalar access plus
        # ``float()``/``_id_to_py()`` per candidate. ``ndarray.tolist()`` is elementwise
        # ``.item()``, which is exactly what ``_id_to_py`` does for a numpy scalar, so the
        # values reaching the heap are unchanged. Same change as in _accumulate_arrays.
        cs = scores[sel].tolist()
        cid = np.asarray(ids)[sel].tolist()
        for j, i in enumerate(sel.tolist()):
            heap.offer_pair(cs[j], cid[j], pairs[i], tf_attr,
                            None if rot is None else rot[i])
        lo = hi
    if scores_out is not None and scores_out[qi] is not None:
        scores_out[qi][start:start + n] = scores


def _normalize_scores_out(scores_out, n_queries):
    if scores_out is None:
        return [None] * n_queries
    if isinstance(scores_out, np.ndarray) and n_queries == 1:
        return [scores_out]
    if isinstance(scores_out, (list, tuple)) and len(scores_out) == n_queries:
        return list(scores_out)
    raise ValueError("scores_out must be None, a single array (1 query), or a list "
                     "of one array per query")


def screen_many(queries: Sequence, store: "ProfileStore", mode: str = "surf_esp", *,
                backend: Optional[str] = None, do_center: Optional[bool] = None,
                top_k: int = 1000, ndev: Optional[int] = None,
                scores_out=None, alpha: Optional[float] = None,
                progress: bool = False, **align_kwargs) -> List[List["Hit"]]:
    """Screen a **panel** of queries against ``store`` in a single streaming pass.

    Each shard is read from disk **once** and aligned against *every* query (so the
    library is streamed once for the whole panel, not once per query). For the fast
    modes (all of ``vol/vol_esp/surf/surf_esp/pharm/vol_color/vol_and_surf_esp``) on a pre-centered
    store, the shard's fit tensors are built once on-device and reused across the panel
    via the direct array->kernel path (no per-molecule ``MoleculeProfile``/``MoleculePair``).

    Returns a list aligned with ``queries``: ``out[j]`` is query ``j``'s ``top_k``
    ``Hit``s (sorted, descending).

    See :func:`screen` for the per-query parameters. ``scores_out`` may be a list of
    one preallocated array per query (single-process only). ``ndev>1`` streams shards
    across one worker process per GPU, spawned on the first such call and kept until
    :func:`close_multigpu_pool` or interpreter exit (fast modes only).
    """
    import torch
    queries = list(queries)
    mode = _canon_mode(mode)                      # accept legacy esp / esp_combo everywhere below
    align_kwargs = _resolve_screen(store, mode, alpha, align_kwargs)
    backend = backend or _default_backend()

    _spec = _spec_of(mode)
    _chans = [_CHANNELS[c] for c in _spec.all_channels()]
    if any(c.basis == "surf" for c in _chans):
        for q in queries:
            qn = getattr(q, "num_surf_points", None)
            if qn is not None and qn != store.num_surf_points:
                raise ValueError(f"query num_surf_points ({qn}) != store "
                                 f"({store.num_surf_points}); ALPHA is calibrated to it")

    # Fast-path query preconditions: name the missing field up front instead of crashing
    # opaquely inside _query_ref_arrays. Each channel's own reader raises a ValueError naming
    # what it needed, so the check is simply "read them all once" -- a bare RDKit-backed
    # Molecule has everything; a MoleculeProfile reconstructed without the mode's arrays does
    # not, and this is where that is reported rather than mid-screen.
    for q in queries:
        for c in _chans:
            if c.is_pair:
                continue
            try:
                c.read(q)
            except ValueError as e:
                raise ValueError(f"{mode} query cannot provide {c.name}: {e}") from e

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

    # The fast CPU path runs the batched kernels through numba. Without it the aligner falls
    # into a per-pair fallback that calls p.align_with_<mode>() -- which the lightweight
    # _FastPair stand-ins on the fast screen path don't have, so it dies with a confusing
    # AttributeError deep inside the aligner. Fail clearly up front instead.
    if fast and device.type == "cpu":
        try:
            import numba  # noqa: F401
        except ImportError:
            raise ImportError(
                "fast CPU screen requires numba (the batched CPU kernels run on it) -- "
                "install it (pip install numba) or screen on a GPU backend") from None

    if ndev and ndev > 1:
        if not fast:
            raise ValueError("ndev>1 requires the fast path (a pre-centered store, a "
                             f"{sorted(_FAST_MODES)} mode, trans_init=False, GPU backend)")
        if scores_out is not None:
            # The multi-GPU workers return only per-query top-K heaps; there is no path
            # for a full score vector back to the parent. Fail loudly rather than silently
            # leave the caller's preallocated array unwritten.
            raise ValueError("scores_out is not supported with ndev>1 (multi-GPU screening "
                             "returns top-K hits only). Run single-process for full score vectors.")
        heaps = _screen_many_multigpu(qs, store.path, mode, ndev,
                                      _fast_batch_kwargs(mode, align_kwargs), top_k, progress)
        # NB the multi-GPU workers rebuild their own device tensors, so the avoid cloud crosses
        # as the plain numpy array in ``align_kwargs`` and is uploaded inside each worker.
        return [h.sorted() for h in heaps]

    so = _normalize_scores_out(scores_out, len(queries))
    if fast:
        qs_ref = [_query_ref_arrays(q, mode) for q in qs]
        batch_kw = _fast_batch_kwargs(mode, align_kwargs)
        batch_kw = _with_avoid(batch_kw, align_kwargs, device)
    else:
        qs_ref = qs
        batch_kw = None
    heaps = _run_shards_inproc(store, range(store.num_shards), qs_ref, mode, device, top_k,
                               batch_kw, align_kwargs, backend, fast, center_profiles, so,
                               progress, len(store))
    return [h.sorted() for h in heaps]


def screen(query, store: "ProfileStore", mode: str = "surf_esp", *,
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
        One of ``vol vol_esp surf surf_esp pharm vol_and_surf_esp vol_color`` (legacy
        ``esp``/``esp_combo`` accepted; must be supported by
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
        Stream shards across this many GPUs, one worker process per device (fast modes only).
        The workers are spawned on the first ``ndev>1`` call and kept for later screens;
        :func:`close_multigpu_pool` releases them (also run at interpreter exit).
    scores_out : np.ndarray, optional
        Preallocated ``(len(store),)`` array (e.g. an ``np.memmap``) written with every
        score in library order. Single-process only.
    alpha : float, optional
        Shape Gaussian width; auto-fills ``ALPHA(num_surf_points)`` for ``surf``/``esp``,
        required for ``vol_and_surf_esp`` (``alpha=0.81`` selects volumetric shape, else surface),
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
# Multi-GPU: ONE persistent worker process per GPU, spawned on first use and kept for the life
# of the calling process, each holding its device and receiving whole screens as jobs. Within a
# screen a worker owns a static share of the shards -- r, r+ndev, r+2*ndev, ... -- and streams
# them with the same read-ahead thread the single-process screen uses, so the disk read of one
# shard overlaps the alignment of the previous one. Both halves were measured before they were
# written (Shepherd-Score-Paper, SI): with a fresh pool per call and a serial read-then-align loop
# per worker, four L40S screened 10^7 vol conformers no faster than one (9.2 s either way) and two
# were slower (13.4 s), because the ~4 s spawn was paid every call and a serial worker cost ~2.2x
# the pipelined single process per shard.
# --------------------------------------------------------------------------- #
_MGPU_POOL = None                        # {"key": (ndev, threads), "procs", "job_qs", "out_q"}


def _screen_worker(rank, threads, store_path, ref_arrays_list, mode, batch_kw, top_k,
                   shard_q, out_q):
    """One device's share of one screen, run inside its worker process.

    ``shard_q`` is either this worker's static LIST of shard indices -- streamed through
    :func:`_iter_shards_prefetched`, so shard i+1 is read while shard i aligns -- or a queue
    yielding indices and then ``None`` (the original work-stealing form, kept for callers and
    tests that drive a worker by hand; it reads and aligns serially). Results go to ``out_q`` as
    ``(rank, per_query_raw_heaps)``; an exception goes there as ``(rank, "__ERR__", traceback)``.
    """
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
        # The canonical store's constant seeds, exactly as the in-process loop sets them (one set
        # per query); without this the worker ran per-molecule seeds on the same store at 2.1x
        # the cost per shard.
        batch_kws = _canonical_batch_kws(store, ref_arrays_list, mode, dev, batch_kw)

        def _drain(q):                                     # the queue form: serial reads
            while True:
                idx = q.get()
                if idx is None:
                    return
                yield store.read_shard(idx)

        if isinstance(shard_q, (list, tuple, range)):
            share = list(shard_q)
            shards = _iter_shards_prefetched(store, share) if share else iter(())
        else:
            shards = _drain(shard_q)
        for _sh, arrs in shards:
            rot = _canonical_rot(store, arrs)      # canonical-frame stores; None otherwise
            if _use_arrays(mode):
                # THE SAME selector the in-process driver uses, deliberately: this branch was
                # hardwired to the vol builder and the vol aligner, so every non-vol mode
                # screened here came back with vol answers under its own name (vol_color
                # measured max|delta| 3.07e-01 and a different top-10; pharm raised KeyError).
                # See _array_dispatch.
                build, align = _array_dispatch(mode)
                ids, fit = build(arrs, dev)
                for qi, ref in enumerate(ref_tensors):
                    scores, se3 = align(ref, fit, batch_kws[qi])
                    _accumulate_arrays(heaps[qi], ids, scores, se3, None, qi, 0, rot)
                torch.cuda.synchronize()
                continue
            ids, pairs = _build_fit_fast_pairs(arrs, mode, dev,
                                               avoid=batch_kw.get("avoid_points"))
            for qi, ref in enumerate(ref_tensors):
                scores = _align_fast(pairs, ref, mode, batch_kw)
                # Same pre-filtered reduce as the in-process driver (scores_out is not
                # supported with ndev>1, hence the None).
                _accumulate(heaps[qi], ids, scores, pairs, tf_attr, None, qi, 0, rot)
            torch.cuda.synchronize()
        out_q.put((rank, [h.raw() for h in heaps]))
    except Exception:                            # noqa: BLE001 - relayed to parent
        import traceback
        out_q.put((rank, "__ERR__", traceback.format_exc()))


def _mgpu_pool_worker(rank, threads, job_q, out_q):
    """The persistent per-device process: one CUDA device, many screens. A job is
    ``(store_path, ref_arrays_list, mode, batch_kw, top_k, shard_list)``; ``None`` ends it."""
    while True:
        job = job_q.get()
        if job is None:
            return
        store_path, ref_arrays_list, mode, batch_kw, top_k, shards = job
        _screen_worker(rank, threads, store_path, ref_arrays_list, mode, batch_kw, top_k,
                       shards, out_q)


def close_multigpu_pool():
    """Shut down the persistent ``screen(ndev>1)`` worker pool, if one is running. Registered
    with :mod:`atexit`; call it yourself to release the devices earlier."""
    global _MGPU_POOL
    pool, _MGPU_POOL = _MGPU_POOL, None
    if pool is None:
        return
    for q in pool["job_qs"]:
        try:
            q.put(None)
        except Exception:                        # noqa: BLE001 - a dead queue is already closed
            pass
    for p in pool["procs"]:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()


def _mgpu_pool(ndev, threads):
    """The pool for ``(ndev, threads)``: reused while it is alive, (re)spawned otherwise. Spawning
    ndev CUDA processes costs seconds (about 4 s for four L40S) and used to be paid on every
    call; a screen now pays it once per process lifetime."""
    global _MGPU_POOL
    pool = _MGPU_POOL
    if pool is not None and pool["key"] == (ndev, threads) and all(p.is_alive() for p in pool["procs"]):
        return pool
    close_multigpu_pool()
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    job_qs, procs = [], []
    for r in range(ndev):
        q = ctx.Queue()
        p = ctx.Process(target=_mgpu_pool_worker, args=(r, threads, q, out_q), daemon=True)
        p.start()
        job_qs.append(q)
        procs.append(p)
    _MGPU_POOL = {"key": (ndev, threads), "procs": procs, "job_qs": job_qs, "out_q": out_q}
    return _MGPU_POOL


def _screen_many_multigpu(qs, store_path, mode, ndev, batch_kw, top_k, progress):
    import os as _os
    import torch
    from queue import Empty

    ndev = max(1, min(ndev, torch.cuda.device_count() if torch.cuda.is_available() else 1))
    try:
        cores = len(_os.sched_getaffinity(0))
    except AttributeError:
        cores = _os.cpu_count() or ndev
    threads = max(1, cores // ndev)
    ref_arrays_list = [_query_ref_arrays(q, mode) for q in qs]

    store = ProfileStore.open(store_path)
    n_shards = store.num_shards

    # The thread caps are read by the workers at spawn; set them around the spawn only.
    _saved = {k: _os.environ.get(k) for k in
              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in _saved:
        _os.environ[k] = str(threads)
    try:
        pool = _mgpu_pool(ndev, threads)
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v

    # Static, interleaved shares: shards are equal-sized except the last, so this is balanced
    # without a shared queue, and a static share is what lets a worker read ahead.
    for r, q in enumerate(pool["job_qs"]):
        q.put((store_path, ref_arrays_list, mode, batch_kw, top_k, list(range(r, n_shards, ndev))))
    results, errs = {}, []
    out_q = pool["out_q"]
    while len(results) + len(errs) < ndev:
        try:
            msg = out_q.get(timeout=5.0)
        except Empty:
            dead = [p.pid for p in pool["procs"] if not p.is_alive()]
            if dead:
                close_multigpu_pool()
                raise RuntimeError(f"multi-GPU screen: worker process(es) {dead} died")
            continue
        if len(msg) == 3 and msg[1] == "__ERR__":
            errs.append((msg[0], msg[2]))
        else:
            results[msg[0]] = msg[1]
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


atexit.register(close_multigpu_pool)
