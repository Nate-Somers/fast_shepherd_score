"""
REAL-molecule workloads for the alignment benchmark.

Provenance of the test molecules
---------------------------------
These are real, marketed small-molecule drugs. The SMILES below are the standard
canonical structures (as found in DrugBank / PubChem / Wikipedia chemical
infoboxes); each is labelled with its name and heavy-atom count. 3-D conformers
are generated with RDKit ETKDG + MMFF94 optimisation
(`shepherd_score.conformer_generation.embed_conformer_from_smiles`), molecular
surfaces with the repo's own surface sampler (Open3D), partial charges with
MMFF94 (RDKit; no xtb needed), and pharmacophores with the repo's RDKit feature
factory. So every array the aligners consume comes from a real molecule, not a
random point cloud.

For a speed/parity benchmark each pair is (ref = real molecule, fit = a rigid
SE(3)-transformed copy of the same molecule). Using a transformed self-copy
keeps a *known* optimum (perfect overlap recoverable) while exercising the real
molecular geometry / surface / ESP / pharmacophore arrays — exactly what the
timing depends on.

Bucketing: the GPU batch path buckets by the mode's point count (band = 16:
surface points for surf/esp, atoms for vol, pharmacophore features for pharm).
``make_real_cohort`` builds either a *same-bucket* cohort (molecules whose count
lands in one band) or a *cross-bucket* cohort (molecules spread across bands),
so the bucketing penalty can be isolated on real data.
"""
from __future__ import annotations

import functools
import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --- self-contained pair/cohort containers + geometry helper ----------------
# (previously imported from the now-removed alignment_bench package)
@dataclass
class PairSpec:
    """One reference/fit pair plus the ground-truth transform used to build it."""
    ref: object
    fit: object
    R: np.ndarray            # (3, 3) rotation applied to make fit
    t: np.ndarray            # (3,)   translation applied to make fit
    n_ref: int               # mode-relevant point count of ref (the "size")
    n_fit: int


@dataclass
class Cohort:
    """A reproducible set of pairs sharing a size/bucket policy."""
    name: str
    mode: str
    pairs: List[PairSpec]
    size_kind: str           # "same" | "cross"
    seed: int
    noise: float
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3x3 proper rotation matrix (via QR of a Gaussian)."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


def _disk_cache_path(smiles: str, surf_per_atom: int, seed: int) -> Optional[str]:
    """Path for a cached _RealMol, or None if disk caching is off.

    Enabled by setting env FSS_MOL_CACHE_DIR (used by the per-cell subprocess
    benchmark so fresh processes don't each rebuild the molecules). Transparent:
    the build is deterministic, so a cache hit is identical to a rebuild.
    """
    d = os.environ.get("FSS_MOL_CACHE_DIR")
    if not d:
        return None
    key = hashlib.md5(f"v1|{smiles}|{surf_per_atom}|{seed}".encode()).hexdigest()
    return os.path.join(d, key + ".pkl")

# (name, SMILES, approx heavy-atom count) — real marketed drugs, small -> large.
DRUGS: List[Tuple[str, str, int]] = [
    ("benzene",        "c1ccccc1", 6),
    ("phenol",         "Oc1ccccc1", 7),
    ("aniline",        "Nc1ccccc1", 7),
    ("paracetamol",    "CC(=O)Nc1ccc(O)cc1", 11),
    ("salicylic_acid", "OC(=O)c1ccccc1O", 11),
    ("aspirin",        "CC(=O)Oc1ccccc1C(=O)O", 13),
    ("ibuprofen",      "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 15),
    ("caffeine",       "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 14),
    ("naproxen",       "COc1ccc2cc(ccc2c1)C(C)C(=O)O", 17),
    ("paracetamol2",   "CC(=O)Nc1ccc(OC)cc1", 12),
    ("warfarin",       "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 25),
    ("diphenhydramine","O(CCN(C)C)C(c1ccccc1)c1ccccc1", 18),
    ("indomethacin",   "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", 28),
    ("sildenafil",     "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1", 33),
    ("imatinib",       "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", 37),
]


@dataclass
class _RealMol:
    """Lightweight, duck-typed molecule exposing exactly what the aligners read."""
    atom_pos: np.ndarray
    surf_pos: np.ndarray
    surf_esp: np.ndarray
    partial_charges: np.ndarray
    pharm_types: np.ndarray
    pharm_ancs: np.ndarray
    pharm_vecs: np.ndarray

    @property
    def _nonH_atoms_idx(self):
        return np.arange(self.atom_pos.shape[0])

    def center_to(self, xyz_mean):
        self.atom_pos = self.atom_pos - xyz_mean
        self.surf_pos = self.surf_pos - xyz_mean
        self.pharm_ancs = self.pharm_ancs - xyz_mean


@functools.lru_cache(maxsize=None)
def _build_molecule(smiles: str, surf_per_atom: int = 3, seed: int = 42) -> _RealMol:
    # Optional cross-process disk cache (FSS_MOL_CACHE_DIR) so per-cell subprocess
    # benchmarks don't rebuild the molecules every process.
    _cpath = _disk_cache_path(smiles, surf_per_atom, seed)
    if _cpath and os.path.exists(_cpath):
        try:
            with open(_cpath, "rb") as _f:
                return pickle.load(_f)
        except Exception:
            pass

    from rdkit import Chem
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule

    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    m = Molecule(rd, num_surf_points=ns, pharm_multi_vector=False)
    mol = _RealMol(
        atom_pos=np.asarray(m.atom_pos, dtype=np.float64),
        surf_pos=np.asarray(m.surf_pos, dtype=np.float64),
        surf_esp=np.asarray(m.surf_esp, dtype=np.float64),
        partial_charges=np.asarray(m.partial_charges, dtype=np.float64),
        pharm_types=np.asarray(m.pharm_types, dtype=np.int64),
        pharm_ancs=np.asarray(m.pharm_ancs, dtype=np.float64),
        pharm_vecs=np.asarray(m.pharm_vecs, dtype=np.float64),
    )
    if _cpath:
        try:
            os.makedirs(os.path.dirname(_cpath), exist_ok=True)
            with open(_cpath, "wb") as _f:
                pickle.dump(mol, _f)
        except Exception:
            pass
    return mol


def _transform(mol: _RealMol, R: np.ndarray, t: np.ndarray) -> _RealMol:
    vecs = mol.pharm_vecs @ R.T
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(n > 0, n, 1.0)
    return _RealMol(
        atom_pos=mol.atom_pos @ R.T + t,
        surf_pos=mol.surf_pos @ R.T + t,
        surf_esp=mol.surf_esp.copy(),
        partial_charges=mol.partial_charges.copy(),
        pharm_types=mol.pharm_types.copy(),
        pharm_ancs=mol.pharm_ancs @ R.T + t,
        pharm_vecs=vecs,
    )


def _count_for_mode(mol: _RealMol, mode: str) -> int:
    if mode in ("surf", "esp"):
        return mol.surf_pos.shape[0]
    if mode == "pharm":
        return mol.pharm_ancs.shape[0]
    return mol.atom_pos.shape[0]


def molecule_table(mode: str, surf_per_atom: int = 3) -> List[Tuple[str, int, int]]:
    """Return [(name, heavy, mode_count_for_bucketing)] for the curated set."""
    out = []
    for name, smi, heavy in DRUGS:
        m = _build_molecule(smi, surf_per_atom=surf_per_atom)
        out.append((name, heavy, _count_for_mode(m, mode)))
    return out


def make_real_cohort(mode: str, *, n_pairs: int, bucket_kind: str,
                     trans_max: float = 3.0, rot_max_deg: float = 60.0,
                     surf_per_atom: int = 3, seed: int = 3) -> Cohort:
    """Cohort of (real molecule, SE(3)-copy) pairs.

    bucket_kind:
      'same'  -> sample only from molecules whose mode-count lands in ONE band
                 (the most populated band) -> single GPU bucket.
      'cross' -> sample across the whole size range -> many buckets.
    """
    from shepherd_score.container._core import _band_key
    rng = np.random.default_rng(seed)

    mols = [_build_molecule(smi, surf_per_atom=surf_per_atom) for _, smi, _ in DRUGS]
    names = [n for n, _, _ in DRUGS]
    bands = [_band_key(_count_for_mode(m, mode)) for m in mols]

    if bucket_kind == "same":
        # pick the most populated band and use only those molecules
        vals, counts = np.unique(bands, return_counts=True)
        target = int(vals[np.argmax(counts)])
        idx_pool = [i for i, b in enumerate(bands) if b == target]
    elif bucket_kind == "cross":
        idx_pool = list(range(len(mols)))
    else:
        raise ValueError(bucket_kind)

    pairs: List[PairSpec] = []
    chosen = rng.choice(idx_pool, size=n_pairs, replace=True)
    for i in chosen:
        ref = mols[int(i)]
        for _ in range(16):
            R = _random_rotation(rng)
            if (np.trace(R) - 1.0) / 2.0 >= np.cos(np.deg2rad(rot_max_deg)):
                break
        t = rng.standard_normal(3)
        t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * trans_max)
        fit = _transform(ref, R, t)
        pairs.append(PairSpec(ref=ref, fit=fit, R=R, t=t,
                              n_ref=_count_for_mode(ref, mode),
                              n_fit=_count_for_mode(fit, mode)))

    name = f"real-{bucket_kind}bucket"
    return Cohort(name=name, mode=mode, pairs=pairs, size_kind=bucket_kind,
                  seed=seed, noise=0.0,
                  meta={"n_pairs": n_pairs, "molecules": names,
                        "bands": bands, "pool": [names[i] for i in idx_pool]})
