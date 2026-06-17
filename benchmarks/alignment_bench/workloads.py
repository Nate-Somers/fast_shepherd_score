"""
Controlled, reproducible workloads for the alignment benchmark.

Design goals (fairness / honesty)
---------------------------------
* **Size control.** Every cohort fixes the per-molecule size distribution so we
  can build a *uniform* cohort (all molecules the same size -> a single GPU
  bucket) and a *mixed* cohort (a spread of sizes -> many GPU buckets).  This is
  exactly the axis along which the GPU batch path's size-bucketing helps or
  hurts, so we test both and report bucket statistics alongside the timings.

* **Known ground truth.** Each pair's *fit* molecule is a rigid SE(3) copy of
  its *ref* molecule (optionally perturbed by Gaussian noise).  With zero noise
  the global optimum is recoverable and the best attainable score equals the
  self-overlap (Tanimoto = 1).  This gives every backend the *same* problem with
  a *known* optimum, so accuracy is measured as "what score did this backend
  actually deliver" rather than bit-for-bit equality of two implementations.

* **Backend-neutral storage.** Workloads are stored as plain numpy arrays.  Each
  backend converts to its own representation (torch cpu/cuda tensors, padded
  batches, JAX arrays) *inside* the timed ``prepare`` step, so device-transfer
  and padding costs are attributed honestly to the backend that incurs them.

The synthetic generator is deliberately decoupled from RDKit/xTB/Open3D so the
suite runs anywhere (including CPU-only boxes) and is fully reproducible.  The
arrays it produces are the exact arrays the alignment kernels consume, so the
compute being benchmarked is identical to production; only the *source* of the
numbers is synthetic.  A real-molecule loader can be dropped in behind the same
``Cohort`` interface (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The alignment objective each "mode" optimises.
MODES = ("vol", "surf", "esp", "pharm")

# How many pharmacophore types exist (must match score.constants.P_TYPES).
try:
    from shepherd_score.score.constants import P_TYPES as _P_TYPES
    _N_PTYPES = len(_P_TYPES)
except Exception:  # pragma: no cover - constants always import, but be safe
    _N_PTYPES = 9


@dataclass
class SyntheticMolecule:
    """Duck-typed stand-in for ``shepherd_score.container.Molecule``.

    Exposes exactly the attributes the alignment code paths read.  All
    positional arrays are float32-ready numpy; ``pharm_types`` is int.
    """

    atom_pos: np.ndarray              # (N, 3)
    surf_pos: np.ndarray              # (S, 3)
    surf_esp: np.ndarray              # (S,)
    partial_charges: np.ndarray       # (N,)
    pharm_types: np.ndarray           # (P,) int
    pharm_ancs: np.ndarray            # (P, 3)
    pharm_vecs: np.ndarray            # (P, 3)

    @property
    def _nonH_atoms_idx(self) -> np.ndarray:
        # Synthetic molecules carry no explicit hydrogens.
        return np.arange(self.atom_pos.shape[0])

    def center_to(self, xyz_mean: np.ndarray) -> None:
        """Mirror Molecule.center_to: shift all positional arrays by -mean."""
        self.atom_pos = self.atom_pos - xyz_mean
        self.surf_pos = self.surf_pos - xyz_mean
        self.pharm_ancs = self.pharm_ancs - xyz_mean


@dataclass
class PairSpec:
    """One reference/fit pair plus the ground-truth transform used to build it."""

    ref: SyntheticMolecule
    fit: SyntheticMolecule
    R: np.ndarray                     # (3, 3) rotation applied to make fit
    t: np.ndarray                     # (3,)   translation applied to make fit
    n_ref: int                        # heavy-atom count of ref (the "size")
    n_fit: int


@dataclass
class Cohort:
    """A reproducible set of pairs sharing a size policy."""

    name: str                         # e.g. "uniform-30" or "mixed-10to60"
    mode: str                         # one of MODES (decides which arrays matter)
    pairs: List[PairSpec]
    size_kind: str                    # "uniform" | "mixed"
    seed: int
    noise: float
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    # -- convenience views the backends consume -----------------------------
    def ref_atom_pos(self) -> List[np.ndarray]:
        return [p.ref.atom_pos for p in self.pairs]

    def fit_atom_pos(self) -> List[np.ndarray]:
        return [p.fit.atom_pos for p in self.pairs]

    def sizes(self) -> np.ndarray:
        return np.array([p.n_ref for p in self.pairs], dtype=int)


# ---------------------------------------------------------------------------
# Random geometry helpers
# ---------------------------------------------------------------------------
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3x3 rotation matrix (via QR of a Gaussian matrix)."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    # Fix sign so det(q) = +1 (proper rotation).
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


def _blob(n: int, rng: np.random.Generator) -> np.ndarray:
    """An asymmetric point cloud of ``n`` points with a realistic extent.

    Asymmetry matters: a symmetric cloud has a degenerate optimum, which would
    make "did the optimiser recover the pose" ill-defined.  We use an
    anisotropic Gaussian (different variance per axis) so the global optimum is
    well separated.
    """
    # Radius of gyration grows mildly with size, like a real molecule.
    scale = 1.4 * (n / 20.0) ** (1.0 / 3.0)
    aniso = np.array([1.0, 0.65, 0.45])  # break rotational symmetry
    pts = rng.standard_normal((n, 3)) * (scale * aniso)
    return (pts - pts.mean(0)).astype(np.float64)


def _make_molecule(n_atoms: int, rng: np.random.Generator) -> SyntheticMolecule:
    atom_pos = _blob(n_atoms, rng)

    # Surface points: ~2.5 per heavy atom, sitting just outside the atoms.
    n_surf = max(24, int(round(2.5 * n_atoms)))
    base = atom_pos[rng.integers(0, n_atoms, size=n_surf)]
    surf_pos = base + 1.4 * rng.standard_normal((n_surf, 3))
    surf_pos = (surf_pos - surf_pos.mean(0)).astype(np.float64)

    # Electrostatic potential: smooth-ish field in a realistic (a.u.-ish) range.
    surf_esp = (0.08 * rng.standard_normal(n_surf)).astype(np.float64)

    # Partial charges per atom, realistic magnitude, sum ~ 0.
    pc = 0.4 * rng.standard_normal(n_atoms)
    pc = pc - pc.mean()
    partial_charges = pc.astype(np.float64)

    # Pharmacophores: ~1 per 3 heavy atoms, at least 3.
    n_pharm = max(3, n_atoms // 3)
    pharm_types = rng.integers(0, _N_PTYPES, size=n_pharm).astype(np.int64)
    pharm_ancs = atom_pos[rng.integers(0, n_atoms, size=n_pharm)].astype(np.float64)
    vecs = rng.standard_normal((n_pharm, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    pharm_vecs = vecs.astype(np.float64)

    return SyntheticMolecule(
        atom_pos=atom_pos,
        surf_pos=surf_pos,
        surf_esp=surf_esp,
        partial_charges=partial_charges,
        pharm_types=pharm_types,
        pharm_ancs=pharm_ancs,
        pharm_vecs=pharm_vecs,
    )


def _transform_molecule(
    mol: SyntheticMolecule,
    R: np.ndarray,
    t: np.ndarray,
    noise: float,
    rng: np.random.Generator,
) -> SyntheticMolecule:
    """Return a rigid SE(3) copy of ``mol`` (positions rotated+translated,
    vectors rotated, scalar fields copied), optionally with Gaussian noise."""
    def xf(p):
        out = p @ R.T + t
        if noise > 0:
            out = out + noise * rng.standard_normal(out.shape)
        return out.astype(np.float64)

    vecs = mol.pharm_vecs @ R.T
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9

    return SyntheticMolecule(
        atom_pos=xf(mol.atom_pos),
        surf_pos=xf(mol.surf_pos),
        surf_esp=mol.surf_esp.copy(),
        partial_charges=mol.partial_charges.copy(),
        pharm_types=mol.pharm_types.copy(),
        pharm_ancs=xf(mol.pharm_ancs),
        pharm_vecs=vecs.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def _sizes_for_cohort(
    size_kind: str,
    n_pairs: int,
    size: int,
    size_range: Tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if size_kind == "uniform":
        return np.full(n_pairs, size, dtype=int)
    if size_kind == "mixed":
        lo, hi = size_range
        return rng.integers(lo, hi + 1, size=n_pairs).astype(int)
    raise ValueError(f"unknown size_kind={size_kind!r}")


def make_cohort(
    mode: str,
    *,
    n_pairs: int = 64,
    size_kind: str = "uniform",
    size: int = 30,
    size_range: Tuple[int, int] = (10, 60),
    rot_max_deg: float = 60.0,
    trans_max: float = 3.0,
    noise: float = 0.0,
    seed: int = 0,
) -> Cohort:
    """Build a reproducible cohort of pairs for one alignment ``mode``.

    Parameters
    ----------
    mode : str
        One of :data:`MODES`.  Decides which arrays are meaningful, but *all*
        arrays are generated so the same cohort object can be reused across
        modes if desired.
    n_pairs : int
        Number of reference/fit pairs.
    size_kind : {"uniform", "mixed"}
        ``uniform`` -> every molecule has ``size`` heavy atoms (single GPU
        bucket).  ``mixed`` -> heavy-atom counts drawn uniformly from
        ``size_range`` (many GPU buckets).
    size, size_range :
        See ``size_kind``.
    rot_max_deg, trans_max :
        Magnitude of the ground-truth pose offset between ref and fit.
    noise : float
        Std of Gaussian position noise added to the fit (0 -> exactly
        recoverable optimum).
    seed : int
        RNG seed for full reproducibility.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode={mode!r}; choose from {MODES}")
    rng = np.random.default_rng(seed)
    sizes = _sizes_for_cohort(size_kind, n_pairs, size, size_range, rng)

    pairs: List[PairSpec] = []
    cos_lim = np.cos(np.deg2rad(rot_max_deg))
    for n in sizes:
        ref = _make_molecule(int(n), rng)
        # Bounded rotation: sample until the rotation angle <= rot_max_deg.
        for _ in range(32):
            R = _random_rotation(rng)
            ang_cos = (np.trace(R) - 1.0) / 2.0
            if ang_cos >= cos_lim:
                break
        t = (rng.standard_normal(3))
        t = (t / (np.linalg.norm(t) + 1e-9)) * (rng.random() * trans_max)
        fit = _transform_molecule(ref, R, t, noise, rng)
        pairs.append(PairSpec(ref=ref, fit=fit, R=R, t=t,
                              n_ref=int(n), n_fit=int(n)))

    if size_kind == "uniform":
        name = f"uniform-{size}"
    else:
        name = f"mixed-{size_range[0]}to{size_range[1]}"

    return Cohort(
        name=name,
        mode=mode,
        pairs=pairs,
        size_kind=size_kind,
        seed=seed,
        noise=noise,
        meta={
            "n_pairs": n_pairs,
            "rot_max_deg": rot_max_deg,
            "trans_max": trans_max,
            "size_hist": dict(zip(*[a.tolist() for a in np.unique(sizes, return_counts=True)])),
        },
    )
