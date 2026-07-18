"""
Molecule class to hold molecule geometries and extract interaction profiles.
MoleculePair class facilitates alignment with interaction profiles.
"""
from typing import Union, List, Optional, Tuple, Iterable
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point3D
import torch

from shepherd_score.score.constants import COULOMB_SCALING, LAM_SCALING, ALPHA  # noqa: F401
from shepherd_score.generate_point_cloud import get_atom_coords, get_atomic_vdw_radii, get_molecular_surface, get_molecular_surface_const_density
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.gaussian_overlap import get_overlap
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores, Pharmacophore
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE, get_overlap_pharm
from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical, optimize_ROCS_esp_overlay, optimize_ROCS_esp_overlay_analytical, optimize_esp_combo_score_overlay
from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical
from shepherd_score.alignment import optimize_vol_color_overlay
from shepherd_score.alignment import optimize_vol_tversky_overlay
from shepherd_score.alignment import optimize_esp_field_overlay
from shepherd_score.alignment.utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np
from shepherd_score.accel import batch as _ba
from shepherd_score.accel._modes import (
    MODE_ATTRS as _MODE_ATTRS,
    CANONICAL_MODES as _CANONICAL_MODES,
    LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES,
)
from shepherd_score.container.profiles import Surface


def _default_seeds(mode: str) -> int:
    """Per-mode default SE(3) seed count (``MODE_SEEDS``) from ``shepherd_score/accel/_modes.py``,
    the single source of truth shared with the batched path. Resolves ``num_repeats=None`` in
    ``align_with_*`` so the per-pair API uses the same per-mode defaults as the batched API."""
    from shepherd_score.accel.batch.aligners import _seeds_for
    return _seeds_for(mode)


def _default_steps(mode: str) -> int:
    """Per-mode default fine-step count (``MODE_STEPS``); see ``_default_seeds``."""
    from shepherd_score.accel.batch.aligners import _steps_for
    return _steps_for(mode)


# Alignment modes tracked by MoleculePair (one AlignmentResult each), in fss canonical names.
# The upstream ``esp``/``esp_combo`` modes are ``surf_esp``/``vol_and_surf_esp`` here, plus the
# new ``vol_color`` mode. The bare ``vol``/``vol_esp`` keys hold the WITH-hydrogen results;
# the ``*_noH`` keys hold the heavy-atom results. Legacy ``esp``/``esp_combo`` attribute names
# are kept as delegating properties on MoleculePair (see the class body).
_ALIGN_KEYS = (
    'vol', 'vol_noH', 'vol_esp', 'vol_esp_noH',
    'surf', 'surf_esp', 'vol_and_surf_esp', 'pharm', 'vol_color', 'vol_tversky',
    'esp_field',
)


def _require_jax():
    """
    Import and return ``jax.numpy``, raising a clear error if JAX is unavailable.
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
    return jnp


@dataclass
class AlignmentResult:
    """
    Result of a single alignment mode: the optimal similarity score and SE(3) transform.

    Attributes
    ----------
    score : np.ndarray or None
        Optimally aligned similarity score. ``None`` until an alignment is run.
    transform : np.ndarray
        SE(3) transformation matrix, shape (4, 4). Defaults to the identity.
    """
    score: Optional[np.ndarray] = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))


def _alignment_property(key: str, field_name: str) -> property:
    """Build a property delegating to ``self._alignments[key].<field_name>``."""
    def getter(self):
        return getattr(self._alignments[key], field_name)

    def setter(self, value) -> None:
        setattr(self._alignments[key], field_name, value)

    return property(getter, setter)


def update_mol_coordinates(mol: Chem.Mol, coordinates: Union[List, np.ndarray]) -> Chem.Mol:
    """
    Updates the coordinates of a 3D RDKit mol object with a new set of coordinates

    Parameters
    ----------
    mol : Chem.Mol
        RDKit mol object with 3D coordinates to be replaced
    coordinates : Union[List, np.ndarray]
        List/array of new [x,y,z] coordinates

    Returns
    -------
    mol_new : Chem.Mol
        deep-copied RDKit mol object with updated 3D coordinates
    """
    mol_new = deepcopy(mol)
    conf = mol_new.GetConformer()
    for i in range(mol_new.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return mol_new


class Molecule:
    """
    Molecule contains ways to hold/generate molecule geometries
    """
    def __init__(self,
                 mol: Chem.rdchem.Mol,
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 probe_radius: Optional[float] = None,
                 surface_points: Optional[np.ndarray] = None,
                 partial_charges : Optional[np.ndarray] = None,
                 electrostatics: Optional[np.ndarray] = None,
                 pharm_multi_vector: Optional[bool] = None,
                 pharm_types: Optional[np.ndarray] = None,
                 pharm_ancs: Optional[np.ndarray] = None,
                 pharm_vecs: Optional[np.ndarray] = None,
                 feature_set: str = 'shepherd',
                 directionless: bool = False,
                 surface_method: str = 'mesh'
                 ):
        """
        Molecule constructor to extract interaction profiles.

        If `partial_charges` are not provided, they will be generated using MMFF94 which may
        result in subpar performance.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
        num_surf_points : Optional[int]
            Number of surface points to sample.
            If ``None``, the surface point cloud is not generated. More efficient if only doing volumentric.
        density : Optional[float]
            Density of points to sample on molecular surface.
            If ``None``, the surface point cloud is not generated. More efficient if only doing volumentric.
            If both ``num_surf_points`` and ``density`` are not ``None``, ``num_surf_points`` supercedes ``density``.
        surface_points : Optional[np.ndarray]
            Surface points if they were previously generated. Shape: (M,3).
        probe_radius : Optional[float]
            The radius of a probe atom to act as a "solvent accessible surface".
            Default is 1.2 if ``None`` is passed.
        partial_charges : Optional[np.ndarray]
            Partial charges for each atom. Shape: (N,).
            If ``None`` is passed and ESP surface is generated, it will default to MMFF94 partial charges.
        electrostatics : Optional[np.ndarray]
            Electrostatic potential if they were previously generated. Shape: (M,).
        pharm_multi_vector : Optional[bool]
            If ``None``, don't generate pharmacophore, else generate
            pharmacophore with/without (``True``/``False``) multi-vectors.
        pharm_types : Optional[np.ndarray]
            Types of pharmacophore. Shape: (P,).
        pharm_ancs : Optional[np.ndarray]
            Anchor positions of pharmacophore. Shape: (P,3).
        pharm_vecs : Optional[np.ndarray]
            Unit vectors relative to anchor positions of pharmacophore. Shape: (P,3).
        feature_set : str
            Which pharmacophore feature definition to use when generating pharmacophores.
            ``'shepherd'`` (default) uses the local ``smarts_features.fdef`` (8 fss types);
            ``'rdkit_base'`` uses RDKit's stock ``BaseFeatures.fdef`` reduced to the 6
            ROCS/ROSHAMBO color types. Only used when pharmacophores are generated (i.e.
            ``pharm_multi_vector`` is not ``None`` and explicit arrays are not provided).
        directionless : bool
            When ``True``, generate isotropic (zero-vector) "color" pharmacophores for all
            families (ROCS/ROSHAMBO style); this overrides ``pharm_multi_vector`` for the
            orientation-capable families. Default ``False`` computes orientation vectors.
            Only used when pharmacophores are generated.
        surface_method : str
            How to generate the surface point cloud when it is generated internally.
            ``'mesh'`` (default, UNCHANGED) uses the original Open3D ball-pivoting + Poisson-disk
            surface. ``'smooth_sdf'`` uses the opt-in, Open3D-free, mesh-free smooth + stochastic
            surfacer (``generate_point_cloud.get_molecular_surface_smooth_sdf``) intended for the
            generative pipeline; it requires ``num_surf_points`` (not ``density``). Opt-in only;
            using it is a distribution shift vs a model trained on the mesh surface (validate first).
        """
        self.mol = mol
        self.atom_pos = Chem.RemoveHs(mol).GetConformer().GetPositions()
        if surface_points is None:
            self.num_surf_points = num_surf_points
        else:
            self.num_surf_points = len(surface_points)
        self.density = density
        self.surface_method = surface_method

        if isinstance(partial_charges, list):
            partial_charges = np.array(partial_charges)

        if isinstance(partial_charges, np.ndarray):
            self.partial_charges = partial_charges
        else:
            self.partial_charges = self.get_partial_charges()
        self.radii = get_atomic_vdw_radii(mol)

        self._surface = Surface(
            positions=None,
            esp=None,
            probe_radius=probe_radius if probe_radius is not None else 1.2,
        )
        if surface_points is None:
            if isinstance(num_surf_points, int):
                self.surf_pos = self.get_pc()
            elif isinstance(density, float):
                self.surf_pos = self.get_pc(use_density=True)
            # else: no point cloud (surf_pos/surf_esp stay None)
        else:
            self.surf_pos = surface_points

        if self.surf_pos is not None and self.partial_charges is not None:
            if not isinstance(electrostatics, np.ndarray):
                self.surf_esp = self.get_electrostatic_potential()
            else:
                self.surf_esp = electrostatics

        # Indices for atoms that aren't hydrogens
        self._nonH_atoms_idx = np.array([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1])

        # ESP field points (Cresset-style): a VARIABLE-LENGTH set of signed extrema of the
        # softened molecular electrostatic potential, from heavy-atom positions + partial charges
        # (no surface / open3d). Computed LAZILY on first ``get_field_points()`` and cached: the
        # grid extremum search is expensive and only the ``esp_field`` mode needs it, so building a
        # Molecule for any other mode must not pay for it. ``field_point_pos`` is (M,3) float32,
        # ``field_point_sign`` is (M,) float32 (+1 = potential maximum, -1 = minimum); M may be 0.
        self.field_point_pos = None
        self.field_point_sign = None

        self.pharm_multi_vector = pharm_multi_vector
        if isinstance(pharm_types, np.ndarray) and isinstance(pharm_ancs, np.ndarray) and isinstance(pharm_vecs, np.ndarray):
            self._pharmacophore = Pharmacophore(types=pharm_types,
                                                  positions=pharm_ancs,
                                                  vectors=pharm_vecs)
        else:
            self._pharmacophore = None
            if self.pharm_multi_vector is not None:
                self.get_pharmacophore(
                    multi_vector=self.pharm_multi_vector,
                    exclude=[],
                    check_access=False,
                    scale=1.,
                    feature_set=feature_set,
                    directionless=directionless
                )


    # Interaction-profile accessors (backwards-compatible with the loose
    # ``surf_pos``/``surf_esp``/``probe_radius`` and ``pharm_*`` attributes)
    @property
    def surface(self) -> Surface:
        """The :class:`Surface` holding surface positions, ESP, and probe radius."""
        return self._surface

    @property
    def surf_pos(self) -> Optional[np.ndarray]:
        return self._surface.positions

    @surf_pos.setter
    def surf_pos(self, value: Optional[np.ndarray]) -> None:
        self._surface.positions = value

    @property
    def surf_esp(self) -> Optional[np.ndarray]:
        return self._surface.esp

    @surf_esp.setter
    def surf_esp(self, value: Optional[np.ndarray]) -> None:
        self._surface.esp = value

    @property
    def probe_radius(self) -> float:
        return self._surface.probe_radius

    @probe_radius.setter
    def probe_radius(self, value: float) -> None:
        self._surface.probe_radius = value

    @property
    def pharmacophore(self) -> Optional[Pharmacophore]:
        """The :class:`Pharmacophore` container, or ``None`` if not generated."""
        return self._pharmacophore

    def _ensure_pharm_container(self) -> Pharmacophore:
        """Lazily create an empty :class:`Pharmacophore` so setters can populate it."""
        if self._pharmacophore is None:
            self._pharmacophore = Pharmacophore(types=None, positions=None, vectors=None)
        return self._pharmacophore

    @property
    def pharm_types(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.types

    @pharm_types.setter
    def pharm_types(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().types = value

    @property
    def pharm_ancs(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.positions

    @pharm_ancs.setter
    def pharm_ancs(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().positions = value

    @property
    def pharm_vecs(self) -> Optional[np.ndarray]:
        return None if self._pharmacophore is None else self._pharmacophore.vectors

    @pharm_vecs.setter
    def pharm_vecs(self, value: Optional[np.ndarray]) -> None:
        self._ensure_pharm_container().vectors = value

    def get_partial_charges(self) -> np.ndarray:
        """
        Get the partial charges on each atom using MMFF.
        """
        mol_copy = deepcopy(self.mol)
        molec_props = Chem.AllChem.MMFFGetMoleculeProperties(mol_copy)
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol_copy.GetAtoms())])
        return charges.astype(np.float32)


    def get_positions(self, no_H: bool = True) -> np.ndarray:
        """
        Get atom coordinates with or without hydrogens.

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return the cached heavy-atom coordinates (``atom_pos``).
            If ``False`` return all-atom coordinates from the conformer (including H).

        Returns
        -------
        np.ndarray
            Atom coordinates. Shape: (N, 3).
        """
        if no_H:
            return self.atom_pos
        return self.mol.GetConformer().GetPositions()


    def get_charges(self, no_H: bool = True) -> np.ndarray:
        """
        Get partial charges with or without hydrogens.

        This slices the already-computed ``partial_charges``; it does not recompute them
        (see :meth:`get_partial_charges` for MMFF computation).

        Parameters
        ----------
        no_H : bool, optional
            If ``True`` (default) return charges for heavy atoms only.
            If ``False`` return charges for all atoms (including H).

        Returns
        -------
        np.ndarray
            Partial charges. Shape: (N,).
        """
        if no_H:
            return self.partial_charges[self._nonH_atoms_idx]
        return self.partial_charges


    def get_field_point_contribs(self,
                                 eps: float = 1.0,
                                 spacing: float = 0.75,
                                 margin: float = 4.0,
                                 rel_thresh: float = 0.15,
                                 shell_min: float = 1.5,
                                 shell_max: float = 4.5,
                                 merge_radius: float = 1.5,
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cresset-style electrostatic-potential (ESP) *field points*: a variable-length set
        of signed extrema of the molecular ESP around the molecule.

        Unlike the per-atom ``partial_charges`` trio, this is a *derived point set* (like the
        pharmacophore) whose length ``M`` depends on the ESP topology, not on the atom count.
        Everything is computed from heavy-atom positions (``atom_pos``) and their MMFF partial
        charges (``partial_charges[_nonH_atoms_idx]``) -- no surface / mesh is required.

        Method
        ------
        1. Softened Coulomb potential ``phi(r) = sum_i q_i / sqrt(|r - r_i|^2 + eps^2)`` over
           heavy atoms (``eps=1.0`` A avoids the singularity at the nuclei).
        2. Evaluate ``phi`` on a regular grid: heavy-atom bounding box + ``margin`` (4.0 A),
           ``spacing`` 0.75 A.
        3. Restrict to the *probe shell* -- grid points whose nearest heavy atom lies in
           ``[shell_min, shell_max]`` (around, not inside, the molecule) -- and keep shell cells
           that are a strict local max or min of ``phi`` among their in-shell 26 neighbours, with
           ``|phi| > rel_thresh * max|phi|``. (The extremum test is restricted to the shell because
           the unrestricted extrema of a superposed softened-Coulomb potential sit *on* the atoms;
           searching within the shell places field points out in space where the potential is
           extremal along the accessible surface -- the Cresset field-point concept.)
        4. Merge extrema within ``merge_radius`` (1.5 A), keeping the strongest by ``|phi|``.

        Parameters
        ----------
        eps : float
            Softening length (A) for the Coulomb potential. Default 1.0.
        spacing : float
            Grid spacing (A). Default 0.75.
        margin : float
            Bounding-box margin (A) around the heavy atoms. Default 4.0.
        rel_thresh : float
            Keep extrema with ``|phi|`` above this fraction of ``max|phi|``. Default 0.15.
        shell_min, shell_max : float
            Retain extrema whose nearest heavy atom lies in ``[shell_min, shell_max]`` A.
            Defaults 1.5 and 4.5.
        merge_radius : float
            Merge extrema closer than this (A), keeping the strongest. Default 1.5.

        Returns
        -------
        field_point_pos : np.ndarray (M, 3) float32
            Coordinates of the field points.
        field_point_sign : np.ndarray (M,) float32
            +1 for a potential maximum, -1 for a minimum. Empty (M=0) if no extrema qualify.
        """
        empty_pos = np.zeros((0, 3), dtype=np.float32)
        empty_sign = np.zeros((0,), dtype=np.float32)

        pos = np.asarray(self.atom_pos, dtype=np.float64)  # heavy-atom coordinates (N,3)
        charges = self.partial_charges[self._nonH_atoms_idx].astype(np.float64)  # (N,)
        if pos.shape[0] == 0 or not np.any(charges != 0.0):
            return empty_pos, empty_sign

        # --- grid over the heavy-atom bounding box + margin ---
        lo = pos.min(axis=0) - margin
        hi = pos.max(axis=0) + margin
        axes = [np.arange(lo[d], hi[d] + spacing, spacing) for d in range(3)]
        nx, ny, nz = (len(a) for a in axes)
        if nx < 3 or ny < 3 or nz < 3:
            return empty_pos, empty_sign
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (G,3)

        # --- softened Coulomb potential + nearest-atom distance at every grid point ---
        d2 = np.sum((grid[:, None, :] - pos[None, :, :]) ** 2, axis=2)  # (G,N)
        phi = np.sum(charges[None, :] / np.sqrt(d2 + eps ** 2), axis=1)  # (G,)
        phi3 = phi.reshape(nx, ny, nz)
        nn3 = np.sqrt(d2).min(axis=1).reshape(nx, ny, nz)  # nearest heavy-atom distance

        max_abs = np.abs(phi).max()
        if max_abs == 0.0:
            return empty_pos, empty_sign

        # --- probe shell: grid cells whose nearest atom is in [shell_min, shell_max] ---
        shell = (nn3 >= shell_min) & (nn3 <= shell_max)
        if not np.any(shell):
            return empty_pos, empty_sign

        # --- strict local extrema restricted to the shell (compare only in-shell neighbours) ---
        # Pad by 1 so out-of-grid neighbours are treated as "not in shell" (ignored).
        phi_pad = np.pad(phi3, 1, constant_values=0.0)
        shell_pad = np.pad(shell, 1, constant_values=False)
        is_max = np.ones_like(shell, dtype=bool)
        is_min = np.ones_like(shell, dtype=bool)
        n_neigh = np.zeros_like(shell, dtype=np.int32)  # in-shell neighbour count
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neigh_shell = shell_pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz]
                    neigh_phi = phi_pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz]
                    n_neigh += neigh_shell.astype(np.int32)
                    # A candidate stays a strict max only if no in-shell neighbour is >= it.
                    is_max &= ~(neigh_shell & (neigh_phi >= phi3))
                    is_min &= ~(neigh_shell & (neigh_phi <= phi3))
        # Require a few in-shell neighbours so isolated cells are not trivially extremal.
        valid = shell & (n_neigh >= 3)
        extrema = valid & (is_max | is_min) & (np.abs(phi3) > rel_thresh * max_abs)
        if not np.any(extrema):
            return empty_pos, empty_sign

        ex_pos = grid.reshape(nx, ny, nz, 3)[extrema]           # (E,3)
        ex_phi = phi3[extrema]                                  # (E,)
        ex_sign = np.where(is_max[extrema], 1.0, -1.0).astype(np.float64)

        # --- merge extrema within merge_radius, keeping the strongest |phi| ---
        order = np.argsort(-np.abs(ex_phi))
        accepted_pos = []
        accepted_sign = []
        for idx in order:
            p = ex_pos[idx]
            if accepted_pos:
                dists = np.sqrt(np.sum((np.asarray(accepted_pos) - p) ** 2, axis=1))
                if np.any(dists < merge_radius):
                    continue
            accepted_pos.append(p)
            accepted_sign.append(ex_sign[idx])

        if not accepted_pos:
            return empty_pos, empty_sign
        return (np.asarray(accepted_pos, dtype=np.float32),
                np.asarray(accepted_sign, dtype=np.float32))


    def get_field_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the ESP field points, computing them lazily on first access and caching.

        Returns
        -------
        field_point_pos : np.ndarray (M, 3) float32
            Field-point coordinates (empty if the molecule has no qualifying extrema).
        field_point_sign : np.ndarray (M,) float32
            +1 for a potential maximum, -1 for a minimum.
        """
        if self.field_point_pos is None:
            self.field_point_pos, self.field_point_sign = self.get_field_point_contribs()
        return self.field_point_pos, self.field_point_sign


    def get_pc(self, use_density=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the point cloud positions.
        """
        self.mol, centers = get_atom_coords(self.mol, MMFF_optimize=False)
        surface_method = self.surface_method
        if use_density:
            if surface_method != 'mesh':
                raise ValueError(
                    f"surface_method={surface_method!r} does not support density-based surfaces; "
                    "the mesh-free path needs num_surf_points. Use surface_method='mesh' with density, "
                    "or pass num_surf_points instead of density."
                )
            positions = get_molecular_surface_const_density(centers,
                                                            self.radii,
                                                            self.density,
                                                            probe_radius=self.probe_radius,
                                                            num_samples_per_atom=25)
        else:
            # num_samples_per_atom left to each method's default: 25 for 'mesh' (unchanged),
            # the sparser SMOOTH_SDF_NSPA for 'smooth_sdf'.
            positions = get_molecular_surface(centers,
                                              self.radii,
                                              num_points=self.num_surf_points,
                                              probe_radius=self.probe_radius,
                                              method=surface_method)
        return positions.astype(np.float32)


    def get_electrostatic_potential(self) -> np.ndarray:
        """
        Get the electrostatic potential at each surface point.
        """
        centers = self.mol.GetConformer().GetPositions()
        distances = np.linalg.norm(self.surf_pos[:, np.newaxis] - centers, axis=2)
        # Calculate the potentials
        E_pot = np.dot(self.partial_charges, 1 / distances.T) * COULOMB_SCALING
        # Ensure that invalid distances (where distance is 0) are handled
        E_pot[np.isinf(E_pot)] = 0
        return E_pot.astype(np.float32)


    def center_to(self, xyz_means: np.ndarray) -> None:
        """
        If you want to center the molecule with respect to a certain coordinate frame.
        """
        self.atom_pos -= xyz_means
        trans = np.eye(4)
        trans[:3,3] = -xyz_means
        Chem.rdMolTransforms.TransformConformer(self.mol.GetConformer(), trans)
        if self.surf_pos is not None:
            self.surf_pos -= xyz_means
        if self.pharm_ancs is not None:
            self.pharm_ancs -= xyz_means
        if self.field_point_pos is not None:
            # Field points already cached from an earlier get_field_points(); they move
            # rigidly with the conformer (the signs are translation-invariant), so shift
            # them in place rather than leaving a stale copy at the old coordinates.
            self.field_point_pos = self.field_point_pos - xyz_means


    def get_pharmacophore(self,
                          multi_vector: bool = True,
                          exclude: List[int] = [],
                          check_access: bool = False,
                          scale: float = 1,
                          feature_set: str = 'shepherd',
                          directionless: bool = False,
                          return_atom_ids: bool = False,
                          priority_atoms: Optional[Iterable[int]] = None,
                          min_ring_priority_atoms: int = 3):
        """
        Get the pharmacophore of the molecule.

        Stores the full :class:`~shepherd_score.pharm_utils.pharmacophore.Pharmacophore`
        container on ``self`` (accessible via :attr:`pharmacophore`). The
        ``pharm_types``/``pharm_ancs``/``pharm_vecs`` properties delegate to it, so
        existing usage is unchanged.

        Parameters
        ----------
        multi_vector : bool, optional
            Whether to represent pharmacophore with multiple vectors. Default ``True``.
        exclude : list, optional
            Hydrogen indices to not include as a HBD. Default ``[]``.
        check_access : bool, optional
            Check if HBD/HBA are accessible to the molecular surface. Default ``False``.
        scale : float, optional
            Length of a pharmacophore vector in Angstroms. Default 1.
        feature_set : str, optional
            ``'shepherd'`` (default, 8 fss types) or ``'rdkit_base'`` (6 ROCS/ROSHAMBO
            color types). See :func:`get_pharmacophores`.
        directionless : bool, optional
            When ``True``, emit isotropic zero-vector "color" pharmacophores, overriding
            ``multi_vector``. Default ``False``.
        return_atom_ids : bool, optional
            Retain per-pharmacophore atom-id sets on ``self.pharmacophore.atom_ids``,
            enabling ``self.pharmacophore.priority_labels(...)``. Default ``False``.
        priority_atoms : iterable of int, optional
            When provided, priority labels are computed and stored on
            ``self.pharmacophore.labels``. Default ``None``.
        min_ring_priority_atoms : int, optional
            Only used when ``priority_atoms`` is provided. See
            :func:`~shepherd_score.pharm_utils.pharmacophore.get_pharmacophore`.
            Default 3.
        """
        self._pharmacophore = get_pharmacophores(
            self.mol,
            multi_vector=multi_vector,
            exclude=exclude,
            check_access=check_access,
            scale=scale,
            feature_set=feature_set,
            directionless=directionless,
            return_atom_ids=return_atom_ids,
            priority_atoms=priority_atoms,
            min_ring_priority_atoms=min_ring_priority_atoms,
        )


def _bind_batch_aligners(cls):
    """Bind ``accel.batch._align_batch_<mode>`` onto ``cls`` as static methods -- one per
    canonical registry mode, plus the legacy-name aliases (esp->surf_esp,
    esp_combo->vol_and_surf_esp) -- so ``MoleculePair._align_batch_vol(pairs, ...)`` etc. still
    resolve here and adding a mode needs no per-mode edit. Exactly mirrors the old explicit
    staticmethod block; driven off accel/_modes so the registry is the single source."""
    for _m in _CANONICAL_MODES:
        setattr(cls, "_align_batch_" + _m, staticmethod(getattr(_ba, "_align_batch_" + _m)))
    for _legacy, _canon in _LEGACY_MODE_ALIASES.items():
        setattr(cls, "_align_batch_" + _legacy, staticmethod(getattr(_ba, "_align_batch_" + _canon)))
    return cls


@_bind_batch_aligners
class MoleculePair:
    """ Pair of Molecule objects to facilitate alignment. """

    def __init__(self,
                 ref_mol: Union[Chem.rdchem.Mol, Molecule],
                 fit_mol: Union[Chem.rdchem.Mol, Molecule],
                 num_surf_points: Optional[int] = None,
                 density: Optional[float] = None,
                 do_center: bool = False,
                 device = -1):
        """
        A pair of molecules. A refence molecule and a fit molecule that can be aligned to the fit.
        There are a number of alignments that can be done:

        - Volumetric (with and without hydrogens)
        - Volumetric with partial charge weighting (with and without hydrogens)
        - Surface
        - Surface with electrostatic potential weighting
        - ShaEP scoring (esp-combo)
        - Pharmacophore (with various settings for using extended points rather than vectors)

        Similarly, you can score with surface, Surf+ESP, and pharmacophore

        Parameters
        ----------
        ref_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Reference molecule.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        fit_mol : Union[rdkit.Chem.rdchem.Mol, container.Molecule]
            Molecule to fit to the reference.
            If a RDKit Mol object is provided, it will be converted to a Molecule
            object. If a Molecule object is given, it will NOT regenerate the surface.
        num_surf_points : Optional[int] (default = None)
            Number of surface points to sample if rdkit Mol objects are given.
            MUST provide a value for surface or ESP alignment.
        density : Optional[float] (default = None)
            Density of points to sample if rdkit Mol objects are given.
            An integer intput for num_surf_points supercedes the density call.
        do_center : bool (default = False)
            THIS IS CRUCIAL
            Whether to initially align molecule centers together. For global optimizations, set to
            True. For scoring of current alignment or local alignment set to False.
        device : pytorch Device (default = -1)
            Device to use if you want to align with PyTorch downstream.
            Default places alignment computation on CPU.
        """
        # Generate surfaces if not a Molecule object
        if not isinstance(ref_mol, Chem.rdchem.Mol):
            self.ref_molec = ref_mol
        else:
            self.ref_molec = Molecule(ref_mol, num_surf_points=num_surf_points, density=density)
        if not isinstance(fit_mol, Chem.rdchem.Mol):
            self.fit_molec = fit_mol
        else:
            self.fit_molec = Molecule(fit_mol, num_surf_points=num_surf_points, density=density)

        self.num_surf_points = num_surf_points
        self.density = density
        if density is not None and num_surf_points is None:
            self.num_surf_points = True
        if not isinstance(device, torch.device):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Center to origin
        if do_center:
            self.ref_molec.center_to(self.ref_molec.atom_pos.mean(0))
            self.fit_molec.center_to(self.fit_molec.atom_pos.mean(0))

        # ----  pre-convert atomic coordinates to torch on the target device  ----
        self._ref_xyz_t = torch.as_tensor(self.ref_molec.atom_pos,
                                          dtype=torch.float32,
                                          device=device)          # (N,3)
        self._fit_xyz_t = torch.as_tensor(self.fit_molec.atom_pos,
                                          dtype=torch.float32,
                                          device=device)          # (M,3)

        # One AlignmentResult per mode (score defaults to None, transform to identity).
        # transform_<mode>/sim_aligned_<mode> are properties delegating to this dict.
        self._alignments = {key: AlignmentResult() for key in _ALIGN_KEYS}


    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a float32 tensor on this pair's device."""
        return torch.from_numpy(arr).to(torch.float32).to(self.device)

    # Alignment-result accessors (backwards-compatible with the loose
    # ``transform_*`` / ``sim_aligned_*`` attributes). One property pair
    # per mode in _ALIGN_KEYS, generated via _alignment_property().
    for _key in _ALIGN_KEYS:
        locals()[f'transform_{_key}'] = _alignment_property(_key, 'transform')
        locals()[f'sim_aligned_{_key}'] = _alignment_property(_key, 'score')
    del _key

    # Legacy result-attribute aliases (renamed modes; old names kept working):
    # esp -> surf_esp, esp_combo -> vol_and_surf_esp. Delegate to the canonical entries.
    transform_esp = _alignment_property('surf_esp', 'transform')
    sim_aligned_esp = _alignment_property('surf_esp', 'score')
    transform_esp_combo = _alignment_property('vol_and_surf_esp', 'transform')
    sim_aligned_esp_combo = _alignment_property('vol_and_surf_esp', 'score')

    # --- batched GPU/Triton aligners -------------------------------------
    # Implemented as free functions in ``accel.batch``; bound as static methods (one per
    # registry mode + legacy aliases) by the ``@_bind_batch_aligners`` class decorator above, so
    # the public seam ``MoleculePair._align_batch_vol(pairs, ...)`` etc. still resolves here.

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = None,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using volumetric similarity.

        Optimally aligned score found in ``self.sim_aligned_vol`` and the optimal SE(3)
        transformation is at ``self.transform_vol``. If ``no_H`` is ``True``, append '_noH' to them.

        Parameters
        ----------
        no_H : bool
            Whether to not include hydrogens in volumetric similarity. Default is ``True``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass (COM) is translated to
            each ``ref_molec``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        use_analytical : bool, optional
            Whether to use analytical gradients instead of PyTorch autograd. Ignored if
            ``use_jax=True``. Default is ``True``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps. Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("vol")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol")
        ref_atom_pos = self.ref_molec.get_positions(no_H)
        fit_atom_pos = self.fit_molec.get_positions(no_H)
        if use_jax: # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay_jax(
                ref_points=jnp.array(ref_atom_pos),
                fit_points=jnp.array(fit_atom_pos),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_fit_points = np.array(aligned_fit_points)
        else:
            # PyTorch
            _vol_fn = optimize_ROCS_overlay_analytical if use_analytical else optimize_ROCS_overlay
            aligned_fit_points, se3_transform, score = _vol_fn(
                ref_points=self._to_tensor(ref_atom_pos),
                fit_points=self._to_tensor(fit_atom_pos),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers = self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            se3_transform = se3_transform.numpy()
            score = score.numpy()
            aligned_fit_points = aligned_fit_points.numpy()
        if no_H:
            self.transform_vol_noH = se3_transform
            self.sim_aligned_vol_noH = score
        else:
            self.transform_vol = se3_transform
            self.sim_aligned_vol = score
        return aligned_fit_points


    def align_with_vol_esp(self,
                           lam: float = 0.1,
                           no_H: bool = True,
                           num_repeats: int = None,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = None,
                           use_jax: bool = False,
                           use_analytical: bool = True,
                           verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using volume similarity weighted by partial charge
        Toggle ``no_H`` parameter for scoring with or without hydrogens.

        ``lam`` DEFAULTS to 0.1 -- the value this library has always documented for
        partial-charge volumetric ESP.

        !! ``lam`` IS A WIDTH, NOT A WEIGHT, AND ITS UNITS DIFFER BY MODE.
        The ESP term is ``exp(-esp_diff_sq / lam)``, so a SMALLER lam means charge
        differences are penalised MORE sharply (a more discriminative ESP), not less.
        Crucially, ``align_with_vol_esp`` takes lam RAW, while ``align_with_surf_esp``
        multiplies its lam by ``LAM_SCALING`` (= COULOMB_SCALING**2, ~207) internally.
        The two lams are therefore ~207x apart in absolute terms and are NOT
        interchangeable. Handing this method surf_esp's 0.3 makes the ESP ~3x too
        permissive; that mistake previously cost ~0.02 ROC-AUC on a 41-target DUDE-Z
        screen, and it is exactly why lam is a DEFAULT here now rather than a required
        argument the caller has to guess.
        Optimally aligned score found in ``self.sim_aligned_vol_esp`` and the optimal SE(3)
        transformation is at ``self.transform_vol_esp``. If ``no_H`` is ``True``, append '_noH' to them.

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter.
        no_H : bool
            Whether to not include hydrogens in volumetric similarity. Default is ``True``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass
            (COM) is translated to each ``ref_molec``'s atoms, with 10 rotations for each translation.
            So the number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("vol_esp")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_esp")
        ref_mol_partial_charges = self.ref_molec.get_charges(no_H)
        fit_mol_partial_charges = self.fit_molec.get_charges(no_H)
        ref_mol_pos = self.ref_molec.get_positions(no_H)
        fit_mol_pos = self.fit_molec.get_positions(no_H)

        if use_jax: # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay_jax(
                ref_points=jnp.array(ref_mol_pos),
                fit_points=jnp.array(fit_mol_pos),
                ref_charges=jnp.array(ref_mol_partial_charges),
                fit_charges=jnp.array(fit_mol_partial_charges),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            se3_transform = np.array(se3_transform)
            score = np.array(score)
            aligned_fit_points = np.array(aligned_fit_points)

        else: # Use Torch implementation
            _esp_fn = optimize_ROCS_esp_overlay_analytical if use_analytical else optimize_ROCS_esp_overlay
            aligned_fit_points, se3_transform, score = _esp_fn(
                ref_points=self._to_tensor(ref_mol_pos),
                fit_points=self._to_tensor(fit_mol_pos),
                ref_charges=self._to_tensor(ref_mol_partial_charges),
                fit_charges=self._to_tensor(fit_mol_partial_charges),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers = self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            se3_transform = se3_transform.numpy()
            score = score.numpy()
            aligned_fit_points = aligned_fit_points.numpy()

        if no_H:
            self.transform_vol_esp_noH = se3_transform
            self.sim_aligned_vol_esp_noH = score
        else:
            self.transform_vol_esp = se3_transform
            self.sim_aligned_vol_esp = score
        return aligned_fit_points


    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = None,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = None,
                        use_jax: bool = False,
                        use_analytical: bool = True,
                        verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using surface similarity.

        Optimally aligned score found in ``self.sim_aligned_surf`` and the optimal SE(3)
        transformation is at ``self.transform_surf``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass
            (COM) is translated to each ``ref_molec``'s atoms, with 10 rotations for each
            translation. So the number of initializations scales as
            (# translation centers * 10 + 5) where 5 is from the identity and 4 PCA with
            aligned COMs. If ``None``, then ``num_repeats`` rotations are done with aligned COMs.
            Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        use_analytical : bool, optional
            Whether to use analytical gradients instead of PyTorch autograd. Ignored if
            ``use_jax=True``. Default is ``True``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps. Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("surf")
        if max_num_steps is None:
            max_num_steps = _default_steps("surf")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_overlay_jax(
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_surf = np.array(se3_transform)
            self.sim_aligned_surf = np.array(score)
            return np.array(aligned_fit_points)
        else:
            # Torch
            _surf_fn = optimize_ROCS_overlay_analytical if use_analytical else optimize_ROCS_overlay
            aligned_fit_points, se3_transform, score = _surf_fn(
                ref_points=self._to_tensor(self.ref_molec.surf_pos),
                fit_points=self._to_tensor(self.fit_molec.surf_pos),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers = self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_surf = se3_transform.numpy()
            self.sim_aligned_surf = score.numpy()
            return aligned_fit_points.numpy()


    def align_with_surf_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = None,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using surface-ESP similarity. ``surf_esp`` is the
        canonical name for the mode formerly called ``esp`` (legacy alias kept below).
        ``lam`` is scaled by ``(1e4/(4*55.263*np.pi))**2`` for correct units.

        Typically, ``lam=0.3`` is used and is scaled internally.

        Optimally aligned score found in ``self.sim_aligned_surf_esp`` and the optimal SE(3)
        transformation is at ``self.transform_surf_esp``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            Weighting factor for ESP scoring. Scaled internally. Default is 0.3.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If None, then num_repeats rotations are done
            with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("surf_esp")
        if max_num_steps is None:
            max_num_steps = _default_steps("surf_esp")
        lam_scaled = LAM_SCALING * lam
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_ROCS_esp_overlay_jax(
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                ref_charges=jnp.array(self.ref_molec.surf_esp),
                fit_charges=jnp.array(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_surf_esp = np.array(se3_transform)
            self.sim_aligned_surf_esp = np.array(score)
            return np.array(aligned_fit_points)
        else: # Use Torch implementation
            _esp_fn = optimize_ROCS_esp_overlay_analytical if use_analytical else optimize_ROCS_esp_overlay
            aligned_fit_points, se3_transform, score = _esp_fn(
                ref_points=self._to_tensor(self.ref_molec.surf_pos),
                fit_points=self._to_tensor(self.fit_molec.surf_pos),
                ref_charges=self._to_tensor(self.ref_molec.surf_esp),
                fit_charges=self._to_tensor(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers = self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_surf_esp = se3_transform.numpy()
            self.sim_aligned_surf_esp = score.numpy()
            return aligned_fit_points.numpy()


    def align_with_vol_and_surf_esp(self,
                             alpha: float,
                             lam: float = 0.001,
                             probe_radius: float = 1.0,
                             esp_weight: float = 0.5,
                             num_repeats: int = None,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = None,
                             use_jax: bool = False,
                             verbose: bool = False):
        """
        Align using ShaEP similarity score. ``vol_and_surf_esp`` is the canonical name
        for the mode formerly called ``esp_combo`` (legacy alias kept below).
        If alpha is 0.81, then it automatically uses volumetric shape similarity.
        Otherwise, it uses surface shape similarity.

        Optimally aligned score found in ``self.sim_aligned_vol_and_surf_esp`` and the optimal
        SE(3) transformation is at ``self.transform_vol_and_surf_esp``.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            ESP weighting parameter. Default is 0.001.
        probe_radius : float, optional
            Surface points found within vdW radii + probe radius will be masked out.
            Surface generation uses a probe radius of 1.2 by default (radius of hydrogen)
            so we use a slightly lower radius for be more tolerant. Default is 1.0.
        esp_weight : float, optional
            How much to weight shape vs esp_combo similarity ([0,1]). Default is 0.5.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated
            to each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats``
            rotations are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray (N, 3)
            Coordinates of transformed atoms. Shape: (N, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("vol_and_surf_esp")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_and_surf_esp")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_esp_combo_score_overlay_jax
            aligned_fit_points, se3_transform, score = optimize_esp_combo_score_overlay_jax(
                ref_centers_w_H=jnp.array(self.ref_molec.mol.GetConformer().GetPositions()),
                fit_centers_w_H=jnp.array(self.fit_molec.mol.GetConformer().GetPositions()),
                ref_centers=jnp.array(self.ref_molec.atom_pos) if alpha == 0.81 else jnp.array(self.ref_molec.surf_pos),
                fit_centers=jnp.array(self.fit_molec.atom_pos) if alpha == 0.81 else jnp.array(self.fit_molec.surf_pos),
                ref_points=jnp.array(self.ref_molec.surf_pos),
                fit_points=jnp.array(self.fit_molec.surf_pos),
                ref_partial_charges=jnp.array(self.ref_molec.partial_charges),
                fit_partial_charges=jnp.array(self.fit_molec.partial_charges),
                ref_surf_esp=jnp.array(self.ref_molec.surf_esp),
                fit_surf_esp=jnp.array(self.fit_molec.surf_esp),
                ref_radii=jnp.array(self.ref_molec.radii),
                fit_radii=jnp.array(self.fit_molec.radii),
                alpha=alpha,
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                num_repeats=num_repeats,
                trans_centers = self.ref_molec.atom_pos if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_vol_and_surf_esp = np.array(se3_transform)
            self.sim_aligned_vol_and_surf_esp = np.array(score)
            return np.array(aligned_fit_points)
        else:
            if alpha == 0.81:
                ref_centers = self._to_tensor(self.ref_molec.atom_pos)
                fit_centers = self._to_tensor(self.fit_molec.atom_pos)
            else:
                ref_centers = self._to_tensor(self.ref_molec.surf_pos)
                fit_centers = self._to_tensor(self.fit_molec.surf_pos)

            aligned_fit_points, se3_transform, score = optimize_esp_combo_score_overlay(
                ref_centers_w_H=self._to_tensor(self.ref_molec.mol.GetConformer().GetPositions()),
                fit_centers_w_H=self._to_tensor(self.fit_molec.mol.GetConformer().GetPositions()),
                ref_centers=ref_centers,
                fit_centers=fit_centers,
                ref_points=self._to_tensor(self.ref_molec.surf_pos),
                fit_points=self._to_tensor(self.fit_molec.surf_pos),
                ref_partial_charges=self._to_tensor(self.ref_molec.partial_charges),
                fit_partial_charges=self._to_tensor(self.fit_molec.partial_charges),
                ref_surf_esp=self._to_tensor(self.ref_molec.surf_esp),
                fit_surf_esp=self._to_tensor(self.fit_molec.surf_esp),
                ref_radii=self._to_tensor(self.ref_molec.radii),
                fit_radii=self._to_tensor(self.fit_molec.radii),
                alpha=alpha,
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                num_repeats=num_repeats,
                trans_centers = self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )
            self.transform_vol_and_surf_esp = se3_transform.numpy()
            self.sim_aligned_vol_and_surf_esp = score.numpy()
            return aligned_fit_points.detach().numpy()

    # legacy method aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp)
    align_with_esp = align_with_surf_esp
    align_with_esp_combo = align_with_vol_and_surf_esp


    def align_with_vol_color(self,
                             color_weight: float = 0.5,
                             alpha: float = 0.81,
                             similarity: _SIM_TYPE = 'tanimoto',
                             directionless: bool = True,
                             extended_points: bool = False,
                             only_extended: bool = False,
                             num_repeats: int = None,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = None,
                             verbose: bool = False) -> np.ndarray:
        """
        Align using a ROCS/ROSHAMBO-style combined atom-centred Gaussian *shape* (volume) +
        directionless *color* (pharmacophore) overlay (a TanimotoCombo analogue).

        The optimized objective is
        ``(1 - color_weight) * shape_Tanimoto + color_weight * color_Tanimoto``. By default
        the color channel is *directionless* (isotropic point Gaussians, ROCS/ROSHAMBO
        "color"); pass ``directionless=False`` to keep fss's orientation-vector weighting. For
        ROCS/ROSHAMBO feature parity, build the ``Molecule`` objects with
        ``feature_set='rdkit_base'``.

        Optimally aligned score is stored in ``self.sim_aligned_vol_color`` and the optimal
        SE(3) transformation in ``self.transform_vol_color``.

        Parameters
        ----------
        color_weight : float, optional
            Weight of the color channel in [0, 1]; shape gets ``1 - color_weight``.
            Default is 0.5 (the ROCS/ROSHAMBO 50/50 combo).
        alpha : float, optional
            Gaussian width for the shape overlap. Default is 0.81 (volumetric, heavy atoms).
        similarity : str, optional
            Similarity for the color channel. Default is 'tanimoto'.
        directionless : bool, optional
            ``True`` (default) scores color as isotropic point Gaussians; ``False`` uses the
            orientation-vector cosine weighting. Same polarity as the extraction-side
            ``directionless`` on :meth:`Molecule.get_pharmacophore`.
        extended_points, only_extended : bool, optional
            Forwarded to the color scorer (ignored when ``directionless=True``).
        num_repeats : int, optional
            Number of SE(3) initializations. Default (``None``) is ``MODE_SEEDS['vol_color']`` (16).
        trans_init : bool, optional
            Translation-seeded initialization from the reference atoms. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default (``None``) is ``MODE_STEPS['vol_color']`` (40).
        verbose : bool, optional
            Print progress. Default is ``False``.

        Returns
        -------
        aligned_fit_centers : np.ndarray (M, 3)
            Transformed fit atom (heavy-atom) coordinates.
        """
        if num_repeats is None:
            num_repeats = _default_seeds("vol_color")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_color")
        if self.ref_molec.pharm_types is None or self.fit_molec.pharm_types is None:
            raise ValueError(
                'Both Molecule objects must have pharmacophores to use align_with_vol_color. '
                "Build them with `pharm_multi_vector` set (and optionally "
                "`feature_set='rdkit_base'` for ROCS/ROSHAMBO color)."
            )

        dev = self.device

        aligned_fit_centers, se3_transform, score = optimize_vol_color_overlay(
            ref_centers=self._ref_xyz_t,
            fit_centers=self._fit_xyz_t,
            ref_pharms=torch.from_numpy(self.ref_molec.pharm_types).to(dev),
            fit_pharms=torch.from_numpy(self.fit_molec.pharm_types).to(dev),
            ref_anchors=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(dev),
            fit_anchors=torch.from_numpy(self.fit_molec.pharm_ancs).to(torch.float32).to(dev),
            ref_vectors=torch.from_numpy(self.ref_molec.pharm_vecs).to(torch.float32).to(dev),
            fit_vectors=torch.from_numpy(self.fit_molec.pharm_vecs).to(torch.float32).to(dev),
            alpha=alpha,
            color_weight=color_weight,
            similarity=similarity,
            directionless=directionless,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=num_repeats,
            trans_centers=self._ref_xyz_t if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )
        self.transform_vol_color = se3_transform.numpy()
        self.sim_aligned_vol_color = score.numpy()
        return aligned_fit_centers.numpy()


    def align_with_esp_field(self,
                             field_weight: float = 0.5,
                             alpha: float = 0.81,
                             alpha_field: float = 0.81,
                             lam: float = 0.1,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False) -> np.ndarray:
        """
        Align using a Cresset-style electrostatic *field-point* overlay: a weighted combination of
        atom-centred Gaussian *shape* (volume) Tanimoto and the overlap of the molecular ESP
        *field points* (signed extrema of the softened electrostatic potential), matched by sign.

        The optimized objective is
        ``(1 - field_weight) * shape_Tanimoto + field_weight * field_Tanimoto`` where the field
        channel scores the field points as signed points via the ESP overlap (same-sign extrema
        reward, opposite-sign penalize). Only the fit molecule is transformed; both its heavy-atom
        centres and its field points move rigidly under the same SE(3) pose. If either molecule has
        no field points the field channel is 0 (shape-only) and the alignment still runs.

        Optimally aligned score is stored in ``self.sim_aligned_esp_field`` and the optimal SE(3)
        transformation in ``self.transform_esp_field``.

        Parameters
        ----------
        field_weight : float, optional
            Weight of the field-point channel in [0, 1]; shape gets ``1 - field_weight``.
            Default is 0.5.
        alpha : float, optional
            Gaussian width for the shape overlap. Default is 0.81 (volumetric, heavy atoms).
        alpha_field : float, optional
            Gaussian width for the field-point positional overlap. Default is 0.81.
        lam : float, optional
            Sign ("charge") weighting for the field-point ESP overlap. Default is 0.1
            (atom-centred/volumetric convention).
        num_repeats : int, optional
            Number of SE(3) initializations. Default is 50.
        trans_init : bool, optional
            Translation-seeded initialization from the reference atoms. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default is 200.
        verbose : bool, optional
            Print progress. Default is ``False``.

        Returns
        -------
        aligned_fit_centers : np.ndarray (M, 3)
            Transformed fit atom (heavy-atom) coordinates.
        """
        dev = self.device
        ref_fp_pos, ref_fp_sign = self.ref_molec.get_field_points()
        fit_fp_pos, fit_fp_sign = self.fit_molec.get_field_points()

        aligned_fit_centers, se3_transform, score = optimize_esp_field_overlay(
            ref_centers=self._ref_xyz_t,
            fit_centers=self._fit_xyz_t,
            ref_fp_pos=self._to_tensor(ref_fp_pos),
            fit_fp_pos=self._to_tensor(fit_fp_pos),
            ref_fp_sign=self._to_tensor(ref_fp_sign),
            fit_fp_sign=self._to_tensor(fit_fp_sign),
            alpha=alpha,
            alpha_field=alpha_field,
            lam=lam,
            field_weight=field_weight,
            num_repeats=num_repeats,
            trans_centers=self._ref_xyz_t if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )
        self.transform_esp_field = se3_transform.numpy()
        self.sim_aligned_esp_field = score.numpy()
        return aligned_fit_centers.numpy()


    def align_with_vol_tversky(self,
                               tversky_alpha: float = 0.95,
                               tversky_beta: float = 0.05,
                               alpha: float = 0.81,
                               no_H: bool = True,
                               num_repeats: int = 50,
                               trans_init: bool = False,
                               lr: float = 0.1,
                               max_num_steps: int = 200,
                               verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using an asymmetric "fits-inside" volumetric *shape* overlay
        scored with **Tversky** rather than Tanimoto.

        The optimized objective is the Tversky shape similarity
        ``AB / (AB + tversky_alpha * (AA - AB) + tversky_beta * (BB - AB))`` where ``AB`` is the
        cross overlap of the reference with the SE(3)-transformed fit, ``AA`` the reference
        self-overlap, and ``BB`` the fit self-overlap (atom-centred Gaussian volume integrals).
        With the defaults (``tversky_alpha=0.95``, ``tversky_beta=0.05``) missing reference volume
        is penalized heavily while extra fit volume is barely penalized, so the score rewards the
        *reference* (query) being contained in the fit -- useful for scaffold hopping / finding
        larger elaborated actives. The objective is asymmetric: swapping ref and fit changes the
        score. Only the fit is transformed.

        Optimally aligned score is stored in ``self.sim_aligned_vol_tversky`` and the optimal
        SE(3) transformation in ``self.transform_vol_tversky``.

        Parameters
        ----------
        tversky_alpha : float, optional
            Weight on missing reference volume ``AA - AB``. Default is 0.95. Named to avoid
            colliding with the Gaussian width ``alpha``.
        tversky_beta : float, optional
            Weight on extra fit volume ``BB - AB``. Default is 0.05.
        alpha : float, optional
            Gaussian width for the shape overlap. Default is 0.81 (volumetric, heavy atoms).
        no_H : bool, optional
            Whether to exclude hydrogens (heavy-atom overlay). Default is ``True``.
        num_repeats : int, optional
            Number of SE(3) initializations. Default is 50.
        trans_init : bool, optional
            Translation-seeded initialization from the reference atoms. Default is ``False``.
        lr : float, optional
            Learning rate. Default is 0.1.
        max_num_steps : int, optional
            Maximum optimization steps. Default is 200.
        verbose : bool, optional
            Print progress. Default is ``False``.

        Returns
        -------
        aligned_fit_points : np.ndarray (M, 3)
            Transformed fit atom (heavy-atom) coordinates.
        """
        ref_atom_pos = self.ref_molec.get_positions(no_H)
        fit_atom_pos = self.fit_molec.get_positions(no_H)

        aligned_fit_points, se3_transform, score = optimize_vol_tversky_overlay(
            ref_points=self._to_tensor(ref_atom_pos),
            fit_points=self._to_tensor(fit_atom_pos),
            alpha=alpha,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            num_repeats=num_repeats,
            trans_centers=self._to_tensor(self.ref_molec.atom_pos) if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )
        self.transform_vol_tversky = se3_transform.numpy()
        self.sim_aligned_vol_tversky = score.numpy()
        return aligned_fit_points.numpy()


    def align_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = None,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = None,
                         use_jax: bool = False,
                         verbose: bool = False,
                         use_vectorized: bool = True,
                         use_analytical: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align fit_molec to ref_molec using pharmacophore similarity.

        Optimally aligned score found in ``self.sim_aligned_pharm`` and the optimal SE(3)
        transformation is at ``self.transform_pharm``.

        Parameters
        ----------
        similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
            Specifies what similarity function to use. Options are:
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
        extended_points : bool, optional
            Whether to score HBA/HBD with gaussian overlaps of extended points. Default is ``False``.
        only_extended : bool, optional
            When ``extended_points`` is ``True``, decide whether to only score the extended points
            (ignore anchor overlaps). Default is ``False``.
        num_repeats : int, optional
            Number of different random initializations of SO(3) transformation parameters.
            Default (``None``) is the per-mode ``MODE_SEEDS`` value in ``shepherd_score/accel/_modes.py``.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s pharmacophore, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats`` rotations
            are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over.
            Default (``None``) is the per-mode ``MODE_STEPS`` value in ``shepherd_score/accel/_modes.py``.
        use_jax : bool, optional
            Whether to use Jax instead of PyTorch. Default is ``False``.
        verbose : bool, optional
            Print initial and final similarity scores with scores every 100 steps.
            Default is ``False``.
        use_vectorized : bool, optional
            Whether to use the vectorized version of the pharmacophore scoring function.
            This is only relevant if ``use_jax=True``.
            Default is ``True``.
        use_analytical : bool, optional
            Whether to use the analytical version of the pharmacophore scoring function.
            Currently only implemented for PyTorch.
            Default is ``True``.
        Returns
        -------
        tuple
            aligned_fit_anchors : np.ndarray
                Aligned coordinates of pharmacophore positions. Shape: (P, 3).
            aligned_fit_vectors : np.ndarray
                Aligned coordinates of pharmacophore vectors. Shape: (P, 3).
        """
        if num_repeats is None:
            num_repeats = _default_seeds("pharm")
        if max_num_steps is None:
            max_num_steps = _default_steps("pharm")
        if use_jax:
            jnp = _require_jax()
            from shepherd_score.alignment_jax import optimize_pharm_overlay_jax, optimize_pharm_overlay_jax_vectorized

            _pharm_fn = optimize_pharm_overlay_jax_vectorized if use_vectorized else optimize_pharm_overlay_jax
            aligned_fit_anchors, aligned_fit_vectors, se3_transform, score = _pharm_fn(
                ref_pharms=jnp.array(self.ref_molec.pharm_types),
                fit_pharms=jnp.array(self.fit_molec.pharm_types),
                ref_anchors=jnp.array(self.ref_molec.pharm_ancs),
                fit_anchors=jnp.array(self.fit_molec.pharm_ancs),
                ref_vectors=jnp.array(self.ref_molec.pharm_vecs),
                fit_vectors=jnp.array(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended,
                num_repeats=num_repeats,
                trans_centers=self.ref_molec.pharm_ancs if trans_init else None,
                lr=lr,
                max_num_steps=max_num_steps,
                verbose=verbose
            )

            self.transform_pharm = np.array(se3_transform)
            self.sim_aligned_pharm = np.array(score)
            return np.array(aligned_fit_anchors), np.array(aligned_fit_vectors)

        # PyTorch
        _pharm_fn = optimize_pharm_overlay_analytical if use_analytical else optimize_pharm_overlay
        aligned_fit_anchors, aligned_fit_vectors, se3_transform, score = _pharm_fn(
            ref_pharms=self._to_tensor(self.ref_molec.pharm_types),
            fit_pharms=self._to_tensor(self.fit_molec.pharm_types),
            ref_anchors=self._to_tensor(self.ref_molec.pharm_ancs),
            fit_anchors=self._to_tensor(self.fit_molec.pharm_ancs),
            ref_vectors=self._to_tensor(self.ref_molec.pharm_vecs),
            fit_vectors=self._to_tensor(self.fit_molec.pharm_vecs),
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=num_repeats,
            trans_centers=self._to_tensor(self.ref_molec.pharm_ancs) if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

        self.transform_pharm = se3_transform.numpy()
        self.sim_aligned_pharm = score.numpy()
        return aligned_fit_anchors.numpy(), aligned_fit_vectors.numpy()


    def score_with_surf(self,
                        alpha: float,
                        use: str = 'np'
                        ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using surface similarity given current alignment.
        By default it uses the numpy implementation.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use == 'jax' or use == 'jnp': # Use Jax optimization implementation
            jnp = _require_jax()
            from shepherd_score.score.gaussian_overlap_jax import get_overlap_jax
            score = get_overlap_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                alpha=alpha,
            )
            return np.array(score)
        elif use == 'torch' or use == 'pytorch':
            # Torch
            score = get_overlap(
                centers_1=self._to_tensor(self.ref_molec.surf_pos),
                centers_2=self._to_tensor(self.fit_molec.surf_pos),
                alpha=alpha,
            )
            return score.cpu().numpy()
        elif use == 'np' or use == 'numpy':
            score = get_overlap_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                alpha=alpha,
            )
            return score


    def score_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       use: str = 'np'
                       ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using ESP+surface similarity given current alignment.
        ``lam`` is scaled by ``(1e4/(4*55.263*np.pi))**2`` for correct units.

        Typically ``lam = 0.3`` is used and is scaled internally.
        By default it uses the numpy implementation.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float, optional
            Weighting factor for ESP scoring. Default is 0.3.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        lam_scaled = LAM_SCALING * lam
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use in ('jax', 'jnp'): # Use Jax implementation
            jnp = _require_jax()
            from shepherd_score.score.electrostatic_scoring_jax import get_overlap_esp_jax
            score = get_overlap_esp_jax(
                centers_1=jnp.array(self.ref_molec.surf_pos),
                centers_2=jnp.array(self.fit_molec.surf_pos),
                charges_1=jnp.array(self.ref_molec.surf_esp),
                charges_2=jnp.array(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
            )
            return np.array(score)
        elif use in ('torch', 'pytorch'): # Use Torch implementation
            score = get_overlap_esp(
                centers_1=self._to_tensor(self.ref_molec.surf_pos),
                centers_2=self._to_tensor(self.fit_molec.surf_pos),
                charges_1=self._to_tensor(self.ref_molec.surf_esp),
                charges_2=self._to_tensor(self.fit_molec.surf_esp),
                alpha=alpha,
                lam=lam_scaled,
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_esp_np(
                centers_1=self.ref_molec.surf_pos,
                centers_2=self.fit_molec.surf_pos,
                charges_1=self.ref_molec.surf_esp,
                charges_2=self.fit_molec.surf_esp,
                alpha=alpha,
                lam=lam_scaled,
            )
            return score


    def score_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         use: str = 'np'
                         ) -> np.ndarray:
        """
        Score fit_molec to ref_molec using pharmacophore similarity given current alignment.
        By default it uses the numpy implementation.

        Parameters
        ----------
        similarity : str from ('tanimoto', 'tversky', 'tversky_ref', 'tversky_fit')
            Specifies what similarity function to use. Options are:
            'tanimoto' -- symmetric scoring function
            'tversky' -- asymmetric -> Uses OpenEye's formulation 95% normalization by molec 1
            'tversky_ref' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 1.
            'tversky_fit' -- asymmetric -> Uses Pharao's formulation 100% normalization by molec 2.
        extended_points : bool, optional
            Whether to score HBA/HBD with gaussian overlaps of extended points.
            Default is ``False``.
        only_extended : bool, optional
            When ``extended_points`` is ``True``, decide whether to only score the extended
            points (ignore anchor overlaps). Default is ``False``.
        use : str, optional
            Specifies what implementation to use. Options are:
            - 'np' or 'numpy' (numpy implementation)
            - 'jax' or 'jnp' (Jax implementation)
            - 'torch' or 'pytorch' (PyTorch implementation)
            Default is 'np'.

        Returns
        -------
        score : np.ndarray
            Similarity score. Shape: (1,).
        """
        use = use.lower()
        accepted_keys = ('jax', 'jnp', 'torch', 'pytorch', 'np', 'numpy')
        if use not in accepted_keys:
            raise ValueError(f"`use` must be in {accepted_keys}. Instead {use} was passed.")
        elif use in ('torch', 'pytorch'):
            # PyTorch
            score = get_overlap_pharm(
                ptype_1=self._to_tensor(self.ref_molec.pharm_types),
                ptype_2=self._to_tensor(self.fit_molec.pharm_types),
                anchors_1=self._to_tensor(self.ref_molec.pharm_ancs),
                anchors_2=self._to_tensor(self.fit_molec.pharm_ancs),
                vectors_1=self._to_tensor(self.ref_molec.pharm_vecs),
                vectors_2=self._to_tensor(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score.cpu().numpy()
        elif use in ('np', 'numpy'):
            score = get_overlap_pharm_np(
                ptype_1=self.ref_molec.pharm_types,
                ptype_2=self.fit_molec.pharm_types,
                anchors_1=self.ref_molec.pharm_ancs,
                anchors_2=self.fit_molec.pharm_ancs,
                vectors_1=self.ref_molec.pharm_vecs,
                vectors_2=self.fit_molec.pharm_vecs,
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return score
        elif use in ('jax', 'jnp'):
            jnp = _require_jax()
            from shepherd_score.score.pharmacophore_scoring_jax import get_overlap_pharm_jax

            score = get_overlap_pharm_jax(
                ptype_1=jnp.array(self.ref_molec.pharm_types),
                ptype_2=jnp.array(self.fit_molec.pharm_types),
                anchors_1=jnp.array(self.ref_molec.pharm_ancs),
                anchors_2=jnp.array(self.fit_molec.pharm_ancs),
                vectors_1=jnp.array(self.ref_molec.pharm_vecs),
                vectors_2=jnp.array(self.fit_molec.pharm_vecs),
                similarity=similarity,
                extended_points=extended_points,
                only_extended=only_extended
            )
            return np.array(score)


    def get_transformed_mol_and_feats(self,
                                      se3_transform: np.ndarray
                                      ) -> Tuple:
        """
        Get an RDKit mol object and applicable features with a transformation applied.

        Parameters
        ----------
        se3_transform : np.ndarray
            SE(3) transformation matrix. Shape: (4,4).

        Returns
        -------
        tuple
            transformed_mol : rdkit.Chem.Mol
                Molecule with transformed coordinates.
            transformed_surf_pos : np.ndarray
                Transformed surface points. Shape: (N, 3).
            transformed_pharm_ancs : np.ndarray
                Transformed pharmacophore anchor positions. Shape: (P, 3).
            transformed_pharm_vecs : np.ndarray
                Transformed pharmacophore vector positions. Shape: (P, 3).
        """
        # Transform mol
        transformed_mol = update_mol_coordinates(
            mol=self.fit_molec.mol,
            coordinates=apply_SE3_transform_np(
                points=self.fit_molec.mol.GetConformer().GetPositions(),
                SE3_transform=se3_transform
            )
        )

        # Transform surface points
        transformed_surf_pos = None
        if self.fit_molec.surf_pos is not None:
            transformed_surf_pos = apply_SE3_transform_np(
                points=self.fit_molec.surf_pos,
                SE3_transform=se3_transform
            )

        # Transform pharmacophore features
        transformed_pharm_ancs = None
        transformed_pharm_vecs = None
        if self.fit_molec.pharm_ancs is not None and self.fit_molec.pharm_vecs is not None:
            transformed_pharm_ancs = apply_SE3_transform_np(
                points=self.fit_molec.pharm_ancs,
                SE3_transform=se3_transform
            )
            transformed_pharm_vecs = apply_SO3_transform_np(
                points=self.fit_molec.pharm_vecs,
                SE3_transform=se3_transform
            )
        return transformed_mol, transformed_surf_pos, transformed_pharm_ancs, transformed_pharm_vecs


    def get_transformed_molecule(self,
                                 se3_transform: np.ndarray
                                 ) -> Molecule:
        """
        Get Molecule object transformation applied to all applicable features for the fit molecule.

        Parameters
        ----------
        se3_transform : np.ndarray
            SE(3) transformation matrix. Shape: (4,4).

        Returns
        -------
        Molecule
            Molecule with transformed features.
        """
        (transformed_mol,
        transformed_surf_pos,
        transformed_pharm_ancs,
        transformed_pharm_vecs) = self.get_transformed_mol_and_feats(se3_transform=se3_transform)

        transformed_fit_molec = Molecule(mol=transformed_mol,
                                         probe_radius=self.fit_molec.probe_radius,
                                         surface_points=transformed_surf_pos,
                                         partial_charges=self.fit_molec.partial_charges,
                                         electrostatics=self.fit_molec.surf_esp,
                                         pharm_multi_vector=self.fit_molec.pharm_multi_vector,
                                         pharm_types=self.fit_molec.pharm_types,
                                         pharm_ancs=transformed_pharm_ancs,
                                         pharm_vecs=transformed_pharm_vecs
                                         )
        return transformed_fit_molec

