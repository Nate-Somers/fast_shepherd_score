"""
Molecule class to hold molecule geometries and extract interaction profiles.
MoleculePair class facilitates alignment with interaction profiles.
"""
from typing import Union, List, Optional, Tuple
from copy import deepcopy
import sys

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
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.score.pharmacophore_scoring import _SIM_TYPE, get_overlap_pharm
from shepherd_score.alignment import optimize_ROCS_overlay, optimize_ROCS_overlay_analytical, optimize_ROCS_esp_overlay, optimize_ROCS_esp_overlay_analytical, optimize_esp_combo_score_overlay
from shepherd_score.alignment import optimize_pharm_overlay, optimize_pharm_overlay_analytical
from shepherd_score.alignment import optimize_vol_color_overlay
from shepherd_score.alignment.utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np
from shepherd_score.accel import batch as _ba
from shepherd_score.accel._modes import (
    MODE_ATTRS as _MODE_ATTRS,
    CANONICAL_MODES as _CANONICAL_MODES,
    LEGACY_MODE_ALIASES as _LEGACY_MODE_ALIASES,
)


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
            If ``None``, don't generate pharmacophores, else generate
            pharmacophores with/without (``True``/``False``) multi-vectors.
        pharm_types : Optional[np.ndarray]
            Types of pharmacophores. Shape: (P,).
        pharm_ancs : Optional[np.ndarray]
            Anchor positions of pharmacophores. Shape: (P,3).
        pharm_vecs : Optional[np.ndarray]
            Unit vectors relative to anchor positions of pharmacophores. Shape: (P,3).
        feature_set : str
            Which pharmacophore feature definition to use when generating pharmacophores.
            ``'shepherd'`` (default) uses the local ``smarts_features.fdef`` (8 fss types);
            ``'rdkit_base'`` uses RDKit's stock ``BaseFeatures.fdef`` reduced to the 6
            ROCS/ROSHAMBO color types. Only used when pharmacophores are generated (i.e.
            ``pharm_multi_vector`` is not ``None`` and explicit arrays are not provided).
        directionless : bool
            When ``True``, generate isotropic (zero-vector) "color" pharmacophores for all
            families (ROCS/ROSHAMBO style). Default ``False`` computes orientation vectors.
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

        if surface_points is None:
            self.probe_radius = probe_radius if probe_radius is not None else 1.2
            if isinstance(num_surf_points, int):
                self.surf_pos = self.get_pc()
            elif isinstance(density, float):
                self.surf_pos = self.get_pc(use_density=True)
            else: # if None then don't generate a point cloud
                self.surf_pos = None
                self.surf_esp = None
        else:
            self.surf_pos = surface_points
            self.probe_radius = probe_radius if probe_radius is not None else 1.2

        if self.surf_pos is not None and self.partial_charges is not None:
            if not isinstance(electrostatics, np.ndarray):
                self.surf_esp = self.get_electrostatic_potential()
            else:
                self.surf_esp = electrostatics

        # Indices for atoms that aren't hydrogens
        self._nonH_atoms_idx = np.array([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1])

        self.pharm_multi_vector = pharm_multi_vector
        self.pharm_feature_set = feature_set
        if isinstance(pharm_types, np.ndarray) and isinstance(pharm_ancs, np.ndarray) and isinstance(pharm_vecs, np.ndarray):
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = pharm_types, pharm_ancs, pharm_vecs
        else:
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = None, None, None
            if self.pharm_multi_vector is not None:
                self.get_pharmacophore(
                    multi_vector=self.pharm_multi_vector,
                    exclude=[],
                    check_access=False,
                    scale=1.,
                    feature_set=feature_set,
                    directionless=directionless
                )

    def get_partial_charges(self) -> np.ndarray:
        """
        Get the partial charges on each atom using MMFF.
        """
        molec_props = Chem.AllChem.MMFFGetMoleculeProperties(self.mol)
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(self.mol.GetAtoms())])
        return charges.astype(np.float32)


    def get_pc(self, use_density=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the point cloud positions.
        """
        self.mol, centers = get_atom_coords(self.mol, MMFF_optimize=False)
        surface_method = getattr(self, 'surface_method', 'mesh')
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


    def get_pharmacophore(self,
                          multi_vector: bool = True,
                          exclude: List[int] = [],
                          check_access: bool = False,
                          scale: float = 1,
                          feature_set: str = 'shepherd',
                          directionless: bool = False):
        """ Get the pharmacophores of the molecule. """
        self.pharm_types, self.pharm_ancs, self.pharm_vecs = get_pharmacophores(
            self.mol,
            multi_vector=multi_vector,
            exclude=exclude,
            check_access=check_access,
            scale=scale,
            feature_set=feature_set,
            directionless=directionless
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
        - Pharmacophores (with various settings for using extended points rather than vectors)

        Similarly, you can score with surface, Surf+ESP, and pharmacophores

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

        # Result slots (transform=identity, score=None until aligned). The 7 canonical modes
        # are driven by the registry (accel/_modes.MODE_ATTRS); adding a mode there auto-creates
        # its slots here. The two no_H-variant slots below are legacy extras written by
        # align_with_vol/align_with_vol_esp(no_H=False) and are intentionally not registry modes.
        for _t, _s in _MODE_ATTRS.values():
            setattr(self, _t, np.eye(4))
            setattr(self, _s, None)
        for _t, _s in (("transform_vol", "sim_aligned_vol"),
                       ("transform_vol_esp", "sim_aligned_vol_esp")):
            setattr(self, _t, np.eye(4))
            setattr(self, _s, None)

    # --- legacy result-attribute aliases (renamed modes; old names kept working) ---
    # esp -> surf_esp, esp_combo -> vol_and_surf_esp. The canonical attributes set in
    # __init__ are the real storage; these properties redirect old reads/writes to them.
    @property
    def sim_aligned_esp(self): return self.sim_aligned_surf_esp
    @sim_aligned_esp.setter
    def sim_aligned_esp(self, v): self.sim_aligned_surf_esp = v
    @property
    def transform_esp(self): return self.transform_surf_esp
    @transform_esp.setter
    def transform_esp(self, v): self.transform_surf_esp = v
    @property
    def sim_aligned_esp_combo(self): return self.sim_aligned_vol_and_surf_esp
    @sim_aligned_esp_combo.setter
    def sim_aligned_esp_combo(self, v): self.sim_aligned_vol_and_surf_esp = v
    @property
    def transform_esp_combo(self): return self.transform_vol_and_surf_esp
    @transform_esp_combo.setter
    def transform_esp_combo(self, v): self.transform_vol_and_surf_esp = v

    # --- batched GPU/Triton aligners -------------------------------------
    # Implemented as free functions in ``accel.batch``; bound as static methods (one per
    # registry mode + legacy aliases) by the ``@_bind_batch_aligners`` class decorator above, so
    # the public seam ``MoleculePair._align_batch_vol(pairs, ...)`` etc. still resolves here.

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
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
            Number of different random initializations of SO(3) transformation parameters. Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass (COM) is translated to
            each ``ref_molec``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over. Default is 200.
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
        if no_H:
            ref_atom_pos = self.ref_molec.atom_pos
            fit_atom_pos = self.fit_molec.atom_pos
        else:
            ref_atom_pos = self.ref_molec.mol.GetConformer().GetPositions()
            # ref_atom_pos -= ref_atom_pos.mean(0)
            fit_atom_pos = self.fit_molec.mol.GetConformer().GetPositions()
            # fit_atom_pos -= fit_atom_pos.mean(0)
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                ref_points=torch.from_numpy(ref_atom_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(fit_atom_pos).to(torch.float32).to(self.device),
                alpha=0.81,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
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
                           lam: float,
                           no_H: bool = True,
                           num_repeats: int = 50,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = 200,
                           use_jax: bool = False,
                           use_analytical: bool = True,
                           verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using volume similarity weighted by partial charge
        Toggle ``no_H`` parameter for scoring with or without hydrogens.

        Typically ``lam=0.1`` is used.
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
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s center of mass
            (COM) is translated to each ``ref_molec``'s atoms, with 10 rotations for each translation.
            So the number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COMs. If ``None``, then ``num_repeats``
            rotations are done with aligned COMs. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over. Default is 200.
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
        if no_H:
            ref_mol_partial_charges = self.ref_molec.partial_charges[self.ref_molec._nonH_atoms_idx]
            fit_mol_partial_charges = self.fit_molec.partial_charges[self.fit_molec._nonH_atoms_idx]
            ref_mol_pos = self.ref_molec.atom_pos
            fit_mol_pos = self.fit_molec.atom_pos
        else:
            ref_mol_partial_charges = self.ref_molec.partial_charges
            fit_mol_partial_charges = self.fit_molec.partial_charges
            ref_mol_pos = self.ref_molec.mol.GetConformer().GetPositions()
            # ref_mol_pos -= ref_mol_pos.mean(0) # move COM to origin
            fit_mol_pos = self.fit_molec.mol.GetConformer().GetPositions()
            # fit_mol_pos -= fit_mol_pos.mean(0)

        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                ref_points=torch.from_numpy(ref_mol_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(fit_mol_pos).to(torch.float32).to(self.device),
                ref_charges=torch.from_numpy(ref_mol_partial_charges).to(torch.float32).to(self.device),
                fit_charges=torch.from_numpy(fit_mol_partial_charges).to(torch.float32).to(self.device),
                alpha=0.81,
                lam=lam,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
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
                        num_repeats: int = 50,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = 200,
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
            Default is 50.
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
            Maximum number of steps to optimize over. Default is 200.
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
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                alpha=alpha,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
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
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       use_fast: bool = False,
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
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If None, then num_repeats rotations are done
            with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over. Default is 200.
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
        lam_scaled = LAM_SCALING * lam
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
        else: # Use Torch implementation (opt-in CUDA fast path via use_fast)
            if use_fast and torch.cuda.is_available():
                try:
                    from shepherd_score.accel.drivers.esp import fast_optimize_ROCS_esp_overlay
                except ImportError:
                    fast_optimize_ROCS_esp_overlay = None

                if fast_optimize_ROCS_esp_overlay is not None:
                    # Prefer cached tensors to avoid repeated host->device transfers.
                    if getattr(self, "_ref_surf_t", None) is None:
                        self._ref_surf_t = torch.as_tensor(self.ref_molec.surf_pos, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_surf_t", None) is None:
                        self._fit_surf_t = torch.as_tensor(self.fit_molec.surf_pos, dtype=torch.float32, device=self.device)
                    if getattr(self, "_ref_surf_esp_t", None) is None:
                        self._ref_surf_esp_t = torch.as_tensor(self.ref_molec.surf_esp, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_surf_esp_t", None) is None:
                        self._fit_surf_esp_t = torch.as_tensor(self.fit_molec.surf_esp, dtype=torch.float32, device=self.device)

                    trans_centers = None
                    if trans_init:
                        trans_centers = self._ref_xyz_t if hasattr(self, "_ref_xyz_t") else torch.as_tensor(
                            self.ref_molec.atom_pos, dtype=torch.float32, device=self.device
                        )

                    aligned_fit_points_t, se3_transform_t, score_t = fast_optimize_ROCS_esp_overlay(
                        ref_points=self._ref_surf_t,
                        fit_points=self._fit_surf_t,
                        ref_charges=self._ref_surf_esp_t,
                        fit_charges=self._fit_surf_esp_t,
                        alpha=alpha,
                        lam=lam_scaled,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers,
                        num_repeats_per_trans=10,
                        topk=30,
                        steps_fine=max_num_steps,
                        lr=lr,
                    )

                    self.transform_surf_esp = se3_transform_t.numpy()
                    self.sim_aligned_surf_esp = score_t.numpy()
                    return aligned_fit_points_t.numpy()

            _esp_fn = optimize_ROCS_esp_overlay_analytical if use_analytical else optimize_ROCS_esp_overlay
            aligned_fit_points, se3_transform, score = _esp_fn(
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                ref_charges=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                fit_charges=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
                alpha=alpha,
                lam=lam_scaled,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
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
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             use_jax: bool = False,
                             use_fast: bool = False,
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
            Number of different random initializations of SO(3) transformation parameters. Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated
            to each ``ref_molecs``'s atoms, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is
            from the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats``
            rotations are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over. Default is 200.
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
        if self.num_surf_points is None:
            raise ValueError('The Molecule objects were initialized with no surface points so this method cannot be used.')
        if use_jax: # Use Jax optimization implementation
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                ref_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.atom_pos).to(torch.float32).to(self.device)
            else:
                ref_centers = torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device)

            if use_fast and torch.cuda.is_available():
                try:
                    from shepherd_score.accel.drivers.esp_combo import fast_optimize_esp_combo_score_overlay
                except ImportError:
                    fast_optimize_esp_combo_score_overlay = None

                if fast_optimize_esp_combo_score_overlay is not None:
                    # Prefer cached tensors (also used by the batch path).
                    if getattr(self, "_ref_surf_t", None) is None:
                        self._ref_surf_t = torch.as_tensor(self.ref_molec.surf_pos, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_surf_t", None) is None:
                        self._fit_surf_t = torch.as_tensor(self.fit_molec.surf_pos, dtype=torch.float32, device=self.device)
                    if getattr(self, "_ref_surf_esp_t", None) is None:
                        self._ref_surf_esp_t = torch.as_tensor(self.ref_molec.surf_esp, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_surf_esp_t", None) is None:
                        self._fit_surf_esp_t = torch.as_tensor(self.fit_molec.surf_esp, dtype=torch.float32, device=self.device)

                    if getattr(self, "_ref_centers_w_H_t", None) is None:
                        self._ref_centers_w_H_t = torch.as_tensor(
                            self.ref_molec.mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device
                        )
                    if getattr(self, "_fit_centers_w_H_t", None) is None:
                        self._fit_centers_w_H_t = torch.as_tensor(
                            self.fit_molec.mol.GetConformer().GetPositions(), dtype=torch.float32, device=self.device
                        )
                    if getattr(self, "_ref_partial_t", None) is None:
                        self._ref_partial_t = torch.as_tensor(self.ref_molec.partial_charges, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_partial_t", None) is None:
                        self._fit_partial_t = torch.as_tensor(self.fit_molec.partial_charges, dtype=torch.float32, device=self.device)
                    if getattr(self, "_ref_radii_t", None) is None:
                        self._ref_radii_t = torch.as_tensor(self.ref_molec.radii, dtype=torch.float32, device=self.device)
                    if getattr(self, "_fit_radii_t", None) is None:
                        self._fit_radii_t = torch.as_tensor(self.fit_molec.radii, dtype=torch.float32, device=self.device)

                    # Centers for shape component
                    if alpha == 0.81:
                        ref_centers_t = self._ref_xyz_t
                        fit_centers_t = self._fit_xyz_t
                    else:
                        ref_centers_t = self._ref_surf_t
                        fit_centers_t = self._fit_surf_t

                    trans_centers = self._ref_xyz_t if trans_init else None

                    aligned_fit_points_t, se3_transform_t, score_t = fast_optimize_esp_combo_score_overlay(
                        ref_centers_w_H=self._ref_centers_w_H_t,
                        fit_centers_w_H=self._fit_centers_w_H_t,
                        ref_centers=ref_centers_t,
                        fit_centers=fit_centers_t,
                        ref_points=self._ref_surf_t,
                        fit_points=self._fit_surf_t,
                        ref_partial_charges=self._ref_partial_t,
                        fit_partial_charges=self._fit_partial_t,
                        ref_surf_esp=self._ref_surf_esp_t,
                        fit_surf_esp=self._fit_surf_esp_t,
                        ref_radii=self._ref_radii_t,
                        fit_radii=self._fit_radii_t,
                        alpha=alpha,
                        lam=lam,
                        probe_radius=probe_radius,
                        esp_weight=esp_weight,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers,
                        num_repeats_per_trans=10,
                        topk=30,
                        steps_fine=max_num_steps,
                        lr=lr,
                    )

                    self.transform_vol_and_surf_esp = se3_transform_t.numpy()
                    self.sim_aligned_vol_and_surf_esp = score_t.numpy()
                    return aligned_fit_points_t.numpy()

            aligned_fit_points, se3_transform, score = optimize_esp_combo_score_overlay(
                ref_centers_w_H=torch.from_numpy(self.ref_molec.mol.GetConformer().GetPositions()).to(torch.float32).to(self.device),
                fit_centers_w_H=torch.from_numpy(self.fit_molec.mol.GetConformer().GetPositions()).to(torch.float32).to(self.device),
                ref_centers=ref_centers,
                fit_centers=fit_centers,
                ref_points=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                fit_points=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                ref_partial_charges=torch.from_numpy(self.ref_molec.partial_charges).to(torch.float32).to(self.device),
                fit_partial_charges=torch.from_numpy(self.fit_molec.partial_charges).to(torch.float32).to(self.device),
                ref_surf_esp=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                fit_surf_esp=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
                ref_radii=torch.from_numpy(self.ref_molec.radii).to(torch.float32).to(self.device),
                fit_radii=torch.from_numpy(self.fit_molec.radii).to(torch.float32).to(self.device),
                alpha=alpha,
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                num_repeats=num_repeats,
                trans_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device) if trans_init else None,
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
                             directional: bool = False,
                             extended_points: bool = False,
                             only_extended: bool = False,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             verbose: bool = False,
                             use_fast: bool = False) -> np.ndarray:
        """
        Align using a ROCS/ROSHAMBO-style combined atom-centred Gaussian *shape* (volume) +
        directionless *color* (pharmacophore) overlay (a TanimotoCombo analogue).

        The optimized objective is
        ``(1 - color_weight) * shape_Tanimoto + color_weight * color_Tanimoto``. By default
        the color channel is *directionless* (isotropic point Gaussians, ROCS/ROSHAMBO
        "color"); pass ``directional=True`` to keep fss's orientation-vector weighting. For
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
        directional : bool, optional
            ``False`` (default) scores color as isotropic point Gaussians; ``True`` uses the
            orientation-vector cosine weighting.
        extended_points, only_extended : bool, optional
            Forwarded to the color scorer (ignored when ``directional=False``).
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
        if self.ref_molec.pharm_types is None or self.fit_molec.pharm_types is None:
            raise ValueError(
                'Both Molecule objects must have pharmacophores to use align_with_vol_color. '
                "Build them with `pharm_multi_vector` set (and optionally "
                "`feature_set='rdkit_base'` for ROCS/ROSHAMBO color)."
            )

        dev = self.device

        # Opt-in per-pair fast path: the batched fused-kernel driver (shape Triton/numba
        # kernel + fused directionless color kernel). Only covers the directionless tanimoto
        # case it implements; otherwise falls through to the torch optimizer.
        if (use_fast and torch.cuda.is_available() and not directional
                and similarity == 'tanimoto' and not extended_points):
            try:
                from shepherd_score.accel.drivers.vol_color import fast_optimize_vol_color_overlay
            except ImportError:
                fast_optimize_vol_color_overlay = None
            if fast_optimize_vol_color_overlay is not None:
                aligned_fit_centers, se3_transform, score = fast_optimize_vol_color_overlay(
                    ref_centers=self._ref_xyz_t,
                    fit_centers=self._fit_xyz_t,
                    ref_types=torch.as_tensor(self.ref_molec.pharm_types, dtype=torch.int64, device=dev),
                    fit_types=torch.as_tensor(self.fit_molec.pharm_types, dtype=torch.int64, device=dev),
                    ref_ancs=torch.as_tensor(self.ref_molec.pharm_ancs, dtype=torch.float32, device=dev),
                    fit_ancs=torch.as_tensor(self.fit_molec.pharm_ancs, dtype=torch.float32, device=dev),
                    alpha=alpha, color_weight=color_weight, num_repeats=num_repeats,
                    trans_centers=self._ref_xyz_t if trans_init else None,
                    steps_fine=max_num_steps, lr=lr,
                )
                self.transform_vol_color = se3_transform.numpy()
                self.sim_aligned_vol_color = score.numpy()
                return aligned_fit_centers.numpy()

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
            directional=directional,
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


    def align_with_pharm(self,
                         similarity: _SIM_TYPE = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = 50,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = 200,
                         use_jax: bool = False,
                         verbose: bool = False,
                         use_vectorized: bool = True,
                         use_analytical: bool = True,
                         use_fast: bool = False,
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
            Default is 50.
        trans_init : bool, optional
            Apply translation initializiation for alignment. ``fit_molec``'s COM is translated to
            each ``ref_molecs``'s pharmacophore, with 10 rotations for each translation. So the
            number of initializations scales as (# translation centers * 10 + 5) where 5 is from
            the identity and 4 PCA with aligned COM's. If ``None``, then ``num_repeats`` rotations
            are done with aligned COM's. Default is ``False``.
        lr : float, optional
            Learning rate or step-size for optimization. Default is 0.1.
        max_num_steps : int, optional
            Maximum number of steps to optimize over. Default is 200.
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
                Aligned coordinates of pharmacophores positions. Shape: (P, 3).
            aligned_fit_vectors : np.ndarray
                Aligned coordinates of pharmacophore vectors. Shape: (P, 3).
        """
        if use_jax:
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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

        # PyTorch (opt-in CUDA fast path via use_fast)
        if use_fast and torch.cuda.is_available():
            try:
                from shepherd_score.accel.drivers.pharm import fast_optimize_pharm_overlay
            except ImportError:
                fast_optimize_pharm_overlay = None

            if fast_optimize_pharm_overlay is not None:
                if getattr(self, "_ref_pharm_types_t", None) is None:
                    self._ref_pharm_types_t = torch.as_tensor(self.ref_molec.pharm_types, dtype=torch.int64, device=self.device)
                if getattr(self, "_fit_pharm_types_t", None) is None:
                    self._fit_pharm_types_t = torch.as_tensor(self.fit_molec.pharm_types, dtype=torch.int64, device=self.device)
                if getattr(self, "_ref_pharm_ancs_t", None) is None:
                    self._ref_pharm_ancs_t = torch.as_tensor(self.ref_molec.pharm_ancs, dtype=torch.float32, device=self.device)
                if getattr(self, "_fit_pharm_ancs_t", None) is None:
                    self._fit_pharm_ancs_t = torch.as_tensor(self.fit_molec.pharm_ancs, dtype=torch.float32, device=self.device)
                if getattr(self, "_ref_pharm_vecs_t", None) is None:
                    self._ref_pharm_vecs_t = torch.as_tensor(self.ref_molec.pharm_vecs, dtype=torch.float32, device=self.device)
                if getattr(self, "_fit_pharm_vecs_t", None) is None:
                    self._fit_pharm_vecs_t = torch.as_tensor(self.fit_molec.pharm_vecs, dtype=torch.float32, device=self.device)

                trans_centers = self._ref_pharm_ancs_t if trans_init else None

                aligned_fit_anchors_t, aligned_fit_vectors_t, se3_transform_t, score_t = fast_optimize_pharm_overlay(
                    ref_pharms=self._ref_pharm_types_t,
                    fit_pharms=self._fit_pharm_types_t,
                    ref_anchors=self._ref_pharm_ancs_t,
                    fit_anchors=self._fit_pharm_ancs_t,
                    ref_vectors=self._ref_pharm_vecs_t,
                    fit_vectors=self._fit_pharm_vecs_t,
                    similarity=similarity,
                    extended_points=extended_points,
                    only_extended=only_extended,
                    num_repeats=num_repeats,
                    trans_centers=trans_centers,
                    num_repeats_per_trans=10,
                    topk=30,
                    steps_fine=max_num_steps,
                    lr=lr,
                )

                self.transform_pharm = se3_transform_t.numpy()
                self.sim_aligned_pharm = score_t.numpy()
                return aligned_fit_anchors_t.numpy(), aligned_fit_vectors_t.numpy()

        # PyTorch
        _pharm_fn = optimize_pharm_overlay_analytical if use_analytical else optimize_pharm_overlay
        aligned_fit_anchors, aligned_fit_vectors, se3_transform, score = _pharm_fn(
            ref_pharms=torch.from_numpy(self.ref_molec.pharm_types).to(torch.float32).to(self.device),
            fit_pharms=torch.from_numpy(self.fit_molec.pharm_types).to(torch.float32).to(self.device),
            ref_anchors=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device),
            fit_anchors=torch.from_numpy(self.fit_molec.pharm_ancs).to(torch.float32).to(self.device),
            ref_vectors=torch.from_numpy(self.ref_molec.pharm_vecs).to(torch.float32).to(self.device),
            fit_vectors=torch.from_numpy(self.fit_molec.pharm_vecs).to(torch.float32).to(self.device),
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=num_repeats,
            trans_centers=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device) if trans_init else None,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

        self.transform_pharm = se3_transform.numpy()
        self.sim_aligned_pharm = score.numpy()
        return aligned_fit_anchors.numpy(), aligned_fit_vectors.numpy()


    def score_with_vol(
        self,
        alpha: float,
        *,
        no_H: bool = True,
        use: str = "torch",
    ) -> np.ndarray:
        """
        Shape (volume) Tanimoto similarity between *ref_molec* and *fit_molec*
        given their **current alignment**.

        Parameters
        ----------
        alpha : float
            Gaussian width parameter used for the overlap calculation.
        no_H : bool (default = True)
            When True, hydrogens are ignored for both molecules.
        use : str (default = 'torch')
            Pick the backend implementation:
              'np'/'numpy'   → NumPy
              'torch'/'pytorch' → PyTorch (recommended, fastest)
              'jax'/'jnp'    → JAX  (if installed)

        Returns
        -------
        score : np.ndarray shape (1,)
            Tanimoto-style similarity score.
        """
        use = use.lower()
        accepted = ("jax", "jnp", "torch", "pytorch", "np", "numpy")
        if use not in accepted:
            raise ValueError(f"`use` must be one of {accepted}, got {use!r}")

        # Choose which atomic coordinates to feed to the overlap kernels.
        def _coords(mol):
            if no_H:
                # Fast path if the class already pre-computed non-H indices
                if hasattr(mol, "_nonH_atoms_idx"):
                    return mol.atom_pos[mol._nonH_atoms_idx]
            return mol.atom_pos

        # -------------------------- JAX -------------------------------------
        if use in ("jax", "jnp"):
            try:
                import jax.numpy as jnp
            except ImportError:
                raise ImportError(
                    "JAX is not installed.  Use `use='torch'` or `use='np'`."
                )
            from shepherd_score.score.gaussian_overlap_jax import (
                get_overlap_jax,
            )

            score = get_overlap_jax(
                centers_1=jnp.array(_coords(self.ref_molec)),
                centers_2=jnp.array(_coords(self.fit_molec)),
                alpha=alpha,
            )
            return np.array(score)  # keep return type consistent

        # -------------------------- PyTorch---------------------------------
        elif use in ("torch", "pytorch"):
            import torch
            from shepherd_score.score.gaussian_overlap import get_overlap

            score = get_overlap(
                centers_1=torch.as_tensor(_coords(self.ref_molec), dtype=torch.float32, device=self.device),
                centers_2=torch.as_tensor(_coords(self.fit_molec), dtype=torch.float32, device=self.device),
                alpha=alpha,
            )
            return score.cpu().numpy()

        # -------------------------- NumPy -----------------------------------
        else:  # 'np' / 'numpy'
            # NB: do NOT `import numpy as np` here — it would shadow the module-level
            # `np` and make the jax branch's `np.array(score)` raise UnboundLocalError.
            from shepherd_score.score.gaussian_overlap_np import get_overlap_np

            score = get_overlap_np(
                centers_1=_coords(self.ref_molec),
                centers_2=_coords(self.fit_molec),
                alpha=alpha,
            )
            return score


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
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                centers_2=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
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
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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
                centers_1=torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device),
                centers_2=torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device),
                charges_1=torch.from_numpy(self.ref_molec.surf_esp).to(torch.float32).to(self.device),
                charges_2=torch.from_numpy(self.fit_molec.surf_esp).to(torch.float32).to(self.device),
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
                ptype_1=torch.from_numpy(self.ref_molec.pharm_types).to(torch.float32).to(self.device),
                ptype_2=torch.from_numpy(self.fit_molec.pharm_types).to(torch.float32).to(self.device),
                anchors_1=torch.from_numpy(self.ref_molec.pharm_ancs).to(torch.float32).to(self.device),
                anchors_2=torch.from_numpy(self.fit_molec.pharm_ancs).to(torch.float32).to(self.device),
                vectors_1=torch.from_numpy(self.ref_molec.pharm_vecs).to(torch.float32).to(self.device),
                vectors_2=torch.from_numpy(self.fit_molec.pharm_vecs).to(torch.float32).to(self.device),
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
            if 'jax' not in sys.modules or 'jax.numpy' not in sys.modules:
                try:
                    import jax.numpy as jnp
                except ImportError:
                    raise ImportError('jax.numpy and torch is required for this function. Install Jax or just use Torch.')
            import jax.numpy as jnp
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

