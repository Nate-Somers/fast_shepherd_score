"""
Molecule class to hold molecule geometries and extract interaction profiles.
MoleculePair class facilitates alignment with interaction profiles.
"""
from typing import Union, List, Optional, Tuple
from copy import deepcopy
import sys
import os

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
from shepherd_score.alignment.utils.se3_np import apply_SE3_transform_np, apply_SO3_transform_np
from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _self_overlap_in_chunks
from shepherd_score.alignment.utils.fast_common import batched_seeds_torch
from shepherd_score.alignment.utils.se3 import quaternion_to_SE3

### BEGIN size_bucketing #####################################################
# Every heavy-atom count 3‒150 is mapped to a “band” of 8 atoms
# (   1-8, 9-16, 17-24, … ).  Pairs that fall in the same band
# share a common padded tensor size → one GPU launch.
_BAND = 16                     # change to 16/32 if you want larger bands

def _band_key(n: int) -> int:
    "return the *upper* bound of the 8-atom band this n falls into"
    return ((n + _BAND - 1) // _BAND) * _BAND
### END size_bucketing #######################################################

# ---- persistent, per-process caches (reused across calls) -------------------
_ALIGN_WORKSPACES: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
_INT_BUFFER_CACHE: dict[int, dict[str, torch.Tensor]] = {}

# Measured fine-loop footprint (bytes per pair) keyed by (mode, N_pad, M_pad,
# num_seeds). Lets the sub-batcher size each bucket's chunk to the GPU.
_PAIR_FOOTPRINT_BYTES: dict[tuple, int] = {}
# Set env SUBBATCH_DEBUG=1 to print the chosen chunk size per bucket.
_SUBBATCH_DEBUG = bool(os.environ.get("SUBBATCH_DEBUG"))


# --- multi-GPU dispatch ------------------------------------------------------
import threading as _threading

_DISPATCH_LOCAL = _threading.local()


def _dev_idx(device: torch.device) -> int:
    """Cache-key component so per-device workspaces/buffers never collide under
    the multi-GPU dispatcher. Constant 0 on a single GPU -> no behaviour change."""
    return device.index if (device.type == "cuda" and device.index is not None) else -1


def _should_distribute(pairs) -> bool:
    """True when `pairs` should be sharded across multiple CUDA devices."""
    if getattr(_DISPATCH_LOCAL, "active", False):
        return False                       # already inside a per-device shard
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return False
    if not pairs or pairs[0].device.type != "cuda":
        return False
    return len(pairs) >= 2 * torch.cuda.device_count()


def _run_distributed(align_fn, pairs, **kwargs):
    """Shard `pairs` across ALL visible CUDA devices and run `align_fn` on each in
    parallel (CUDA ops release the GIL); results are written in-place to the pairs.
    The per-device workspace/footprint caches are device-keyed so concurrent
    shards never collide. NOTE: the single-GPU path (via _should_distribute) is
    validated; the multi-GPU concurrency path needs multi-GPU hardware to benchmark."""
    ndev = torch.cuda.device_count()
    shards = [sh for sh in (pairs[i::ndev] for i in range(ndev)) if sh]
    errs = {}

    def _worker(dev_idx, shard):
        _DISPATCH_LOCAL.active = True
        try:
            with torch.cuda.device(dev_idx):
                dev = torch.device("cuda", dev_idx)
                for p in shard:
                    p.device = dev          # align_fn moves this shard's tensors to `dev`
                align_fn(shard, **kwargs)
        except Exception as e:              # noqa: BLE001 - re-raised after join
            import traceback
            errs[dev_idx] = (repr(e), traceback.format_exc())
        finally:
            _DISPATCH_LOCAL.active = False

    threads = [_threading.Thread(target=_worker, args=(k, sh), daemon=True)
               for k, sh in enumerate(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise RuntimeError(f"multi-GPU align failed on device(s) {list(errs)}: {errs}")


def _subbatched_align(process, K: int, *, key: tuple,
                      safety: float = 0.7, init_cap: int = 1024):
    """Drive ``process(start, count) -> (scores, q, t)`` over ``K`` independent
    pairs in GPU-memory-safe sub-batches and concatenate the per-pair results.

    Because pairs are independent (each result is its own max over seeds),
    chunking + concatenation is *exactly equivalent* to one big call -- it only
    bounds peak memory, so it never changes a score.

    Sizing is dynamic and per-bucket: bytes-per-pair is measured from the fine
    loop's peak allocation and cached per ``key=(mode, N_pad, M_pad, num_seeds)``
    (so a band-112 / pharm bucket -- whose footprint grows ~quadratically with
    pad size -- gets a much smaller chunk than a cheap band-32 surf bucket). Each
    chunk is sized so its peak stays under ``safety`` x (free device memory +
    torch's reusable cache). A previously-unseen shape starts at ``init_cap``
    pairs, then grows once calibrated (only chunks at least a quarter of the
    target size update the footprint, so a tiny trailing remainder cannot inflate
    it); an OOM halves the chunk and retries. Off CUDA (or if a single pair won't
    fit) it just calls ``process`` once.
    """
    if not torch.cuda.is_available():
        return process(0, K)

    key = (torch.cuda.current_device(),) + tuple(key)   # device-scope the footprint cache

    def _budget() -> float:
        free, _ = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return safety * (free + max(0, reusable))

    fp = _PAIR_FOOTPRINT_BYTES.get(key)
    need_resize = fp is None
    K_sub = max(1, min(K, int(_budget() // fp))) if fp else min(K, init_cap)
    if _SUBBATCH_DEBUG:
        print(f"[subbatch] key={key} K={K} init_fp={fp} K_sub0={K_sub} "
              f"free={torch.cuda.mem_get_info()[0]//(1024*1024)}MiB", flush=True)

    sc_parts, q_parts, t_parts = [], [], []
    s = 0
    _nchunks = 0; _noom = 0; _ks = []                            # diag (SUBBATCH_DEBUG)
    while s < K:
        k = min(K_sub, K - s)
        try:
            torch.cuda.reset_peak_memory_stats()
            sc, q, t = process(s, k)
            peak = int(torch.cuda.max_memory_allocated())
            # Fold a chunk into the per-pair footprint only when it is large enough
            # that the fixed workspace overhead (seed/autotune scratch -- tens of MB,
            # independent of k) is amortised. peak/k = fixed/k + per_pair, so a tiny
            # trailing remainder (e.g. k=7) yields a wildly inflated bytes/pair that
            # max() would lock in, collapsing every later chunk to a fraction of its
            # right size (pharm was observed going 2 -> 16 -> 82 chunks this way). The
            # first chunk has k == K_sub so it always qualifies; calibration is never
            # starved.
            if k >= max(1, K_sub // 4):
                fp_meas = max(1, -(-peak // k))                  # ceil bytes/pair
                _PAIR_FOOTPRINT_BYTES[key] = max(_PAIR_FOOTPRINT_BYTES.get(key, 0), fp_meas)
            sc_parts.append(sc); q_parts.append(q); t_parts.append(t)
            s += k
            _nchunks += 1
            if _SUBBATCH_DEBUG:
                _ks.append(k)
            if need_resize:   # first success -> we now know the real footprint
                fp = _PAIR_FOOTPRINT_BYTES[key]
                remaining = K - s
                if remaining > 0:
                    K_sub = max(1, min(remaining, int(_budget() // fp)))
                need_resize = False
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # Some OOMs surface as a plain RuntimeError; only treat those as OOM.
            if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                    and "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            _noom += 1
            if _SUBBATCH_DEBUG:
                print(f"[subbatch] OOM at k={k} (after {_nchunks} ok) "
                      f"free={torch.cuda.mem_get_info()[0]//(1024*1024)}MiB -> K_sub={max(1, k // 2)}",
                      flush=True)
            if k <= 1:
                raise
            K_sub = max(1, k // 2)
    if _SUBBATCH_DEBUG:
        print(f"[subbatch] DONE key={key} K={K} nchunks={_nchunks} noom={_noom} "
              f"ks={_ks} final_fp={_PAIR_FOOTPRINT_BYTES.get(key)}", flush=True)
    return torch.cat(sc_parts), torch.cat(q_parts), torch.cat(t_parts)

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
                 pharm_vecs: Optional[np.ndarray] = None
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
        """
        self.mol = mol
        self.atom_pos = Chem.RemoveHs(mol).GetConformer().GetPositions()
        if surface_points is None:
            self.num_surf_points = num_surf_points
        else:
            self.num_surf_points = len(surface_points)
        self.density = density

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
        if isinstance(pharm_types, np.ndarray) and isinstance(pharm_ancs, np.ndarray) and isinstance(pharm_vecs, np.ndarray):
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = pharm_types, pharm_ancs, pharm_vecs
        else:
            self.pharm_types, self.pharm_ancs, self.pharm_vecs = None, None, None
            if self.pharm_multi_vector is not None:
                self.get_pharmacophore(
                    multi_vector=self.pharm_multi_vector,
                    exclude=[],
                    check_access=False,
                    scale=1.
                )

        self._shape_cache: dict[float, float] = {}

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
        if use_density:
            positions = get_molecular_surface_const_density(centers,
                                                            self.radii,
                                                            self.density,
                                                            probe_radius=self.probe_radius,
                                                            num_samples_per_atom=25)
        else:
            positions = get_molecular_surface(centers,
                                              self.radii,
                                              num_points=self.num_surf_points,
                                              probe_radius=self.probe_radius,
                                              num_samples_per_atom = 25)
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
                          scale: float = 1):
        """ Get the pharmacophores of the molecule. """
        self.pharm_types, self.pharm_ancs, self.pharm_vecs = get_pharmacophores(
            self.mol,
            multi_vector=multi_vector,
            exclude=exclude,
            check_access=check_access,
            scale=scale
        )


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

        self.transform_vol = np.eye(4)
        self.sim_aligned_vol = None

        self.transform_vol_noH = np.eye(4)
        self.sim_aligned_vol_noH = None

        self.transform_surf = np.eye(4)
        self.sim_aligned_surf = None

        self.transform_esp = np.eye(4)
        self.sim_aligned_esp = None

        self.transform_vol_esp = np.eye(4)
        self.sim_aligned_vol_esp = None

        self.transform_vol_esp_noH = np.eye(4)
        self.sim_aligned_vol_esp_noH = None

        self.transform_esp_combo = np.eye(4)
        self.sim_aligned_esp_combo = None

        self.transform_pharm = np.eye(4)
        self.sim_aligned_pharm = None

    @staticmethod
    def align_batch_vol(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
        """
        Batched alignment with workspace reuse & reduced per-pair transfers.
        """

        global _ALIGN_WORKSPACES, _INT_BUFFER_CACHE
        
        if not pairs:
            return
        if _should_distribute(pairs):
            return _run_distributed(MoleculePair.align_batch_vol, pairs,
                                    alpha=alpha, steps_fine=steps_fine)

        device = pairs[0].device
        # --- move coords once (skip if already there & right dtype) -------------
        for p in pairs:
            rx = p._ref_xyz_t
            fx = p._fit_xyz_t
            if rx.device != device:
                p._ref_xyz_t = rx.to(device, non_blocking=True)
            if fx.device != device:
                p._fit_xyz_t = fx.to(device, non_blocking=True)

        # --- result accumulators (GPU first; host copy only once) ---------------
        all_pairs: list[MoleculePair] = []
        all_scores: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []

        # --- band bucketing -----------------------------------------------------
        buckets: dict[tuple[int, int], list[MoleculePair]] = {}
        for p in pairs:
            n_band = _band_key(p._ref_xyz_t.shape[0])
            m_band = _band_key(p._fit_xyz_t.shape[0])
            buckets.setdefault((n_band, m_band), []).append(p)

        for (N_pad, M_pad), bucket in buckets.items():
            K = len(bucket)

            # ---- integer buffers (reuse) ---------------------------------------
            ib_key = (_dev_idx(device), K)
            int_buf = _INT_BUFFER_CACHE.get(ib_key)
            if int_buf is None:
                int_buf = {
                    'N': torch.empty(K, dtype=torch.int32, device=device),
                    'M': torch.empty(K, dtype=torch.int32, device=device),
                }
                _INT_BUFFER_CACHE[ib_key] = int_buf
            N_real = int_buf['N']
            M_real = int_buf['M']

            # Fill in-place (no new tensor creation)
            for i, p in enumerate(bucket):
                N_real[i] = p._ref_xyz_t.shape[0]
                M_real[i] = p._fit_xyz_t.shape[0]

            # ---- workspaces (reuse & grow) -------------------------------------
            ws_key = (_dev_idx(device), N_pad, M_pad)
            ws = _ALIGN_WORKSPACES.get(ws_key)
            if ws is None or ws['ref'].shape[0] < K:
                # allocate at least K; allow some headroom (optional)
                ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
                fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
                _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
            ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
            fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

            # We only write the valid prefix; no need to .zero_ entire array
            # but we do clear the padding slices for deterministic results.
            ref_pad.zero_()
            fit_pad.zero_()
            # Batch .item() calls to reduce GPU→CPU sync overhead
            n_list = N_real.tolist()
            m_list = M_real.tolist()
            for i, (p, n, m) in enumerate(zip(bucket, n_list, m_list)):
                ref_pad[i, :n] = p._ref_xyz_t
                fit_pad[i, :m] = p._fit_xyz_t

            # ---- self-overlaps (reused kernel) ---------------------------------
            VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
            VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

            # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
            # memory-pressured chunking never re-pays the launch-bound seed-gen.
            seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=50)

            # ---- coarse + fine alignment, in GPU-memory-safe sub-batches -------
            def _proc(_s, _k):
                sl = slice(_s, _s + _k)
                return coarse_fine_align_many(
                    ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                    N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                    seeds=(seeds_q[sl], seeds_t[sl]))
            scores, q_batch, t_batch = _subbatched_align(
                _proc, K, key=("vol", N_pad, M_pad, 50))

            all_pairs.extend(bucket)
            all_scores.append(scores)
            all_q.append(q_batch)
            all_t.append(t_batch)

        # ---- final host transfer (single) --------------------------------------
        scores_cpu = torch.cat(all_scores).cpu()
        q_cpu = torch.cat(all_q).cpu()
        t_cpu = torch.cat(all_t).cpu()

        for p, s, q, t in zip(all_pairs, scores_cpu, q_cpu, t_cpu):
            # quaternion_to_SE3 unchanged
            p.transform_vol_noH = quaternion_to_SE3(q, t)
            p.sim_aligned_vol_noH = float(s)

    @staticmethod
    def align_batch_surf(pairs: list["MoleculePair"], *, alpha: float = 0.81, steps_fine: int = 100):
        """
        Batched alignment over *surface point clouds* using Gaussian-overlap
        surface similarity (ROCS-style), modeled after `align_batch_vol`.

        Inputs
        ------
        pairs : list[MoleculePair]
            Each pair must provide surface point clouds for reference/fit:
            • prefer:   _ref_surf_t, _fit_surf_t  (torch.float32, (N/M, 3))
            • fallback: ref_molec.surf_pos, fit_molec.surf_pos (numpy, (N/M, 3))
        alpha : float
            Gaussian width parameter (same meaning as in `align_with_surf`).

        Side effects
        ------------
        Writes:
        • p.transform_surf      ← best SE(3) as 4×4 (via quaternion_to_SE3)
        • p.sim_aligned_surf    ← best Tanimoto surface score (float)
        """

        # Reuse the persistent, per-process workspace/int-buffer caches (same
        # ref/fit scratch-buffer layout as align_batch_vol). The previous local
        # re-declarations here shadowed the module globals, so the surf path
        # never reused workspaces across calls. Buffers are zeroed before use,
        # so cross-call (and cross-modality) reuse is safe.
        global _ALIGN_WORKSPACES, _INT_BUFFER_CACHE

        if not pairs:
            return
        if _should_distribute(pairs):
            return _run_distributed(MoleculePair.align_batch_surf, pairs,
                                    alpha=alpha, steps_fine=steps_fine)

        device = pairs[0].device

        # --- ensure/prepare surface tensors on the right device --------------------
        for p in pairs:
            # Prefer already-prepared torch tensors
            r = getattr(p, "_ref_surf_t", None)
            f = getattr(p, "_fit_surf_t", None)

            if r is None or f is None:
                # Fallback: build from numpy surface arrays if present
                if not hasattr(p, "ref_molec") or not hasattr(p.ref_molec, "surf_pos"):
                    raise ValueError(
                        "Surface points missing: MoleculePair must have _ref/_fit_surf_t "
                        "or ref_molec/fit_molec with .surf_pos."
                    )
                r_np = p.ref_molec.surf_pos
                f_np = p.fit_molec.surf_pos
                if r_np is None or f_np is None:
                    raise ValueError("Surface points are None; cannot run align_batch_surf.")
                p._ref_surf_t = torch.as_tensor(r_np, dtype=torch.float32, device=device)
                p._fit_surf_t = torch.as_tensor(f_np, dtype=torch.float32, device=device)
            else:
                # move to target device if needed
                if r.device != device:
                    p._ref_surf_t = r.to(device, non_blocking=True)
                if f.device != device:
                    p._fit_surf_t = f.to(device, non_blocking=True)

        # --- result accumulators (GPU first; host copy only once) ------------------
        all_pairs: list[MoleculePair] = []
        all_scores: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []

        # --- band bucketing (by padded N/M) ---------------------------------------
        buckets: dict[tuple[int, int], list[MoleculePair]] = {}
        for p in pairs:
            n_band = _band_key(p._ref_surf_t.shape[0])
            m_band = _band_key(p._fit_surf_t.shape[0])
            buckets.setdefault((n_band, m_band), []).append(p)

        for (N_pad, M_pad), bucket in buckets.items():
            K = len(bucket)

            # ---- integer buffers (reuse) ------------------------------------------
            ib_key = (_dev_idx(device), K)
            int_buf = _INT_BUFFER_CACHE.get(ib_key)
            if int_buf is None:
                int_buf = {
                    'N': torch.empty(K, dtype=torch.int32, device=device),
                    'M': torch.empty(K, dtype=torch.int32, device=device),
                }
                _INT_BUFFER_CACHE[ib_key] = int_buf
            N_real = int_buf['N']
            M_real = int_buf['M']

            # Fill in-place
            for i, p in enumerate(bucket):
                N_real[i] = p._ref_surf_t.shape[0]
                M_real[i] = p._fit_surf_t.shape[0]

            # ---- workspaces (reuse & grow) ----------------------------------------
            ws_key = (_dev_idx(device), N_pad, M_pad)
            ws = _ALIGN_WORKSPACES.get(ws_key)
            if ws is None or ws['ref'].shape[0] < K:
                ref_pad = torch.empty(K, N_pad, 3, device=device, dtype=torch.float32)
                fit_pad = torch.empty(K, M_pad, 3, device=device, dtype=torch.float32)
                _ALIGN_WORKSPACES[ws_key] = {'ref': ref_pad, 'fit': fit_pad}
            ref_pad = _ALIGN_WORKSPACES[ws_key]['ref'][:K]
            fit_pad = _ALIGN_WORKSPACES[ws_key]['fit'][:K]

            # Clear padding slices for determinism; write valid prefixes
            ref_pad.zero_()
            fit_pad.zero_()
            # Batch .item() calls to reduce GPU→CPU sync overhead
            n_list = N_real.tolist()
            m_list = M_real.tolist()
            for i, (p, n, m) in enumerate(zip(bucket, n_list, m_list)):
                ref_pad[i, :n] = p._ref_surf_t
                fit_pad[i, :m] = p._fit_surf_t

            # ---- self-overlaps on surface point clouds ----------------------------
            VAA = _self_overlap_in_chunks(ref_pad, N_real, alpha)
            VBB = _self_overlap_in_chunks(fit_pad, M_real, alpha)

            # ---- seeds ONCE per band (hoisted out of the sub-batch loop) so
            # memory-pressured chunking never re-pays the launch-bound seed-gen.
            seeds_q, seeds_t = batched_seeds_torch(ref_pad, fit_pad, N_real, M_real, num_seeds=50)

            # ---- coarse + fine alignment (same engine as volumetric), processed in
            # GPU-memory-safe sub-batches sized per bucket (pairs are independent)
            def _proc(_s, _k):
                sl = slice(_s, _s + _k)
                return coarse_fine_align_many(
                    ref_pad[sl], fit_pad[sl], VAA[sl], VBB[sl],
                    N_real=N_real[sl], M_real=M_real[sl], alpha=alpha, steps_fine=steps_fine,
                    seeds=(seeds_q[sl], seeds_t[sl]))
            scores, q_batch, t_batch = _subbatched_align(
                _proc, K, key=("surf", N_pad, M_pad, 50))

            all_pairs.extend(bucket)
            all_scores.append(scores)
            all_q.append(q_batch)
            all_t.append(t_batch)

        # ---- final host transfer (single) -----------------------------------------
        scores_cpu = torch.cat(all_scores).cpu()
        q_cpu = torch.cat(all_q).cpu()
        t_cpu = torch.cat(all_t).cpu()

        for p, s, q, t in zip(all_pairs, scores_cpu, q_cpu, t_cpu):
            p.transform_surf = quaternion_to_SE3(q, t)
            p.sim_aligned_surf = float(s)

    @staticmethod
    def align_batch_esp(
        pairs: list["MoleculePair"],
        *,
        alpha: float,
        lam: float,
        trans_init: bool = False,
        num_repeats: int = 50,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
    ) -> None:
        """
        Batched ESP alignment using the fused ESP Triton kernel.

        Side effects
        ------------
        Writes:
        - p.transform_esp
        - p.sim_aligned_esp
        """
        if not pairs:
            return
        if _should_distribute(pairs):
            return _run_distributed(MoleculePair.align_batch_esp, pairs,
                                    alpha=alpha, lam=lam, trans_init=trans_init,
                                    num_repeats=num_repeats,
                                    num_repeats_per_trans=num_repeats_per_trans,
                                    topk=topk, steps_fine=steps_fine, lr=lr)

        from shepherd_score.score.constants import LAM_SCALING
        from shepherd_score.alignment.utils.fast_esp_se3 import fast_optimize_ROCS_esp_overlay_batch

        device = pairs[0].device
        lam_scaled = LAM_SCALING * lam

        # Ensure surface tensors (+ ESP) exist on device
        for p in pairs:
            r = getattr(p, "_ref_surf_t", None)
            f = getattr(p, "_fit_surf_t", None)
            rc = getattr(p, "_ref_surf_esp_t", None)
            fc = getattr(p, "_fit_surf_esp_t", None)

            if r is None or f is None or rc is None or fc is None:
                if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
                    raise ValueError("Surface points are None; cannot run align_batch_esp.")
                if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
                    raise ValueError("Surface ESP is None; cannot run align_batch_esp.")

                p._ref_surf_t = torch.as_tensor(p.ref_molec.surf_pos, dtype=torch.float32, device=device)
                p._fit_surf_t = torch.as_tensor(p.fit_molec.surf_pos, dtype=torch.float32, device=device)
                p._ref_surf_esp_t = torch.as_tensor(p.ref_molec.surf_esp, dtype=torch.float32, device=device)
                p._fit_surf_esp_t = torch.as_tensor(p.fit_molec.surf_esp, dtype=torch.float32, device=device)
            else:
                if r.device != device:
                    p._ref_surf_t = r.to(device, non_blocking=True)
                if f.device != device:
                    p._fit_surf_t = f.to(device, non_blocking=True)
                if rc.device != device:
                    p._ref_surf_esp_t = rc.to(device, non_blocking=True)
                if fc.device != device:
                    p._fit_surf_esp_t = fc.to(device, non_blocking=True)

            # Translation centers must be available on device for trans_init.
            if trans_init and getattr(p, "_ref_xyz_t", None) is None:
                p._ref_xyz_t = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=device)

        all_pairs: list[MoleculePair] = []
        all_scores: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []

        # Bucket by padded surface sizes; for translation-seeded mode, also bucket by exact
        # number of translation centers (legacy uses 10*P + 5 seeds).
        buckets: dict[tuple[int, int, int], list[MoleculePair]] = {}
        for p in pairs:
            n_band = _band_key(p._ref_surf_t.shape[0])
            m_band = _band_key(p._fit_surf_t.shape[0])
            tc = int(p._ref_xyz_t.shape[0]) if trans_init else 0
            buckets.setdefault((n_band, m_band, tc), []).append(p)

        # Workspace caches keyed by (N_pad, M_pad, K)
        workspaces: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        int_buffers: dict[int, dict[str, torch.Tensor]] = {}

        for (N_pad, M_pad, tc), bucket in buckets.items():
            K = len(bucket)

            ib = int_buffers.get(K)
            if ib is None:
                ib = {
                    "N": torch.empty(K, dtype=torch.int32, device=device),
                    "M": torch.empty(K, dtype=torch.int32, device=device),
                }
                int_buffers[K] = ib
            N_real = ib["N"]
            M_real = ib["M"]

            for i, p in enumerate(bucket):
                N_real[i] = p._ref_surf_t.shape[0]
                M_real[i] = p._fit_surf_t.shape[0]

            ws_key = (N_pad, M_pad, K)
            ws = workspaces.get(ws_key)
            if ws is None:
                ws = {
                    "ref": torch.empty(K, N_pad, 3, device=device, dtype=torch.float32),
                    "fit": torch.empty(K, M_pad, 3, device=device, dtype=torch.float32),
                    "ref_c": torch.empty(K, N_pad, device=device, dtype=torch.float32),
                    "fit_c": torch.empty(K, M_pad, device=device, dtype=torch.float32),
                }
                workspaces[ws_key] = ws

            ref_pad = ws["ref"]
            fit_pad = ws["fit"]
            ref_c_pad = ws["ref_c"]
            fit_c_pad = ws["fit_c"]

            ref_pad.zero_()
            fit_pad.zero_()
            ref_c_pad.zero_()
            fit_c_pad.zero_()
            # Batch .item() calls to reduce GPU→CPU sync overhead
            n_list = N_real.tolist()
            m_list = M_real.tolist()
            for i, (p, n, m) in enumerate(zip(bucket, n_list, m_list)):
                ref_pad[i, :n] = p._ref_surf_t
                fit_pad[i, :m] = p._fit_surf_t
                ref_c_pad[i, :n] = p._ref_surf_esp_t
                fit_c_pad[i, :m] = p._fit_surf_esp_t

            trans_centers_batch = None
            trans_centers_real = None
            if trans_init:
                # NOTE: this bucket key uses exact translation center count (tc), so
                # the legacy seed count is identical for all pairs in this bucket.
                trans_centers_batch = torch.empty(K, tc, 3, device=device, dtype=torch.float32)
                for i, p in enumerate(bucket):
                    trans_centers_batch[i] = p._ref_xyz_t
                trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

            # NOTE: seed-gen hoist intentionally NOT applied to esp. Unlike surf/vol
            # (where it cleanly helps under memory pressure), for esp's heavier
            # per-chunk footprint the held full-band seeds shaved enough free memory
            # to tip the sub-batcher into OOM-retry thrash under pressure (esp-same
            # large-batch went 1912 -> 273 mol/s). Clean-process esp is already fast;
            # the per-cell subprocess benchmark removes the pressure entirely.
            def _proc(_s, _k):
                sl = slice(_s, _s + _k)
                tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
                tcr = trans_centers_real[sl] if trans_centers_real is not None else None
                _, q, t, sc = fast_optimize_ROCS_esp_overlay_batch(
                    ref_pad[sl], fit_pad[sl], ref_c_pad[sl], fit_c_pad[sl],
                    alpha=alpha, lam=lam_scaled,
                    N_real=N_real[sl], M_real=M_real[sl],
                    trans_centers_batch=tcb, trans_centers_real=tcr,
                    num_repeats_per_trans=num_repeats_per_trans,
                    topk=topk, steps_fine=steps_fine, lr=lr,
                )
                return sc, q, t
            scores, q_batch, t_batch = _subbatched_align(
                _proc, K, key=("esp", N_pad, M_pad, 50))

            all_pairs.extend(bucket)
            all_scores.append(scores)
            all_q.append(q_batch)
            all_t.append(t_batch)

        scores_cpu = torch.cat(all_scores).cpu()
        q_cpu = torch.cat(all_q).cpu()
        t_cpu = torch.cat(all_t).cpu()

        for p, s, q, t in zip(all_pairs, scores_cpu, q_cpu, t_cpu):
            p.transform_esp = quaternion_to_SE3(q, t)
            p.sim_aligned_esp = float(s)

    @staticmethod
    def align_batch_esp_combo(
        pairs: list["MoleculePair"],
        *,
        alpha: float,
        lam: float = 0.001,
        probe_radius: float = 1.0,
        esp_weight: float = 0.5,
        trans_init: bool = False,
        num_repeats: int = 50,
        num_repeats_per_trans: int = 10,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
    ) -> None:
        """
        Batched ESP-combo alignment (ShaEP-style) with padding-safe masks.

        Side effects
        ------------
        Writes:
        - p.transform_esp_combo
        - p.sim_aligned_esp_combo
        """
        if not pairs:
            return
        if _should_distribute(pairs):
            return _run_distributed(MoleculePair.align_batch_esp_combo, pairs,
                                    alpha=alpha, lam=lam, probe_radius=probe_radius,
                                    esp_weight=esp_weight, trans_init=trans_init,
                                    num_repeats=num_repeats,
                                    num_repeats_per_trans=num_repeats_per_trans,
                                    topk=topk, steps_fine=steps_fine, lr=lr)

        from shepherd_score.alignment.utils.fast_esp_combo_se3 import fast_optimize_esp_combo_score_overlay_batch

        device = pairs[0].device

        # Ensure required tensors exist on device
        for p in pairs:
            if p.ref_molec.surf_pos is None or p.fit_molec.surf_pos is None:
                raise ValueError("Surface points are None; cannot run align_batch_esp_combo.")
            if p.ref_molec.surf_esp is None or p.fit_molec.surf_esp is None:
                raise ValueError("Surface ESP is None; cannot run align_batch_esp_combo.")

            if getattr(p, "_ref_surf_t", None) is None:
                p._ref_surf_t = torch.as_tensor(p.ref_molec.surf_pos, dtype=torch.float32, device=device)
            if getattr(p, "_fit_surf_t", None) is None:
                p._fit_surf_t = torch.as_tensor(p.fit_molec.surf_pos, dtype=torch.float32, device=device)
            if getattr(p, "_ref_surf_esp_t", None) is None:
                p._ref_surf_esp_t = torch.as_tensor(p.ref_molec.surf_esp, dtype=torch.float32, device=device)
            if getattr(p, "_fit_surf_esp_t", None) is None:
                p._fit_surf_esp_t = torch.as_tensor(p.fit_molec.surf_esp, dtype=torch.float32, device=device)

            if getattr(p, "_ref_centers_w_H_t", None) is None:
                p._ref_centers_w_H_t = torch.as_tensor(
                    p.ref_molec.mol.GetConformer().GetPositions(), dtype=torch.float32, device=device
                )
            if getattr(p, "_fit_centers_w_H_t", None) is None:
                p._fit_centers_w_H_t = torch.as_tensor(
                    p.fit_molec.mol.GetConformer().GetPositions(), dtype=torch.float32, device=device
                )
            if getattr(p, "_ref_partial_t", None) is None:
                p._ref_partial_t = torch.as_tensor(p.ref_molec.partial_charges, dtype=torch.float32, device=device)
            if getattr(p, "_fit_partial_t", None) is None:
                p._fit_partial_t = torch.as_tensor(p.fit_molec.partial_charges, dtype=torch.float32, device=device)
            if getattr(p, "_ref_radii_t", None) is None:
                p._ref_radii_t = torch.as_tensor(p.ref_molec.radii, dtype=torch.float32, device=device)
            if getattr(p, "_fit_radii_t", None) is None:
                p._fit_radii_t = torch.as_tensor(p.fit_molec.radii, dtype=torch.float32, device=device)

            # Translation centers must be available on device for trans_init.
            if trans_init and getattr(p, "_ref_xyz_t", None) is None:
                p._ref_xyz_t = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=device)
            if trans_init and getattr(p, "_fit_xyz_t", None) is None:
                p._fit_xyz_t = torch.as_tensor(p.fit_molec.atom_pos, dtype=torch.float32, device=device)

        all_pairs: list[MoleculePair] = []
        all_scores: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []

        buckets: dict[tuple[int, int, int, int, int, int, int], list[MoleculePair]] = {}
        for p in pairs:
            n_surf_band = _band_key(p._ref_surf_t.shape[0])
            m_surf_band = _band_key(p._fit_surf_t.shape[0])
            n_wH_band = _band_key(p._ref_centers_w_H_t.shape[0])
            m_wH_band = _band_key(p._fit_centers_w_H_t.shape[0])

            # Shape "centers" use volumetric atoms when alpha==0.81, else surface points.
            if alpha == 0.81:
                n_cent_band = _band_key(p._ref_xyz_t.shape[0])
                m_cent_band = _band_key(p._fit_xyz_t.shape[0])
            else:
                n_cent_band = n_surf_band
                m_cent_band = m_surf_band

            tc = int(p._ref_xyz_t.shape[0]) if trans_init else 0
            buckets.setdefault((n_wH_band, m_wH_band, n_cent_band, m_cent_band, n_surf_band, m_surf_band, tc), []).append(p)

        for (n_wH_pad, m_wH_pad, n_cent_pad, m_cent_pad, n_surf_pad, m_surf_pad, tc), bucket in buckets.items():
            K = len(bucket)

            # Allocate padded blocks
            centers_w_H_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
            centers_w_H_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)
            partial_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
            partial_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)
            radii_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
            radii_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)

            centers_1 = torch.zeros(K, n_cent_pad, 3, device=device, dtype=torch.float32)
            centers_2 = torch.zeros(K, m_cent_pad, 3, device=device, dtype=torch.float32)

            points_1 = torch.zeros(K, n_surf_pad, 3, device=device, dtype=torch.float32)
            points_2 = torch.zeros(K, m_surf_pad, 3, device=device, dtype=torch.float32)
            point_charges_1 = torch.zeros(K, n_surf_pad, device=device, dtype=torch.float32)
            point_charges_2 = torch.zeros(K, m_surf_pad, device=device, dtype=torch.float32)

            N_real_atoms_w_H_1 = torch.empty(K, device=device, dtype=torch.int32)
            M_real_atoms_w_H_2 = torch.empty(K, device=device, dtype=torch.int32)
            N_real_centers = torch.empty(K, device=device, dtype=torch.int32)
            M_real_centers = torch.empty(K, device=device, dtype=torch.int32)
            N_real_surf_1 = torch.empty(K, device=device, dtype=torch.int32)
            M_real_surf_2 = torch.empty(K, device=device, dtype=torch.int32)

            for i, p in enumerate(bucket):
                n_wH = p._ref_centers_w_H_t.shape[0]
                m_wH = p._fit_centers_w_H_t.shape[0]
                n_surf = p._ref_surf_t.shape[0]
                m_surf = p._fit_surf_t.shape[0]
                N_real_atoms_w_H_1[i] = n_wH
                M_real_atoms_w_H_2[i] = m_wH
                N_real_surf_1[i] = n_surf
                M_real_surf_2[i] = m_surf

                centers_w_H_1[i, :n_wH] = p._ref_centers_w_H_t
                centers_w_H_2[i, :m_wH] = p._fit_centers_w_H_t
                partial_1[i, :n_wH] = p._ref_partial_t
                partial_2[i, :m_wH] = p._fit_partial_t
                radii_1[i, :n_wH] = p._ref_radii_t
                radii_2[i, :m_wH] = p._fit_radii_t

                points_1[i, :n_surf] = p._ref_surf_t
                points_2[i, :m_surf] = p._fit_surf_t
                point_charges_1[i, :n_surf] = p._ref_surf_esp_t
                point_charges_2[i, :m_surf] = p._fit_surf_esp_t

                if alpha == 0.81:
                    n_cent = p._ref_xyz_t.shape[0]
                    m_cent = p._fit_xyz_t.shape[0]
                    centers_1[i, :n_cent] = p._ref_xyz_t
                    centers_2[i, :m_cent] = p._fit_xyz_t
                    N_real_centers[i] = n_cent
                    M_real_centers[i] = m_cent
                else:
                    centers_1[i, :n_surf] = p._ref_surf_t
                    centers_2[i, :m_surf] = p._fit_surf_t
                    N_real_centers[i] = n_surf
                    M_real_centers[i] = m_surf

            trans_centers_batch = None
            trans_centers_real = None
            if trans_init:
                trans_centers_batch = torch.empty(K, tc, 3, device=device, dtype=torch.float32)
                for i, p in enumerate(bucket):
                    trans_centers_batch[i] = p._ref_xyz_t
                trans_centers_real = torch.full((K,), tc, device=device, dtype=torch.int32)

            _, q_batch, t_batch, scores = fast_optimize_esp_combo_score_overlay_batch(
                centers_w_H_1,
                centers_w_H_2,
                centers_1,
                centers_2,
                points_1,
                points_2,
                partial_1,
                partial_2,
                point_charges_1,
                point_charges_2,
                radii_1,
                radii_2,
                alpha,
                lam=lam,
                probe_radius=probe_radius,
                esp_weight=esp_weight,
                N_real_atoms_w_H_1=N_real_atoms_w_H_1,
                M_real_atoms_w_H_2=M_real_atoms_w_H_2,
                N_real_centers=N_real_centers,
                M_real_centers=M_real_centers,
                N_real_surf_1=N_real_surf_1,
                M_real_surf_2=M_real_surf_2,
                trans_centers_batch=trans_centers_batch,
                trans_centers_real=trans_centers_real,
                num_repeats_per_trans=num_repeats_per_trans,
                topk=topk,
                steps_fine=steps_fine,
                lr=lr,
            )

            all_pairs.extend(bucket)
            all_scores.append(scores)
            all_q.append(q_batch)
            all_t.append(t_batch)

        scores_cpu = torch.cat(all_scores).cpu()
        q_cpu = torch.cat(all_q).cpu()
        t_cpu = torch.cat(all_t).cpu()

        for p, s, q, t in zip(all_pairs, scores_cpu, q_cpu, t_cpu):
                p.transform_esp_combo = quaternion_to_SE3(q, t)
                p.sim_aligned_esp_combo = float(s)

    @staticmethod
    def align_batch_pharm(
        pairs: list["MoleculePair"],
        *,
        similarity: _SIM_TYPE = "tanimoto",
        extended_points: bool = False,
        only_extended: bool = False,
        trans_init: bool = False,
        num_repeats: int = 50,
        topk: int = 30,
        steps_fine: int = 100,
        lr: float = 0.075,
    ):
        """
        Batched pharmacophore alignment using the fast GPU pathway when available.

        Writes per-pair:
        - p.transform_pharm (4x4)
        - p.sim_aligned_pharm (float)
        """
        if not pairs:
            return
        if _should_distribute(pairs):
            return _run_distributed(MoleculePair.align_batch_pharm, pairs,
                                    similarity=similarity, extended_points=extended_points,
                                    only_extended=only_extended, trans_init=trans_init,
                                    num_repeats=num_repeats, topk=topk,
                                    steps_fine=steps_fine, lr=lr)

        if not torch.cuda.is_available():
            # Slow fallback: per-pair legacy optimizer
            for p in pairs:
                p.align_with_pharm(
                    similarity=similarity,
                    extended_points=extended_points,
                    only_extended=only_extended,
                    trans_init=trans_init,
                    num_repeats=num_repeats,
                    lr=lr,
                    max_num_steps=steps_fine,
                    use_jax=False,
                    verbose=False,
                )
            return

        try:
            from shepherd_score.alignment.utils.fast_pharm_se3 import fast_optimize_pharm_overlay_batch
        except ImportError:
            fast_optimize_pharm_overlay_batch = None

        if fast_optimize_pharm_overlay_batch is None:
            for p in pairs:
                p.align_with_pharm(
                    similarity=similarity,
                    extended_points=extended_points,
                    only_extended=only_extended,
                    trans_init=trans_init,
                    num_repeats=num_repeats,
                    lr=lr,
                    max_num_steps=steps_fine,
                    use_jax=False,
                    verbose=False,
                )
            return

        device = pairs[0].device

        # Ensure per-pair cached tensors exist on device.
        for p in pairs:
            if getattr(p, "_ref_pharm_types_t", None) is None:
                p._ref_pharm_types_t = torch.as_tensor(p.ref_molec.pharm_types, dtype=torch.int64, device=device)
            if getattr(p, "_fit_pharm_types_t", None) is None:
                p._fit_pharm_types_t = torch.as_tensor(p.fit_molec.pharm_types, dtype=torch.int64, device=device)
            if getattr(p, "_ref_pharm_ancs_t", None) is None:
                p._ref_pharm_ancs_t = torch.as_tensor(p.ref_molec.pharm_ancs, dtype=torch.float32, device=device)
            if getattr(p, "_fit_pharm_ancs_t", None) is None:
                p._fit_pharm_ancs_t = torch.as_tensor(p.fit_molec.pharm_ancs, dtype=torch.float32, device=device)
            if getattr(p, "_ref_pharm_vecs_t", None) is None:
                p._ref_pharm_vecs_t = torch.as_tensor(p.ref_molec.pharm_vecs, dtype=torch.float32, device=device)
            if getattr(p, "_fit_pharm_vecs_t", None) is None:
                p._fit_pharm_vecs_t = torch.as_tensor(p.fit_molec.pharm_vecs, dtype=torch.float32, device=device)

        all_pairs: list[MoleculePair] = []
        all_scores: list[torch.Tensor] = []
        all_q: list[torch.Tensor] = []
        all_t: list[torch.Tensor] = []

        buckets: dict[tuple[int, int, int], list[MoleculePair]] = {}
        for p in pairs:
            n_band = _band_key(p._ref_pharm_ancs_t.shape[0])
            m_band = _band_key(p._fit_pharm_ancs_t.shape[0])
            tc = int(p._ref_pharm_ancs_t.shape[0]) if trans_init else 0
            buckets.setdefault((n_band, m_band, tc), []).append(p)

        for (N_pad, M_pad, tc), bucket in buckets.items():
            K = len(bucket)

            ref_types = torch.zeros(K, N_pad, device=device, dtype=torch.int64)
            fit_types = torch.zeros(K, M_pad, device=device, dtype=torch.int64)
            ref_ancs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_ancs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)
            ref_vecs = torch.zeros(K, N_pad, 3, device=device, dtype=torch.float32)
            fit_vecs = torch.zeros(K, M_pad, 3, device=device, dtype=torch.float32)

            N_real = torch.empty(K, device=device, dtype=torch.int32)
            M_real = torch.empty(K, device=device, dtype=torch.int32)

            for i, p in enumerate(bucket):
                n = p._ref_pharm_ancs_t.shape[0]
                m = p._fit_pharm_ancs_t.shape[0]
                N_real[i] = n
                M_real[i] = m

                ref_types[i, :n] = p._ref_pharm_types_t
                fit_types[i, :m] = p._fit_pharm_types_t
                ref_ancs[i, :n] = p._ref_pharm_ancs_t
                fit_ancs[i, :m] = p._fit_pharm_ancs_t
                ref_vecs[i, :n] = p._ref_pharm_vecs_t
                fit_vecs[i, :m] = p._fit_pharm_vecs_t

            trans_centers_batch = ref_ancs if trans_init else None
            trans_centers_real = N_real if trans_init else None

            # GPU-memory-safe sub-batching per bucket (independent pairs). Pharm's
            # analytical fine loop has the largest (~N_pad*M_pad) footprint, so
            # this is where the dynamic cap matters most.
            def _proc(_s, _k):
                sl = slice(_s, _s + _k)
                tcb = trans_centers_batch[sl] if trans_centers_batch is not None else None
                tcr = trans_centers_real[sl] if trans_centers_real is not None else None
                _, _, q, t, sc = fast_optimize_pharm_overlay_batch(
                    ref_types[sl], fit_types[sl], ref_ancs[sl], fit_ancs[sl],
                    ref_vecs[sl], fit_vecs[sl],
                    similarity=similarity, extended_points=extended_points,
                    only_extended=only_extended, num_repeats=num_repeats,
                    trans_centers_batch=tcb, trans_centers_real=tcr,
                    num_repeats_per_trans=10, N_real=N_real[sl], M_real=M_real[sl],
                    topk=topk, steps_fine=steps_fine, lr=lr,
                )
                return sc, q, t
            scores, q_batch, t_batch = _subbatched_align(
                _proc, K, key=("pharm", N_pad, M_pad, num_repeats))

            all_pairs.extend(bucket)
            all_scores.append(scores)
            all_q.append(q_batch)
            all_t.append(t_batch)

        scores_cpu = torch.cat(all_scores).cpu()
        q_cpu = torch.cat(all_q).cpu()
        t_cpu = torch.cat(all_t).cpu()

        for p, s, q, t in zip(all_pairs, scores_cpu, q_cpu, t_cpu):
            p.transform_pharm = quaternion_to_SE3(q, t)
            p.sim_aligned_pharm = float(s)

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


    def align_with_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = 50,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = 200,
                       use_jax: bool = False,
                       use_analytical: bool = True,
                       verbose: bool = False) -> np.ndarray:
        """
        Align fit_molec to ref_molec using ESP+surface similarity.
        ``lam`` is scaled by ``(1e4/(4*55.263*np.pi))**2`` for correct units.

        Typically, ``lam=0.3`` is used and is scaled internally.

        Optimally aligned score found in ``self.sim_aligned_esp`` and the optimal SE(3)
        transformation is at ``self.transform_esp``.

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
            self.transform_esp = np.array(se3_transform)
            self.sim_aligned_esp = np.array(score)
            return np.array(aligned_fit_points)
        else: # Use Torch implementation (fast path on CUDA if available)
            if torch.cuda.is_available():
                try:
                    from shepherd_score.alignment.utils.fast_esp_se3 import fast_optimize_ROCS_esp_overlay
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

                    self.transform_esp = se3_transform_t.numpy()
                    self.sim_aligned_esp = score_t.numpy()
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

            self.transform_esp = se3_transform.numpy()
            self.sim_aligned_esp = score.numpy()
            return aligned_fit_points.numpy()


    def align_with_esp_combo(self,
                             alpha: float,
                             lam: float = 0.001,
                             probe_radius: float = 1.0,
                             esp_weight: float = 0.5,
                             num_repeats: int = 50,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = 200,
                             use_jax: bool = False,
                             verbose: bool = False):
        """
        Align using ShaEP similarity score.
        If alpha is 0.81, then it automatically uses volumetric shape similarity.
        Otherwise, it uses surface shape similarity.

        Optimally aligned score found in ``self.sim_aligned_esp_combo`` and the optimal SE(3)
        transformation is at ``self.transform_esp_combo``.

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
            self.transform_esp_combo = np.array(se3_transform)
            self.sim_aligned_esp_combo = np.array(score)
            return np.array(aligned_fit_points)
        else:
            if alpha == 0.81:
                ref_centers = torch.from_numpy(self.ref_molec.atom_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.atom_pos).to(torch.float32).to(self.device)
            else:
                ref_centers = torch.from_numpy(self.ref_molec.surf_pos).to(torch.float32).to(self.device)
                fit_centers = torch.from_numpy(self.fit_molec.surf_pos).to(torch.float32).to(self.device)

            if torch.cuda.is_available():
                try:
                    from shepherd_score.alignment.utils.fast_esp_combo_se3 import fast_optimize_esp_combo_score_overlay
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

                    self.transform_esp_combo = se3_transform_t.numpy()
                    self.sim_aligned_esp_combo = score_t.numpy()
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
            self.transform_esp_combo = se3_transform.numpy()
            self.sim_aligned_esp_combo = score.numpy()
            return aligned_fit_points.detach().numpy()


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

        # PyTorch (fast path on CUDA if available)
        if torch.cuda.is_available():
            try:
                from shepherd_score.alignment.utils.fast_pharm_se3 import fast_optimize_pharm_overlay
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
            import numpy as np
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

