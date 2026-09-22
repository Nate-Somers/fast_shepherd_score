"""Reference-mode gates for ``vol_fukui`` (shape + condensed-Fukui field), which rides the
``vol_lipo`` objective with the Fukui field in place of Crippen logP. The real field needs xTB,
so a deterministic synthetic signed field is injected via ``Molecule(fukui=...)``; the overlap
math does not depend on how the field was produced.
"""
import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.alignment._torch import (
    objective_vol_lipo_overlay,
    optimize_vol_lipo_overlay,
)
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.alignment.utils.se3 import get_SE3_transform, apply_SE3_transform


MODE = "vol_fukui"
SEED = 0


def _embed(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0
    return m


def _synthetic_fukui(m):
    """Deterministic signed per-atom field in with-H order; a function of atom order so copies match."""
    z = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=np.float32)
    return (0.2 * ((z % 4) - 1.5)).astype(np.float32)   # values in {-0.3, -0.1, 0.1, 0.3}


def deepcopy_mol(m):
    return Chem.Mol(m)


def _make_pair(smi="c1ccccc1CCO"):
    """A (ref, fit) pair where fit is a copy of ref, each carrying the synthetic Fukui field."""
    m = _embed(smi)
    ref = Molecule(deepcopy_mol(m), fukui=_synthetic_fukui(m))
    fit = Molecule(deepcopy_mol(m), fukui=_synthetic_fukui(m))
    return MoleculePair(ref, fit)


def _fukui_inputs(molec):
    """Extract the (shape_centres, fukui_centres, fukui) float32 tensors a Molecule feeds vol_fukui."""
    centers = torch.as_tensor(molec.atom_pos, dtype=torch.float32)
    fukui_pos = torch.as_tensor(
        np.ascontiguousarray(molec.mol.GetConformer().GetPositions()[molec._nonH_atoms_idx]),
        dtype=torch.float32,
    )
    fukui = torch.as_tensor(molec.get_fukui(no_H=True), dtype=torch.float32)
    return centers, fukui_pos, fukui


# --- Gate 1: self-overlap is exactly 1.000 ----------------------------------------------------
def test_self_overlap_is_one():
    pair = _make_pair()
    getattr(pair, f"align_with_{MODE}")()
    score = getattr(pair, f"sim_aligned_{MODE}")
    assert np.isclose(float(score), 1.0, atol=1e-4), f"self-overlap {score} != 1.000"


# --- Gate 2: autograd gradient matches finite differences -------------------------------------
def test_autograd_matches_finite_difference():
    """Analytic gradient with the Fukui field matches finite differences at a non-identity pose."""
    pair = _make_pair()
    ref_centers, ref_fukui_pos, ref_fukui = _fukui_inputs(pair.ref_molec)
    fit_centers, fit_fukui_pos, fit_fukui = _fukui_inputs(pair.fit_molec)

    def obj(se3):
        return objective_vol_lipo_overlay(
            se3, ref_centers, fit_centers, ref_fukui_pos, fit_fukui_pos,
            ref_fukui, fit_fukui, alpha=0.81, lam=0.1, lipo_weight=0.5,
        )

    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    val = obj(se3)
    val.backward()
    ana = se3.grad.clone()
    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data); d[i] = eps
        hi = obj((se3.data + d).clone().requires_grad_(False))
        lo = obj((se3.data - d).clone().requires_grad_(False))
        num[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"autograd {ana} vs FD {num}"


# --- Gate 3: the optimizer recovers a planted rotation ----------------------------------------
def test_optimizer_recovers_planted_pose():
    pair = _make_pair()
    ref_centers, ref_fukui_pos, ref_fukui = _fukui_inputs(pair.ref_molec)
    fit_centers, fit_fukui_pos, fit_fukui = _fukui_inputs(pair.fit_molec)

    se3 = torch.tensor([0.924, 0.383, 0.0, 0.0, 1.0, -0.5, 0.3], dtype=torch.float32)
    T = get_SE3_transform(se3)
    fit_centers = apply_SE3_transform(fit_centers, T)
    fit_fukui_pos = apply_SE3_transform(fit_fukui_pos, T)

    _, _, score = optimize_vol_lipo_overlay(
        ref_centers=ref_centers, fit_centers=fit_centers,
        ref_lipo_pos=ref_fukui_pos, fit_lipo_pos=fit_fukui_pos,
        ref_lipo=ref_fukui, fit_lipo=fit_fukui,
        alpha=0.81, lam=0.1, lipo_weight=0.5,
        num_repeats=20, max_num_steps=200,
    )
    assert float(score) > 0.95, f"planted-pose recovery only reached {float(score):.3f}"


# --- Gate 4: determinism ----------------------------------------------------------------------
def test_deterministic_given_seed():
    torch.manual_seed(SEED)
    pair_a = _make_pair()
    getattr(pair_a, f"align_with_{MODE}")()
    torch.manual_seed(SEED)
    pair_b = _make_pair()
    getattr(pair_b, f"align_with_{MODE}")()
    assert np.isclose(
        float(getattr(pair_a, f"sim_aligned_{MODE}")),
        float(getattr(pair_b, f"sim_aligned_{MODE}")),
        atol=0.0,
    )
    assert np.array_equal(
        getattr(pair_a, f"transform_{MODE}"),
        getattr(pair_b, f"transform_{MODE}"),
    )


# --- Gate 5: retained-H molecule --------------------------------------------------------------
def test_retained_h_molecule():
    """A retained-H molecule must not desync the heavy Fukui field from its centres."""
    m = Chem.AddHs(Chem.MolFromSmiles("[2H]OC(=O)c1ccccc1"))     # deuterium survives RemoveHs
    params = AllChem.ETKDGv3(); params.randomSeed = 0
    assert AllChem.EmbedMolecule(m, params) == 0
    mol = Molecule(m, fukui=_synthetic_fukui(m))
    assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
        "test premise broken: this molecule no longer retains an H after RemoveHs"

    # the per-atom Fukui slice must match the true-heavy count, not atom_pos
    fukui = mol.get_fukui(no_H=True)
    assert len(fukui) == len(mol._nonH_atoms_idx)
    assert len(fukui) != mol.atom_pos.shape[0]

    # the shape centres (N rows) and Fukui centres (N-1 rows) differ in length; a wrong pairing
    # would raise a broadcast error here
    centers, fukui_pos, fukui_t = _fukui_inputs(mol)
    assert centers.shape[0] != fukui_pos.shape[0]                 # shape N vs fukui N-1
    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1], dtype=torch.float32)
    loss = objective_vol_lipo_overlay(
        se3, centers, centers, fukui_pos, fukui_pos, fukui_t, fukui_t,
        alpha=0.81, lam=0.1, lipo_weight=0.5,
    )
    assert 0.0 <= float(1 - loss) <= 1.0

    # end-to-end on the same deuterated molecule
    m2 = Chem.AddHs(Chem.MolFromSmiles("[2H]OC(=O)c1ccccc1"))
    params2 = AllChem.ETKDGv3(); params2.randomSeed = 2
    assert AllChem.EmbedMolecule(m2, params2) == 0
    mol2 = Molecule(m2, fukui=_synthetic_fukui(m2))
    assert mol2.atom_pos.shape[0] != len(mol2._nonH_atoms_idx)   # still retains the D
    pair = MoleculePair(mol2, Molecule(Chem.Mol(m2), fukui=_synthetic_fukui(m2)))
    pair.align_with_vol_fukui(num_repeats=10, max_num_steps=100)
    assert np.isclose(float(pair.sim_aligned_vol_fukui), 1.0, atol=1e-4)
