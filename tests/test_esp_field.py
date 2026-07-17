"""Reference-mode tests for the ``esp_field`` alignment mode.

``esp_field`` is a Cresset-style electrostatic FIELD-POINT overlay: a weighted combination of
atom-centred Gaussian shape (volume) Tanimoto and the overlap of the molecular ESP *field points*
(signed extrema of the softened electrostatic potential), matched by sign. The field points are a
new, VARIABLE-LENGTH ``Molecule`` feature computed from partial charges + heavy-atom positions
(no surface / open3d).

The four correctness gates: self-overlap == 1.000, a sane non-empty field-point set, autograd ==
finite-difference at a non-identity pose (float32), planted-rotation recovery, and determinism.
"""
import numpy as np
import pytest
import torch

from shepherd_score.alignment._torch import (
    objective_esp_field_overlay,
    optimize_esp_field_overlay,
)
from shepherd_score.alignment.utils.se3 import get_SE3_transform, apply_SE3_transform
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair

MODE = "esp_field"
SEED = 0
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _make_mol():
    m = embed_conformer_from_smiles(IBUPROFEN, MMFF_optimize=True, random_seed=0)
    # No surface: field points come from charges + heavy-atom positions only.
    return Molecule(m)


def _make_pair():
    """A (ref, fit) MoleculePair where fit is a copy of ref (self-overlap)."""
    ref = _make_mol()
    fit = _make_mol()
    return MoleculePair(ref, fit)


# --- Gate 0: the new Molecule feature is sane -------------------------------------------------
def test_field_points_are_reasonable():
    mol = _make_mol()
    pos, sign = mol.get_field_points()
    assert pos.dtype == np.float32 and sign.dtype == np.float32
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert sign.shape == (pos.shape[0],)
    # Non-empty and a sane count (a handful to a few dozen).
    assert 1 <= pos.shape[0] <= 60, f"unexpected field-point count {pos.shape[0]}"
    # Signs are strictly +/-1.
    assert set(np.unique(sign).tolist()).issubset({-1.0, 1.0})
    # Field points sit in the probe shell (1.5-4.5 A from the nearest heavy atom).
    nn = np.sqrt(((pos[:, None, :] - mol.atom_pos[None, :, :]) ** 2).sum(-1)).min(1)
    assert nn.min() >= 1.5 - 1e-4 and nn.max() <= 4.5 + 1e-4


# --- Gate 1: self-overlap is exactly 1.000 ----------------------------------------------------
def test_self_overlap_is_one():
    pair = _make_pair()
    getattr(pair, f"align_with_{MODE}")()
    score = getattr(pair, f"sim_aligned_{MODE}")
    assert np.isclose(float(score), 1.0, atol=1e-4), f"self-overlap {score} != 1.000"


# --- Gate 2: autograd gradient matches finite differences -------------------------------------
def test_autograd_matches_finite_difference():
    mol = _make_mol()
    centers = torch.as_tensor(mol.atom_pos, dtype=torch.float32)
    fp_pos, fp_sign = mol.get_field_points()
    fp_pos = torch.as_tensor(fp_pos, dtype=torch.float32)
    fp_sign = torch.as_tensor(fp_sign, dtype=torch.float32)

    def obj(se3):
        return objective_esp_field_overlay(
            se3, centers, centers, fp_pos, fp_pos, fp_sign, fp_sign,
            alpha=0.81, alpha_field=0.81, lam=0.1, field_weight=0.5,
        )

    # Non-identity pose (small rotation quat + translation) so the gradient is genuinely nonzero.
    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    val = obj(se3)
    val.backward()
    ana = se3.grad.clone()

    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data)
        d[i] = eps
        hi = obj((se3.data + d).clone().requires_grad_(False))
        lo = obj((se3.data - d).clone().requires_grad_(False))
        num[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"autograd {ana} vs FD {num}"


# --- Gate 3: the optimizer recovers a planted rotation ----------------------------------------
def test_optimizer_recovers_planted_pose():
    mol = _make_mol()
    centers = torch.as_tensor(mol.atom_pos, dtype=torch.float32)
    fp_pos, fp_sign = mol.get_field_points()
    fp_pos = torch.as_tensor(fp_pos, dtype=torch.float32)
    fp_sign = torch.as_tensor(fp_sign, dtype=torch.float32)

    # Plant a known SE(3) on the fit (rotate + translate a copy of the molecule).
    planted = torch.tensor([0.7071, 0.7071, 0.0, 0.0, 1.0, -0.5, 0.3], dtype=torch.float32)
    T = get_SE3_transform(planted)
    fit_centers = apply_SE3_transform(centers, T)
    fit_fp_pos = apply_SE3_transform(fp_pos, T)

    _, _, score = optimize_esp_field_overlay(
        ref_centers=centers,
        fit_centers=fit_centers,
        ref_fp_pos=fp_pos,
        fit_fp_pos=fit_fp_pos,
        ref_fp_sign=fp_sign,
        fit_fp_sign=fp_sign,
        num_repeats=50,
        max_num_steps=200,
    )
    assert float(score) > 0.95, f"planted-pose recovery only reached {float(score)}"


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
