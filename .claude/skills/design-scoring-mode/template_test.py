"""Reference-mode test template.

Copy this to ``tests/test_<mode>.py`` and replace the token ``YOURMODE`` (find/replace) with your
canonical mode id, then fill in the ``# TODO`` markers.

Gates 1-4 are the correctness gate for every reference mode. Gate 5 (retained-H) is required only
if the mode reads a per-atom field; gate 6 (asymmetry) only if the mode is a Tversky reduction.
Delete the ones that do not apply. All applicable gates must pass before handing off to
``accelerate-scoring-mode``.

Working models in the tree, in order of usefulness:
  * ``tests/test_vol_fukui.py``  -- per-atom field, reuses another mode's optimizer, injects a
    synthetic field so the test needs no xtb binary. The closest thing to this template, filled in.
  * ``tests/test_vol_lipo.py``   -- per-atom field computed by the library itself.
  * ``tests/test_vol_esp_tversky.py`` -- carries both gate 5 and gate 6.
  * ``tests/test_si_modes.py``   -- nine modes sharing one parametrized harness; copy this shape if
    you are adding several related modes rather than one.

Note the repo's gate numbering is not perfectly consistent between files (``test_vol_tversky.py``
numbers asymmetry as gate 5). Use the names, not the numbers.

This file is a template: it is valid Python (so it byte-compiles) but is not meant to run until
YOURMODE is a real mode. Keep it under ``.claude/skills/`` -- do not ship it as a live test.
"""
import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

# --- imports the real test needs (uncomment once the mode exists) -----------------------------
# from shepherd_score.alignment._torch import (
#     objective_YOURMODE_overlay,
#     optimize_YOURMODE_overlay,
# )
# from shepherd_score.container import Molecule, MoleculePair
# from shepherd_score.alignment.utils.se3 import get_SE3_transform, apply_SE3_transform


MODE = "YOURMODE"           # canonical mode id, e.g. "vol_color"
SEED = 0                    # any fixed seed; the point is determinism


def _embed(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0
    return m


def _make_pair(smi="c1ccccc1CCO"):
    """Build a (ref, fit) MoleculePair where fit is a copy of ref, so the self-overlap gates work.

    TODO: construct two Molecule objects carrying whatever your mode reads. Use ``Chem.Mol(m)`` to
    copy rather than deepcopy. If your mode reads a field that needs an external binary (xtb), pass
    a deterministic SYNTHETIC field through the constructor instead -- the correctness of the
    overlap math is independent of how the field was produced, and the test then runs everywhere.
    """
    pytest.fail("fill in _make_pair for YOURMODE")


def _inputs(molec):
    """Extract the float32 tensors a Molecule feeds YOURMODE.

    TODO: shape centres come from ``molec.atom_pos`` (or ``molec.surf_pos`` for a surface mode).
    A per-atom field's centres come from its ``get_<feature>_positions()`` accessor -- NEVER from
    ``atom_pos``, which is the RemoveHs set and retains isotope-labelled H. See pitfalls.md.
    """
    centers = torch.as_tensor(molec.atom_pos, dtype=torch.float32)
    # field_pos = torch.as_tensor(np.ascontiguousarray(molec.get_YOURFEATURE_positions()),
    #                             dtype=torch.float32)
    # field = torch.as_tensor(molec.get_YOURFEATURE(no_H=True), dtype=torch.float32)
    return centers


# --- Gate 1: self-overlap is exactly 1.000 ----------------------------------------------------
def test_self_overlap_is_one():
    # Use a generous explicit budget; this optimizer check complements a fixed-pose
    # normalization test. Adapt the expected self-score for non-Tanimoto objectives.
    pair = _make_pair()                       # ref and fit are the same molecule
    getattr(pair, f"align_with_{MODE}")(num_repeats=20, max_num_steps=200)
    score = getattr(pair, f"sim_aligned_{MODE}")
    assert np.isclose(float(score), 1.0, atol=1e-4), f"self-overlap {score} != 1.000"


# --- Gate 2: autograd gradient matches central finite differences ------------------------------
def test_autograd_matches_finite_difference():
    """Checks the reference derivative numerically at the chosen inputs and pose.

    Two harness gotchas, both in pitfalls.md, neither a mode bug:
      * float32. ``get_SE3_transform`` builds the rotation in float32, so float64 inputs raise a
        dtype mismatch. float32 finite differences are noisy, hence eps=1e-3 / atol=2e-3.
      * a NON-identity pose. A self-overlap objective at the identity sits at its optimum where the
        gradient is ~0 in every direction, so the comparison passes vacuously.
    """
    pytest.fail("fill in the objective inputs for YOURMODE")
    # pair = _make_pair()
    #
    # def obj(se3):
    #     return objective_YOURMODE_overlay(se3, ...)     # TODO: ref/fit inputs, alpha, weights
    #
    # se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
    #                    dtype=torch.float32, requires_grad=True)
    # val = obj(se3)
    # val.backward()
    # ana = se3.grad.clone()
    # num = torch.zeros_like(ana)
    # eps = 1e-3
    # for i in range(se3.numel()):
    #     d = torch.zeros_like(se3.data); d[i] = eps
    #     hi = obj((se3.data + d).clone().requires_grad_(False))
    #     lo = obj((se3.data - d).clone().requires_grad_(False))
    #     num[i] = (hi - lo) / (2 * eps)
    # assert torch.allclose(ana, num, atol=2e-3), f"autograd {ana} vs FD {num}"


# --- Gate 3: the optimizer recovers a planted rotation ----------------------------------------
def test_optimizer_recovers_planted_pose():
    """Rotate a copy of a molecule by a known SE(3), then check the optimizer aligns it back.

    This exercises the multi-start optimizer rather than just the objective, and complements
    direct checks of feature-to-atom assignment; a unit self-score alone is insufficient.
    Transform every coordinate channel by the same SE(3), not just the shape centres.
    """
    pytest.fail("fill in the planted-pose recovery for YOURMODE")
    # pair = _make_pair()
    # se3 = torch.tensor([0.924, 0.383, 0.0, 0.0, 1.0, -0.5, 0.3], dtype=torch.float32)
    # T = get_SE3_transform(se3)
    # fit_centers = apply_SE3_transform(fit_centers, T)
    # # ... rotate/translate other fit positions; rotate vectors WITHOUT translation
    # _, _, score = optimize_YOURMODE_overlay(..., num_repeats=20, max_num_steps=200)
    # assert float(score) > 0.95, f"planted-pose recovery only reached {float(score):.3f}"


# --- Gate 4: determinism ----------------------------------------------------------------------
def test_deterministic_given_seed():
    """Two runs with the same inputs must give the same score AND the same transform.
    A non-deterministic reference cannot serve as the accel skill's parity oracle.
    """
    torch.manual_seed(SEED)
    pair_a = _make_pair()
    getattr(pair_a, f"align_with_{MODE}")()
    torch.manual_seed(SEED)
    pair_b = _make_pair()
    getattr(pair_b, f"align_with_{MODE}")()
    assert np.isclose(
        float(getattr(pair_a, f"sim_aligned_{MODE}")),
        float(getattr(pair_b, f"sim_aligned_{MODE}")),
        atol=0.0, rtol=0.0,
    )
    assert np.array_equal(
        getattr(pair_a, f"transform_{MODE}"),
        getattr(pair_b, f"transform_{MODE}"),
    )


# --- Gate 5: retained-H molecule -- ONLY if your mode reads a per-atom field -------------------
def test_retained_h_molecule():
    """``Chem.RemoveHs`` RETAINS isotope-labelled hydrogen, so on a deuterated molecule
    ``atom_pos`` is longer than the strict-heavy set ``_nonH_atoms_idx`` selects -- which is what
    every per-atom field is sliced to. Pairing the two broadcasts (N) against (N-1) and crashes.

    Field centres must come from ``get_<feature>_positions()``. This gate proves they do. Every
    plain heavy-atom molecule sails through gates 1-4 with the bug present, so this is the only
    gate that sees it. Delete it if your mode reads no per-atom field.
    """
    m = _embed("[2H]OC(=O)c1ccccc1")          # deuterium survives RemoveHs
    # mol = Molecule(m)                        # TODO: pass whatever your mode reads
    # assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
    #     "test premise broken: this molecule no longer retains an H after RemoveHs"
    #
    # # the field slice matches the true-heavy count, not atom_pos
    # field = mol.get_YOURFEATURE(no_H=True)
    # assert len(field) == len(mol._nonH_atoms_idx)
    # assert len(field) != mol.atom_pos.shape[0]
    #
    # # exercise the full overlap path at a NON-identity pose: a wrong pairing raises here
    # # ... then end to end, a self-pair on the same deuterated molecule still scores 1.000
    pytest.fail("fill in the per-atom-data call for YOURMODE, or delete this gate if N/A")


# --- Gate 6: asymmetry -- ONLY if your mode is a Tversky reduction -----------------------------
def test_tversky_is_asymmetric():
    """With ta != tb the objective is deliberately not symmetric: O(a,b) != O(b,a). That asymmetry
    IS the mode, so assert it rather than tolerating it. Gaussian-overlap Tversky can exceed one, whereas pharmacophore Tversky
    clamps at one. Test the range and clamp of the actual objective. Delete this gate if your mode is Tanimoto.
    """
    pytest.fail("fill in the asymmetry check for YOURMODE, or delete this gate if N/A")
    # pair_ab = MoleculePair(mol_a, mol_b); pair_ba = MoleculePair(mol_b, mol_a)
    # getattr(pair_ab, f"align_with_{MODE}")(); getattr(pair_ba, f"align_with_{MODE}")()
    # assert not np.isclose(float(getattr(pair_ab, f"sim_aligned_{MODE}")),
    #                       float(getattr(pair_ba, f"sim_aligned_{MODE}")), atol=1e-3)
