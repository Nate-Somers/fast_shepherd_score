"""Reference-mode test template.

Copy this to ``tests/test_<mode>.py`` and replace the token ``YOURMODE`` (find/replace)
with your canonical mode id, then fill in the ``# TODO`` markers. The four checks below are
the correctness gate for a reference mode; all must pass before handing off to
``accelerate-scoring-mode``.

This file is a template: it is valid Python (so it byte-compiles) but is not meant to run until
YOURMODE is a real mode. Keep it under ``.claude/skills/`` — do not ship it as a live test.
"""
import numpy as np
import pytest
import torch

# --- imports the real test needs (uncomment once the mode exists) -----------------------------
# from shepherd_score.alignment._torch import (
#     objective_YOURMODE_overlay,
#     optimize_YOURMODE_overlay,
# )
# from shepherd_score.container import Molecule, MoleculePair


MODE = "YOURMODE"          # canonical mode id, e.g. "vol_color"
SEED = 0                    # any fixed seed; the point is determinism


def _make_pair():
    """Build a (ref, fit) MoleculePair from a small test molecule.

    TODO: construct two Molecule objects with whatever inputs your mode reads
    (positions, charges, pharmacophores, ...). For the self-overlap check, `fit`
    is a copy of `ref`.
    """
    pytest.skip("fill in _make_pair for YOURMODE")


# --- Gate 1: self-overlap is exactly 1.000 ----------------------------------------------------
def test_self_overlap_is_one():
    pair = _make_pair()  # ref and fit are the same molecule
    getattr(pair, f"align_with_{MODE}")()
    score = getattr(pair, f"sim_aligned_{MODE}")
    assert np.isclose(float(score), 1.0, atol=1e-4), f"self-overlap {score} != 1.000"


# --- Gate 2: autograd gradient matches finite differences -------------------------------------
def test_autograd_matches_finite_difference():
    """The objective's autograd gradient must match a central finite difference.

    TODO: build the objective inputs, then compare d(objective)/d(se3_params) from
    autograd against a central difference. This proves the eager objective — the oracle
    the accel kernels are validated against — is analytically correct.

    NOTE: run this in float32. `get_SE3_transform` builds the rotation matrix in float32, so
    float64 inputs raise a dtype mismatch inside the transform; float32 finite differences are
    noisy, hence eps=1e-3 and a loose atol=2e-3 (see pitfalls.md).
    """
    pytest.skip("fill in the objective inputs for YOURMODE")
    # se3 = torch.zeros(7, dtype=torch.float32, requires_grad=True)
    # se3.data[0] = 1.0  # identity quat + zero trans
    # val = objective_YOURMODE_overlay(se3, ...)
    # val.backward()
    # ana = se3.grad.clone()
    # num = torch.zeros_like(ana)
    # eps = 1e-3
    # for i in range(se3.numel()):
    #     d = torch.zeros_like(se3.data); d[i] = eps
    #     hi = objective_YOURMODE_overlay((se3.data + d).requires_grad_(False), ...)
    #     lo = objective_YOURMODE_overlay((se3.data - d).requires_grad_(False), ...)
    #     num[i] = (hi - lo) / (2 * eps)
    # assert torch.allclose(ana, num, atol=2e-3)


# --- Gate 3: the optimizer recovers a planted rotation ----------------------------------------
def test_optimizer_recovers_planted_pose():
    """Rotate a copy of a molecule by a known SE(3), then check the optimizer aligns it back
    (score returns to ~1.000). This tests the multi-start optimizer, not just the objective.
    TODO: implement using optimize_YOURMODE_overlay with a fixed num_repeats.
    """
    pytest.skip("fill in the planted-pose recovery for YOURMODE")


# --- Gate 4: determinism ----------------------------------------------------------------------
def test_deterministic_given_seed():
    """Two runs with the same inputs and seed count must give the same score and transform.
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
        atol=0.0,
    )
