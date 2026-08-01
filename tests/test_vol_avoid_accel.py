"""Accel parity gates for ``vol_avoid`` (shape Tanimoto MINUS a linear hard-sphere excluded-volume
penalty against a FIXED avoid-point cloud).

Unlike the other modes, vol_avoid takes a THIRD input (``avoid_points``) that is neither molecule, so
it is pairwise-only (no screen wiring). Gates here:
  * self-copy with the avoid cloud far away  -> pure shape self-overlap == 1.000 (gate 4);
  * batched-numba == per-pair ACCEL (same seeds) -> the true "batched == per-pair" gate (bit-level);
  * batched-numba ~= the per-pair AUTOGRAD reference for a realistic MILD avoid (basin tol 1e-2)
    -- the reference and accel use DIFFERENT multi-start seed sets, and the penalty landscape is
    piecewise-linear, so a strong/degenerate avoid cloud (e.g. one covering the whole molecule) can
    land the two seed sets in different near-degenerate optima; that is seed coverage, not a kernel
    error (the kernel itself is validated bit-for-bit by the value+FD-grad gate).
  * Triton == numba (CUDA-only; the new hard-sphere kernel's GPU twin).
"""
import warnings
import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")

IBU = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAF = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _mol(smiles):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=0)
    return Molecule(rd, pharm_multi_vector=False)


def _far_cloud(m):
    """A single avoid point 100 A away -- never intrudes, so the penalty is 0."""
    return (np.asarray(m.atom_pos).mean(0) + 100.0).astype(np.float32)[None, :]


def _mild_cloud():
    """A small fixed 'wall' ~5 A to one side: a realistic, mild excluded region that nudges the pose
    without dominating, so the reference and accel agree tightly."""
    return np.array([[5.0, 0.0, 0.0], [5.0, 2.0, 0.0], [5.0, -2.0, 0.0]], dtype=np.float32)


def test_self_copy_far_avoid_is_one():
    """Gate 4: a molecule aligned to a copy of itself with the avoid cloud far away scores 1.0
    (the penalty term is 0, so only the shape self-overlap remains)."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    m = _mol(IBU)
    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(IBU), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = b.align_with_vol_avoid(_far_cloud(m), backend="numba")
    assert np.isclose(float(scores[0]), 1.0, atol=1e-2), \
        f"vol_avoid self-copy (far avoid) {scores[0]} != 1.0"


@pytest.mark.parametrize("cloud_name", ["far", "mild", "strong"])
def test_numba_batched_matches_per_pair_accel(cloud_name):
    """Gate 3 (batched == per-pair): the batched-numba path equals the per-pair accel entry point
    (fast_optimize_vol_avoid_overlay) to convergence -- both use the SAME structured seed set, so
    this is bit-level even on a strong/degenerate avoid cloud."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    from shepherd_score.accel.drivers.avoid import fast_optimize_vol_avoid_overlay
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    nr, ns = MODE_SEEDS["vol_avoid"], MODE_STEPS["vol_avoid"]

    ref_m, fit_m = _mol(IBU), _mol(CAF)
    if cloud_name == "far":
        avoid = _far_cloud(ref_m)
    elif cloud_name == "mild":
        avoid = _mild_cloud()
    else:
        avoid = np.asarray(ref_m.atom_pos, dtype=np.float32)      # covers the whole reference shape

    mp = MoleculePair(_mol(IBU), _mol(CAF), do_center=True, device=torch.device("cpu"))
    rc = torch.tensor(mp.ref_molec.atom_pos, dtype=torch.float32)
    fc = torch.tensor(mp.fit_molec.atom_pos, dtype=torch.float32)
    _, _, sc_pp = fast_optimize_vol_avoid_overlay(
        rc, fc, torch.tensor(avoid, dtype=torch.float32),
        avoid_weight=1.0, avoid_min_dist=2.0, num_repeats=nr, steps_fine=ns, lr=0.1)

    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                        device=torch.device("cpu"))])
    sc_b, _ = b.align_with_vol_avoid(avoid, backend="numba")
    assert abs(float(sc_pp) - float(sc_b[0])) < 1e-3, \
        f"vol_avoid [{cloud_name}]: per-pair-accel {float(sc_pp):.5f} vs batched {float(sc_b[0]):.5f}"


def test_numba_batched_matches_reference_mild():
    """The batched-numba accel finds the per-pair AUTOGRAD reference's optimum for a realistic MILD
    avoid cloud (basin tolerance 1e-2; different seed sets, same optimum)."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    nr, ns = MODE_SEEDS["vol_avoid"], MODE_STEPS["vol_avoid"]
    avoid = _mild_cloud()

    mp = MoleculePair(_mol(IBU), _mol(CAF), do_center=True, device=torch.device("cpu"))
    mp.align_with_vol_avoid(avoid, avoid_weight=1.0, avoid_min_dist=2.0,
                            num_repeats=nr, max_num_steps=ns)
    ref = float(mp.sim_aligned_vol_avoid)

    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                        device=torch.device("cpu"))])
    sc_b, _ = b.align_with_vol_avoid(avoid, backend="numba")
    assert abs(ref - float(sc_b[0])) < 1e-2, \
        f"vol_avoid mild: per-pair reference {ref:.5f} vs batched {float(sc_b[0]):.5f}"


def test_avoid_penalty_lowers_score():
    """Sanity: a strong avoid cloud (covering the reference shape) drives the best score well below
    the no-penalty (far avoid) score -- the penalty actually steers the pose."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    ref_m = _mol(IBU)
    far = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                          device=torch.device("cpu"))])
    s_far, _ = far.align_with_vol_avoid(_far_cloud(ref_m), backend="numba")
    strong = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                             device=torch.device("cpu"))])
    s_strong, _ = strong.align_with_vol_avoid(np.asarray(ref_m.atom_pos, dtype=np.float32),
                                              backend="numba")
    assert float(s_strong[0]) < float(s_far[0]) - 0.05, \
        f"avoid penalty did not lower score: strong {s_strong[0]} vs far {s_far[0]}"


@pytest.mark.cuda
@pytest.mark.skipif(not (TORCH and torch.cuda.is_available()), reason="CUDA required")
def test_triton_matches_numba():
    """Gate 2 (Triton == numba): the new hard-sphere avoid kernel's GPU twin agrees with the numba
    kernel end-to-end. CPU-only boxes skip it."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    avoid = _mild_cloud()
    bn = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cpu"))])
    sc_n, _ = bn.align_with_vol_avoid(avoid, backend="numba")
    bt = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cuda"))])
    sc_t, _ = bt.align_with_vol_avoid(avoid, backend="triton")
    assert abs(float(sc_n[0]) - float(sc_t[0])) < 1e-2, \
        f"vol_avoid: numba {float(sc_n[0]):.5f} vs triton {float(sc_t[0]):.5f}"
