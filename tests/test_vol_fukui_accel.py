"""Batched accel parity gates for ``vol_fukui``: numba self-copy = 1.0, batched-numba vs the
per-pair torch reference, and Triton = numba (CUDA-only). A deterministic synthetic Fukui field
is injected via ``Molecule(fukui=...)`` so no xtb binary is needed.
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


def _synthetic_fukui(rd):
    """Deterministic signed per-atom field in with-H order; a function of atom order so copies match."""
    z = np.array([a.GetAtomicNum() for a in rd.GetAtoms()], dtype=np.float32)
    return (0.2 * ((z % 4) - 1.5)).astype(np.float32)


def _mol(smiles):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=0)
    return Molecule(rd, pharm_multi_vector=False, fukui=_synthetic_fukui(rd))


MODE, ATTR = "vol_fukui", "sim_aligned_vol_fukui"


def test_numba_self_copy_is_one():
    """A molecule aligned to a copy of itself scores 1.0 under the numba backend."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(IBU), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = b.align_with_vol_fukui(backend="numba")
    assert np.isclose(float(scores[0]), 1.0, atol=1e-4), \
        f"vol_fukui numba self-copy {scores[0]} != 1.0"


def test_numba_batched_matches_per_pair():
    """Batched-numba lands in the per-pair reference's basin at the shipped seed/step budget."""
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    nr, ns = MODE_SEEDS[MODE], MODE_STEPS[MODE]

    mp = MoleculePair(_mol(IBU), _mol(CAF), do_center=True, device=torch.device("cpu"))
    mp.align_with_vol_fukui(num_repeats=nr, max_num_steps=ns)
    ref = float(getattr(mp, ATTR))

    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = b.align_with_vol_fukui(backend="numba")
    bat = float(scores[0])

    assert abs(ref - bat) < 1e-2, f"vol_fukui: per-pair {ref:.5f} vs batched-numba {bat:.5f}"


@pytest.mark.cuda
@pytest.mark.skipif(not (TORCH and torch.cuda.is_available()), reason="CUDA required")
def test_triton_matches_numba():
    """The Triton batched driver agrees with the numba one on the same pair."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    bn = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cpu"))])
    sc_n, _ = bn.align_with_vol_fukui(backend="numba")
    bt = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cuda"))])
    sc_t, _ = bt.align_with_vol_fukui(backend="triton")
    assert abs(float(sc_n[0]) - float(sc_t[0])) < 1e-2, (
        f"vol_fukui: numba {float(sc_n[0]):.5f} vs triton {float(sc_t[0]):.5f}")
