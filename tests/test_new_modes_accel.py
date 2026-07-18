"""Batched (numba) accel regression tests for the vol_tversky and esp_field modes.

Covers the fast-path gates that survive without a GPU: self-copy overlap == 1.0 under the
numba backend, and batched-numba vs the per-pair torch reference on a distinct pair (matched
seed/step budget, loose tolerance to absorb the different multi-start seed set). Triton parity
and GPU throughput are validated separately on a CUDA box.
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
    return Molecule(rd)


@pytest.fixture(scope="module")
def mols():
    return _mol(IBU), _mol(CAF)


# per mode: (align kwarg name for repeats, steps) come from MODE_SEEDS/MODE_STEPS
MODES = [
    ("vol_tversky", "sim_aligned_vol_tversky"),
    ("esp_field", "sim_aligned_esp_field"),
]


@pytest.mark.parametrize("mode,attr", MODES)
def test_numba_self_copy_is_one(mode, attr, mols):
    """A molecule aligned to a copy of itself scores 1.0 under the numba backend."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    ibu, _ = mols
    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(IBU), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = getattr(b, f"align_with_{mode}")(backend="numba")
    assert np.isclose(float(scores[0]), 1.0, atol=1e-4), f"{mode} numba self-copy {scores[0]} != 1.0"


@pytest.mark.parametrize("mode,attr", MODES)
def test_numba_batched_matches_per_pair(mode, attr, mols):
    """Batched-numba matches the per-pair torch reference on a distinct pair at the shipped
    (MODE_SEEDS, MODE_STEPS) budget. Loose tolerance: the batched and per-pair paths use
    different multi-start seed sets, so they land in the same basin, not bit-identical."""
    from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    ibu, caf = mols
    nr, ns = MODE_SEEDS[mode], MODE_STEPS[mode]

    mp = MoleculePair(_mol(IBU), _mol(CAF), do_center=True, device=torch.device("cpu"))
    getattr(mp, f"align_with_{mode}")(num_repeats=nr, max_num_steps=ns)
    ref = float(getattr(mp, attr))

    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = getattr(b, f"align_with_{mode}")(backend="numba")
    bat = float(scores[0])

    assert abs(ref - bat) < 1e-2, f"{mode}: per-pair {ref:.5f} vs batched-numba {bat:.5f}"
