"""Batched (numba) accel regression tests for the vol_tversky and vol_lipo modes.

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
    # pharm_multi_vector=False builds pharmacophores (needed by the pharm/colour SI modes;
    # harmless for the shape/ESP/lipophilicity modes, which ignore them).
    return Molecule(rd, pharm_multi_vector=False)


@pytest.fixture(scope="module")
def mols():
    return _mol(IBU), _mol(CAF)


# per mode: (align kwarg name for repeats, steps) come from MODE_SEEDS/MODE_STEPS
MODES = [
    ("vol_tversky", "sim_aligned_vol_tversky"),
    ("vol_lipo", "sim_aligned_vol_lipo"),
    ("vol_esp_tversky", "sim_aligned_vol_esp_tversky"),
    # SI experimental modes that need NO molecular surface (validatable on a CPU-only box).
    # The surface modes (surf_tversky, surf_esp_tversky, vol_and_surf_esp_tversky) require Open3D
    # and are validated on the GPU cluster.
    ("vol_mr", "sim_aligned_vol_mr"),
    ("vol_lipo_tversky", "sim_aligned_vol_lipo_tversky"),
    ("vol_color_tversky", "sim_aligned_vol_color_tversky"),
    ("vol_atomtype", "sim_aligned_vol_atomtype"),
    ("vol_pharm", "sim_aligned_vol_pharm"),
    ("pharm_tversky", "sim_aligned_pharm_tversky"),
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


@pytest.mark.cuda
@pytest.mark.skipif(not (TORCH and torch.cuda.is_available()), reason="CUDA required")
@pytest.mark.parametrize("mode,attr", MODES)
def test_triton_matches_numba(mode, attr, mols):
    """Gate 2 (Triton == numba): the GPU (Triton) batched driver agrees with the CPU (numba)
    one on the same pair. These modes REUSE the shape + ESP kernels, whose Triton twins are
    already parity-validated; this end-to-end check confirms the driver's blend dispatches
    device-consistently. CPU-only boxes skip it (the rest of the gates still run)."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    bn = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cpu"))])
    sc_n, _ = getattr(bn, f"align_with_{mode}")(backend="numba")
    bt = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cuda"))])
    sc_t, _ = getattr(bt, f"align_with_{mode}")(backend="triton")
    assert abs(float(sc_n[0]) - float(sc_t[0])) < 1e-2, (
        f"{mode}: numba {float(sc_n[0]):.5f} vs triton {float(sc_t[0]):.5f}")
