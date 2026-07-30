"""Batched accel parity gates for the ``vol_fukui`` mode (shape + condensed-Fukui reactivity field).

Mirrors ``test_new_modes_accel.py`` (gate 1: numba self-copy == 1.0; gate 3: batched-numba vs the
per-pair torch reference; gate 2: Triton == numba, CUDA-only) but INJECTS a deterministic synthetic
Fukui field via ``Molecule(fukui=...)`` -- the real ``f+ - f-`` field needs the xtb binary (three
gfn2-xTB single points), and the *accel* correctness is independent of how the field was produced.
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
    """Deterministic SIGNED per-atom field (full with-H order) standing in for the xTB Fukui dual
    descriptor; a pure function of atom order so a molecule and its copy get the same field."""
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
    """Gate 1/4: a molecule aligned to a copy of itself scores 1.0 under the numba backend."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    b = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(IBU), do_center=True,
                                        device=torch.device("cpu"))])
    scores, _ = b.align_with_vol_fukui(backend="numba")
    assert np.isclose(float(scores[0]), 1.0, atol=1e-4), \
        f"vol_fukui numba self-copy {scores[0]} != 1.0"


def test_numba_batched_matches_per_pair():
    """Gate 3: batched-numba matches the per-pair torch reference on a distinct pair at the shipped
    (MODE_SEEDS, MODE_STEPS) budget (loose tol: different multi-start seed sets, same basin)."""
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
    """Gate 2 (Triton == numba): reuses the already-parity-validated shape + ESP kernels, so this
    end-to-end check confirms the vol_lipo-driver blend dispatches device-consistently for the
    Fukui field. CPU-only boxes skip it."""
    from shepherd_score.container import MoleculePair, MoleculePairBatch
    bn = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cpu"))])
    sc_n, _ = bn.align_with_vol_fukui(backend="numba")
    bt = MoleculePairBatch([MoleculePair(_mol(IBU), _mol(CAF), do_center=True,
                                         device=torch.device("cuda"))])
    sc_t, _ = bt.align_with_vol_fukui(backend="triton")
    assert abs(float(sc_n[0]) - float(sc_t[0])) < 1e-2, (
        f"vol_fukui: numba {float(sc_n[0]):.5f} vs triton {float(sc_t[0]):.5f}")
