"""Reference-mode tests for the asymmetric "fits-inside" ``vol_esp_tversky`` overlay.

``vol_esp_tversky`` is the ``vol_esp`` electrostatic-weighted volumetric overlap scored with a
**Tversky** reduction instead of Tanimoto -- exactly what ``vol_tversky`` is to ``vol``.

Correctness gates (per the design-scoring-mode skill):
  * self-overlap == 1.000 (Tversky(A,A) = 1 for any weights),
  * autograd gradient matches central finite differences at a NON-identity pose (float32),
  * the multi-start optimizer recovers a planted rotation (score -> ~1.0),
  * determinism given a fixed seed,
  * retained-H molecule (deuterium survives ``Chem.RemoveHs``): heavy centres come from the
    with-H conformer indexed by ``_nonH_atoms_idx`` (1:1 with the heavy charges), NOT ``atom_pos``
    -- REQUIRED because this mode reads per-atom partial charges,
  * Tversky asymmetry: a small query contained in a bigger molecule scores HIGHER when the small
    molecule is the REFERENCE (query -> big fit) than in the reverse direction.
"""
import warnings

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - skipping vol_esp_tversky tests")

if TORCH_AVAILABLE:
    from shepherd_score.alignment._torch import (
        objective_vol_esp_tversky_overlay,
        optimize_vol_esp_tversky_overlay,
    )
    from shepherd_score.container import Molecule, MoleculePair

MODE = "vol_esp_tversky"
SEED = 0

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")


def _embed(smiles):
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    try:
        mol = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=0)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"could not embed {smiles!r}: {e}")
    if mol is None:
        pytest.skip(f"embedding returned None for {smiles!r}")
    return mol


@pytest.fixture(scope="module")
def ibuprofen():
    return _embed("CC(C)Cc1ccc(cc1)C(C)C(=O)O")


def _make_pair(ibuprofen):
    """A (ref, fit) MoleculePair where fit is a copy of ref (for the self-overlap gate)."""
    ref = Molecule(ibuprofen)
    fit = Molecule(ibuprofen)
    return MoleculePair(ref, fit, do_center=True, device=torch.device('cpu'))


def _heavy_pos_charges(ibuprofen):
    """Strict-heavy centres (with-H conformer indexed by _nonH_atoms_idx) + heavy charges,
    the exact inputs the mode consumes -- centred to the heavy COM."""
    m = Molecule(ibuprofen)
    pos = m.mol.GetConformer().GetPositions()[m._nonH_atoms_idx].astype(np.float32)
    pos = pos - pos.mean(0)
    chg = np.asarray(m.get_charges(no_H=True), dtype=np.float32)
    return pos, chg


# --- Gate 1: self-overlap is exactly 1.000 ----------------------------------------------------
def test_self_overlap_is_one(ibuprofen):
    pair = _make_pair(ibuprofen)
    pair.align_with_vol_esp_tversky(num_repeats=20, max_num_steps=200)
    score = float(pair.sim_aligned_vol_esp_tversky)
    assert np.isclose(score, 1.0, atol=1e-4), f"self-overlap {score} != 1.000"


# --- Gate 2: autograd gradient matches finite differences (float32, non-identity pose) --------
def test_autograd_matches_finite_difference(ibuprofen):
    pos, chg = _heavy_pos_charges(ibuprofen)
    ref = torch.tensor(pos, dtype=torch.float32)
    fit = torch.tensor(pos, dtype=torch.float32)
    rc = torch.tensor(chg, dtype=torch.float32)
    fc = torch.tensor(chg, dtype=torch.float32)

    # non-identity pose (normalized quat + translation) so the gradient is genuinely nonzero
    q = torch.tensor([0.966, 0.259, 0.0, 0.0], dtype=torch.float32)
    q = q / q.norm()
    se3 = torch.cat([q, torch.tensor([0.3, -0.2, 0.1], dtype=torch.float32)])
    se3 = se3.clone().requires_grad_(True)

    def _obj(p):
        return objective_vol_esp_tversky_overlay(p, ref, fit, rc, fc, alpha=0.81, lam=0.1,
                                                 tversky_alpha=0.95, tversky_beta=0.05)

    val = _obj(se3)
    val.backward()
    ana = se3.grad.clone()

    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data)
        d[i] = eps
        hi = _obj((se3.data + d).clone().requires_grad_(False))
        lo = _obj((se3.data - d).clone().requires_grad_(False))
        num[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"autograd {ana} vs finite-diff {num}"


# --- Gate 3: the optimizer recovers a planted rotation ----------------------------------------
def test_optimizer_recovers_planted_pose(ibuprofen):
    pos, chg = _heavy_pos_charges(ibuprofen)
    ref = torch.tensor(pos, dtype=torch.float32)

    # plant a known proper rotation on the fit
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    fit = torch.tensor(pos @ Q.T, dtype=torch.float32)
    rc = torch.tensor(chg, dtype=torch.float32)
    fc = torch.tensor(chg, dtype=torch.float32)

    _, _, score = optimize_vol_esp_tversky_overlay(ref, fit, rc, fc, num_repeats=20,
                                                   max_num_steps=200)
    assert float(score) > 0.95, f"planted-pose recovery only {float(score):.3f}"


# --- Gate 4: determinism ----------------------------------------------------------------------
def test_deterministic_given_seed(ibuprofen):
    torch.manual_seed(SEED)
    pair_a = _make_pair(ibuprofen)
    pair_a.align_with_vol_esp_tversky(num_repeats=20, max_num_steps=100)
    torch.manual_seed(SEED)
    pair_b = _make_pair(ibuprofen)
    pair_b.align_with_vol_esp_tversky(num_repeats=20, max_num_steps=100)
    assert np.isclose(float(pair_a.sim_aligned_vol_esp_tversky),
                      float(pair_b.sim_aligned_vol_esp_tversky), atol=0.0)


# --- Gate 5: retained-H molecule (REQUIRED -- this mode reads per-atom charges) ----------------
def test_retained_h_molecule():
    """A molecule whose ``Chem.RemoveHs`` RETAINS an H (isotope-labelled deuterium) has
    ``atom_pos`` longer than the true-heavy set ``_nonH_atoms_idx`` selects. This mode pairs
    heavy charges with heavy centres taken from ``GetConformer().GetPositions()[_nonH_atoms_idx]``
    (never ``atom_pos``), so it must not desync/crash -- and the self-copy still scores ~1.0."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("[2H]OC(=O)c1ccccc1"))       # deuterium survives RemoveHs
    params = AllChem.ETKDGv3(); params.randomSeed = 0
    assert AllChem.EmbedMolecule(m, params) == 0
    AllChem.MMFFOptimizeMolecule(m)
    mol = Molecule(m)
    assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
        "test premise broken: this molecule no longer retains an H after RemoveHs"

    pair = MoleculePair(mol, Molecule(m), do_center=True, device=torch.device('cpu'))
    pair.align_with_vol_esp_tversky(num_repeats=10, max_num_steps=100)     # must not raise
    score = float(pair.sim_aligned_vol_esp_tversky)
    assert np.isclose(score, 1.0, atol=1e-3), f"retained-H self-overlap {score} != 1.000"


# --- Gate 6: Tversky asymmetry (fits-inside) --------------------------------------------------
def test_tversky_asymmetry_fits_inside():
    """A small query contained in a larger molecule should score HIGHER when it is the
    REFERENCE (query -> big fit) than in the reverse direction, under the default weights."""
    phenol = _embed("Oc1ccccc1")
    naphthol = _embed("Oc1ccc2ccccc2c1")

    small = Molecule(phenol)
    big = Molecule(naphthol)

    # query (small) as ref, big as fit
    mp_fwd = MoleculePair(small, big, do_center=True, device=torch.device('cpu'))
    mp_fwd.align_with_vol_esp_tversky(num_repeats=30, max_num_steps=200)
    fwd = float(mp_fwd.sim_aligned_vol_esp_tversky)

    # reverse: big as ref, query (small) as fit
    mp_rev = MoleculePair(big, small, do_center=True, device=torch.device('cpu'))
    mp_rev.align_with_vol_esp_tversky(num_repeats=30, max_num_steps=200)
    rev = float(mp_rev.sim_aligned_vol_esp_tversky)

    assert fwd > rev, f"expected fits-inside fwd ({fwd:.3f}) > reverse ({rev:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
