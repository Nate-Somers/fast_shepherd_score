"""Reference-layer correctness gates for the SI experimental alignment modes.

Covers the nine modes added for the supplementary information:
  vol_pharm, vol_atomtype, vol_mr,
  surf_tversky, surf_esp_tversky, vol_and_surf_esp_tversky,
  vol_color_tversky, vol_lipo_tversky, pharm_tversky.

Design-scoring-mode gates enforced here: self-overlap = 1.000 (gate 1), autograd == finite
difference at a non-identity pose (gate 2, for the modes with new objective math), determinism
(gate 4), and the retained-H basis (gate 5, for the per-atom-field modes). The surface modes need
Open3D; those tests ``importorskip`` it, but ``vol_and_surf_esp_tversky``'s NEW blend math is also
exercised with synthetic surface data so it is validated even without Open3D.
"""
import numpy as np
import pytest
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.alignment._torch import (
    objective_vol_atomtype_overlay,
    objective_vol_color_tversky_overlay,
    objective_vol_lipo_tversky_overlay,
    optimize_vol_and_surf_esp_tversky_overlay,
)

torch.set_num_threads(2)

# Modes that need no molecular surface (Open3D-free) -> (align method, score attr).
NONSURF_MODES = {
    "vol_pharm":         "sim_aligned_vol_pharm",
    "vol_atomtype":      "sim_aligned_vol_atomtype",
    "vol_mr":            "sim_aligned_vol_mr",
    "vol_color_tversky": "sim_aligned_vol_color_tversky",
    "vol_lipo_tversky":  "sim_aligned_vol_lipo_tversky",
    "pharm_tversky":     "sim_aligned_pharm_tversky",
}
SURF_MODES = {
    "surf_tversky":             "sim_aligned_surf_tversky",
    "surf_esp_tversky":         "sim_aligned_surf_esp_tversky",
    "vol_and_surf_esp_tversky": "sim_aligned_vol_and_surf_esp_tversky",
}
_KW = dict(num_repeats=10, max_num_steps=60)


def _mol(smiles, surface=False):
    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=42)
    kw = dict(pharm_multi_vector=False)
    if surface:
        kw.update(num_surf_points=100)
    return Molecule(rd, **kw)


def _run(pair, mode):
    getattr(pair, f"align_with_{mode}")(**_KW)
    return float(getattr(pair, NONSURF_MODES.get(mode) or SURF_MODES[mode]))


# --- Gate 1: self-overlap = 1.000 (non-surface modes) -----------------------------------------
@pytest.mark.parametrize("mode", list(NONSURF_MODES))
def test_self_overlap_is_one(mode):
    m = _mol("CC(=O)Oc1ccccc1C(=O)O")
    score = _run(MoleculePair(m, m, do_center=True, device=torch.device("cpu")), mode)
    assert np.isclose(score, 1.0, atol=5e-3), f"{mode} self-overlap {score} != 1.000"


# --- Gate 1 + sanity: a real pair scores in [0, 1] --------------------------------------------
@pytest.mark.parametrize("mode", list(NONSURF_MODES))
def test_pair_in_unit_interval(mode):
    a, b = _mol("CC(=O)Oc1ccccc1C(=O)O"), _mol("COC(=O)c1ccccc1O")
    s = _run(MoleculePair(a, b, do_center=True, device=torch.device("cpu")), mode)
    assert -1e-4 <= s <= 1.0 + 1e-4, f"{mode} pair score {s} outside [0,1]"


# --- Gate 4: determinism ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["vol_atomtype", "vol_color_tversky", "vol_mr"])
def test_deterministic(mode):
    a, b = _mol("CC(=O)Oc1ccccc1C(=O)O"), _mol("COC(=O)c1ccccc1O")
    s1 = _run(MoleculePair(a, b, do_center=True, device=torch.device("cpu")), mode)
    s2 = _run(MoleculePair(a, b, do_center=True, device=torch.device("cpu")), mode)
    assert s1 == s2, f"{mode} not deterministic: {s1} != {s2}"


# --- Gate 2: autograd == finite difference at a NON-identity pose (new-objective modes) --------
def _fd_grad_check(obj_fn):
    # small rotation quat (r,i,j,k) + translation -> genuinely nonzero gradient (float32, eps=1e-3)
    se3 = torch.tensor([0.966, 0.259, 0.0, 0.0, 0.3, -0.2, 0.1],
                       dtype=torch.float32, requires_grad=True)
    val = obj_fn(se3)
    val.backward()
    ana = se3.grad.clone()
    num = torch.zeros_like(ana)
    eps = 1e-3
    for i in range(se3.numel()):
        d = torch.zeros_like(se3.data); d[i] = eps
        hi = obj_fn((se3.data + d).clone().requires_grad_(False))
        lo = obj_fn((se3.data - d).clone().requires_grad_(False))
        num[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(ana, num, atol=2e-3), f"grad mismatch\nana={ana}\nnum={num}"


def _T(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32)


def _heavy(m):
    return m.mol.GetConformer().GetPositions()[m._nonH_atoms_idx]


def test_gradient_vol_atomtype():
    a, b = _mol("CC(=O)Oc1ccccc1C(=O)O"), _mol("COC(=O)c1ccccc1O")
    rc, fc = _T(a.atom_pos), _T(b.atom_pos)
    rtp, ftp = _T(_heavy(a)), _T(_heavy(b))
    rl, fl = _T(a.get_atomic_numbers(True)), _T(b.get_atomic_numbers(True))
    _fd_grad_check(lambda se3: objective_vol_atomtype_overlay(
        se3, rc, fc, rtp, ftp, rl, fl, alpha=0.81, atomtype_weight=0.5))


def test_gradient_vol_color_tversky():
    a, b = _mol("CC(=O)Oc1ccccc1C(=O)O"), _mol("COC(=O)c1ccccc1O")
    _fd_grad_check(lambda se3: objective_vol_color_tversky_overlay(
        se3, _T(a.atom_pos), _T(b.atom_pos),
        torch.from_numpy(a.pharm_types), torch.from_numpy(b.pharm_types),
        _T(a.pharm_ancs), _T(b.pharm_ancs), _T(a.pharm_vecs), _T(b.pharm_vecs),
        alpha=0.81, color_weight=0.5))


def test_gradient_vol_lipo_tversky():
    a, b = _mol("CC(=O)Oc1ccccc1C(=O)O"), _mol("COC(=O)c1ccccc1O")
    _fd_grad_check(lambda se3: objective_vol_lipo_tversky_overlay(
        se3, _T(a.atom_pos), _T(b.atom_pos), _T(_heavy(a)), _T(_heavy(b)),
        _T(a.get_lipophilicity(True)), _T(b.get_lipophilicity(True)),
        alpha=0.81, lam=0.1, lipo_weight=0.5))


# --- Gate 5: retained-H basis (per-atom-field modes must not desync N vs N-1) ------------------
@pytest.mark.parametrize("mode", ["vol_mr", "vol_atomtype", "vol_lipo_tversky"])
def test_retained_h_molecule(mode):
    m = Chem.AddHs(Chem.MolFromSmiles("[2H]OC(=O)c1ccccc1"))   # deuterium survives RemoveHs
    params = AllChem.ETKDGv3(); params.randomSeed = 0
    assert AllChem.EmbedMolecule(m, params) == 0
    AllChem.MMFFOptimizeMolecule(m)
    mol = Molecule(m, pharm_multi_vector=False)
    assert mol.atom_pos.shape[0] != len(mol._nonH_atoms_idx), \
        "test premise broken: this molecule no longer retains an H after RemoveHs"
    # must not raise a broadcast error (N vs N-1) -- the retained-H trap
    s = _run(MoleculePair(mol, mol, do_center=True, device=torch.device("cpu")), mode)
    assert np.isclose(s, 1.0, atol=5e-3)


# --- vol_and_surf_esp_tversky: NEW blend math, exercised with SYNTHETIC surface data ----------
def _synthetic_esp_inputs(seed=0):
    """A real molecule's atom set + a fake surface point cloud/ESP so the vol_and_surf_esp_tversky
    optimizer runs without Open3D. The points sit on a shell OUTSIDE the vdW+probe envelope (so
    ``_esp_comparison`` never masks them, exactly like a real molecular surface) -- required for the
    self-overlap = 1.000 gate: a masked point contributes 0 to the numerator but 1 to the point
    count, so in-volume points would depress the ESP-agreement channel even for a self-pair."""
    rng = np.random.default_rng(seed)
    m = _mol("CC(=O)Oc1ccccc1C(=O)O")
    w_h = m.mol.GetConformer().GetPositions().astype(np.float32)
    heavy = _heavy(m).astype(np.float32)
    charges = m.partial_charges.astype(np.float32)      # MMFF fallback (no xTB needed)
    radii = m.radii.astype(np.float32)
    com = w_h.mean(0)
    shell_r = float(np.linalg.norm(w_h - com, axis=1).max()) + 4.0  # > max(radii)+probe past every atom
    dirs = rng.normal(size=(120, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    surf = (com + shell_r * dirs).astype(np.float32)
    # ESP at each surface point = Coulomb potential of the molecule's charges (the SAME formula
    # _esp_comparison recomputes internally). It must be the real potential, not random, or a
    # self-pair's stored-vs-recomputed ESP would differ and the agreement channel would be < 1.
    from shepherd_score.score.constants import COULOMB_SCALING
    d = np.linalg.norm(surf[:, None, :] - w_h[None, :, :], axis=2)   # (120, n_wh)
    surf_esp = ((charges[None, :] / d).sum(axis=1) * COULOMB_SCALING).astype(np.float32)
    return dict(w_h=w_h, heavy=heavy, charges=charges, radii=radii, surf=surf, surf_esp=surf_esp)


def _run_vas_tversky(ref, fit, **over):
    kw = dict(alpha=0.81, lam=0.001, probe_radius=1.0, esp_weight=0.5,
              num_repeats=8, max_num_steps=50)
    kw.update(over)
    _, _, score = optimize_vol_and_surf_esp_tversky_overlay(
        ref_centers_w_H=_T(ref["w_h"]), fit_centers_w_H=_T(fit["w_h"]),
        ref_centers=_T(ref["heavy"]), fit_centers=_T(fit["heavy"]),
        ref_points=_T(ref["surf"]), fit_points=_T(fit["surf"]),
        ref_partial_charges=_T(ref["charges"]), fit_partial_charges=_T(fit["charges"]),
        ref_surf_esp=_T(ref["surf_esp"]), fit_surf_esp=_T(fit["surf_esp"]),
        ref_radii=_T(ref["radii"]), fit_radii=_T(fit["radii"]), **kw)
    return float(score)


def test_vol_and_surf_esp_tversky_self_overlap_synthetic():
    d = _synthetic_esp_inputs()
    s = _run_vas_tversky(d, d)
    assert np.isclose(s, 1.0, atol=5e-3), f"vol_and_surf_esp_tversky self-overlap {s} != 1.000"


def test_vol_and_surf_esp_tversky_deterministic_synthetic():
    d, e = _synthetic_esp_inputs(0), _synthetic_esp_inputs(1)
    assert _run_vas_tversky(d, e) == _run_vas_tversky(d, e)


# --- Surface modes end-to-end (Open3D required) -----------------------------------------------
@pytest.mark.parametrize("mode", list(SURF_MODES))
def test_surf_modes_self_overlap(mode):
    pytest.importorskip("open3d")
    m = _mol("CC(=O)Oc1ccccc1C(=O)O", surface=True)
    pair = MoleculePair(m, m, do_center=True, device=torch.device("cpu"))
    getattr(pair, f"align_with_{mode}")(**_KW)
    s = float(getattr(pair, SURF_MODES[mode]))
    assert np.isclose(s, 1.0, atol=5e-3), f"{mode} self-overlap {s} != 1.000"
