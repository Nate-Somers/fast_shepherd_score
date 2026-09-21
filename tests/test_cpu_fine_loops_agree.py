"""Gate 3b on CPU: the fused numba fine loop and the eager torch fine loop must agree.

`drivers/engine.align` picks the fused loop when the mode's ModeSpec allows it and the padded
widths fit, and falls back to the eager loop on ANY exception::

    try:
        best, bq, bt = run_fused(...)
    except Exception:
        best = None          # fused failed -> eager

That fallback is silent, so a mode whose two loops disagree would return different scores
depending on whether an unrelated failure fired -- and nothing measured the gap. The engine
rewrite put 19 of the 21 modes on the fused loop (4 before), so the question went from narrow
to library-wide.

The gap is ENVIRONMENT-DEPENDENT, which the first version of this file got wrong. With SVML the
fused loop swaps in the fp32 SoA kernels (`kernels/cpu_soa.py`, ~1e-6 value / ~1e-4 gradient
relative error by construction); without it the fp64 AoS kernels run. A bound fitted to one
environment fails in the other, so these are fitted to BOTH.

MEASURED over 6 molecules, every mode, in both:

| | worst, no SVML (Windows) | worst, SVML (node3509) |
|---|---|---|
| every mode except `surf_esp` | 1.699e-06 / 0.0004% | 2.086e-06 / 0.0005% |
| `surf_esp` | 1.699e-06 / 0.0004% | **1.782e-04 / 0.0375%** |

`surf_esp` is the outlier by 75x, and it is the mode already singled out as the most
shape-degenerate in the library -- the reason its spec carries `cpu_fused_max_pad=100`. It gets
its own bound rather than loosening every mode to fit it. The pharmacophore family sets
`cpu_fused=False` and is identical either way, because both runs take the eager loop.
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

# (max |delta|, max relative %) per mode, defaulting to the tight pair. Roughly an order of
# magnitude above the worse of the two measured environments, so a failure means two loops
# have genuinely diverged rather than that fp32 drifted.
TOL_DEFAULT = (1.0e-5, 0.01)
TOL = {"surf_esp": (1.0e-3, 0.2)}

SMILES = ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
          "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccc2O"]
N_SURF = 60

# Modes whose call needs an explicit keyword; everything else takes the registry defaults.
KW = {"surf": {"alpha": 0.81}, "surf_esp": {"alpha": 0.81, "lam": 0.3},
      "surf_tversky": {"alpha": 0.81}, "surf_esp_tversky": {"alpha": 0.81, "lam": 0.3},
      "vol_and_surf_esp": {"alpha": 0.81}, "vol_and_surf_esp_tversky": {"alpha": 0.81}}


def _mol(smiles, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        assert AllChem.EmbedMolecule(rd, p) == 0, smiles
    rng = np.random.default_rng(seed)
    n = rd.GetNumAtoms()
    return Molecule(rd, pharm_multi_vector=False,
                    surface_points=(rng.standard_normal((N_SURF, 3)) * 3.0).astype(np.float32),
                    electrostatics=rng.standard_normal(N_SURF).astype(np.float32),
                    fukui=(rng.standard_normal(n) * 0.1).astype(np.float32),
                    partial_charges=(rng.standard_normal(n) * 0.2).astype(np.float32))


@pytest.fixture(scope="module")
def mols():
    return [_mol(s, i) for i, s in enumerate(SMILES)]


def _scores(mode, mols, force_eager):
    """Align mols[0] against every mol, optionally with the fused loop disabled."""
    from shepherd_score.accel._modes import MODE_ATTRS
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    real = cpu_fused.run_fused
    if force_eager:
        def boom(*a, **k):
            raise RuntimeError("fused disabled by the test")
        cpu_fused.run_fused = boom
    try:
        pairs = [MoleculePair(mols[0], f, do_center=True, device=torch.device("cpu"))
                 for f in mols]
        batch = MoleculePairBatch(pairs)
        args = ()
        if mode == "vol_avoid":                       # the one mode with a third input
            args = (np.asarray(mols[0].atom_pos, dtype=np.float32) + 40.0,)
        getattr(batch, "align_with_" + mode)(*args, backend="numba", **KW.get(mode, {}))
        return [float(getattr(p, MODE_ATTRS[mode][1])) for p in pairs]
    finally:
        cpu_fused.run_fused = real


def _all_modes():
    from shepherd_score.accel._modes import CANONICAL_MODES
    return list(CANONICAL_MODES)


@pytest.mark.parametrize("mode", _all_modes())
def test_fused_and_eager_cpu_loops_agree(mode, mols):
    from shepherd_score.accel._modes import spec_of

    fused = _scores(mode, mols, force_eager=False)
    eager = _scores(mode, mols, force_eager=True)

    assert all(np.isfinite(fused)) and all(np.isfinite(eager)), (fused, eager)
    d = max(abs(a - b) for a, b in zip(fused, eager))
    rel = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(fused, eager)) * 100

    max_abs, max_rel = TOL.get(mode, TOL_DEFAULT)

    assert d <= max_abs and rel <= max_rel, (
        f"{mode}: fused vs eager max|d| {d:.3e} ({rel:.4f}% relative) exceeds "
        f"{max_abs:.1e} / {max_rel}%.\n  fused {fused}\n  eager {eager}\n"
        "  (SVML widens this gap; see the module docstring for both measurements)")

    if not spec_of(mode).cpu_fused:
        assert fused == eager, \
            f"{mode} sets cpu_fused=False, so both runs take the eager loop and must be identical"


def test_disabling_the_fused_loop_actually_changes_the_path(mols, monkeypatch):
    """Guard the guard: if forcing the fused loop off stopped reaching the eager loop, every
    test above would compare a path against itself and pass vacuously."""
    from shepherd_score.accel.drivers import engine

    seen = []
    real = engine._eager

    def spy(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(engine, "_eager", spy)
    _scores("vol", mols, force_eager=False)
    assert not seen, "vol took the eager loop even with the fused loop available"
    _scores("vol", mols, force_eager=True)
    assert seen, "forcing the fused loop off did not reach the eager loop"
