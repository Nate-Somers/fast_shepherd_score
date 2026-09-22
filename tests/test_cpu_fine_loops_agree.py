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


def _mol(smiles, seed, n_surf=None):
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
    ns = N_SURF if n_surf is None else int(n_surf)
    return Molecule(rd, pharm_multi_vector=False,
                    surface_points=(rng.standard_normal((ns, 3)) * 3.0).astype(np.float32),
                    electrostatics=rng.standard_normal(ns).astype(np.float32),
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


# --- the regime the lifted surf_esp cap opened ------------------------------------------------
# surf_esp used to carry cpu_fused_max_pad=100, which refused the fused loop above ~96 surface
# points and so never fired at the 200-point default. The tests above run at N_SURF, which is
# below that, so they would not have noticed either the exclusion or its removal.

@pytest.mark.parametrize("n_surf", [128, 200, 300])
def test_surf_esp_stays_fused_above_the_old_cap(n_surf, monkeypatch):
    """Above 100 surface points surf_esp must take the FUSED loop. If someone reinstates a cap,
    this fails rather than quietly costing 13x on CPU."""
    from shepherd_score.accel.drivers import engine
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    seen = []
    rf, re_ = cpu_fused.run_fused, engine._eager
    monkeypatch.setattr(cpu_fused, "run_fused",
                        lambda *a, **k: (seen.append("fused"), rf(*a, **k))[1])
    monkeypatch.setattr(engine, "_eager", lambda *a, **k: (seen.append("eager"), re_(*a, **k))[1])

    a, b = _mol(SMILES[0], 0, n_surf=n_surf), _mol(SMILES[1], 1, n_surf=n_surf)
    pair = MoleculePair(a, b, do_center=True, device=torch.device("cpu"))
    MoleculePairBatch([pair]).align_with_surf_esp(backend="numba", alpha=0.81)
    assert "fused" in seen and "eager" not in seen, (
        "surf_esp at %d surface points took %s; it must take the fused CPU loop"
        % (n_surf, seen or ["?"]))


@pytest.mark.parametrize("n_surf", [128, 200])
def test_surf_esp_pose_agrees_above_the_old_cap(n_surf):
    """The cap's stated reason was pose-exactness, not score agreement: the most shape-degenerate
    mode might settle in a different basin under the fused loop. Measured on real open3d surfaces
    it does not (0.045 deg at 200 points), and this holds the claim on the synthetic fixture too.
    Compares the returned 4x4 transforms, since scores can agree while poses diverge."""
    from shepherd_score.accel.kernels import cpu_fused
    from shepherd_score.container import MoleculePair, MoleculePairBatch

    mols = [_mol(s, i, n_surf=n_surf) for i, s in enumerate(SMILES)]

    def poses(force_eager):
        real = cpu_fused.run_fused
        if force_eager:
            def boom(*a, **k):
                raise RuntimeError("fused disabled by the test")
            cpu_fused.run_fused = boom
        try:
            pairs = [MoleculePair(mols[0], f, do_center=True, device=torch.device("cpu"))
                     for f in mols]
            MoleculePairBatch(pairs).align_with_surf_esp(backend="numba", alpha=0.81)
            return [np.asarray(p.transform_surf_esp, dtype=np.float64) for p in pairs]
        finally:
            cpu_fused.run_fused = real

    worst = 0.0
    for A, B in zip(poses(False), poses(True)):
        R = A[:3, :3] @ B[:3, :3].T
        worst = max(worst, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))))
    assert worst < 5.0, (
        "surf_esp fused vs eager rotated by %.3f deg at %d surface points: the fused loop has "
        "landed in a different basin, which is what the old cpu_fused_max_pad=100 guarded "
        "against. Measured 0.045 deg on real surfaces when the cap was lifted." % (worst, n_surf))
