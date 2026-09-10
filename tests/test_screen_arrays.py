"""Gate 5 for the array-native screen path (``FSS_SCREEN_ARRAYS=1``).

``ea46a2e`` added ``accel/batch/_arrays.py`` plus the ``screen.py`` dispatch and asserted
"bit-identical" in its commit message, but shipped no test; ``_arrays.py`` still says "Default OFF
until the gates pass". This is that gate: the array path must produce byte-for-byte the same
scores as the object path it replaces, or it is not the re-expression it claims to be.

WHY THE SPIES. A parity test here can pass while proving nothing. ``_use_arrays`` gates on
``mode in _ARRAY_MODES`` AND ``_arrays.ENABLED``, so a mistake in either silently compares the
object path against itself and reports success. These tests therefore assert WHICH BUILDER RAN,
not merely that two vectors matched -- and each mode has its own builder, so the spy set has to
cover ``_build_fit_arrays_vol``, ``_build_fit_arrays_vol_color`` and ``_build_fit_fast_pairs``.

WHY THE SIZES ARE WHAT THEY ARE. The array path re-derives bucket membership as spans over an
index array rather than Python lists of pair objects, and ``_pad._band_key`` bands molecules by
``((n + 15) // 16) * 16``. A library of drug-like molecules is NOT enough: 3-15 heavy atoms all
land in band 16, giving one cell and leaving the span/pad arithmetic untested (measured -- an
earlier version of this file asserted >=2 buckets and failed on exactly that). The alkanes below
exist to push the library across band boundaries: heavy counts 3..36 span bands 16/32/48.
"""
import os

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

from rdkit import Chem                                          # noqa: E402
from rdkit.Chem import AllChem                                  # noqa: E402

from shepherd_score.container import Molecule                   # noqa: E402
from shepherd_score.screen import ProfileStore, screen          # noqa: E402

_BAND = 16          # mirrors accel.batch._pad._BAND; asserted against the real one below

#: the documented library defaults, matching aligners/fss.py::prepare_screen. vol_esp takes
#: lam RAW (surf_esp scales it x207) and screen() refuses to run vol_esp without it.
_MODE_KW = {"vol_esp": {"lam": 0.1}, "vol_and_surf_esp": {"alpha": 0.81}}

#: Modes whose self-copy reaches ~1.0 on THIS fixture. vol_and_surf_esp is excluded, and not
#: arbitrarily: ``_build_molecule`` gives every molecule a SYNTHETIC random surface (so the test
#: needs no Open3D), which is fine for shape/pharm channels but is not the surface implied by the
#: molecule's own atoms -- and vol_and_surf_esp scores surface and ESP channels against fields
#: derived from those atoms, so a self-copy need not reach 1.0. The repo's own
#: ``test_screen.py::test_self_screen_recovers_one`` excludes it from exactly this assertion for
#: the same reason. Measured here: it ranks itself FIRST but scores 0.5886, identically on both
#: paths (the parity test compares the full score vector, self-copy included).
_SELF_SCORES_ONE = ("vol", "vol_color", "pharm", "vol_esp")
#: vol_and_surf_esp has NO alpha default -- _fast_batch_kwargs reads ak["alpha"] directly.


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


# Drug-like molecules (band 16) + n-alkanes chosen so the library also occupies bands 32 and 48.
# "C" * n has exactly n heavy atoms, so the band coverage is exact rather than eyeballed.
_SMILES = [
    "CCO",                                  # 3
    "C1CCCCC1",                             # 6
    "c1ccccc1O",                            # 7
    "CC(=O)Nc1ccc(O)cc1",                   # 11
    "CC(=O)Oc1ccccc1C(=O)O",                # 13
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",           # 15
    "C" * 18,                               # 18 -> band 32
    "C" * 22,                               # 22 -> band 32
    "C" * 26,                               # 26 -> band 32
    "C" * 30,                               # 30 -> band 32
    "C" * 34,                               # 34 -> band 48
    "C" * 36,                               # 36 -> band 48
]


def _build_molecule(smi, seed, S=64):
    """Real RDKit conformer + a *synthetic* surface (so the test needs no Open3D)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    surf = rng.standard_normal((S, 3)).astype(np.float32) * 3.0
    esp = rng.standard_normal((S,)).astype(np.float32)
    return Molecule(m, surface_points=surf, electrostatics=esp, pharm_multi_vector=False)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


@pytest.fixture(scope="module")
def store_path(tmp_path_factory, molecules):
    p = os.path.join(tmp_path_factory.mktemp("arrays"), "lib.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol", "vol_color", "pharm", "vol_esp", "vol_and_surf_esp"),
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def _screen_recording(monkeypatch, store_path, query, *, enabled, mode="vol", steps=30):
    """Run one screen with the array path forced on/off, recording which builder ran.

    ``_use_arrays`` re-imports ``_arrays`` and reads ``.ENABLED`` on every call
    (screen.py::_use_arrays), so patching the ATTRIBUTE steers the dispatch. Patching the env var
    would not: ``ENABLED`` is resolved from the environment once, at import.
    """
    from shepherd_score.accel.batch import _arrays
    import shepherd_score.screen as screenmod

    monkeypatch.setattr(_arrays, "ENABLED", enabled, raising=True)

    seen = {"arrays": 0, "objects": 0, "buckets": 0}
    real_obj = screenmod._build_fit_fast_pairs
    real_spans = _arrays.plan_spans

    def _count_arr(real):
        def f(*a, **k):
            seen["arrays"] += 1
            return real(*a, **k)
        return f

    def spy_obj(*a, **k):
        seen["objects"] += 1
        return real_obj(*a, **k)

    real_spans_multi = _arrays.plan_spans_multi

    def spy_spans(*a, **k):
        order, buckets = real_spans(*a, **k)
        seen["buckets"] = max(seen["buckets"], len(buckets))
        return order, buckets

    def spy_spans_multi(*a, **k):
        # vol_and_surf_esp keys SIX dims, so it plans through plan_spans_multi and never
        # touches plan_spans. Counting only the latter made the bucket tripwire fire on a
        # perfectly good path -- and, worse, short-circuited the parity check behind it.
        buckets = real_spans_multi(*a, **k)
        seen["buckets"] = max(seen["buckets"], len(buckets))
        return buckets

    # Both call sites resolve these as module globals (screen.py:1315/1322, _arrays.py:167),
    # so attribute patches genuinely intercept them.
    # EVERY array builder is wrapped, and the _ARRAY_BUILDERS table is re-pointed at the
    # wrapped versions -- the dispatch reads the table, so patching only the module globals
    # would leave the table holding unwrapped functions and the spy would count zero.
    for _name in ("_build_fit_arrays_vol", "_build_fit_arrays_vol_color",
                  "_build_fit_arrays_pharm", "_build_fit_arrays_vol_esp",
                  "_build_fit_arrays_vol_and_surf_esp"):
        monkeypatch.setattr(screenmod, _name, _count_arr(getattr(screenmod, _name)), raising=True)
    monkeypatch.setattr(screenmod, "_ARRAY_BUILDERS",
                        {m: _count_arr(fn) for m, fn in screenmod._ARRAY_BUILDERS.items()},
                        raising=True)
    monkeypatch.setattr(screenmod, "_build_fit_fast_pairs", spy_obj, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans", spy_spans, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans_multi", spy_spans_multi, raising=True)

    n = len(ProfileStore.open(store_path))
    scores = np.full(n, np.nan, dtype=float)
    hits = screen(query, ProfileStore.open(store_path), mode=mode, backend="triton",
                  top_k=n, max_num_steps=steps, scores_out=scores, **_MODE_KW.get(mode, {}))
    return scores, hits, seen


def test_library_spans_multiple_pad_bands(molecules):
    """The parity test is only meaningful on a library that crosses ``_band_key`` boundaries.

    Guards the fixture itself: if someone trims the alkanes, the parity test silently stops
    exercising the multi-cell span arithmetic instead of failing here.
    """
    from shepherd_score.accel.batch._pad import _BAND as REAL_BAND, _band_key

    assert REAL_BAND == _BAND, "band width changed upstream; revisit the library sizes"
    sizes = [int(m.atom_pos.shape[0]) for m in molecules]
    bands = {_band_key(n) for n in sizes}
    assert len(set(sizes)) >= 6, f"library sizes are not varied enough: {sorted(set(sizes))}"
    assert len(bands) >= 3, f"library occupies only bands {sorted(bands)}; need >=3"


@pytest.mark.cuda
@pytest.mark.parametrize("mode",
                         ["vol", "vol_color", "pharm", "vol_esp", "vol_and_surf_esp"])
def test_array_path_is_bit_identical_to_object_path(monkeypatch, store_path, molecules, mode):
    """The array path is a re-expression of the object path, so scores must match EXACTLY."""
    _require_fast_cuda()
    query = molecules[1]

    with monkeypatch.context() as mp:
        s_on, h_on, seen_on = _screen_recording(mp, store_path, query, enabled=True, mode=mode)
    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _screen_recording(mp, store_path, query, enabled=False, mode=mode)

    # --- the two runs really took different paths -------------------------------------------
    assert seen_on["arrays"] > 0, "ENABLED=True did not reach the array builder"
    assert seen_on["objects"] == 0, "ENABLED=True still built _FastPair objects"
    assert seen_off["objects"] > 0, "ENABLED=False did not reach the object builder"
    assert seen_off["arrays"] == 0, "ENABLED=False still reached the array builder"
    assert seen_on["buckets"] >= 1, "plan_spans never ran on the array path"

    # --- scores are real, not a degenerate all-equal / all-NaN vector -----------------------
    assert np.isfinite(s_on).all(), "array path produced non-finite scores"
    assert np.isfinite(s_off).all(), "object path produced non-finite scores"
    assert s_on.max() - s_on.min() > 1e-6, "scores are degenerate; parity would be vacuous"

    # --- the gate itself --------------------------------------------------------------------
    np.testing.assert_array_equal(s_on, s_off)
    assert [h.id for h in h_on] == [h.id for h in h_off], "top-k identity/order diverged"
    np.testing.assert_array_equal(np.array([h.score for h in h_on], dtype=float),
                                  np.array([h.score for h in h_off], dtype=float))


@pytest.mark.cuda
@pytest.mark.parametrize("mode",
                         ["vol", "vol_color", "pharm", "vol_esp", "vol_and_surf_esp"])
def test_array_path_recovers_the_self_copy(monkeypatch, store_path, molecules, mode):
    """Independent anchor: the array path must be a CORRECT screen, not merely a consistent one.
    Two paths agreeing on nonsense would satisfy parity by itself."""
    _require_fast_cuda()
    with monkeypatch.context() as mp:
        scores, hits, seen = _screen_recording(mp, store_path, molecules[1],
                                               enabled=True, mode=mode)
    assert seen["arrays"] > 0
    # RANKING is the anchor that holds for every mode: a self-copy is its own best match.
    assert hits[0].id == 1, "query is in the library at id=1 and must rank first"
    if mode in _SELF_SCORES_ONE:
        assert hits[0].score == pytest.approx(1.0, abs=1e-2), "self-copy must score ~1.0"


def test_array_path_is_off_by_default():
    """The flag is opt-in. Compares against the env the module actually saw at import, so this
    stays honest when the suite is deliberately run with FSS_SCREEN_ARRAYS=1 exported."""
    from shepherd_score.accel.batch import _arrays
    assert _arrays.ENABLED == (os.environ.get("FSS_SCREEN_ARRAYS") == "1")


def test_use_arrays_gates_on_mode_and_reads_enabled_live(monkeypatch):
    """``_use_arrays`` must gate on BOTH the mode and the live flag -- the two ways this file
    could silently stop testing anything.

    The mode list is asserted against ``_ARRAY_MODES`` rather than hardcoded, but every mode
    NOT in it is checked explicitly: adding a mode to the tuple without an array-native aligner
    would otherwise route it into a function that does not exist."""
    from shepherd_score.accel.batch import _arrays
    import shepherd_score.screen as screenmod

    monkeypatch.setattr(_arrays, "ENABLED", True, raising=True)
    for mode in screenmod._ARRAY_MODES:
        assert screenmod._use_arrays(mode) is True
    for mode in ("surf", "surf_esp"):
        assert mode not in screenmod._ARRAY_MODES
        assert screenmod._use_arrays(mode) is False, f"{mode} has no array-native aligner"

    monkeypatch.setattr(_arrays, "ENABLED", False, raising=True)
    for mode in screenmod._ARRAY_MODES:
        assert screenmod._use_arrays(mode) is False, "ENABLED is read at call time, not import"
