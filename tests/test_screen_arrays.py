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


# ------------------------------------------------------------------------------------------------
# Canonical-frame stores: the transform must come back in the MOLECULE's frame, not the store's.
# ------------------------------------------------------------------------------------------------
# A canonical store rotates every library molecule into its own principal frame at build time, so
# the pose the optimizer finds is expressed against those rotated coordinates. Scores and RANKING
# are correct either way -- which is exactly why this needs its own test: every score-based check
# in this file passes on a canonical store whose transforms are silently in the wrong frame.
#
# The check does not compare against the legacy store's transform. It cannot: the two stores seed
# the optimizer differently, so they reach different optima and their poses legitimately differ.
# Instead it re-scores: apply the returned transform to the molecule's OWN centred coordinates and
# recompute the Tanimoto directly. A pose in the wrong frame does not reproduce its own score.


def _canonical_store(tmp_path, molecules):
    p = os.path.join(tmp_path, "canon.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float32",
                             pre_centered=True, canonical=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def _plain_store(tmp_path, molecules):
    """Same library, same modes, canonical OFF -- the control leg."""
    p = os.path.join(tmp_path, "plain.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float32",
                             pre_centered=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def test_canonical_store_records_its_rotation(tmp_path, molecules):
    """The store must carry ``rot``, and it must be a PROPER rotation.

    Without this the composition below is a no-op that the re-scoring test would still pass on a
    store that quietly fell back to non-canonical -- the same vacuity the spies guard elsewhere.
    """
    store = ProfileStore.open(_canonical_store(str(tmp_path), molecules))
    assert store.canonical is True
    _, arrs = store.read_shard(0)
    R = np.asarray(arrs["rot"])
    assert R.shape == (len(molecules), 3, 3)
    assert R.dtype == np.float32                       # float32 regardless of store dtype
    assert np.allclose(np.matmul(R, np.transpose(R, (0, 2, 1))),
                       np.eye(3)[None], atol=2e-5), "rot is not orthogonal"
    assert np.all(np.linalg.det(R) > 0), "rot contains a reflection"


def _rescore_hits(hits, molecules, query):
    """``(reported, re-scored)`` for each hit: apply its own transform, recompute the Tanimoto.

    The store is pre-centred and ``screen`` centres the query (``_centered_copy``), so both sides
    are centred here to match. ``points @ R.T + t`` is the repo's convention
    (``alignment/utils/se3.py::apply_SE3_transform``).
    """
    from shepherd_score.score.gaussian_overlap import shape_tanimoto
    q = np.asarray(query.atom_pos, dtype=np.float64)
    qt = torch.as_tensor(q - q.mean(0), dtype=torch.float64)
    out = []
    for h in hits:
        fit = np.asarray(molecules[h.id].atom_pos, dtype=np.float64)
        fit = fit - fit.mean(0)
        T = np.asarray(h.transform, dtype=np.float64)
        posed = fit @ T[:3, :3].T + T[:3, 3]
        out.append((h.id, h.score, float(shape_tanimoto(qt, torch.as_tensor(posed), 0.81))))
    return out


@pytest.mark.parametrize("canonical", [False, True])
def test_transform_is_in_the_molecule_frame_cpu(tmp_path, molecules, canonical):
    """A returned pose must reproduce its own score -- on BOTH store kinds, on the CPU route.

    This is the test that fails if ``_compose_rot`` is removed: the canonical store's pose would
    be expressed against the rotated coordinates, so re-scoring it in the molecule's own frame
    gives a different Tanimoto than the one reported. Measured with the composition disabled:
    the canonical leg drifts up to 0.22 while the legacy leg stays at 1.7e-07 -- so this also
    pins the composition as a genuine no-op on non-canonical stores rather than a global fudge.

    Running BOTH legs matters. Score-based checks cannot see this defect at all: with the
    composition disabled the self-copy still REPORTS 1.000000 while its pose re-scores to 0.9068.

    CPU (``backend="torch"``, the non-fast object route) so it runs everywhere; the GPU routes
    are covered by the ``cuda``-marked twin below.
    """
    path = _canonical_store(str(tmp_path), molecules) if canonical         else _plain_store(str(tmp_path), molecules)
    hits = screen(molecules[0], ProfileStore.open(path), mode="vol", backend="torch", top_k=4)
    assert hits, "screen returned nothing"
    for mol_id, reported, rescored in _rescore_hits(hits, molecules, molecules[0]):
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f} "
            f"-- the transform is not in the molecule's frame (canonical={canonical})")


def test_canonical_store_screens_without_the_array_path(monkeypatch, tmp_path, molecules):
    """A canonical store must not REQUIRE ``FSS_SCREEN_ARRAYS``.

    ``const_seeds`` is a parameter of ``align_batch_vol_arrays`` alone; the object path's
    ``_align_batch_vol`` has no such keyword. Before the gate, a canonical store raised
    ``TypeError: _align_batch_vol() got an unexpected keyword argument 'const_seeds'`` on every
    route but the array one -- caught only because this suite exercises the CPU route, since the
    benchmarks always run with the array path enabled.
    """
    from shepherd_score.accel.batch import _arrays
    monkeypatch.setattr(_arrays, "ENABLED", False, raising=True)
    hits = screen(molecules[0], ProfileStore.open(_canonical_store(str(tmp_path), molecules)),
                  mode="vol", backend="torch", top_k=3)
    assert len(hits) == 3


@pytest.mark.cuda
def test_canonical_transform_is_in_the_molecule_frame(tmp_path, molecules):
    """Applying the returned transform to the molecule's own coords must reproduce its score.

    GPU twin of ``test_transform_is_in_the_molecule_frame_cpu``. Same invariant, but over the
    triton route, where the pose comes from the array path's batched SE(3) epilogue rather than
    from a pair object -- a different composition site (``offer_row`` vs ``offer_pair``).
    """
    _require_fast_cuda()
    path = _canonical_store(str(tmp_path), molecules)
    hits = screen(molecules[0], ProfileStore.open(path), mode="vol", backend="triton", top_k=5)
    assert hits, "screen returned nothing"
    for mol_id, reported, rescored in _rescore_hits(hits, molecules, molecules[0]):
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f} "
            f"-- the transform is not in the molecule's frame")
