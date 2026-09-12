"""Gate 5 for the array-native screen path (``FSS_SCREEN_ARRAYS=1``).

``ea46a2e`` added ``accel/batch/_arrays.py`` plus the ``screen.py`` dispatch and asserted
"bit-identical" in its commit message, but shipped no test; ``_arrays.py`` still says "Default OFF
until the gates pass". This is that gate: the array path must produce byte-for-byte the same
scores as the object path it replaces, or it is not the re-expression it claims to be.

EVERY ARRAY MODE, DERIVED. ``_ARRAY_MODES`` grew from five to eleven (``vol_tversky``,
``vol_esp_tversky``, ``surf``, ``surf_esp``, ``vol_lipo``, ``vol_fukui`` joined), and this file
covered the old five BY NAME -- three hardcoded lists plus a store fixture built for five modes,
so the six new ones had no repo coverage at all and the negative control below asserted that two
of them were *not* array modes. Everything mode-shaped here is now read off
``screen.py::_ARRAY_MODES`` at import, and the store is built for that same tuple, so a mode
added there is covered by this file on the next run instead of being forgotten. The six were
measured bit-identical at N=99,984 on an L40S (jobs 22640575 / 22641030 / 22641516: 0 of 99,984
scores moved, max|delta| 0.000e+00); this file is the standing check that they stay that way.

WHY THE SPIES. A parity test here can pass while proving nothing. ``_use_arrays`` gates on
``mode in _ARRAY_MODES`` AND ``_arrays.ENABLED``, so a mistake in either silently compares the
object path against itself and reports success. These tests therefore assert WHICH BUILDER RAN,
not merely that two vectors matched -- and each mode has its own builder, so the spy set is read
off ``_ARRAY_BUILDERS`` (every entry wrapped) rather than named one by one.

WHY THE PARITY GATE ALSO RUNS WITHOUT A GPU. ``fast`` (screen.py::screen_many) needs a
pre-centered store and a ``_FAST_MODES`` mode; it does NOT need CUDA, and ``_use_arrays`` never
looks at the device -- so ``backend="numba"`` takes the same ``_array_dispatch`` branch on the
CPU. Measured here on this fixture: all eleven modes, both legs, 0 of 12 scores moved with
max|delta| 0.000e+00 and identical top-k order, the spies confirming the ON leg entered the array
builder and the OFF leg the object builder. The CPU twin is what gives the six new modes a gate
on a machine with no GPU; the ``cuda``-marked twin stays because the GPU is where the kernels
differ.

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

import shepherd_score.screen as screenmod                       # noqa: E402
from shepherd_score.accel._modes import CANONICAL_MODES         # noqa: E402
from shepherd_score.container import Molecule                   # noqa: E402
from shepherd_score.screen import ProfileStore, screen          # noqa: E402

_BAND = 16          # mirrors accel.batch._pad._BAND; asserted against the real one below

#: THE list every mode-shaped thing in this file is driven off: both parametrize lists, the store
#: fixture's ``modes=``, and the negative control's complement. Snapshotted at import because
#: ``pytest.mark.parametrize`` needs a concrete sequence; the store guard below re-reads
#: ``screenmod._ARRAY_MODES`` so a mode added after import still fails loudly rather than
#: silently going untested.
_MODES = tuple(screenmod._ARRAY_MODES)

#: the documented library defaults, matching aligners/fss.py::prepare_screen. vol_esp takes
#: lam RAW (surf_esp scales it x207) and screen() refuses to run vol_esp without it.
#: Nothing else needs an entry: ``_resolve_screen`` *requires* only ``lam`` for vol_esp and
#: ``alpha`` for vol_and_surf_esp; surf/surf_esp take alpha from ``ALPHA(num_surf_points)`` and
#: the Tversky / lipo / fukui modes all carry defaults in ``_fast_batch_kwargs``.
_MODE_KW = {"vol_esp": {"lam": 0.1}, "vol_and_surf_esp": {"alpha": 0.81}}

#: Modes whose self-copy does NOT reach ~1.0 on THIS fixture, and why. An EXCLUSION list, not the
#: positive list this file used to carry: a mode that joins ``_ARRAY_MODES`` is asserted at 1.0 by
#: default and has to be excused on purpose, which is how the positive list came to silently drop
#: six modes.
#: ``vol_and_surf_esp`` is excused because ``_build_molecule`` gives every molecule a SYNTHETIC
#: random surface (so the test needs no Open3D), which is fine for the shape/pharm channels but is
#: not the surface implied by the molecule's own atoms -- and vol_and_surf_esp scores surface and
#: ESP channels against fields derived from those atoms, so a self-copy need not reach 1.0. The
#: repo's own ``test_screen.py::test_self_screen_recovers_one`` excludes it from exactly this
#: assertion for the same reason. It still ranks itself FIRST; its self-score measured 0.5865
#: (numba) / 0.5886 (triton) on an L40S, job 22642878, and 0.5864 on a Windows CPU box -- route
#: dependent, which is why only this one mode is excused rather than the number being asserted.
#: Every other mode, both Tversky modes included, measured EXACTLY 1.000000 on both routes there.
_SELF_SCORE_NOT_ONE = {"vol_and_surf_esp"}


def _scored_asymmetrically(mode: str) -> bool:
    """Whether ``mode`` scores through an ASYMMETRIC Tversky reduction -- read off screen.py.

    Such a mode cannot promise that the self-copy RANKS first: Tversky alpha=0.95 / beta=0.05
    rewards a fit molecule that CONTAINS the query, so a bigger library molecule legitimately
    outscores it and the scores are not capped at 1. Measured on this fixture (query = library
    id 1, job 22642878, L40S): ``vol_tversky`` puts id=4 first at 1.2065 and ``vol_esp_tversky``
    puts id=5 first at 1.0709, on BOTH the numba and the triton route, while the self-copy still
    scores exactly 1.000000 in each case (so the score half of the anchor stands for them).

    Derived rather than listed, and this is why: ``vol_esp_tversky`` ranked the self-copy first
    on one CPU box (a different optimizer landing, same asymmetric score), so a list written from
    that box's numbers would have excused the wrong mode and asserted a coincidence on the other.
    ``_fast_batch_kwargs`` is the authority on which modes carry an unequal Tversky alpha/beta,
    so a future Tversky-reduced mode is excused the day it lands and a symmetric one never is.
    """
    kw = screenmod._fast_batch_kwargs(mode, dict(_MODE_KW.get(mode, {})))
    return kw.get("tversky_alpha") != kw.get("tversky_beta")


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


def _require_numba():
    """The fast CPU path runs the batched kernels through numba, and ``screen()`` raises
    ImportError without it (screen.py::screen_many) -- a missing dependency, not a failure."""
    pytest.importorskip("numba")


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
    """Real RDKit conformer + a *synthetic* surface (so the test needs no Open3D).

    ``fukui=`` is injected for the same reason the surface is synthetic. ``Molecule.fukui`` is
    generated lazily from THREE gfn2-xTB single points (neutral/cation/anion), which needs an
    ``xtb`` binary this suite cannot assume -- and unlike ``partial_charges`` it has no MMFF
    fallback, so building a ``vol_fukui`` store would die in the fixture. The constructor
    documents ``fukui=`` as exactly this seam. It must be a full ``(N,)`` array in with-H order
    (the same basis as ``partial_charges``): ``get_fukui(no_H=True)`` slices it with
    ``_nonH_atoms_idx``, so a heavy-length array would silently misalign with its centres.
    ``lipophilicity`` needs no such help -- the constructor computes it from RDKit's Crippen
    contributions, offline and for free.
    """
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    surf = rng.standard_normal((S, 3)).astype(np.float32) * 3.0
    esp = rng.standard_normal((S,)).astype(np.float32)
    fukui = rng.standard_normal((m.GetNumAtoms(),)).astype(np.float32) * 0.1
    return Molecule(m, surface_points=surf, electrostatics=esp, pharm_multi_vector=False,
                    fukui=fukui)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


@pytest.fixture(scope="module")
def store_path(tmp_path_factory, molecules):
    """One store that serves EVERY array mode.

    ``modes=_MODES`` rather than a five-name list, because the store is the binding constraint on
    what this file can test at all: a store built without the ``lipophilicity`` / ``fukui``
    schema flags makes ``store.supports("vol_lipo")`` False and ``screen()`` raises in
    ``_resolve_screen`` before any path is chosen, whatever the parametrize list says. Handing
    ``_schema_from_modes`` the array-mode tuple means a future mode's channels are stored the
    moment it joins ``_ARRAY_MODES``.
    """
    p = os.path.join(tmp_path_factory.mktemp("arrays"), "lib.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=_MODES,
                             dtype="float32", pre_centered=True) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


def _screen_recording(monkeypatch, store_path, query, *, enabled, mode="vol", steps=30,
                      backend="triton"):
    """Run one screen with the array path forced on/off, recording which builder ran.

    ``_use_arrays`` re-imports ``_arrays`` and reads ``.ENABLED`` on every call
    (screen.py::_use_arrays), so patching the ATTRIBUTE steers the dispatch. Patching the env var
    would not: ``ENABLED`` is resolved from the environment once, at import.
    """
    from shepherd_score.accel.batch import _arrays

    monkeypatch.setattr(_arrays, "ENABLED", enabled, raising=True)
    # The seam has to BITE FOR THIS MODE, and that is checked before the screen runs rather than
    # inferred from the spy counts afterwards. A mode outside ``_ARRAY_MODES`` -- or a gate that
    # stopped reading ENABLED live -- would send both legs down the same path, and the parity
    # assertion behind them would then compare the object path against itself and pass.
    assert screenmod._use_arrays(mode) is enabled, (
        f"_arrays.ENABLED={enabled} but _use_arrays({mode!r}) is "
        f"{screenmod._use_arrays(mode)}; the two legs would take the same path")

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
    # The builder names come from the TABLE (deduped, order preserved), not from a literal list:
    # that list named five builders while the table held eleven, so a new mode's builder went
    # unwrapped. Both are still patched because ``_build_fit_arrays_vol_tversky`` reaches
    # ``_build_fit_arrays_vol`` through the module global -- which double-counts that one mode,
    # harmless against assertions that read >0 rather than an exact count.
    for _fn in dict.fromkeys(screenmod._ARRAY_BUILDERS.values()):
        monkeypatch.setattr(screenmod, _fn.__name__, _count_arr(_fn), raising=True)
    monkeypatch.setattr(screenmod, "_ARRAY_BUILDERS",
                        {m: _count_arr(fn) for m, fn in screenmod._ARRAY_BUILDERS.items()},
                        raising=True)
    monkeypatch.setattr(screenmod, "_build_fit_fast_pairs", spy_obj, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans", spy_spans, raising=True)
    monkeypatch.setattr(_arrays, "plan_spans_multi", spy_spans_multi, raising=True)

    n = len(ProfileStore.open(store_path))
    scores = np.full(n, np.nan, dtype=float)
    hits = screen(query, ProfileStore.open(store_path), mode=mode, backend=backend,
                  top_k=n, max_num_steps=steps, scores_out=scores, **_MODE_KW.get(mode, {}))
    return scores, hits, seen


def _assert_paths_diverged(seen_on, seen_off):
    """The two legs really ran DIFFERENT builders. Everything else here is worthless without it."""
    assert seen_on["arrays"] > 0, "ENABLED=True did not reach the array builder"
    assert seen_on["objects"] == 0, "ENABLED=True still built _FastPair objects"
    assert seen_off["objects"] > 0, "ENABLED=False did not reach the object builder"
    assert seen_off["arrays"] == 0, "ENABLED=False still reached the array builder"
    assert seen_on["buckets"] >= 1, "plan_spans never ran on the array path"


def _assert_bit_identical(s_on, h_on, s_off, h_off):
    # --- scores are real, not a degenerate all-equal / all-NaN vector -----------------------
    assert np.isfinite(s_on).all(), "array path produced non-finite scores"
    assert np.isfinite(s_off).all(), "object path produced non-finite scores"
    assert s_on.max() - s_on.min() > 1e-6, "scores are degenerate; parity would be vacuous"

    # --- the gate itself --------------------------------------------------------------------
    np.testing.assert_array_equal(s_on, s_off)
    assert [h.id for h in h_on] == [h.id for h in h_off], "top-k identity/order diverged"
    np.testing.assert_array_equal(np.array([h.score for h in h_on], dtype=float),
                                  np.array([h.score for h in h_off], dtype=float))


def _assert_self_copy(mode, scores, hits):
    """Independent anchor: the array path must be a CORRECT screen, not merely a consistent one.

    Two paths agreeing on nonsense would satisfy parity by itself. The query is the library's own
    id=1, so RANKING is the anchor that holds for every symmetric mode, and the self-copy's own
    score is asserted at ~1.0 unless the mode is excused above. Both checks are ON by default for
    a mode that joins ``_ARRAY_MODES``.
    """
    if not _scored_asymmetrically(mode):
        assert hits[0].id == 1, "query is in the library at id=1 and must rank first"
    if mode not in _SELF_SCORE_NOT_ONE:
        assert scores[1] == pytest.approx(1.0, abs=1e-2), "self-copy must score ~1.0"


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


def test_store_fixture_serves_every_array_mode(store_path):
    """Guards the OTHER fixture, and it is the one that binds.

    A store is built from a schema (``_schema_from_modes``), so a mode whose channels are absent
    is refused by ``_resolve_screen`` before any path is chosen: the parametrized tests would
    then fail with "does not support" rather than on parity, and a store built for five modes
    cannot test eleven however the parametrize list is written. Re-reads
    ``screenmod._ARRAY_MODES`` rather than the import-time snapshot so a mode added to screen.py
    is caught here first.

    ``_FAST_MODES`` is checked in the same breath because this file's OFF leg depends on it:
    ``fast`` is what routes to ``_build_fit_fast_pairs``, and a mode in ``_ARRAY_MODES`` but not
    in ``_FAST_MODES`` would send the ENABLED=False leg down the MoleculePair profile path
    instead, where the object spy sees nothing.
    """
    store = ProfileStore.open(store_path)
    assert set(screenmod._ARRAY_MODES) == set(_MODES), \
        "_ARRAY_MODES changed after import; the store was built for the import-time tuple"
    for mode in screenmod._ARRAY_MODES:
        assert store.supports(mode), (
            f"the fixture store was built for {store.modes} and cannot serve {mode!r}; "
            f"its parity test would fail in _resolve_screen, not on parity")
        assert mode in screenmod._FAST_MODES, (
            f"{mode!r} is an array mode but not a fast mode; the ENABLED=False leg would not be "
            f"the object path this file compares against")


def test_the_self_copy_rank_anchor_is_not_excused_away():
    """``_scored_asymmetrically`` decides which modes skip the rank half of the anchor, so a
    change that made it answer True everywhere would delete that half for every mode silently.

    Deliberately one-sided: it pins that the anchor still applies to SOMEONE (and to ``vol``, a
    plain symmetric Tanimoto, by name) without pinning how many modes are excused -- a count
    would go red the day a Tversky mode is added to or dropped from ``_ARRAY_MODES``, which is
    the rot this file is being fixed for.
    """
    excused = [m for m in _MODES if _scored_asymmetrically(m)]
    assert not _scored_asymmetrically("vol"), "vol is a symmetric Tanimoto; nothing excuses it"
    assert len(excused) < len(_MODES), \
        f"every array mode is excused from the rank anchor ({excused}); it now asserts nothing"


@pytest.mark.cuda
@pytest.mark.parametrize("mode", _MODES)
def test_array_path_is_bit_identical_to_object_path(monkeypatch, store_path, molecules, mode):
    """The array path is a re-expression of the object path, so scores must match EXACTLY."""
    _require_fast_cuda()
    query = molecules[1]

    with monkeypatch.context() as mp:
        s_on, h_on, seen_on = _screen_recording(mp, store_path, query, enabled=True, mode=mode)
    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _screen_recording(mp, store_path, query, enabled=False, mode=mode)

    _assert_paths_diverged(seen_on, seen_off)
    _assert_bit_identical(s_on, h_on, s_off, h_off)


@pytest.mark.parametrize("mode", _MODES)
def test_array_path_is_bit_identical_to_object_path_cpu(monkeypatch, store_path, molecules, mode):
    """The same gate on the CPU route, where it runs on every box instead of only a GPU one.

    ``fast`` needs a pre-centered store and a ``_FAST_MODES`` mode, not a device, and
    ``_use_arrays`` never looks at one -- so ``backend="numba"`` reaches the identical
    ``_array_dispatch`` branch with ``device=cpu``. That is what gives the six modes that joined
    ``_ARRAY_MODES`` a parity gate on a machine with no GPU; the cuda twin above stays for the
    triton kernels, which this leg never enters.

    The self-copy anchor rides along here instead of in a CPU test of its own: it asserts on the
    ON-leg vector this test already has, and a second CPU screen per mode would cost the file
    eleven more screens to check the same numbers.
    """
    _require_numba()
    query = molecules[1]

    with monkeypatch.context() as mp:
        s_on, h_on, seen_on = _screen_recording(mp, store_path, query, enabled=True, mode=mode,
                                                backend="numba")
    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _screen_recording(mp, store_path, query, enabled=False,
                                                   mode=mode, backend="numba")

    _assert_paths_diverged(seen_on, seen_off)
    _assert_bit_identical(s_on, h_on, s_off, h_off)
    _assert_self_copy(mode, s_on, h_on)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", _MODES)
def test_array_path_recovers_the_self_copy(monkeypatch, store_path, molecules, mode):
    """The same anchor on the GPU route, where the scores come out of the triton kernels."""
    _require_fast_cuda()
    with monkeypatch.context() as mp:
        scores, hits, seen = _screen_recording(mp, store_path, molecules[1],
                                               enabled=True, mode=mode)
    assert seen["arrays"] > 0
    _assert_self_copy(mode, scores, hits)


def test_use_arrays_gates_on_mode_and_reads_enabled_live(monkeypatch):
    """``_use_arrays`` must gate on BOTH the mode and the live flag -- the two ways this file
    could silently stop testing anything.

    THE NEGATIVE CONTROL IS COMPUTED. It used to name ``surf`` and ``surf_esp`` as modes with no
    array-native aligner; both have one now, so it asserted something false and took the suite
    red. Naming two other modes would rot the same way on the next port, so the non-array modes
    are derived as ``CANONICAL_MODES`` minus ``_ARRAY_MODES`` (10 of the 21 today: ``vol_mr``,
    the surf/pharm/color Tversky family, ``vol_atomtype``, ``vol_pharm``, ``vol_avoid``, ...).
    An unknown name is checked alongside them so the discrimination is still proven on the day
    that difference empties out. Legacy aliases are deliberately no part of this: ``_use_arrays``
    does not canonicalize, and callers reach it past ``_canon_mode``.
    """
    from shepherd_score.accel.batch import _arrays

    non_array = tuple(m for m in CANONICAL_MODES if m not in screenmod._ARRAY_MODES)

    monkeypatch.setattr(_arrays, "ENABLED", True, raising=True)
    for mode in screenmod._ARRAY_MODES:
        assert screenmod._use_arrays(mode) is True
    for mode in non_array + ("__not_a_mode__",):
        assert screenmod._use_arrays(mode) is False, f"{mode} has no array-native aligner"
    assert set(screenmod._ARRAY_BUILDERS) == set(screenmod._ARRAY_MODES), \
        "a mode in _ARRAY_MODES with no builder entry would KeyError mid-screen"

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
