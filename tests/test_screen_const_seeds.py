"""Gate for the canonical store's constant seed set on EVERY mode that seeds from the atom cloud.

``_common.canonical_seed_quats`` used to reach one mode: ``screen._canonical_batch_kw`` gated on
``mode == "vol"``, and no other driver could take an external seed set at all. On the same
canonical store every other mode still ran ``batched_seeds_torch`` -- the per-molecule float64
eigensolve, measured at 0.82-0.91 us/mol for vol / vol_color / vol_esp -- on coordinates already
rotated into their principal frame, to rediscover a near-constant answer. The gate is now
``accel._modes.CONST_SEED_MODES`` and the drivers accept ``seeds=``.

WHAT THIS FILE ASSERTS, and why each half is needed:

* WHICH generator ran. A parity number alone cannot tell the constant-seed path from the old one
  (both produce a finite score vector and a sane ranking), so each test spies on
  ``canonical_seed_quats`` AND on ``batched_seeds_torch`` at every module that binds it, and reads
  the ``batch_kw`` the array aligner actually received. The control leg -- the same screen with
  the gate emptied -- must show the OPPOSITE counts, or the spies prove nothing.
* That the screen stays CORRECT under the constant set: the library's own molecule ranks first
  and scores ~1 (the same anchors gate 5 uses), a returned pose re-scores to its reported score
  (the transform is in the molecule's frame, which no score-based check can see), and a panel of
  queries gets the same answer per query as the single-query screen.
* That the modes seeding from OTHER clouds (surface, anchors) are left exactly as they were.

Parity with the per-molecule seeds is NOT asserted bit-for-bit: the two seed sets differ by each
molecule's own rotation, so scores legitimately move at the 1e-3 level (Shepherd-Score-Paper
fig2_speed/validate_canonical.py measured 1.3e-3 mean for vol at N=99,000 with no enrichment
change over 27 DUDE-Z targets). What is asserted is the top-1 and a rank correlation, on this
12-molecule fixture, plus the correctness anchors above.
"""
import os

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
torch = pytest.importorskip("torch")

import shepherd_score.screen as screenmod                                       # noqa: E402
from shepherd_score.accel._modes import CONST_SEED_MODES, MODE_SEEDS            # noqa: E402
from shepherd_score.screen import ProfileStore, screen, screen_many             # noqa: E402

from tests.test_screen_arrays import (                                          # noqa: E402
    _MODE_KW, _SELF_SCORE_NOT_ONE, _SMILES, _build_molecule, _esp_molecules,
    _require_fast_cuda, _require_numba, _rescore_hits_vol_esp, _scored_asymmetrically,
)

#: Read off the registry and the dispatch table, so a mode added to either is covered next run.
_CLASS_A = tuple(m for m in screenmod._ARRAY_MODES if m in CONST_SEED_MODES)
_CLASS_B = tuple(m for m in screenmod._ARRAY_MODES if m not in CONST_SEED_MODES)

#: Every module that binds ``batched_seeds_torch`` by name. The drivers import it at module level
#: (``from ._common import batched_seeds_torch``), so patching ``_common`` alone would miss them;
#: ``_arrays`` imports it inside each function, so ``_common`` is what catches those.
_SEED_MODULES = ("shepherd_score.accel.drivers._common", "shepherd_score.accel.drivers.shape",
                 "shepherd_score.accel.drivers.esp", "shepherd_score.accel.drivers.vol_color",
                 "shepherd_score.accel.drivers.vol_lipo", "shepherd_score.accel.drivers.esp_combo",
                 "shepherd_score.accel.drivers.vol_tversky",
                 "shepherd_score.accel.drivers.vol_esp_tversky",
                 "shepherd_score.accel.drivers.pharm")


def test_the_two_classes_partition_the_array_modes():
    assert set(_CLASS_A) | set(_CLASS_B) == set(screenmod._ARRAY_MODES)
    assert "vol" in _CLASS_A and {"surf", "surf_esp", "pharm"} <= set(_CLASS_B)
    assert set(CONST_SEED_MODES) <= set(screenmod._ARRAY_MODES), \
        "a constant-seed mode must have an array aligner to hand const_seeds to"


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


def _store(path, molecules, canonical):
    with ProfileStore.create(path, num_surf_points=64, modes=tuple(screenmod._ARRAY_MODES),
                             dtype="float32", pre_centered=True, canonical=canonical) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return path


@pytest.fixture(scope="module")
def canon_store(tmp_path_factory, molecules):
    return _store(os.path.join(tmp_path_factory.mktemp("cs"), "canon.fss"), molecules, True)


@pytest.fixture(scope="module")
def plain_store(tmp_path_factory, molecules):
    return _store(os.path.join(tmp_path_factory.mktemp("cs"), "plain.fss"), molecules, False)


def _spied_screen(monkeypatch, store_path, queries, mode, *, backend, gate_on=True, **kw):
    """Screen ``queries`` (one or a panel) recording which seed generator ran and what
    ``batch_kw`` reached the array aligner. Returns ``(scores (Q,N), hits, seen)``."""
    import importlib
    from shepherd_score.accel.drivers import _common

    seen = {"canonical": 0, "batched": 0, "kw": []}
    real_c = _common.canonical_seed_quats

    def spy_c(*a, **k):
        seen["canonical"] += 1
        return real_c(*a, **k)

    def _wrap(fn, counter):
        def spy_b(*a, **k):
            counter["batched"] += 1
            return fn(*a, **k)
        return spy_b

    # Wrap whatever each module currently binds -- no identity guard. The guard version silently
    # skipped the driver modules under pytest (their binding was not ``real_b`` by then), which
    # left both legs reporting zero generator calls and proved nothing; wrapping a wrapper is
    # harmless, since the inner spy counts and then calls the outer one.
    for name in _SEED_MODULES:
        mod = importlib.import_module(name)
        if hasattr(mod, "batched_seeds_torch"):
            monkeypatch.setattr(mod, "batched_seeds_torch", _wrap(mod.batched_seeds_torch, seen),
                                raising=True)
    monkeypatch.setattr(_common, "canonical_seed_quats", spy_c, raising=True)

    real_align = screenmod._ARRAY_ALIGNERS[mode]

    def spy_align(ref, fit, batch_kw):
        seen["kw"].append(batch_kw)
        return real_align(ref, fit, batch_kw)

    monkeypatch.setattr(screenmod, "_ARRAY_ALIGNERS",
                        {**screenmod._ARRAY_ALIGNERS, mode: spy_align}, raising=True)
    if not gate_on:
        monkeypatch.setattr(screenmod, "_CONST_SEED_MODES", (), raising=True)

    qs = list(queries)
    n = len(ProfileStore.open(store_path))
    scores = [np.full(n, np.nan, dtype=float) for _ in qs]
    hits = screen_many(qs, ProfileStore.open(store_path), mode=mode, backend=backend, top_k=n,
                       scores_out=scores, **{**_MODE_KW.get(mode, {}), **kw})
    scores = np.stack(scores)
    assert np.isfinite(scores).all(), "screen left holes or produced non-finite scores"
    return scores, hits, seen


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------------
# 1) which generator ran
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", _CLASS_A)
def test_constant_seeds_replace_the_eigensolve(monkeypatch, canon_store, molecules, mode):
    """On a canonical store a class-A mode must build ONE constant set per query and never call
    the per-molecule generator; with the gate emptied the same screen must do the reverse.
    Both legs on the numba route, which takes the same array dispatch as triton."""
    _require_numba()
    s_on, h_on, seen_on = _spied_screen(monkeypatch, canon_store, [molecules[1]], mode,
                                        backend="numba")
    assert seen_on["canonical"] == 1, f"{mode}: expected one constant set, got {seen_on}"
    assert seen_on["batched"] == 0, f"{mode}: the per-molecule eigensolve still ran: {seen_on}"
    assert seen_on["kw"], "the aligner spy saw no call"
    cs = seen_on["kw"][0].get("const_seeds")
    assert cs is not None and tuple(cs.shape) == (MODE_SEEDS[mode], 4), \
        f"{mode}: const_seeds missing or the wrong shape: {None if cs is None else cs.shape}"
    assert torch.allclose(cs.norm(dim=1), torch.ones(cs.shape[0], device=cs.device), atol=1e-5)

    with monkeypatch.context() as mp:
        s_off, h_off, seen_off = _spied_screen(mp, canon_store, [molecules[1]], mode,
                                               backend="numba", gate_on=False)
    assert seen_off["canonical"] == 0 and seen_off["batched"] >= 1, \
        f"{mode}: the control leg did not run the per-molecule generator: {seen_off}"
    assert "const_seeds" not in seen_off["kw"][0]

    # Same screen, two seed sets: they must agree on WHAT the answer is, not bit for bit.
    assert h_on[0][0].id == h_off[0][0].id == (1 if not _scored_asymmetrically(mode) else h_off[0][0].id)
    assert _spearman(s_on[0], s_off[0]) > 0.9, f"{mode}: ranking diverged"
    assert np.abs(s_on[0] - s_off[0]).mean() < 5e-2, f"{mode}: mean |delta| too large"


@pytest.mark.parametrize("mode", _CLASS_B)
def test_other_cloud_modes_keep_their_own_generator(monkeypatch, canon_store, molecules, mode):
    """surf / surf_esp / pharm seed from the surface or anchor PCA, which the store does not
    canonicalise, so on the SAME canonical store they must run their generator untouched."""
    _require_numba()
    _, hits, seen = _spied_screen(monkeypatch, canon_store, [molecules[1]], mode, backend="numba")
    assert seen["canonical"] == 0, f"{mode}: built a constant seed set it cannot use"
    assert seen["batched"] >= 1
    assert "const_seeds" not in seen["kw"][0]
    assert hits[0][0].id == 1


def test_vol_and_surf_esp_uses_constant_seeds_only_at_the_volumetric_alpha(monkeypatch,
                                                                            canon_store,
                                                                            molecules):
    """At alpha == 0.81 its driver seeds from the atom clouds; at any other alpha from the
    surfaces. The constant set is valid for the first and must be withheld from the second."""
    _require_numba()
    _, _, seen = _spied_screen(monkeypatch, canon_store, [molecules[1]], "vol_and_surf_esp",
                               backend="numba")
    assert seen["canonical"] == 1 and seen["batched"] == 0
    with monkeypatch.context() as mp:
        _, _, seen2 = _spied_screen(mp, canon_store, [molecules[1]], "vol_and_surf_esp",
                                    backend="numba", alpha=0.5)
    assert seen2["canonical"] == 0 and seen2["batched"] >= 1
    assert "const_seeds" not in seen2["kw"][0]


# ---------------------------------------------------------------------------------------------
# 2) the screen is still a correct screen
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", _CLASS_A)
def test_self_copy_anchor_holds_under_constant_seeds(monkeypatch, canon_store, molecules, mode):
    """Gate 5's independent anchor, on the constant-seed path: the library's own id=1 ranks first
    (symmetric modes) and scores ~1 (unless excused there)."""
    _require_numba()
    scores, hits, seen = _spied_screen(monkeypatch, canon_store, [molecules[1]], mode,
                                       backend="numba")
    assert seen["canonical"] == 1, "the anchor must be tested ON the constant-seed path"
    if not _scored_asymmetrically(mode):
        assert hits[0][0].id == 1, f"{mode}: self-copy did not rank first"
    if mode not in _SELF_SCORE_NOT_ONE:
        assert scores[0, 1] == pytest.approx(1.0, abs=1e-2), f"{mode}: self-copy scored {scores[0, 1]}"


@pytest.mark.parametrize("backend", ["numba", pytest.param("triton", marks=pytest.mark.cuda)])
def test_vol_esp_transform_is_in_the_molecule_frame_under_constant_seeds(monkeypatch, tmp_path,
                                                                          backend):
    """A returned pose must reproduce its own score, on the constant-seed path, INCLUDING the
    molecule whose RemoveHs retained an H (whose strict-heavy frame differs from the atom frame
    the store canonicalised -- the one case where the constant set is a perturbed seed rather
    than an exact one). No score-based check can see a transform in the wrong frame."""
    if backend == "triton":
        _require_fast_cuda()
    else:
        _require_numba()
    mols, retained = _esp_molecules()
    p = os.path.join(str(tmp_path), "esp_canon.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol", "vol_esp"),
                             dtype="float32", pre_centered=True, canonical=True) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)
    assert "xyz_noH" in ProfileStore.open(p).read_shard(0)[1]
    _, hits, seen = _spied_screen(monkeypatch, p, [mols[0]], "vol_esp", backend=backend)
    assert seen["canonical"] == 1 and seen["batched"] == 0
    seen_ids = set()
    for mol_id, reported, rescored in _rescore_hits_vol_esp(hits[0], mols, mols[0]):
        seen_ids.add(mol_id)
        assert abs(reported - rescored) < 2e-3, (
            f"id={mol_id} reported {reported:.6f} but its own pose re-scores to {rescored:.6f}")
    assert retained <= seen_ids, "the retained-H molecule was not among the hits"


def test_a_panel_gets_one_constant_set_per_query(monkeypatch, canon_store, molecules):
    """``screen_many`` over two queries must derive a seed set from EACH query's frame and give
    every query the answer its single-query screen gives. Before, the gate demanded one query
    and a panel silently fell back to the per-molecule generator."""
    _require_numba()
    qs = [molecules[1], molecules[4]]
    s_panel, h_panel, seen = _spied_screen(monkeypatch, canon_store, qs, "vol_color",
                                           backend="numba")
    assert seen["canonical"] == 2 and seen["batched"] == 0
    a, b = (kw["const_seeds"] for kw in seen["kw"][:2])
    assert not torch.allclose(a, b), "two different queries received the same seed set"
    for qi, q in enumerate(qs):
        with monkeypatch.context() as mp:
            s_one, h_one, _ = _spied_screen(mp, canon_store, [q], "vol_color", backend="numba")
        np.testing.assert_allclose(s_panel[qi], s_one[0], atol=1e-6, rtol=0)
        assert [h.id for h in h_panel[qi]] == [h.id for h in h_one[0]]


def test_constant_seed_screen_is_deterministic(monkeypatch, canon_store, molecules):
    """The set depends on the query alone, never on batch composition, so two runs agree bit
    for bit -- the reproducibility the per-molecule generator could not promise
    (``_masked_principal_axes`` is batch-size dependent)."""
    _require_numba()
    s1, _, _ = _spied_screen(monkeypatch, canon_store, [molecules[3]], "vol_lipo", backend="numba")
    with monkeypatch.context() as mp:
        s2, _, _ = _spied_screen(mp, canon_store, [molecules[3]], "vol_lipo", backend="numba")
    assert np.array_equal(s1, s2)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", _CLASS_A)
def test_constant_seeds_on_the_triton_route(monkeypatch, canon_store, molecules, mode):
    """The GPU twin of the generator + anchor checks: same dispatch, but the kernels differ."""
    _require_fast_cuda()
    scores, hits, seen = _spied_screen(monkeypatch, canon_store, [molecules[1]], mode,
                                       backend="triton")
    assert seen["canonical"] == 1 and seen["batched"] == 0, f"{mode}: {seen}"
    if not _scored_asymmetrically(mode):
        assert hits[0][0].id == 1
    if mode not in _SELF_SCORE_NOT_ONE:
        assert scores[0, 1] == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------------------------------------
# 3) the store default follows the registry
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", tuple(screenmod._ARRAY_MODES))
def test_store_is_canonical_by_default_exactly_for_the_constant_seed_modes(tmp_path, molecules,
                                                                             mode):
    """``canonical=None`` must resolve to True for a store serving ANY constant-seed mode -- a
    vol_esp-only store used to stay raw and forgo the speedup -- and to False for one serving only
    the other-cloud modes, which would pay the score movement for nothing."""
    p = os.path.join(str(tmp_path), f"{mode}.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=(mode,), dtype="float32") as store:
        store.add(molecules[0], id=0)
    assert ProfileStore.open(p).canonical is (mode in CONST_SEED_MODES)


def test_legacy_mode_names_resolve_in_the_store_default(tmp_path, molecules):
    """``esp_combo`` is the legacy name of vol_and_surf_esp, a constant-seed mode."""
    p = os.path.join(str(tmp_path), "legacy.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("esp_combo",), dtype="float32") as store:
        store.add(molecules[0], id=0)
    assert ProfileStore.open(p).canonical is True
