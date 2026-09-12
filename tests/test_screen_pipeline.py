"""Gate for the screen's pipeline changes: pose-capped sub-batching, deferred canonical
composition, pinned device-side shard upload, and the memory-mappable shard format.

Every one of these is a claim of the form "same answer, less time", so what has to be pinned is
the SAME ANSWER half. Three of them cannot change a score even in principle (they move work
between host and device, defer it, or change where bytes are stored) and one -- the pose cap --
changes WHICH FINE LOOP RUNS,
which is exactly why it needs a test: before it, a sub-batch was sized from free GPU memory
alone, so a bucket that happened to fit in one chunk exceeded ``graph_cap`` and silently took
the eager loop instead of the CUDA graph. The two paths do not agree (the graph replay carries
``_GRAPH_ES_MARGIN`` extra blocks of early-stop patience), so the screen's scores depended on
what the allocator was holding. ``test_pose_cap_graphs_every_chunk_on_a_narrow_fixture`` is the
assertion that the choice is now a function of the band alone -- note it is NOT an assertion that
everything graphs, which is false above N_pad*M_pad = 3,662; see that test's docstring.

The CPU-only tests here run everywhere; the rest need CUDA + Triton and skip without them.
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
import shepherd_score.screen as scr                             # noqa: E402


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


# Alkanes so the library crosses _band_key's multiples of 16 (bands 16/32/48), the only place
# the span/pad arithmetic differs -- see test_screen_arrays.py for why drug-like sizes alone
# leave that untested.
_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"] + \
          ["C" * n for n in (12, 18, 20, 26, 30, 34)]


def _build_molecule(smi, seed, S=64):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
    rng = np.random.default_rng(seed)
    return Molecule(m,
                    surface_points=rng.standard_normal((S, 3)).astype(np.float32) * 3.0,
                    electrostatics=rng.standard_normal((S,)).astype(np.float32),
                    pharm_multi_vector=False)


@pytest.fixture(scope="module")
def molecules():
    return [_build_molecule(smi, seed=i) for i, smi in enumerate(_SMILES)]


@pytest.fixture(scope="module")
def canon_store(tmp_path_factory, molecules):
    p = os.path.join(tmp_path_factory.mktemp("pipeline"), "canon.fss")
    with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float16",
                             pre_centered=True, canonical=True, shard_size=4) as store:
        for i, m in enumerate(molecules):
            store.add(m, id=i)
    return p


# ---------------------------------------------------------------------------------------
# 1) deferred canonical composition  (no GPU needed -- it is pure heap bookkeeping)
# ---------------------------------------------------------------------------------------

def test_deferred_composition_matches_composing_at_offer_time():
    """``offer_row`` now stores the pending ``(row, rot_row)``; ``sorted()`` composes it.

    The check is against the old behaviour computed inline, and it is only meaningful if the
    composition actually does something -- so the rotation is deliberately non-identity and the
    test asserts the composed transform DIFFERS from the raw row. Without that, a regression
    that dropped the composition entirely would still pass.
    """
    rng = np.random.default_rng(0)
    K = 25
    T = rng.standard_normal((K, 4, 4)).astype(np.float32)
    # proper rotations via QR, so `rot` is the kind of matrix a canonical store writes
    R = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(K)]).astype(np.float32)
    R[np.linalg.det(R) < 0] *= -1.0
    scores = rng.random(K)

    heap = scr._TopK(5)
    for i in range(K):
        heap.offer_row(float(scores[i]), i, T, i, R)
    got = {h.id: np.asarray(h.transform) for h in heap.sorted()}

    ref = scr._TopK(5)
    for i in range(K):
        if len(ref.heap) < ref.k or float(scores[i]) > ref.heap[0][0]:
            ref._push(float(scores[i]), i, scr._compose_rot(T[i], R[i]))
    want = {h.id: np.asarray(h.transform) for h in ref.sorted()}

    assert set(got) == set(want)
    for k in want:
        assert np.array_equal(got[k], want[k]), f"transform {k} differs"
        assert not np.array_equal(got[k][:3, :3], T[k][:3, :3]), \
            "composition was a no-op -- this test would not catch its removal"


def test_deferred_composition_is_a_noop_without_a_rotation():
    """A non-canonical store passes ``rot=None`` and must still get the raw row back."""
    rng = np.random.default_rng(1)
    T = rng.standard_normal((8, 4, 4)).astype(np.float32)
    heap = scr._TopK(4)
    for i in range(8):
        heap.offer_row(float(i), i, T, i, None)
    for h in heap.sorted():
        assert np.array_equal(np.asarray(h.transform), T[h.id])


def test_raw_and_merge_raw_round_trip_composed_transforms():
    """``raw()`` crosses a process boundary in multi_gpu, so it must not emit pending tuples."""
    rng = np.random.default_rng(2)
    T = rng.standard_normal((6, 4, 4)).astype(np.float32)
    R = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(6)]).astype(np.float32)
    a = scr._TopK(3)
    for i in range(6):
        a.offer_row(float(i), i, T, i, R)
    raw = a.raw()
    for (_s, _i, t) in raw:
        assert isinstance(t, np.ndarray), "raw() leaked a deferred (transform, rot) tuple"
    b = scr._TopK(3)
    b.merge_raw(raw)
    assert [np.asarray(h.transform).tolist() for h in b.sorted()] == \
           [np.asarray(h.transform).tolist() for h in a.sorted()]


# ---------------------------------------------------------------------------------------
# 2) device-side float16 widening
# ---------------------------------------------------------------------------------------

def test_to_device_matches_host_widening_exactly():
    """float16 -> float32 is exact, so widening on the device must give the same bits."""
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((257, 3)) * 10).astype(np.float16)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    got = scr._to_device(a, dev, dtype=torch.float32)
    want = torch.as_tensor(a, dtype=torch.float32, device=dev)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    # already float32 -> passed straight through, not re-cast
    b = a.astype(np.float32)
    assert torch.equal(scr._to_device(b, dev, dtype=torch.float32),
                       torch.as_tensor(b, device=dev))


# ---------------------------------------------------------------------------------------
# 3) pose cap
# ---------------------------------------------------------------------------------------

@pytest.mark.cuda
def test_pose_cap_bounds_the_subbatch_and_only_when_asked():
    """``pose_cap`` is what arms it; without it the schedule is memory-derived as before.

    Opt-in matters here, not just tidiness: armed on every mode the cap measured 1.2985x on
    vol, 0.9981x on vol_esp and **0.6496x on pharm** (job 22598857), because each driver graphs
    below its own ``graph_cap`` budget. A cap that defaulted on would be a regression for two
    of the three."""
    _require_fast_cuda()
    from shepherd_score.accel.batch import _pad as padmod

    dev = torch.device("cuda")
    seen = []

    def process(start, k):
        seen.append(k)
        z = torch.zeros(k, device=dev)
        return z, torch.zeros(k, 4, device=dev), torch.zeros(k, 3, device=dev)

    cap, seeds, K = 640, 10, 1000
    try:
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        padmod._subbatched_align(process, K, key=("t", 1, 1, seeds), device=dev,
                                 pose_cap=cap, seeds=seeds)
        assert seen, "process never ran"
        assert max(seen) <= cap // seeds, f"chunk {max(seen)} exceeds the {cap // seeds} cap"
        assert sum(seen) == K

        seen.clear()
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        padmod._subbatched_align(process, K, key=("t", 1, 1, seeds), device=dev)
        assert max(seen) > cap // seeds, "cap applied without pose_cap= -- it is not opt-in"
    finally:
        padmod._PAIR_FOOTPRINT_BYTES.clear()


@pytest.mark.cuda
def test_pose_cap_graphs_every_chunk_on_a_narrow_fixture(monkeypatch, canon_store, molecules):
    """With the cap, every fine-loop call of THIS fixture reaches the CUDA graph.

    The regression the cap exists for: with the memory-derived schedule alone, whether a bucket
    graphs or falls back to eager depends on free GPU memory at that moment, and the two paths
    score differently (the graph replay carries the early-stop margin).

    READ THE PRECONDITION. This does NOT show that the cap makes graphing unconditional -- that
    is false and was measured false: a capped chunk is 81,920 poses, graph_cap(work) =
    max(2000, min(262144, 300_000_000 // work)), so it fits only while N_pad*M_pad <= 3,662.
    Band 48 graphs; band 64 (work 4,096) and band 112 both run EAGER under the same cap.
    This fixture's molecules are all narrow enough to pad to band 48, which is the ONLY reason
    graphed == calls holds here -- so the precondition is asserted below rather than assumed.
    Widen the fixture past 48 heavy atoms and this test SHOULD go red; that is the signal, not
    a bug.
    """
    _require_fast_cuda()
    from shepherd_score.accel.batch import _arrays
    from shepherd_score.accel.drivers import shape as shapemod
    from shepherd_score.accel.drivers._graphed import reset_graph_cache
    from shepherd_score.accel.batch import _pad as padmod

    monkeypatch.setattr(_arrays, "ENABLED", True)
    seen = {"calls": 0, "graphed": 0}
    _cfa, _rgf = shapemod.coarse_fine_align_many, shapemod._run_graphed_fine

    def cfa(*a, **k):
        seen["calls"] += 1
        return _cfa(*a, **k)

    def rgf(*a, **k):
        seen["graphed"] += 1
        return _rgf(*a, **k)

    monkeypatch.setattr(shapemod, "coarse_fine_align_many", cfa)
    monkeypatch.setattr(shapemod, "_run_graphed_fine", rgf)

    store = ProfileStore.open(canon_store)
    q = molecules[0]

    # The precondition this test rests on, asserted rather than assumed: every molecule must
    # pad to band 48 (work 2,304 <= 3,662) or a capped chunk legitimately runs eager and the
    # graphed == calls assertion below would be testing a false claim.
    _heavy = max(int((m.atom_pos_noH if hasattr(m, "atom_pos_noH") else m.atom_pos).shape[0])
                 for m in molecules + [q])
    assert _heavy <= 48, (
        f"fixture has a {_heavy}-heavy-atom molecule; above 48 the band exceeds graph_cap and "
        "eager chunks are CORRECT. Re-derive the expectation instead of widening this bound.")

    reset_graph_cache()
    hits = screen(q, store, mode="vol", backend="triton", top_k=5)
    assert seen["calls"] > 0, "the fine loop never ran -- the spy missed the call site"
    assert seen["graphed"] == seen["calls"], \
        f"{seen['calls'] - seen['graphed']} of {seen['calls']} fine-loop calls skipped the graph"

    # the cap is a schedule, not a score: a different (smaller) cap must give the same answer
    n = len(store)
    a = np.full(n, np.nan); screen(q, store, mode="vol", backend="triton", top_k=5, scores_out=a)
    old = padmod._FINE_CHUNK_POSES
    try:
        padmod._FINE_CHUNK_POSES = max(10, old // 8)
        padmod._PAIR_FOOTPRINT_BYTES.clear()
        reset_graph_cache()
        b = np.full(n, np.nan)
        hits2 = screen(q, store, mode="vol", backend="triton", top_k=5, scores_out=b)
    finally:
        padmod._FINE_CHUNK_POSES = old
        padmod._PAIR_FOOTPRINT_BYTES.clear()
    ok = ~(np.isnan(a) | np.isnan(b))
    assert np.array_equal(a[ok], b[ok]), "the sub-batch size changed a score"
    assert [h.id for h in hits] == [h.id for h in hits2]


# ---------------------------------------------------------------------------------------
# 4) upload-ahead
# ---------------------------------------------------------------------------------------

@pytest.mark.cuda
def test_shard_upload_stages_through_pinned_memory(monkeypatch, canon_store):
    """The shard upload must go through a PINNED buffer and retype on the device.

    Not a style point. Pageable is what made it slow (0.149 us/mol = 12.1% of the wall at
    N=1,000,000 at an effective ~1.2 GB/s), and a float32 request here would widen on the host
    and push twice the bytes. The float16 -> float32 result must still be exact.
    """
    _require_fast_cuda()
    store = ProfileStore.open(canon_store)
    _, arrs = store.read_shard(0)
    dev = torch.device("cuda")
    scr._PIN_STAGE.clear()

    pinned = []
    real_empty = torch.empty

    def spy(*a, **k):
        if k.get("pin_memory"):
            pinned.append(True)
        return real_empty(*a, **k)

    monkeypatch.setattr(torch, "empty", spy)
    ids, pos, off = scr._build_fit_arrays_vol(arrs, dev)
    monkeypatch.undo()
    assert pinned, "the shard upload did not stage through pinned memory"
    assert pos.is_cuda and pos.dtype == torch.float32
    assert off.is_cuda and off.dtype == torch.int64
    # exact against the host-widened path it replaces
    want = torch.as_tensor(arrs["atom_pos"], dtype=torch.float32, device=dev)
    assert torch.equal(pos, want)
    assert torch.equal(off, torch.as_tensor(arrs["atom_off"], dtype=torch.long, device=dev))


@pytest.mark.cuda
def test_memory_mapped_shards_match_the_zip_format(tmp_path, molecules):
    """The .npy shard format must hand back exactly what the .npz one did.

    It is a format change, so the check is on the BYTES, and on both shards' worth: a store
    written either way must read back identical arrays and screen to identical scores.
    """
    out = {}
    for fmt in ("npz", "npy"):
        p = os.path.join(str(tmp_path), f"{fmt}.fss")
        with ProfileStore.create(p, num_surf_points=64, modes=("vol",), dtype="float16",
                                 pre_centered=True, canonical=True, shard_size=4,
                                 shard_format=fmt) as st:
            for i, m in enumerate(molecules):
                st.add(m, id=i)
        out[fmt] = ProfileStore.open(p)
    a, b = out["npz"], out["npy"]
    assert a.num_shards == b.num_shards > 1
    for i in range(a.num_shards):
        _, ra = a.read_shard(i)
        _, rb = b.read_shard(i)
        assert set(ra) == set(rb)
        for k in ra:
            assert np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])), k

    _require_fast_cuda()
    from shepherd_score.accel.batch import _arrays
    from shepherd_score.accel.drivers._graphed import reset_graph_cache
    prev, _arrays.ENABLED = _arrays.ENABLED, True
    try:
        q = molecules[0]
        sa = np.full(len(a), np.nan); reset_graph_cache()
        ha = screen(q, a, mode="vol", backend="triton", top_k=5, scores_out=sa)
        sb = np.full(len(b), np.nan); reset_graph_cache()
        hb = screen(q, b, mode="vol", backend="triton", top_k=5, scores_out=sb)
    finally:
        _arrays.ENABLED = prev
    ok = ~(np.isnan(sa) | np.isnan(sb))
    assert np.array_equal(sa[ok], sb[ok]), "the shard format changed a score"
    assert [h.id for h in ha] == [h.id for h in hb]
