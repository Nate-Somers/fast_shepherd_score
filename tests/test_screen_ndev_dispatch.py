"""Regression: the multi-GPU screen worker must dispatch PER MODE, like the in-process driver.

``screen(ndev>1)`` fans shards out to :func:`shepherd_score.screen._screen_worker`, one process
per GPU. On the array-native path that worker called ``_build_fit_arrays_vol`` and the vol
aligner UNCONDITIONALLY -- it had no mode branch at all -- while :func:`_run_shards_inproc`
branched correctly. Every non-vol array mode therefore came back with vol answers under its own
name. Replaying the worker's two lines on CPU for ``vol_color`` gave max|delta| 3.0654e-01
against a real vol_color screen, a completely different top-10, and results bit-identical to a
real vol screen; ``pharm`` was the only mode that failed loudly (KeyError, its ref dict carrying
no ``_ref_xyz_t``). ``_arrays.ENABLED`` is True in production and nothing reads the environment,
so that was every multi-GPU screen in the four non-vol array modes -- silently wrong, not slow.

WHY THIS RUNS WITHOUT A GPU. The defect is a dispatch defect, not a kernel defect, and the lines
that carry it sit above every CUDA call in the worker. So the worker is driven here IN PROCESS
with its GPU-facing edges stubbed (device selection, thread capping, the store read, the
builders, the aligners, the reduce), and the assertion is on WHICH CALLABLE PAIR it selected --
which must be the pair :func:`_run_shards_inproc` selects for the same mode. A test that needed
two real GPUs would never have run on the machine this bug was found on, and is worth little as
the guard; the one genuinely end-to-end check below is marked ``cuda`` and skipped without two.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import shepherd_score.screen as scr                               # noqa: E402

try:                                    # same two-step import screen.py itself uses
    from shepherd_score.accel.batch import _DISPATCH_LOCAL        # noqa: E402
except Exception:                                                 # pragma: no cover
    from shepherd_score.container._core import _DISPATCH_LOCAL    # noqa: E402

#: Only ``steps_fine`` is read unconditionally by the real aligners; the recorders below ignore
#: it. Present so the call shape matches what ``_fast_batch_kwargs`` hands the drivers.
_BATCH_KW = {"steps_fine": 1}


class _FakeStore:
    """One empty shard, never canonical. The builders are stubbed, so the arrays are unused."""

    canonical = False
    path = "<fake>"

    @staticmethod
    def open(path):
        return _FakeStore()

    def read_shard(self, idx):
        return {"start": 0, "n": 1}, {}

    def __len__(self):
        return 1


def _install_recorders(mp, seen):
    """Point every builder and aligner at a recorder that reports WHICH MODE's entry ran.

    The tables are replaced wholesale AND the two pre-fix module globals are replaced with the
    same recorders, so the pre-fix code records its (wrong) choice and fails on the assertion
    instead of dying inside a Triton kernel it could never reach on a CPU box.
    """
    def builder(key):
        def build(arrs, device):
            seen.append(("build", key))
            return "ids", "fit0", "fit1"
        return build

    def aligner(key):
        # ``*a`` on purpose: the pre-fix vol aligner took
        # ``(ref_xyz, fit_flat, fit_off, mode, batch_kw)`` and the fixed one takes
        # ``(ref, fit, batch_kw)``. The recorder has to survive both call shapes or a
        # regression would surface as a TypeError rather than as the assertion below.
        def align(*a, **k):
            seen.append(("align", key))
            return np.zeros(1), np.zeros((1, 4, 4), dtype=np.float32)
        return align

    mp.setattr(scr, "_ARRAY_BUILDERS", {m: builder(m) for m in scr._ARRAY_MODES}, raising=True)
    mp.setattr(scr, "_ARRAY_ALIGNERS", {m: aligner(m) for m in scr._ARRAY_MODES}, raising=True)
    mp.setattr(scr, "_build_fit_arrays_vol", builder("vol"), raising=True)
    mp.setattr(scr, "_align_fast_arrays", aligner("vol"), raising=False)   # gone post-fix
    mp.setattr(scr, "_accumulate_arrays", lambda *a, **k: None, raising=True)
    mp.setattr(scr, "_ref_tensors_from_arrays",
               lambda ra, mode, device: {"_ref_xyz_t": None}, raising=True)


def _run_worker(mp, mode):
    """Drive ``_screen_worker`` for one mode on one shard and return what it selected."""
    seen = []
    _install_recorders(mp, seen)
    mp.setattr(scr, "ProfileStore", _FakeStore, raising=True)
    mp.setattr(torch.cuda, "set_device", lambda *a, **k: None, raising=True)
    mp.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=True)
    # _cap_threads calls torch.set_num_threads, which is process-global. The worker normally
    # owns its own spawned process; here it is running inside the test process.
    import shepherd_score.accel.multi_gpu as mg
    mp.setattr(mg, "_cap_threads", lambda threads: None, raising=True)

    class _ShardQ:
        def __init__(self):
            self._q = [0, None]                 # one shard, then the stop sentinel

        def get(self):
            return self._q.pop(0)

    class _OutQ:
        def __init__(self):
            self.msgs = []

        def put(self, msg):
            self.msgs.append(msg)

    out = _OutQ()
    # The worker sets _DISPATCH_LOCAL.active and never restores it -- correct in the spawned
    # process it was written for, not here.
    prev = getattr(_DISPATCH_LOCAL, "active", False)
    try:
        scr._screen_worker(0, 1, "<fake>", [{}], mode, _BATCH_KW, 1, _ShardQ(), out)
    finally:
        _DISPATCH_LOCAL.active = prev

    assert out.msgs, f"_screen_worker returned nothing for mode={mode}"
    msg = out.msgs[0]
    if len(msg) == 3 and msg[1] == "__ERR__":
        pytest.fail(f"_screen_worker raised for mode={mode}:\n{msg[2]}")
    return seen


def _run_inproc(mp, mode):
    """Drive ``_run_shards_inproc`` over the same one shard and return what it selected."""
    seen = []
    _install_recorders(mp, seen)
    scr._run_shards_inproc(_FakeStore(), [0], [{}], mode, torch.device("cpu"), 1,
                           _BATCH_KW, {}, "numba", True, False, None, False, 1)
    return seen


def test_array_tables_cover_every_array_mode():
    """``_array_dispatch`` is the one selector, so the tables it reads must be complete.

    A mode added to ``_ARRAY_MODES`` without both entries would raise KeyError mid-screen, and
    the identity check pins the helper to the tables -- tests/test_screen_arrays.py intercepts
    the builders by patching ``_ARRAY_BUILDERS``, which only works while the dispatch reads it.
    """
    assert set(scr._ARRAY_BUILDERS) == set(scr._ARRAY_MODES)
    assert set(scr._ARRAY_ALIGNERS) == set(scr._ARRAY_MODES)
    for mode in scr._ARRAY_MODES:
        assert scr._array_dispatch(mode) == (scr._ARRAY_BUILDERS[mode],
                                             scr._ARRAY_ALIGNERS[mode])


@pytest.mark.parametrize("mode", scr._ARRAY_MODES)
def test_worker_selects_the_same_pair_as_the_inproc_driver(monkeypatch, mode):
    """THE regression. The ndev>1 worker must pick the mode's own builder and aligner.

    Pre-fix this failed for every mode but ``vol``: the worker recorded
    ``[("build", "vol"), ("align", "vol")]`` whatever mode it was handed.
    """
    with monkeypatch.context() as mp:
        worker_seen = _run_worker(mp, mode)
    with monkeypatch.context() as mp:
        inproc_seen = _run_inproc(mp, mode)

    assert worker_seen == [("build", mode), ("align", mode)], \
        f"the ndev>1 worker selected {worker_seen} for mode={mode}"
    assert worker_seen == inproc_seen, \
        f"worker selected {worker_seen}, in-process driver selected {inproc_seen}"


def test_use_arrays_is_on_for_every_mode_this_file_drives():
    """Both drivers only reach the dispatch behind ``_use_arrays``; if that is off in this
    environment the parametrized test above is asserting on a branch nobody took."""
    from shepherd_score.accel.batch import _arrays
    assert _arrays.ENABLED, "the array path is disabled; the dispatch test would be vacuous"
    for mode in scr._ARRAY_MODES:
        assert scr._use_arrays(mode) is True


# ------------------------------------------------------------------------------------------
# End-to-end, and genuinely GPU-only: it needs TWO devices for _screen_many_multigpu to run.
# ------------------------------------------------------------------------------------------
_SMILES = ["CCO", "C1CCCCC1", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1",
           "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
           "C" * 18, "C" * 26, "C" * 34]


def _require_two_gpus():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("ndev>1 needs two CUDA devices")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


@pytest.fixture(scope="module")
def two_gpu_store(tmp_path_factory):
    """Same construction as tests/test_screen_arrays.py: real conformers, synthetic surfaces."""
    _require_two_gpus()             # skip BEFORE embedding nine conformers on a CPU-only box
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from shepherd_score.container import Molecule

    mols = []
    for i, smi in enumerate(_SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        params = AllChem.ETKDGv3()
        params.randomSeed = i
        assert AllChem.EmbedMolecule(m, params) == 0, f"embed failed for {smi}"
        rng = np.random.default_rng(i)
        mols.append(Molecule(m,
                             surface_points=rng.standard_normal((64, 3)).astype(np.float32) * 3.0,
                             electrostatics=rng.standard_normal((64,)).astype(np.float32),
                             pharm_multi_vector=False))
    p = os.path.join(tmp_path_factory.mktemp("ndev"), "lib.fss")
    # shard_size=3 over 9 molecules so both workers actually get shards off the queue.
    with scr.ProfileStore.create(p, num_surf_points=64, modes=("vol", "vol_color"),
                                 dtype="float32", pre_centered=True, shard_size=3) as store:
        for i, m in enumerate(mols):
            store.add(m, id=i)
    return p, mols


@pytest.mark.cuda
def test_two_gpu_screen_matches_the_single_process_screen(two_gpu_store):
    """A vol_color screen must not depend on how many GPUs it was spread across.

    The vacuousness guard matters here: pre-fix, ndev=2 vol_color returned the VOL answer, so
    the test also asserts the two modes disagree on this library. If they ever agree, this
    check proves nothing and should be given a harder query.
    """
    _require_two_gpus()
    store_path, mols = two_gpu_store
    query = mols[1]
    n = len(scr.ProfileStore.open(store_path))

    one = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol_color",
                     backend="triton", top_k=n, max_num_steps=30)
    two = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol_color",
                     backend="triton", top_k=n, max_num_steps=30, ndev=2)
    vol = scr.screen(query, scr.ProfileStore.open(store_path), mode="vol",
                     backend="triton", top_k=n, max_num_steps=30)

    got = {h.id: h.score for h in two}
    want = {h.id: h.score for h in one}
    volw = {h.id: h.score for h in vol}
    assert set(got) == set(want)
    assert max(abs(got[i] - want[i]) for i in want) < 1e-6, \
        f"ndev=2 disagrees with ndev=1: max|delta| {max(abs(got[i] - want[i]) for i in want):.4e}"
    assert max(abs(volw[i] - want[i]) for i in want) > 1e-3, \
        "vol and vol_color score this library identically; the check above is vacuous"
