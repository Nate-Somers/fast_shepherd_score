"""``_batch_upload`` keys its cache on the MOLECULE, not the pair.

WHY THIS EXISTS. An all-vs-all workload draws K pairs from a much smaller set of distinct
molecules: ``workloads.pairwise_pairs`` picks the smallest ``m`` with ``m*(m-1) >= K``, so
K=100,000 pairs come from m=317 compounds and every molecule appears in ~632 pairs. Keyed on the
pair, ``_batch_upload`` concatenated, uploaded and cloned each molecule once per appearance.
Measured share of a pairwise batch spent in that one call: 61.3% for ``vol_color`` (six arrays per
pair) and 11.8% for ``vol`` (two).

WHAT MUST HOLD, and what each test pins:

  1. BIT-IDENTICAL. Only the concat order and length change; the dtype cast is elementwise and
     goes through torch, so every molecule's bytes convert exactly as before. A change here moves
     P3's published scores, so "close enough" is not the bar -- the test asserts EQUALITY.
  2. THE DEDUP ACTUALLY HAPPENS. A parity test alone would pass just as happily if the keying
     silently did nothing, so the tests count what reached ``np.concatenate`` and assert pairs
     sharing a molecule share one tensor OBJECT.
  3. NO CROSS-MOLECULE ALIASING. Distinct molecules must still get distinct storage -- that is
     the half of the old "no shared-buffer aliasing" rule that still matters.
  4. THE SCREEN PATH IS UNTOUCHED. ``build_fit`` pre-warms these attributes, so ``cold`` is empty
     and the new branch never runs; a warm pair must not be re-uploaded.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from shepherd_score.accel.batch import aligners as amod          # noqa: E402


class _Mol:
    """Minimal stand-in for a Molecule: ``_batch_upload`` only ever reads array attributes."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.atom_pos = rng.standard_normal((n, 3)).astype(np.float64)   # float64 on purpose:
        self.pharm_types = rng.integers(0, 5, size=n).astype(np.int64)   # exercises the torch cast
        self.pharm_ancs = rng.standard_normal((n, 3)).astype(np.float64)


class _Pair:
    __slots__ = ("ref_molec", "fit_molec", "_ref_xyz_t", "_fit_xyz_t",
                 "_ref_pharm_types_t", "_fit_pharm_types_t")

    def __init__(self, ref, fit):
        self.ref_molec = ref
        self.fit_molec = fit
        self._ref_xyz_t = self._fit_xyz_t = None
        self._ref_pharm_types_t = self._fit_pharm_types_t = None


def _library(n_mols=6, seed=0):
    return [_Mol(n=4 + i, seed=seed + i) for i in range(n_mols)]


def _all_vs_all(mols, k=None):
    """The real workload's shape: every ordered (i, j), i != j -- so molecules REPEAT."""
    prs = [_Pair(mols[i], mols[j])
           for i in range(len(mols)) for j in range(len(mols)) if i != j]
    return prs if k is None else prs[:k]


def _count_concat_inputs(monkeypatch):
    """Record how many arrays each ``np.concatenate`` received -- the anti-vacuity probe."""
    seen = []
    real = np.concatenate

    def spy(arrs, *a, **k):
        arrs = list(arrs)
        seen.append(len(arrs))
        return real(arrs, *a, **k)

    monkeypatch.setattr(amod.np, "concatenate", spy)
    return seen


def test_uploads_once_per_molecule_not_once_per_pair(monkeypatch):
    """30 pairs over 6 molecules must concatenate 6 arrays, not 30."""
    mols = _library(6)
    pairs = _all_vs_all(mols)
    assert len(pairs) == 30, "fixture must have molecule REUSE or it proves nothing"
    seen = _count_concat_inputs(monkeypatch)
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos,
                       torch.float32, torch.device("cpu"))
    assert seen == [6], f"expected one array per distinct molecule, got {seen}"


def test_pairs_sharing_a_molecule_share_one_tensor(monkeypatch):
    mols = _library(6)
    pairs = _all_vs_all(mols)
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos,
                       torch.float32, torch.device("cpu"))
    by_mol = {}
    for p in pairs:
        by_mol.setdefault(id(p.ref_molec), []).append(p._ref_xyz_t)
    for tensors in by_mol.values():
        assert all(t is tensors[0] for t in tensors), "same molecule got multiple tensors"
    # ...and DIFFERENT molecules must not alias each other's storage.
    firsts = [ts[0] for ts in by_mol.values()]
    ptrs = {t.data_ptr() for t in firsts}
    assert len(ptrs) == len(firsts), "distinct molecules alias the same storage"


@pytest.mark.parametrize("attr,src,dtype", [
    ("_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32),
    ("_fit_xyz_t", lambda p: p.fit_molec.atom_pos, torch.float32),
    ("_ref_pharm_types_t", lambda p: p.ref_molec.pharm_types, torch.int64),
])
def test_bit_identical_to_the_per_pair_upload(attr, src, dtype):
    """Every pair must end up with EXACTLY the tensor the old per-pair path produced.

    The reference here is ``torch.as_tensor(src, dtype=..., device=...)``, which is what the
    docstring's rules were written to reproduce -- not the previous implementation, so this stays
    a check against the contract rather than against whatever the code happens to do.
    """
    dev = torch.device("cpu")
    mols = _library(6)
    pairs = _all_vs_all(mols)
    amod._batch_upload(pairs, attr, src, dtype, dev)
    for p in pairs:
        want = torch.as_tensor(src(p), dtype=dtype, device=dev)
        got = getattr(p, attr)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want), f"{attr} differs from the per-pair upload"


def test_warm_pairs_are_not_reuploaded(monkeypatch):
    """The screen path pre-warms these attributes; a warm pair must not touch the upload at all."""
    mols = _library(4)
    pairs = _all_vs_all(mols)
    dev = torch.device("cpu")
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, dev)
    before = {id(p): p._ref_xyz_t for p in pairs}
    seen = _count_concat_inputs(monkeypatch)
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, dev)
    assert seen == [], "a warm cache still concatenated something"
    for p in pairs:
        assert p._ref_xyz_t is before[id(p)], "a warm pair was re-uploaded"


def test_dedup_survives_a_partially_warm_batch():
    """A mixed batch (some pairs warm, some cold) must still land every pair on the right data.

    This is the case a naive rewrite gets wrong: ``cold`` is a SUBSET, so keying must be computed
    over that subset while the warm pairs are left exactly as they were.
    """
    dev = torch.device("cpu")
    mols = _library(5)
    pairs = _all_vs_all(mols)
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, dev)
    warm = {id(p): p._ref_xyz_t for p in pairs}
    for p in pairs[::2]:                                   # cool half of them back down
        p._ref_xyz_t = None
    amod._batch_upload(pairs, "_ref_xyz_t", lambda p: p.ref_molec.atom_pos, torch.float32, dev)
    for p in pairs:
        want = torch.as_tensor(p.ref_molec.atom_pos, dtype=torch.float32, device=dev)
        assert torch.equal(p._ref_xyz_t, want)
    for p in pairs[1::2]:
        assert p._ref_xyz_t is warm[id(p)], "an untouched warm pair was replaced"
