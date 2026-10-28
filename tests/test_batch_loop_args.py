"""MoleculePairBatch forwards num_repeats / lr / trans_init to the batched drivers, resolves
``None`` to the registry defaults, and warns when ``backend='jax'`` has no JAX kernel."""
import warnings

import pytest

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from shepherd_score.accel._modes import SPECS
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch


def _backend():
    """The default batched backend, skipping when it is not installed."""
    import torch
    if torch.cuda.is_available():
        pytest.importorskip("triton")
        return 'triton'
    pytest.importorskip("numba")
    return 'numba'


def _mol(smiles, seed):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(m, params)
    return Molecule(m, num_surf_points=40, charge_model='mmff')


def _batch():
    return MoleculePairBatch([MoleculePair(_mol('c1ccccc1O', 1), _mol('c1ccccc1N', 2),
                                           num_surf_points=40)])


class _Recorded(Exception):
    pass


def _record(monkeypatch, mode):
    """Replace ``MoleculePair._align_batch_<mode>`` with a stub that records its keywords."""
    seen = {}

    def stub(pairs, **kw):
        seen.update(kw)
        raise _Recorded

    monkeypatch.setattr(MoleculePair, f'_align_batch_{mode}', staticmethod(stub))
    return seen


@pytest.mark.parametrize('mode', ['vol', 'surf'])
def test_explicit_loop_args_reach_driver(monkeypatch, mode):
    seen = _record(monkeypatch, mode)
    kw = {'alpha': 0.81} if mode == 'surf' else {}
    with pytest.raises(_Recorded):
        getattr(_batch(), f'align_with_{mode}')(
            num_repeats=7, lr=0.2, trans_init=True, max_num_steps=11, backend=_backend(), **kw)
    assert seen['num_repeats'] == 7
    assert seen['lr'] == 0.2
    assert seen['trans_init'] is True
    assert seen['steps_fine'] == 11


@pytest.mark.parametrize('mode', ['vol', 'surf', 'vol_esp', 'vol_color'])
def test_none_loop_args_take_registry_defaults(monkeypatch, mode):
    seen = _record(monkeypatch, mode)
    kw = {'alpha': 0.81} if mode == 'surf' else {'lam': 0.1} if mode == 'vol_esp' else {}
    with pytest.raises(_Recorded):
        getattr(_batch(), f'align_with_{mode}')(backend=_backend(), **kw)
    spec = SPECS[mode]
    assert seen['num_repeats'] == spec.seeds
    assert seen['steps_fine'] == spec.steps
    assert seen['lr'] == spec.params['lr']


def test_engine_honors_num_repeats(monkeypatch):
    from shepherd_score.accel.drivers import engine
    seeds = []
    real = engine.align

    def spy(*a, **kw):
        seeds.append(kw['num_seeds'])
        return real(*a, **kw)

    monkeypatch.setattr(engine, 'align', spy)
    _batch().align_with_vol(num_repeats=3, max_num_steps=2, backend=_backend())
    assert seeds and set(seeds) == {3}


def test_jax_backend_warns_without_jax_kernel():
    batch = _batch()
    with pytest.warns(RuntimeWarning, match='no JAX implementation'):
        handled, _ = batch._run_fast_or_fallthrough(
            'jax', MoleculePair._align_batch_vol_color, {}, None, None, None, False)
    assert not handled
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        batch._run_fast_or_fallthrough(
            'jax', MoleculePair._align_batch_vol, {}, None, None, None, False, has_jax=True)
