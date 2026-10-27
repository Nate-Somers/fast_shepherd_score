"""``MoleculePair.align`` / ``score`` with named and user-defined modes, and the ``similarity``
argument that replaces the separate ``*_tversky`` modes."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest
import torch

from shepherd_score.container import MoleculePairBatch
from shepherd_score.modes import MODES, Combined, Mode, Vol, blend, get_mode
from shepherd_score.score.gaussian_overlap import VAB_2nd_order
from shepherd_score.score.pharmacophore_scoring import tversky_func
from shepherd_score.score.similarity import similarity_from_overlaps

from .legacy_fixture import make_pair, load_molecules, NUM_SURF_POINTS

FAST = dict(num_repeats=8, max_num_steps=40)


def _self_pair():
    from shepherd_score.container import MoleculePair
    m = load_molecules()[0]
    m.fukui = np.linspace(-0.2, 0.2, m.mol.GetNumAtoms())   # the real field needs xTB
    return MoleculePair(m, m, num_surf_points=NUM_SURF_POINTS, device=torch.device('cpu'))


@pytest.mark.parametrize('name', sorted(MODES))
@pytest.mark.parametrize('similarity', ['tanimoto', 'tversky'])
def test_every_mode_scores_one_on_itself(name, similarity):
    pair = _self_pair()
    assert np.isclose(pair.score(name, similarity=similarity), 1.0, atol=1e-5)
    res = pair.align(name, similarity=similarity, **FAST)
    assert np.isclose(res.score, 1.0, atol=5e-3)
    assert res.transform.shape == (4, 4)


@pytest.mark.parametrize('name, legacy, kw', [
    ('vol', 'vol_tversky', {}),
    ('vol_esp', 'vol_esp_tversky', {}),
    ('surf', 'surf_tversky', {}),
    ('surf_esp', 'surf_esp_tversky', {}),
    ('vol_lipo', 'vol_lipo_tversky', {}),
    ('vol_and_surf_esp', 'vol_and_surf_esp_tversky', {}),
    ('pharm', 'pharm_tversky', {}),
])
def test_similarity_argument_matches_tversky_methods(name, legacy, kw):
    pair = make_pair(0, 1)
    res = pair.align(name, similarity='tversky', **FAST, **kw)
    getattr(pair, f'align_with_{legacy}')(**FAST)
    assert np.allclose(res.score, getattr(pair, f'sim_aligned_{legacy}'), atol=1e-6)
    assert np.allclose(res.transform, getattr(pair, f'transform_{legacy}'), atol=1e-5)


def test_tversky_name_and_weights_agree():
    pair = make_pair(0, 1)
    assert np.isclose(pair.score('vol', similarity='tversky'), pair.score('vol', similarity=(0.95, 0.05)))
    assert not np.isclose(pair.score('vol', similarity='tversky'), pair.score('vol', similarity='tversky_fit'))


def test_tversky_matches_sigma_form_when_weights_sum_to_one():
    a, b = torch.randn(9, 3), torch.randn(7, 3) + 0.3
    AB, AA, BB = VAB_2nd_order(a, b, 0.81), VAB_2nd_order(a, a, 0.81), VAB_2nd_order(b, b, 0.81)
    for sigma in (0.95, 0.5, 0.05):
        assert torch.isclose(similarity_from_overlaps(AB, AA, BB, (sigma, 1 - sigma)),
                             tversky_func(AB, AA, BB, sigma))


def test_score_matches_legacy_scorers():
    pair = make_pair(0, 1)
    assert np.isclose(pair.score('surf'), pair.score_with_surf(0.81), atol=1e-5)
    assert np.isclose(pair.score('surf_esp', lam=0.3), pair.score_with_esp(0.81, lam=0.3), atol=1e-5)
    assert np.isclose(pair.score('pharm'), pair.score_with_pharm(), atol=1e-5)


def test_backends():
    pair = make_pair(0, 1)
    auto = pair.align('vol', **FAST)
    assert np.allclose(auto.score, pair.align('vol', backend='analytical', **FAST).score)
    torch_res = pair.align('vol', backend='torch', **FAST)
    assert np.isclose(auto.score, torch_res.score, atol=1e-3)
    with pytest.raises(ValueError, match='not available'):
        pair.align('vol', similarity='tversky', backend='analytical')


def test_bad_arguments():
    with pytest.raises(ValueError, match='Unknown mode'):
        get_mode('volume')
    with pytest.raises(TypeError):
        get_mode('vol', lam=0.3)
    with pytest.raises(TypeError):
        get_mode(Vol(), alpha=1.0)
    with pytest.raises(ValueError, match='Unknown similarity'):
        make_pair().score('vol', similarity='dice')


@dataclass(frozen=True)
class HeavyAtomCount(Mode):
    """A user mode defined outside the package: Gaussian overlap of nitrogen and oxygen atoms."""
    name: ClassVar[str] = 'n_o'
    alpha: float = 0.5
    similarity: str = 'tanimoto'

    def inputs(self, mol):
        z = mol.get_atomic_numbers(no_H=False)
        return {'points': mol.get_positions(no_H=False)[(z == 7) | (z == 8)]}

    def overlap(self, a, b):
        return VAB_2nd_order(a['points'], b['points'], self.alpha)


def test_user_defined_mode_and_blend():
    pair = make_pair(0, 1)
    res = pair.align(HeavyAtomCount(), **FAST)
    assert 0 < res.score <= 1
    mixed = blend(Vol(), HeavyAtomCount(similarity='tversky'), 0.3, 'vol_n_o')
    assert isinstance(mixed, Combined)
    res = pair.align(mixed, **FAST)
    assert 0 < res.score <= 1 and len(res.aligned) == 2


def test_batch_align_dispatches_similarity():
    pairs = [make_pair(0, 1), make_pair(2, 3)]
    MoleculePairBatch(pairs).align('vol', similarity='tversky', backend='numba')
    assert all(p.sim_aligned_vol_tversky is not None for p in pairs)
    assert all(p.sim_aligned_vol_noH is None for p in pairs)
