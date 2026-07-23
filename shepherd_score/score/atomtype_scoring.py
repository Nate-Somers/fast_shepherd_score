"""
Atom-identity ("atom-type") categorical overlap scoring with PyTorch.

This is the scoring channel for the ``vol_atomtype`` alignment mode: a Gaussian volume overlap
that is *partitioned by a categorical per-atom label* (the atomic number / element identity), so
only atoms of the SAME element contribute to the cross overlap. It is the same construction the
pharmacophore "color" channel uses -- per-type masked Gaussian overlap, summed over types, then a
self-normalised Tanimoto (or Tversky) -- but over an arbitrary categorical label instead of the
eight hardcoded pharmacophore types in :mod:`shepherd_score.score.pharmacophore_scoring`.

The overlap primitive is the shared atom-centred Gaussian ``VAB_2nd_order`` from
:mod:`shepherd_score.score.gaussian_overlap`, so the width convention matches the ``vol`` shape
channel (``alpha=0.81`` volumetric). Labels are compared by equality only (their numeric value is
never used as a coordinate), so any integer-coded categorical scheme works; the ``vol_atomtype``
mode passes heavy-atom atomic numbers (:meth:`shepherd_score.container.Molecule.get_atomic_numbers`).
"""
from typing import Literal, Union

import numpy as np
import torch

from shepherd_score.score.gaussian_overlap import VAB_2nd_order

_SIM_TYPE = Literal['tanimoto', 'tversky', 'tversky_ref', 'tversky_fit']


def _sigma_for(similarity: str) -> Union[float, None]:
    """Map a similarity name to its Tversky ``sigma`` (weight on the reference self-overlap), or
    ``None`` for symmetric Tanimoto. Mirrors :func:`pharmacophore_scoring.get_overlap_pharm`."""
    s = similarity.lower()
    if s == 'tanimoto':
        return None
    if s == 'tversky':
        return 0.95
    if s == 'tversky_ref':
        return 1.0
    if s == 'tversky_fit':
        return 0.05
    raise ValueError('Argument `similarity` must be one of (tanimoto, tversky, tversky_ref, tversky_fit).')


def get_overlap_atomtype(labels_1: torch.Tensor,
                         labels_2: torch.Tensor,
                         centers_1: torch.Tensor,
                         centers_2: torch.Tensor,
                         alpha: float = 0.81,
                         similarity: _SIM_TYPE = 'tanimoto',
                         ) -> torch.Tensor:
    """
    Compute the atom-identity (categorical) Gaussian overlap score.

    Only atoms sharing the same categorical ``label`` (e.g. atomic number) contribute to the cross
    overlap. For each label present, the per-label cross/self Gaussian overlaps are summed, then the
    totals are reduced to a Tanimoto (or Tversky) similarity::

        Tanimoto = Σ_t VAB_t / (Σ_t VAA_t + Σ_t VBB_t - Σ_t VAB_t)
        Tversky  = Σ_t VAB_t / (sigma·Σ_t VAA_t + (1-sigma)·Σ_t VBB_t)   (clamped to 1)

    A label present in only one molecule contributes to that molecule's self-overlap sum (the
    denominator) but not to the cross overlap -- exactly the pharmacophore-colour convention -- so
    an element in the reference that the fit lacks correctly penalises the similarity. A molecule
    scored against a copy of itself gives 1.000 for either reduction.

    Only the ``centers`` are geometric; ``labels`` are used for equality masking only. Supports both
    single-instance and batched (``B`` poses of the SAME label sets) inputs: pass 1-D ``labels`` and
    optionally batched ``(B,N,3)`` centers (the multi-start optimiser repeats the poses, not the
    labels).

    Parameters
    ----------
    labels_1 : torch.Tensor (N,)
        Integer-coded categorical label per reference atom (e.g. atomic number).
    labels_2 : torch.Tensor (M,)
        Integer-coded categorical label per fit atom.
    centers_1 : torch.Tensor (N,3) or (B,N,3)
        Reference atom coordinates.
    centers_2 : torch.Tensor (M,3) or (B,M,3)
        Fit atom coordinates (already SE(3)-transformed by the caller).
    alpha : float, optional
        Gaussian width for the overlap. Default 0.81 (volumetric, heavy atoms).
    similarity : str, optional
        'tanimoto' (default, symmetric) or a Tversky variant ('tversky', 'tversky_ref',
        'tversky_fit').

    Returns
    -------
    torch.Tensor
        Score(s): scalar for single-instance input, shape ``(B,)`` for batched centers.
    """
    if isinstance(labels_1, np.ndarray):
        labels_1 = torch.as_tensor(labels_1)
    if isinstance(labels_2, np.ndarray):
        labels_2 = torch.as_tensor(labels_2)

    batched = centers_1.dim() == 3
    # Running totals accumulate as python 0. -> tensor after the first label; a batch keeps a (B,)
    # vector because each VAB_2nd_order over batched centers returns (B,).
    VAB = torch.zeros(centers_1.shape[0], device=centers_1.device) if batched else torch.zeros((), device=centers_1.device)
    VAA = torch.zeros_like(VAB)
    VBB = torch.zeros_like(VAB)

    # Union of labels present in either molecule; each contributes to the relevant self-overlap sum.
    unique_labels = torch.unique(torch.cat((labels_1.reshape(-1), labels_2.reshape(-1))))
    for lbl in unique_labels:
        idx1 = torch.where(labels_1 == lbl)[0]
        idx2 = torch.where(labels_2 == lbl)[0]
        has1 = idx1.numel() > 0
        has2 = idx2.numel() > 0
        if has1:
            c1 = centers_1[..., idx1, :]
            VAA = VAA + VAB_2nd_order(c1, c1, alpha)
        if has2:
            c2 = centers_2[..., idx2, :]
            VBB = VBB + VAB_2nd_order(c2, c2, alpha)
        if has1 and has2:
            VAB = VAB + VAB_2nd_order(centers_1[..., idx1, :], centers_2[..., idx2, :], alpha)

    sigma = _sigma_for(similarity)
    if sigma is None:
        return VAB / (VAA + VBB - VAB)
    return torch.clamp_max(VAB / (sigma * VAA + (1 - sigma) * VBB), max=1.0)
