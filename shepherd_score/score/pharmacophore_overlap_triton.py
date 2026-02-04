# shepherd_score/score/pharmacophore_overlap_triton.py
# Fast GPU-accelerated pharmacophore overlap scoring.
# Uses batched PyTorch operations optimized for GPU execution.
# Full Triton kernel would be complex due to type-specific alphas and vector weighting.

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .constants import P_TYPES, P_ALPHAS

# Lowercase type names for indexing
P_TYPES_LOWER = tuple(map(str.lower, P_TYPES))

# Pre-compute alpha array for GPU indexing
# Order matches P_TYPES: Acceptor, Donor, Aromatic, Hydrophobe, Halogen, Cation, Anion, ZnBinder
ALPHA_VALUES = torch.tensor([
    P_ALPHAS['acceptor'],   # 0
    P_ALPHAS['donor'],      # 1
    P_ALPHAS['aromatic'],   # 2
    P_ALPHAS['hydrophobe'], # 3
    P_ALPHAS['halogen'],    # 4
    P_ALPHAS['cation'],     # 5
    P_ALPHAS['anion'],      # 6
    P_ALPHAS['znbinder'],   # 7
], dtype=torch.float32)

# Types that use vector cosine weighting
VECTOR_TYPES = {2, 0, 1, 4}  # aromatic, acceptor, donor, halogen
# Types where antiparallel vectors are acceptable
ANTIPARALLEL_TYPES = {2}  # aromatic only


@torch.no_grad()
def _gaussian_overlap_kernel(r2: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian overlap: k * exp(-alpha/2 * r^2)
    where k = (pi / (2*alpha))^1.5

    Parameters
    ----------
    r2 : torch.Tensor (..., N, M)
        Squared distances
    alpha : torch.Tensor (..., N, M) or scalar
        Alpha values (can be type-specific)

    Returns
    -------
    overlap : torch.Tensor (..., N, M)
        Gaussian overlap values
    """
    k_const = (math.pi / (2.0 * alpha)) ** 1.5
    return k_const * torch.exp(-0.5 * alpha * r2)


@torch.no_grad()
def _batch_pharm_overlap_typed(
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        vectors_1: torch.Tensor,
        vectors_2: torch.Tensor,
        types_1: torch.Tensor,
        types_2: torch.Tensor,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute batched pharmacophore overlaps with type matching and vector weighting.

    Parameters
    ----------
    anchors_1 : torch.Tensor (B, N, 3)
        Reference pharmacophore anchor positions
    anchors_2 : torch.Tensor (B, M, 3)
        Fit pharmacophore anchor positions
    vectors_1 : torch.Tensor (B, N, 3)
        Reference pharmacophore vectors (normalized)
    vectors_2 : torch.Tensor (B, M, 3)
        Fit pharmacophore vectors (normalized)
    types_1 : torch.Tensor (B, N) int
        Reference pharmacophore type indices
    types_2 : torch.Tensor (B, M) int
        Fit pharmacophore type indices
    N_real, M_real : torch.Tensor (B,) optional
        Real point counts for padded batches

    Returns
    -------
    VAB : torch.Tensor (B,)
        Cross-overlap scores
    VAA : torch.Tensor (B,)
        Reference self-overlap scores
    VBB : torch.Tensor (B,)
        Fit self-overlap scores
    """
    device = anchors_1.device
    dtype = anchors_1.dtype
    B, N, _ = anchors_1.shape
    _, M, _ = anchors_2.shape

    # Move alpha values to device
    alphas = ALPHA_VALUES.to(device=device, dtype=dtype)

    # Normalize vectors
    vectors_1 = F.normalize(vectors_1, p=2, dim=-1)
    vectors_2 = F.normalize(vectors_2, p=2, dim=-1)

    # Create masks for valid points
    if N_real is None:
        mask_1 = torch.ones(B, N, device=device, dtype=dtype)
    else:
        idx = torch.arange(N, device=device).unsqueeze(0)
        mask_1 = (idx < N_real.unsqueeze(1)).float()

    if M_real is None:
        mask_2 = torch.ones(B, M, device=device, dtype=dtype)
    else:
        idx = torch.arange(M, device=device).unsqueeze(0)
        mask_2 = (idx < M_real.unsqueeze(1)).float()

    # Initialize accumulators
    VAB = torch.zeros(B, device=device, dtype=dtype)
    VAA = torch.zeros(B, device=device, dtype=dtype)
    VBB = torch.zeros(B, device=device, dtype=dtype)

    # Process each pharmacophore type
    for ptype_idx in range(len(P_TYPES)):
        alpha = alphas[ptype_idx]
        ptype_name = P_TYPES_LOWER[ptype_idx]
        use_vectors = ptype_idx in VECTOR_TYPES
        allow_antiparallel = ptype_idx in ANTIPARALLEL_TYPES

        # Type masks
        type_mask_1 = (types_1 == ptype_idx).float() * mask_1  # (B, N)
        type_mask_2 = (types_2 == ptype_idx).float() * mask_2  # (B, M)

        # Skip if no features of this type
        if type_mask_1.sum() == 0 and type_mask_2.sum() == 0:
            continue

        # Compute squared distances for cross-overlap (1 vs 2)
        # anchors_1: (B, N, 3), anchors_2: (B, M, 3)
        diff_12 = anchors_1.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, N, M, 3)
        r2_12 = (diff_12 ** 2).sum(dim=-1)  # (B, N, M)

        # Compute Gaussian overlap
        g_12 = _gaussian_overlap_kernel(r2_12, alpha)

        # Apply vector weighting if needed
        if use_vectors:
            # Cosine similarity: (B, N, M)
            cos_sim = torch.einsum('bni,bmi->bnm', vectors_1, vectors_2)

            if allow_antiparallel:
                # Use absolute value of cosine
                weight = (torch.abs(cos_sim) + 2.0) / 3.0
            else:
                # Only positive alignment
                weight = (cos_sim.clamp(min=0) + 2.0) / 3.0

            g_12 = g_12 * weight

        # Apply type matching mask (both points must be same type)
        pair_mask_12 = type_mask_1.unsqueeze(2) * type_mask_2.unsqueeze(1)  # (B, N, M)
        VAB += (g_12 * pair_mask_12).sum(dim=(1, 2))

        # Self-overlap for reference (1 vs 1)
        diff_11 = anchors_1.unsqueeze(2) - anchors_1.unsqueeze(1)  # (B, N, N, 3)
        r2_11 = (diff_11 ** 2).sum(dim=-1)
        g_11 = _gaussian_overlap_kernel(r2_11, alpha)

        if use_vectors:
            cos_sim_11 = torch.einsum('bni,bmi->bnm', vectors_1, vectors_1)
            if allow_antiparallel:
                weight_11 = (torch.abs(cos_sim_11) + 2.0) / 3.0
            else:
                weight_11 = (cos_sim_11.clamp(min=0) + 2.0) / 3.0
            g_11 = g_11 * weight_11

        pair_mask_11 = type_mask_1.unsqueeze(2) * type_mask_1.unsqueeze(1)
        VAA += (g_11 * pair_mask_11).sum(dim=(1, 2))

        # Self-overlap for fit (2 vs 2)
        diff_22 = anchors_2.unsqueeze(2) - anchors_2.unsqueeze(1)  # (B, M, M, 3)
        r2_22 = (diff_22 ** 2).sum(dim=-1)
        g_22 = _gaussian_overlap_kernel(r2_22, alpha)

        if use_vectors:
            cos_sim_22 = torch.einsum('bni,bmi->bnm', vectors_2, vectors_2)
            if allow_antiparallel:
                weight_22 = (torch.abs(cos_sim_22) + 2.0) / 3.0
            else:
                weight_22 = (cos_sim_22.clamp(min=0) + 2.0) / 3.0
            g_22 = g_22 * weight_22

        pair_mask_22 = type_mask_2.unsqueeze(2) * type_mask_2.unsqueeze(1)
        VBB += (g_22 * pair_mask_22).sum(dim=(1, 2))

    return VAB, VAA, VBB


@torch.no_grad()
def batch_pharm_tanimoto(
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        vectors_1: torch.Tensor,
        vectors_2: torch.Tensor,
        types_1: torch.Tensor,
        types_2: torch.Tensor,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute batched pharmacophore Tanimoto similarity.

    Returns
    -------
    scores : torch.Tensor (B,)
        Tanimoto similarity scores
    """
    VAB, VAA, VBB = _batch_pharm_overlap_typed(
        anchors_1, anchors_2,
        vectors_1, vectors_2,
        types_1, types_2,
        N_real, M_real)

    # Tanimoto: VAB / (VAA + VBB - VAB)
    # Handle division by zero
    denom = VAA + VBB - VAB
    scores = torch.where(denom > 1e-8, VAB / denom, torch.zeros_like(VAB))

    return scores


@torch.no_grad()
def batch_pharm_overlap_with_transform(
        anchors_1: torch.Tensor,
        anchors_2: torch.Tensor,
        vectors_1: torch.Tensor,
        vectors_2: torch.Tensor,
        types_1: torch.Tensor,
        types_2: torch.Tensor,
        q: torch.Tensor,
        t: torch.Tensor,
        N_real: Optional[torch.Tensor] = None,
        M_real: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute pharmacophore overlap after applying SE(3) transform to fit molecule.

    Parameters
    ----------
    q : torch.Tensor (B, 4)
        Quaternions (w, x, y, z)
    t : torch.Tensor (B, 3)
        Translations

    Returns
    -------
    VAB, VAA, VBB : torch.Tensor (B,)
        Overlap scores
    """
    from ..alignment_utils.fast_common import apply_se3_transform, apply_so3_transform

    # Transform fit molecule
    anchors_2_t = apply_se3_transform(anchors_2, q, t)
    vectors_2_t = apply_so3_transform(vectors_2, q)

    return _batch_pharm_overlap_typed(
        anchors_1, anchors_2_t,
        vectors_1, vectors_2_t,
        types_1, types_2,
        N_real, M_real)


@torch.no_grad()
def batch_pharm_self_overlap(
        anchors: torch.Tensor,
        vectors: torch.Tensor,
        types: torch.Tensor,
        N_real: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute batched pharmacophore self-overlap.

    Returns
    -------
    V : torch.Tensor (B,)
        Self-overlap scores
    """
    _, V, _ = _batch_pharm_overlap_typed(
        anchors, anchors,
        vectors, vectors,
        types, types,
        N_real, N_real)

    return V
