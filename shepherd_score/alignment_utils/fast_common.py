# shepherd_score/alignment_utils/fast_common.py
# Common utilities shared across fast GPU-accelerated alignment methods.

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def check_gpu_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return torch.cuda.is_available()


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions (or batches of quaternions).

    Parameters
    ----------
    q, r : torch.Tensor (..., 4)
        Quaternions in (w, x, y, z) format

    Returns
    -------
    torch.Tensor (..., 4)
        Quaternion product q * r
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion(s) to rotation matrix/matrices.

    Parameters
    ----------
    q : torch.Tensor (..., 4)
        Quaternion(s) in (w, x, y, z) format (must be normalized)

    Returns
    -------
    torch.Tensor (..., 3, 3)
        Rotation matrix/matrices
    """
    # Normalize for safety
    q = F.normalize(q, p=2, dim=-1)

    w, x, y, z = q.unbind(-1)
    two = 2.0

    # Rotation matrix elements
    r00 = 1 - two*(y*y + z*z)
    r01 = two*(x*y - z*w)
    r02 = two*(x*z + y*w)
    r10 = two*(x*y + z*w)
    r11 = 1 - two*(x*x + z*z)
    r12 = two*(y*z - x*w)
    r20 = two*(x*z - y*w)
    r21 = two*(y*z + x*w)
    r22 = 1 - two*(x*x + y*y)

    # Stack into matrix
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

    return R


def apply_se3_transform(points: torch.Tensor,
                        q: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
    """
    Apply SE(3) transformation (rotation + translation) to points.

    Parameters
    ----------
    points : torch.Tensor (N, 3) or (B, N, 3)
        Points to transform
    q : torch.Tensor (4,) or (B, 4)
        Quaternion(s) for rotation
    t : torch.Tensor (3,) or (B, 3)
        Translation vector(s)

    Returns
    -------
    torch.Tensor
        Transformed points (same shape as input)
    """
    R = quaternion_to_rotation_matrix(q)

    if points.dim() == 2:
        # Single point cloud: (N, 3)
        return points @ R.T + t
    else:
        # Batched: (B, N, 3)
        # R: (B, 3, 3), t: (B, 3)
        return torch.einsum('bni,bji->bnj', points, R) + t.unsqueeze(1)


def apply_so3_transform(vectors: torch.Tensor,
                        q: torch.Tensor) -> torch.Tensor:
    """
    Apply SO(3) rotation to vectors (no translation).

    Parameters
    ----------
    vectors : torch.Tensor (N, 3) or (B, N, 3)
        Vectors to rotate
    q : torch.Tensor (4,) or (B, 4)
        Quaternion(s) for rotation

    Returns
    -------
    torch.Tensor
        Rotated vectors (same shape as input)
    """
    R = quaternion_to_rotation_matrix(q)

    if vectors.dim() == 2:
        return vectors @ R.T
    else:
        return torch.einsum('bni,bji->bnj', vectors, R)


def legacy_seeds_torch(ref_xyz: torch.Tensor,
                       fit_xyz: torch.Tensor,
                       *,
                       num_repeats: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the exact legacy seeds as (q, t) on the same device/dtype as ref_xyz.

    Parameters
    ----------
    ref_xyz : torch.Tensor (N, 3)
        Reference coordinates
    fit_xyz : torch.Tensor (M, 3)
        Fit coordinates
    num_repeats : int
        Number of random seeds

    Returns
    -------
    q : torch.Tensor (num_repeats, 4)
        Quaternions
    t : torch.Tensor (num_repeats, 3)
        Translations
    """
    from ..alignment import _initialize_se3_params as _legacy_init

    # Legacy helper wants CPU tensors
    ref_cpu = ref_xyz.detach().cpu()
    fit_cpu = fit_xyz.detach().cpu()

    se3 = _legacy_init(ref_points=ref_cpu,
                       fit_points=fit_cpu,
                       num_repeats=num_repeats)  # (R, 7) float32 CPU

    # Move back to caller's device/dtype
    se3 = se3.to(dtype=ref_xyz.dtype, device=ref_xyz.device)
    q, t = se3[:, :4], se3[:, 4:]
    return F.normalize(q, dim=1), t


def build_coarse_grid(A_batch: torch.Tensor,
                      B_batch: torch.Tensor,
                      N_real: torch.Tensor,
                      M_real: torch.Tensor,
                      num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a coarse grid of 500 pose hypotheses (250 rotations × 2 translations).

    Parameters
    ----------
    A_batch : torch.Tensor (B, N_pad, 3)
        Reference coordinates (padded)
    B_batch : torch.Tensor (B, M_pad, 3)
        Fit coordinates (padded)
    N_real : torch.Tensor (B,)
        True point counts for A
    M_real : torch.Tensor (B,)
        True point counts for B
    num_seeds : int
        Base number of seeds (typically 50)

    Returns
    -------
    q_grid : torch.Tensor (B, 500, 4)
        Quaternion grid
    t_grid : torch.Tensor (B, 500, 3)
        Translation grid
    """
    device = A_batch.device
    BATCH = A_batch.shape[0]

    # Generate seeds for each pair
    qs, ts = [], []
    for i in range(BATCH):
        q_i, t_i = legacy_seeds_torch(
            A_batch[i, :N_real[i]], B_batch[i, :M_real[i]], num_repeats=num_seeds)
        qs.append(q_i)
        ts.append(t_i)
    quats = torch.stack(qs, dim=0)    # (B, 50, 4)
    t_seeds = torch.stack(ts, dim=0)  # (B, 50, 3)

    # π-axis flips to get 5x more rotations
    qx = torch.tensor([0., 1., 0., 0.], device=device)
    qy = torch.tensor([0., 0., 1., 0.], device=device)
    qz = torch.tensor([0., 0., 0., 1.], device=device)
    flips = torch.stack([qx, qy, qz, quat_mul(qx, qy)], 0)  # (4, 4)

    q_base = quats.reshape(-1, 4)
    q_base = torch.cat([
        q_base,
        quat_mul(flips[:, None], q_base[None]).reshape(-1, 4)
    ], dim=0).view(BATCH, -1, 4)  # (B, 250, 4)

    # Two translations per pair: COM→COM and tip→COM
    com_trans = t_seeds[:, :1, :]  # (B, 1, 3)
    tips = A_batch[torch.arange(BATCH),
                   A_batch.norm(dim=2).argmax(dim=1)]  # (B, 3)
    extra_t = (tips - B_batch.mean(1)).unsqueeze(1)    # (B, 1, 3)
    t_base = torch.cat([com_trans, extra_t], dim=1)    # (B, 2, 3)

    # Cartesian product: 250 rotations × 2 translations = 500
    q_grid = q_base[:, :, None, :].expand(-1, -1, 2, -1).reshape(BATCH, -1, 4)
    t_grid = t_base[:, None, :, :].expand(-1, 250, -1, -1).reshape(BATCH, -1, 3)

    return q_grid, t_grid
