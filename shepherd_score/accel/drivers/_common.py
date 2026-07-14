# shepherd_score/accel/drivers/_common.py
# Common utilities shared across fast GPU-accelerated alignment methods.

import math
import os
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

_TWO_PI_3 = 2.0 * math.pi / 3.0

# Shared early-stop patience override for the esp/pharm fine loops (None -> use the
# call's default). Lever 2: patience=5 over-runs ~25 steps after convergence on the
# fast-converging self-copy benchmark. Set via speedlab; accuracy-gated. (surf/vol
# uses fast_se3._ES_PATIENCE.)


def check_gpu_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return torch.cuda.is_available()


def _update_best(score, q_k, t_k, best_score, best_q, best_t):
    """Track the best (q, t) per pose by score. Uses ``torch.where`` so the update is
    fixed-shape and sync-free, and RETURNS new ``(best_score, best_q, best_t)`` tensors
    rather than mutating in place -- the fine loops are CUDA-graph captured and must not
    do data-dependent or in-place best-pose updates."""
    better = score > best_score
    best_score = torch.where(better, score, best_score)
    mask_q = better.unsqueeze(1)
    best_q = torch.where(mask_q, q_k, best_q)
    best_t = torch.where(mask_q, t_k, best_t)
    return best_score, best_q, best_t


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


def legacy_seeds_with_translations_torch(
    ref_xyz: torch.Tensor,
    fit_xyz: torch.Tensor,
    trans_centers: torch.Tensor,
    *,
    num_repeats_per_trans: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return legacy translation-seeded initializations as (q, t) on the same device/dtype.

    This mirrors `alignment._initialize_se3_params_with_translations`, which is used when
    legacy code is called with `trans_centers!=None` (aka `trans_init=True`).
    """
    from ...alignment._torch import _initialize_se3_params_with_translations as _legacy_init_trans

    ref_cpu = ref_xyz.detach().cpu()
    fit_cpu = fit_xyz.detach().cpu()
    trans_cpu = trans_centers.detach().cpu()

    se3 = _legacy_init_trans(
        ref_points=ref_cpu,
        fit_points=fit_cpu,
        trans_centers=trans_cpu,
        num_repeats_per_trans=num_repeats_per_trans,
    )

    se3 = se3.to(dtype=ref_xyz.dtype, device=ref_xyz.device)
    if se3.dim() == 1:
        se3 = se3.unsqueeze(0)
    q, t = se3[:, :4], se3[:, 4:]
    return F.normalize(q, dim=1), t


def _fallback_quats(num: int, device, dtype) -> torch.Tensor:
    """Deterministic set of 'reasonable' rotations (matches fast_se3._fallback_quats)."""
    import math
    s2 = math.sqrt(0.5)
    base = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0], [s2, s2, 0.0, 0.0], [s2, 0.0, s2, 0.0],
        [s2, 0.0, 0.0, s2], [0.0, s2, s2, 0.0], [0.0, s2, 0.0, s2],
        [0.0, 0.0, s2, s2],
    ], device=device, dtype=dtype)
    if base.size(0) >= num:
        q = base[:num].clone()
    else:
        reps = (num + base.size(0) - 1) // base.size(0)
        q = base.repeat(reps, 1)[:num].clone()
    return F.normalize(q, dim=1)


def _analytic_sym3x3_axes(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Eigenvectors (rows, descending eigenvalue) of a batch of symmetric 3x3
    matrices ``M`` (K,3,3), returned as (K,3,3) with rows = longest-axis-first.
    Closed form, fully vectorized, no host sync:

      * eigenvalues via the trigonometric (Cardano) solution of the characteristic
        cubic of a symmetric 3x3 (Smith 1961);
      * each eigenvector as the max-norm column of ``(M - lam_j I)(M - lam_k I)``
        (whose columns are all parallel to the remaining eigenvector), with the
        middle axis recovered by a cross product so the frame is orthonormal;
      * near-spherical / rank-deficient rows (both product norms ~0) fall back to
        the identity frame -- their axes are ill-defined anyway and the downstream
        Fibonacci seeds + fine optimizer recover them (score-parity validated).
    """
    a00 = M[:, 0, 0]; a11 = M[:, 1, 1]; a22 = M[:, 2, 2]
    a01 = M[:, 0, 1]; a02 = M[:, 0, 2]; a12 = M[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = torch.sqrt((p2 / 6.0).clamp_min(eps)); ip = 1.0 / p
    b00 = (a00 - q) * ip; b11 = (a11 - q) * ip; b22 = (a22 - q) * ip
    b01 = a01 * ip; b02 = a02 * ip; b12 = a12 * ip
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = (detB * 0.5).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)                              # largest
    e3 = q + 2.0 * p * torch.cos(phi + _TWO_PI_3)                  # smallest
    e2 = 3.0 * q - e1 - e3                                         # middle
    I = torch.eye(3, device=M.device, dtype=M.dtype).expand_as(M)
    ar = torch.arange(M.shape[0], device=M.device)

    def evec(lam_j, lam_k):
        P = torch.bmm(M - lam_j.view(-1, 1, 1) * I, M - lam_k.view(-1, 1, 1) * I)
        idx = P.norm(dim=1).argmax(dim=1)                          # max-norm column
        col = P[ar, :, idx]
        nrm = col.norm(dim=1, keepdim=True)
        return col / nrm.clamp_min(eps), nrm.squeeze(1)

    v1, n1 = evec(e2, e3)                                          # eigenvector of e1
    v3, n3 = evec(e1, e2)                                          # eigenvector of e3
    v2 = torch.linalg.cross(v3, v1, dim=1)
    v2 = v2 / v2.norm(dim=1, keepdim=True).clamp_min(eps)
    v1 = torch.linalg.cross(v2, v3, dim=1)                         # re-orthogonalize v1
    v1 = v1 / v1.norm(dim=1, keepdim=True).clamp_min(eps)
    R = torch.stack([v1, v2, v3], dim=1)                          # rows = descending axes
    bad = (n1 < 1e-9) | (n3 < 1e-9) | ~torch.isfinite(R).all(dim=2).all(dim=1)
    return torch.where(bad.view(-1, 1, 1), I, R)                   # sync-free degenerate fallback


def _masked_principal_axes(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row principal axes (rows = axes, longest first) computed over the
    REAL (unmasked) points only.

    points : (K, P, 3)   mask : (K, P) in {0,1}
    Returns (K, 3, 3). Division by N is omitted because it scales eigenvalues
    uniformly and does not change eigenvectors or their ordering. Padding rows
    are zeroed after centering so they contribute nothing to the inertia tensor.
    """
    n = mask.sum(1).clamp(min=1.0)                                  # (K,)
    com = (points * mask.unsqueeze(-1)).sum(1) / n.unsqueeze(-1)    # (K,3)
    centered = (points - com.unsqueeze(1)) * mask.unsqueeze(-1)     # (K,P,3)
    A = (centered ** 2).sum((1, 2))                                 # (K,)
    Bmat = torch.bmm(centered.transpose(1, 2), centered)            # (K,3,3)
    eye = torch.eye(3, device=points.device, dtype=points.dtype)
    inertia = A.view(-1, 1, 1) * eye - Bmat                         # (K,3,3)
    # Closed-form eigensolver, not ``torch.linalg.eigh``: batched cuSOLVER eigh is a
    # stream-synchronizing barrier that stalls the otherwise-async seed-gen prologue,
    # and it fails with CUSOLVER_STATUS_INVALID_VALUE for K > ~8192 (the fit-side call
    # below passes a 4K-row batch, so that limit is routinely exceeded).
    return _analytic_sym3x3_axes(inertia)


def batched_seeds_torch(A_batch: torch.Tensor,
                        B_batch: torch.Tensor,
                        N_real: torch.Tensor,
                        M_real: torch.Tensor,
                        num_seeds: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-native, fully batched replacement for the per-pair seed loop.

    Produces the SAME seed set as calling ``_initialize_se3_params`` per pair
    (identity + 4 principal-component-alignment quaternions + Fibonacci-sampled
    rotations, all with COM-aligning translations), but for the whole cohort in
    one vectorized pass with NO ``.cpu()``/numpy round-trip. The principal-axis
    PCA is done in float64 for near-degenerate stability, via an analytic closed-form
    3x3 eigensolver rather than a synchronizing cuSOLVER call.

    Pairs with < 3 real points or non-finite coordinates fall back to a fixed
    deterministic rotation set + COM-to-COM translation (matching the per-pair
    fallback in ``fast_se3._legacy_seeds_torch``).

    Parameters
    ----------
    A_batch, B_batch : (K, Npad, 3) / (K, Mpad, 3)  padded coordinates
    N_real, M_real   : (K,)  true point counts
    num_seeds        : int   number of base seeds per pair (default 50)

    Returns
    -------
    quats : (K, num_seeds, 4)   t : (K, num_seeds, 3)
    """
    from shepherd_score.alignment._torch import _get_45_fibo, _quats_from_fibo

    device = A_batch.device
    dtype = A_batch.dtype
    K = A_batch.shape[0]
    Npad = A_batch.shape[1]
    Mpad = B_batch.shape[1]

    N_real = N_real.to(device)
    M_real = M_real.to(device)

    # Match _initialize_se3_params seed-count semantics: num_repeats==1 is a
    # single identity seed; 1<num_repeats<5 is bumped to 5 (so the 4 PCA seeds
    # always fit). Everything >=5 is identity + 4 PCA + (num_seeds-5) Fibonacci.
    if num_seeds == 1:
        q = torch.zeros(A_batch.shape[0], 1, 4, device=device, dtype=dtype)
        q[:, :, 0] = 1.0
        t = torch.zeros(A_batch.shape[0], 1, 3, device=device, dtype=dtype)
        return q, t
    if num_seeds < 5:
        num_seeds = 5

    mask_n = (torch.arange(Npad, device=device)[None] < N_real[:, None]).to(dtype)
    mask_m = (torch.arange(Mpad, device=device)[None] < M_real[:, None]).to(dtype)
    nreal = mask_n.sum(1).clamp(min=1.0)
    mreal = mask_m.sum(1).clamp(min=1.0)

    ref_com = (A_batch * mask_n.unsqueeze(-1)).sum(1) / nreal.unsqueeze(-1)   # (K,3)
    fit_com = (B_batch * mask_m.unsqueeze(-1)).sum(1) / mreal.unsqueeze(-1)   # (K,3)

    # ---- 4 principal-component-alignment quaternions per pair ----
    # PCA runs in float64 for near-degenerate stability.
    _wd = torch.float64
    A64 = torch.nan_to_num(A_batch.to(_wd))
    B64 = torch.nan_to_num(B_batch.to(_wd))
    mask_n64 = mask_n.to(_wd)
    mask_m64 = mask_m.to(_wd)

    ref_axes = _masked_principal_axes(A64, mask_n64)                 # (K,3,3)
    ref_axes4 = ref_axes.unsqueeze(1).repeat(1, 4, 1, 1)            # (K,4,3,3)
    ref_axes4[:, 1, 0] = -ref_axes4[:, 1, 0]                         # flip longest
    ref_axes4[:, 2, 1] = -ref_axes4[:, 2, 1]                         # flip 2nd-longest
    ref_axes4[:, 3, 0] = -ref_axes4[:, 3, 0]                         # flip both
    ref_axes4[:, 3, 1] = -ref_axes4[:, 3, 1]
    ref_axes_f = ref_axes4.reshape(4 * K, 3, 3)

    fit_c = (B64 - fit_com.to(_wd).unsqueeze(1)) * mask_m64.unsqueeze(-1)
    fit4 = fit_c.unsqueeze(1).repeat(1, 4, 1, 1).reshape(4 * K, Mpad, 3)
    mask_m4 = mask_m64.unsqueeze(1).repeat(1, 4, 1).reshape(4 * K, Mpad)

    quat_order = [None, None]
    for ax in range(2):
        fit_axes = _masked_principal_axes(fit4, mask_m4)            # (4K,3,3)
        v1 = fit_axes[:, ax]                                        # (4K,3)
        v2 = ref_axes_f[:, ax]                                      # (4K,3)
        cos = torch.clamp((v1 * v2).sum(1, keepdim=True), -1.0, 1.0)
        angle = torch.acos(cos)                                    # (4K,1)
        axis = torch.linalg.cross(v1, v2, dim=1)                   # (4K,3)
        axis_norm = axis.norm(dim=1, keepdim=True)
        # Degenerate (parallel/antiparallel) axes -> safe default [1,0,0]; these
        # few seeds are non-optimal but recovered by the Fibonacci seeds + the
        # coarse grid + fine optimisation (validated by the score-parity gate).
        axis = torch.where(axis_norm < 1e-8,
                           torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=device),
                           axis / axis_norm.clamp(min=1e-12))
        half = angle * 0.5
        q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=1)  # (4K,4)
        quat_order[ax] = q
        R = quaternion_to_rotation_matrix(q)                       # (4K,3,3)
        fit4 = torch.einsum('kni,kji->knj', fit4, R) * mask_m4.unsqueeze(-1)

    pca_quats = quat_mul(quat_order[1], quat_order[0]).reshape(K, 4, 4).to(dtype)

    # ---- assemble seeds: identity + 4 PCA (+ structured axis swaps) + Fibonacci fill ----
    identity = torch.zeros(K, 1, 4, device=device, dtype=dtype)
    identity[:, :, 0] = 1.0

    # Structured seeds: +/-90deg rotations about each ref principal axis, composed onto
    # the base PCA alignment. The 4 PCA quats already cover axis SIGN-flips (180deg); these
    # add the axis SWAPS that PCA-alignment alone misses. Vectorized and reuses the
    # already-computed principal axes, so the cost is negligible next to the float64 PCA
    # eigensolve it rides on.
    n_struct = min(max(num_seeds - 5, 0), 6)
    if n_struct > 0:
        base_pca = pca_quats[:, 0]                                  # (K,4) no-sign-flip PCA align
        axes = ref_axes.to(dtype)                                   # (K,3,3) rows = principal axes
        cos_h = math.cos(math.pi / 4)                              # +/-90deg half-angle
        sgn = torch.tensor([math.sin(math.pi / 4), -math.sin(math.pi / 4)], device=device, dtype=dtype)
        w = torch.full((K, 3, 2, 1), cos_h, device=device, dtype=dtype)
        xyz = sgn.view(1, 1, 2, 1) * axes.view(K, 3, 1, 3)         # (K,3,2,3) per axis, +/-90
        q_ax = torch.cat([w, xyz], dim=-1).reshape(K, 6, 4)[:, :n_struct]
        base_rep = base_pca.unsqueeze(1).expand(K, n_struct, 4).reshape(-1, 4)
        struct_quats = quat_mul(q_ax.reshape(-1, 4), base_rep).reshape(K, n_struct, 4)
    else:
        struct_quats = identity.new_zeros(K, 0, 4)

    n_fibo = max(num_seeds - 5 - n_struct, 0)
    if n_fibo > 0:
        fibo_b = _quats_from_fibo(n_fibo).to(device=device, dtype=dtype).unsqueeze(0).expand(K, -1, -1)
    else:
        fibo_b = identity.new_zeros(K, 0, 4)

    # slice handles num_seeds < 5 too (identity + leading PCA quats)
    quats = torch.cat([identity, pca_quats, struct_quats, fibo_b], dim=1)[:, :num_seeds]
    quats = F.normalize(quats, p=2, dim=-1)

    # ---- translations: t = ref_com - R(q) @ fit_com  (COM alignment) ----
    R_all = quaternion_to_rotation_matrix(quats.reshape(-1, 4)).reshape(K, num_seeds, 3, 3)
    rot_fit_com = torch.einsum('ksij,kj->ksi', R_all, fit_com)
    trans = ref_com.unsqueeze(1) - rot_fit_com                     # (K, num_seeds, 3)

    # ---- fallback for degenerate pairs (< 3 pts / non-finite) ----
    valid = ((N_real >= 3) & (M_real >= 3)
             & torch.isfinite(A_batch).all(dim=2).all(dim=1)
             & torch.isfinite(B_batch).all(dim=2).all(dim=1))
    if not bool(valid.all()):
        fb_q = _fallback_quats(num_seeds, device, dtype).unsqueeze(0).expand(K, -1, -1)
        fb_t = (ref_com - fit_com).unsqueeze(1).expand(-1, num_seeds, -1)
        vmask = valid.view(K, 1, 1)
        quats = torch.where(vmask, quats, fb_q)
        trans = torch.where(vmask, trans, fb_t)

    return quats, trans


def build_coarse_grid(A_batch: torch.Tensor,
                      B_batch: torch.Tensor,
                      N_real: torch.Tensor,
                      M_real: torch.Tensor,
                      num_seeds: int = 50,
                      *,
                      trans_centers_batch: Optional[torch.Tensor] = None,
                      trans_centers_real: Optional[torch.Tensor] = None,
                      num_repeats_per_trans: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
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
    q_grid : torch.Tensor (B, G, 4)
        Quaternion grid. When `trans_centers_batch` is None, G=500. When
        `trans_centers_batch` is provided, G is the legacy translation-seeded
        initialization count (= 10*P + 5 for P translation centers).
    t_grid : torch.Tensor (B, G, 3)
        Translation grid
    """
    device = A_batch.device
    BATCH = A_batch.shape[0]

    if trans_centers_batch is not None:
        if trans_centers_real is None:
            trans_centers_real = torch.full(
                (BATCH,), trans_centers_batch.shape[1], device=device, dtype=torch.int32
            )

        qs, ts = [], []
        expected_G: Optional[int] = None
        for i in range(BATCH):
            p = int(trans_centers_real[i].item())
            q_i, t_i = legacy_seeds_with_translations_torch(
                A_batch[i, :N_real[i]],
                B_batch[i, :M_real[i]],
                trans_centers_batch[i, :p],
                num_repeats_per_trans=num_repeats_per_trans,
            )
            if expected_G is None:
                expected_G = q_i.shape[0]
            elif q_i.shape[0] != expected_G:
                raise ValueError(
                    "Translation-seeded coarse grids require equal seed counts per pair. "
                    "Bucket pairs by trans_centers_real before calling."
                )
            qs.append(q_i)
            ts.append(t_i)

        return torch.stack(qs, dim=0), torch.stack(ts, dim=0)

    # GPU-native, fully batched seeds (no per-pair CPU/numpy PCA round-trip).
    quats, t_seeds = batched_seeds_torch(A_batch, B_batch, N_real, M_real, num_seeds=num_seeds)

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

    # Cartesian product: n_rot rotations × 2 translations.  n_rot is derived from
    # the actual rotation count (5*num_seeds) rather than hardcoded, so denser
    # grids (num_seeds != 50) stay self-consistent between q_grid and t_grid.
    n_rot = q_base.size(1)
    q_grid = q_base[:, :, None, :].expand(-1, -1, 2, -1).reshape(BATCH, -1, 4)
    t_grid = t_base[:, None, :, :].expand(-1, n_rot, -1, -1).reshape(BATCH, -1, 3)

    return q_grid, t_grid
