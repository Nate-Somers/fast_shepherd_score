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
                        num_seeds: int = 50,
                        *,
                        ref_shared: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU-native, fully batched replacement for the per-pair seed loop.

    Seed set: identity + 4 principal-component-alignment quaternions + up to 6
    STRUCTURED seeds (+/-90 degree rotations about each reference principal axis, which
    cover the axis SWAPS that PCA alignment alone misses) + a Fibonacci fill for any
    remaining budget -- all with COM-aligning translations.

    NOTE: this is NOT the same seed set as ``alignment._torch._initialize_se3_params``
    (the per-pair / JAX path), which goes straight from the 4 PCA quaternions to a
    Fibonacci fill with no structured axis-swap seeds. At the shipped per-mode seed
    counts the structured seeds absorb most of the budget, so the accelerated backends
    explore DIFFERENT orientations than ``backend="jax"`` and return different (not
    worse) scores. Do not expect cross-backend score equality.

    Computed for the whole cohort in one vectorized pass with NO ``.cpu()``/numpy
    round-trip. The principal-axis
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
    ref_shared       : bool  CALLER GUARANTEE that every row of ``A_batch`` (and of ``N_real``)
        is identical -- a screen broadcasting ONE query across the bucket. The reference
        principal axes are then solved on row 0 and expanded instead of being solved K times.
        Default False, which is byte-identical to the pre-existing code path.

        On CUDA this is NOT bit-identical to the K-row solve. MEASURED on an L40S over a
        100,000-molecule ``vol`` screen, twice, identically: 8 scores of 100,000 move, max
        |delta| 4.1723e-07 (0.00006% of a score), top-1000 unchanged in membership AND
        bit-identical in value. Throughput 2.542 -> 2.462 us/mol.

        On CPU it IS bitwise exact -- 0.0 over a full screen and over K = 8..257 in
        ``tests/test_seed_dedup.py`` -- so the numba backend and the CPU figure panels are
        untouched.

        THE MECHANISM IS NOT ESTABLISHED. Do not repeat the plausible story. A dedicated probe
        (results/batchdep.json, job 21977539) recomputed every intermediate of
        ``_masked_principal_axes`` on a real molecule row at K = 1 vs 300 / 3050 / 12200 /
        96650 and found EVERY stage bitwise identical, and a second probe found all K output
        rows equal to each other and to the K=1 solve at P = 32 and 40. So neither "batch size
        changes the reduction order" nor "row position changes it" reproduces in isolation:
        the effect is specific to some reference clouds and has not been isolated. What is
        established is the magnitude, its reproducibility, and its absence on CPU.

        Callers must establish the guarantee by OBJECT IDENTITY of the tensor they actually
        pass here (``a is b``), not by value and not by mode. ``_scatter_fill`` is a pure copy,
        so one shared source object provably yields bitwise-identical rows. A value-based check
        would be self-validating but would force a host sync HERE, stalling the async prologue;
        the identity check is free host-side. (Note the function already syncs once, at the
        ``bool(valid.all())`` degenerate-pair guard near the end -- that is not a licence to add
        an EARLIER one.)

        Set ``FSS_SEED_REF_DEDUP=verify`` to make every firing also run the full K-row solve and
        raise if the two disagree beyond ``FSS_SEED_REF_DEDUP_TOL`` (default 1e-6). That is the
        only check that tests the CALLER'S PREDICATE rather than this function's handling of the
        flag; a wrongly-broadcast reference shows up as O(1) axis disagreement, not rounding.

    Returns
    -------
    quats : (K, num_seeds, 4)   t : (K, num_seeds, 3)
    """
    from shepherd_score.alignment._torch import _quats_from_fibo

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
    # ---- reference axes: solved ONCE when the caller guarantees a shared query ----
    # In a screen every row of A_batch is the same broadcast query, so the K-row eigensolve
    # returns K copies of one answer. A_batch/mask_n reach the eigensolve at exactly this one
    # site (A64 and mask_n64 have no other consumer), so slicing to row 0 here is complete and
    # nothing downstream sees a shape change -- ref_axes is expanded back to (K,3,3) below.
    _dedup = bool(ref_shared) and K > 1
    A64 = torch.nan_to_num((A_batch[:1] if _dedup else A_batch).to(_wd))
    B64 = torch.nan_to_num(B_batch.to(_wd))
    mask_n64 = (mask_n[:1] if _dedup else mask_n).to(_wd)
    mask_m64 = mask_m.to(_wd)

    ref_axes = _masked_principal_axes(A64, mask_n64)                 # (1,3,3) if _dedup else (K,3,3)
    if _dedup:
        if os.environ.get("FSS_SEED_REF_DEDUP") == "verify":
            # Tests the CALLER'S guarantee, which no unit test can reach: the identity
            # predicate lives in the aligners and MoleculePair builds a fresh ref tensor per
            # pair, so a wrong predicate is only observable on a real screen. A legitimate
            # firing differs from the full solve by rounding; a wrongly-broadcast reference
            # differs by O(1).
            _full = _masked_principal_axes(torch.nan_to_num(A_batch.to(_wd)), mask_n.to(_wd))
            _tol = float(os.environ.get("FSS_SEED_REF_DEDUP_TOL", "1e-6"))
            _dev = float((_full - ref_axes.expand(K, 3, 3)).abs().max())
            if _dev > _tol:
                raise RuntimeError(
                    f"batched_seeds_torch(ref_shared=True): reference rows are NOT identical -- "
                    f"row-0 axes disagree with the full {K}-row solve by {_dev:.6e} > {_tol:.1e}. "
                    f"The caller's identity predicate is wrong. Unset FSS_SEED_REF_DEDUP to "
                    f"silence this check; fix the caller to actually silence the bug.")
        # .contiguous(): ref_axes is later re-dtyped and .view()ed for the structured seeds,
        # and a stride-0 expanded tensor cannot serve a view. K*9 elements, so this is free.
        ref_axes = ref_axes.expand(K, 3, 3).contiguous()             # (K,3,3)
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
        if ax == 0:
            # ax=0 ONLY: fit4 is still four IDENTICAL copies of fit_c here (it is first rotated
            # at the bottom of this loop), so the (4K,Mpad,3) eigensolve computes the same answer
            # four times per molecule. Solve the K distinct rows and expand.
            #
            # BIT-IDENTICAL, verified: torch.equal(_masked_principal_axes(fit4, mask_m4),
            # _masked_principal_axes(fit_c, mask_m64).unsqueeze(1).repeat(1,4,1,1).reshape(4K,3,3))
            # over randomized clouds -- max|diff| exactly 0.
            #
            # This is float64 work on a GPU that runs fp64 at 1/64 the fp32 rate, and seed
            # generation measured 2.079 device us/mol -- 49% of fss's entire GPU budget for a vol
            # screen, more than the optimizer it feeds. At ax=1 the four copies HAVE been rotated
            # by four different ref sign-flips, so that solve is genuine work and stays.
            #
            # fit_c depends only on the library molecule (its own coords and COM), never on the
            # query, so this same tensor is also precomputable at store-build time.
            _K = fit_c.shape[0]
            fit_axes = (_masked_principal_axes(fit_c, mask_m64)
                        .unsqueeze(1).repeat(1, 4, 1, 1).reshape(4 * _K, 3, 3))
        else:
            fit_axes = _masked_principal_axes(fit4, mask_m4)        # (4K,3,3)
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


# --------------------------------------------------------------------------------------------
# Canonical-frame seeds: the same rotations for EVERY library molecule.
# --------------------------------------------------------------------------------------------
#: Proper sign-flip combinations of a principal frame. PCA fixes the axes only up to sign, which
#: is exactly what batched_seeds_torch's "4 principal-component-alignment quaternions" enumerate.
#: det = +1 for all four, so each is a rotation rather than a reflection.
_SIGN_FLIPS = (
    ((1, 1, 1)), ((1, -1, -1)), ((-1, 1, -1)), ((-1, -1, 1)),
)
#: +/-90 degree rotations about each canonical axis -- the axis SWAPS that sign flips alone miss.
#: Same role as the "STRUCTURED seeds" in the per-molecule generator.
_AXIS_SWAPS = (
    (0, 90), (0, -90), (1, 90), (1, -90), (2, 90), (2, -90),
)


def canonical_seed_quats(ref_points, n_real, num_seeds: int, device):
    """Constant seed rotations for a CANONICAL store, returned as ``(num_seeds, 4)``.

    On a canonical store every library molecule is already expressed in its own principal frame,
    so the rotation that carries a fit molecule's axes onto the query's is the SAME for all of
    them: ``R_query^T`` composed with a sign-flip / axis-swap. The per-molecule eigensolve that
    ``batched_seeds_torch`` performs -- measured at 44.1% of a vol screen, 0.839 us/mol -- is
    therefore redundant, and these seeds cost one 3x3 solve per SCREEN instead of one per
    molecule.

    Seed ORDER and composition deliberately mirror the per-molecule generator: the four proper
    sign flips first (its "4 PCA quaternions"), then the +/-90 axis swaps (its "STRUCTURED
    seeds"), then a Fibonacci fill for any remaining budget. Same coverage, constant cost.

    This is NOT bit-identical to the per-molecule seeds -- the frames differ by each molecule's
    own rotation -- so scores move and enrichment has to be revalidated.
    """
    import numpy as np
    import torch

    pts = ref_points[:int(n_real)] if n_real is not None else ref_points
    pts = np.asarray(pts, dtype=np.float64)
    c = pts - pts.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    v = v[:, ::-1]                                   # descending eigenvalue order
    if np.linalg.det(v) < 0:
        v[:, 2] = -v[:, 2]                           # keep it a proper rotation
    Rq = v.T                                         # query original -> query canonical

    mats = []
    for sx, sy, sz in _SIGN_FLIPS:
        mats.append(np.diag([sx, sy, sz]).astype(np.float64))
    for ax, deg in _AXIS_SWAPS:
        th = np.deg2rad(deg)
        ca, sa = np.cos(th), np.sin(th)
        R = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][ax]
        R[i, i] = ca; R[j, j] = ca
        R[i, j] = -sa if ax != 1 else sa
        R[j, i] = sa if ax != 1 else -sa
        mats.append(R)
    if len(mats) < num_seeds:                        # Fibonacci fill, same role as the generator
        k = num_seeds - len(mats)
        ga = np.pi * (3.0 - np.sqrt(5.0))
        for t in range(k):
            z = 1.0 - 2.0 * (t + 0.5) / k
            r = np.sqrt(max(0.0, 1.0 - z * z))
            a = ga * t
            axis = np.array([r * np.cos(a), r * np.sin(a), z])
            axis /= max(np.linalg.norm(axis), 1e-12)
            th = np.pi * (t + 1) / (k + 1)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            mats.append(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))

    quats = np.empty((num_seeds, 4), dtype=np.float64)
    for i, M in enumerate(mats[:num_seeds]):
        R = Rq.T @ M                                 # canonical fit axes -> query frame
        tr = np.trace(R)
        if tr > 0:
            sq = np.sqrt(tr + 1.0) * 2
            q = [0.25 * sq, (R[2, 1] - R[1, 2]) / sq, (R[0, 2] - R[2, 0]) / sq,
                 (R[1, 0] - R[0, 1]) / sq]
        else:
            i0 = int(np.argmax(np.diag(R)))
            i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
            sq = np.sqrt(max(1e-12, 1.0 + R[i0, i0] - R[i1, i1] - R[i2, i2])) * 2
            q = [0.0, 0.0, 0.0, 0.0]
            q[0] = (R[i2, i1] - R[i1, i2]) / sq
            q[i0 + 1] = 0.25 * sq
            q[i1 + 1] = (R[i1, i0] + R[i0, i1]) / sq
            q[i2 + 1] = (R[i2, i0] + R[i0, i2]) / sq
        q = np.asarray(q, dtype=np.float64)
        quats[i] = q / max(np.linalg.norm(q), 1e-12)
    return torch.as_tensor(quats, dtype=torch.float32, device=device)
