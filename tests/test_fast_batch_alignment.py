import math

import pytest
import torch


def _require_fast_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")


@pytest.mark.cuda
def test_fast_esp_batch_matches_single():
    _require_fast_cuda()

    from shepherd_score.alignment_utils.fast_esp_se3 import (
        fast_optimize_ROCS_esp_overlay,
        fast_optimize_ROCS_esp_overlay_batch,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    N, M = 32, 27
    ref = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.5
    fit = torch.randn(M, 3, device=device, dtype=torch.float32) * 0.5
    ref_c = torch.randn(N, device=device, dtype=torch.float32) * 0.1
    fit_c = torch.randn(M, device=device, dtype=torch.float32) * 0.1

    alpha = 0.81
    lam = 0.01

    _, _, score_single = fast_optimize_ROCS_esp_overlay(
        ref,
        fit,
        ref_c,
        fit_c,
        alpha=alpha,
        lam=lam,
        num_repeats=1,
        topk=3,
        steps_fine=5,
        lr=0.075,
    )

    ref_b = ref.unsqueeze(0)
    fit_b = fit.unsqueeze(0)
    ref_c_b = ref_c.unsqueeze(0)
    fit_c_b = fit_c.unsqueeze(0)
    N_real = torch.tensor([N], device=device, dtype=torch.int32)
    M_real = torch.tensor([M], device=device, dtype=torch.int32)

    _, _, _, score_batch = fast_optimize_ROCS_esp_overlay_batch(
        ref_b,
        fit_b,
        ref_c_b,
        fit_c_b,
        alpha=alpha,
        lam=lam,
        N_real=N_real,
        M_real=M_real,
        topk=3,
        steps_fine=5,
        lr=0.075,
    )

    s1 = float(score_single)
    s2 = float(score_batch[0])
    assert math.isfinite(s1) and math.isfinite(s2)
    assert abs(s1 - s2) < 1e-4


@pytest.mark.cuda
def test_fast_esp_combo_padding_masks_stable():
    _require_fast_cuda()

    from shepherd_score.alignment_utils.fast_esp_combo_se3 import (
        fast_optimize_esp_combo_score_overlay_batch,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")

    # Construct two pairs with different atom counts so padding is required.
    K = 2
    n_wH = [9, 2]
    m_wH = [8, 3]
    n_surf = [40, 40]
    m_surf = [41, 41]

    n_wH_pad = max(n_wH)
    m_wH_pad = max(m_wH)
    n_surf_pad = max(n_surf)
    m_surf_pad = max(m_surf)

    centers_w_H_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
    centers_w_H_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)
    partial_1 = torch.zeros(K, n_wH_pad, device=device, dtype=torch.float32)
    partial_2 = torch.zeros(K, m_wH_pad, device=device, dtype=torch.float32)
    radii_1 = torch.ones(K, n_wH_pad, device=device, dtype=torch.float32) * 1.5
    radii_2 = torch.ones(K, m_wH_pad, device=device, dtype=torch.float32) * 1.5

    # Shape centers for alpha==0.81 use volumetric atoms.
    centers_1 = torch.zeros(K, n_wH_pad, 3, device=device, dtype=torch.float32)
    centers_2 = torch.zeros(K, m_wH_pad, 3, device=device, dtype=torch.float32)

    # Surface points: choose points near origin so that unmasked padded atoms at origin would matter.
    points_1 = torch.zeros(K, n_surf_pad, 3, device=device, dtype=torch.float32)
    points_2 = torch.zeros(K, m_surf_pad, 3, device=device, dtype=torch.float32)
    point_charges_1 = torch.zeros(K, n_surf_pad, device=device, dtype=torch.float32)
    point_charges_2 = torch.zeros(K, m_surf_pad, device=device, dtype=torch.float32)

    for i in range(K):
        centers_w_H_1[i, : n_wH[i]] = torch.randn(n_wH[i], 3, device=device) * 0.2
        centers_w_H_2[i, : m_wH[i]] = torch.randn(m_wH[i], 3, device=device) * 0.2
        centers_1[i, : n_wH[i]] = centers_w_H_1[i, : n_wH[i]]
        centers_2[i, : m_wH[i]] = centers_w_H_2[i, : m_wH[i]]

        points_1[i, : n_surf[i]] = torch.randn(n_surf[i], 3, device=device) * 0.3
        points_2[i, : m_surf[i]] = torch.randn(m_surf[i], 3, device=device) * 0.3
        point_charges_1[i, : n_surf[i]] = torch.randn(n_surf[i], device=device) * 0.05
        point_charges_2[i, : m_surf[i]] = torch.randn(m_surf[i], device=device) * 0.05

    N_real_atoms_w_H_1 = torch.tensor(n_wH, device=device, dtype=torch.int32)
    M_real_atoms_w_H_2 = torch.tensor(m_wH, device=device, dtype=torch.int32)
    N_real_centers = N_real_atoms_w_H_1.clone()
    M_real_centers = M_real_atoms_w_H_2.clone()
    N_real_surf_1 = torch.tensor(n_surf, device=device, dtype=torch.int32)
    M_real_surf_2 = torch.tensor(m_surf, device=device, dtype=torch.int32)

    alpha = 0.81
    scores_full = fast_optimize_esp_combo_score_overlay_batch(
        centers_w_H_1,
        centers_w_H_2,
        centers_1,
        centers_2,
        points_1,
        points_2,
        partial_1,
        partial_2,
        point_charges_1,
        point_charges_2,
        radii_1,
        radii_2,
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1,
        M_real_atoms_w_H_2=M_real_atoms_w_H_2,
        N_real_centers=N_real_centers,
        M_real_centers=M_real_centers,
        N_real_surf_1=N_real_surf_1,
        M_real_surf_2=M_real_surf_2,
        topk=3,
        steps_fine=5,
        lr=0.075,
    )[3]

    # Slice-out runs should match per-item results when padding is mask-safe.
    scores_slice0 = fast_optimize_esp_combo_score_overlay_batch(
        centers_w_H_1[:1],
        centers_w_H_2[:1],
        centers_1[:1],
        centers_2[:1],
        points_1[:1],
        points_2[:1],
        partial_1[:1],
        partial_2[:1],
        point_charges_1[:1],
        point_charges_2[:1],
        radii_1[:1],
        radii_2[:1],
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1[:1],
        M_real_atoms_w_H_2=M_real_atoms_w_H_2[:1],
        N_real_centers=N_real_centers[:1],
        M_real_centers=M_real_centers[:1],
        N_real_surf_1=N_real_surf_1[:1],
        M_real_surf_2=M_real_surf_2[:1],
        topk=3,
        steps_fine=5,
        lr=0.075,
    )[3]

    scores_slice1 = fast_optimize_esp_combo_score_overlay_batch(
        centers_w_H_1[1:2],
        centers_w_H_2[1:2],
        centers_1[1:2],
        centers_2[1:2],
        points_1[1:2],
        points_2[1:2],
        partial_1[1:2],
        partial_2[1:2],
        point_charges_1[1:2],
        point_charges_2[1:2],
        radii_1[1:2],
        radii_2[1:2],
        alpha,
        lam=0.001,
        probe_radius=1.0,
        esp_weight=0.5,
        N_real_atoms_w_H_1=N_real_atoms_w_H_1[1:2],
        M_real_atoms_w_H_2=M_real_atoms_w_H_2[1:2],
        N_real_centers=N_real_centers[1:2],
        M_real_centers=M_real_centers[1:2],
        N_real_surf_1=N_real_surf_1[1:2],
        M_real_surf_2=M_real_surf_2[1:2],
        topk=3,
        steps_fine=5,
        lr=0.075,
    )[3]

    s_full = scores_full.detach().cpu().tolist()
    assert all(math.isfinite(x) for x in s_full)
    assert abs(float(scores_slice0[0]) - s_full[0]) < 1e-4
    assert abs(float(scores_slice1[0]) - s_full[1]) < 1e-4


def test_pharm_similarity_matches_legacy():
    from shepherd_score.score.pharmacophore_overlap_triton import batch_pharm_similarity
    from shepherd_score.score.pharmacophore_scoring import get_overlap_pharm

    torch.manual_seed(0)

    N, M = 7, 6
    anchors_1 = torch.randn(N, 3, dtype=torch.float32)
    anchors_2 = torch.randn(M, 3, dtype=torch.float32)
    vectors_1 = torch.randn(N, 3, dtype=torch.float32)
    vectors_2 = torch.randn(M, 3, dtype=torch.float32)

    vectors_1 = vectors_1 / (vectors_1.norm(dim=1, keepdim=True) + 1e-8)
    vectors_2 = vectors_2 / (vectors_2.norm(dim=1, keepdim=True) + 1e-8)

    types_1 = torch.randint(0, 8, (N,), dtype=torch.int64)
    types_2 = torch.randint(0, 8, (M,), dtype=torch.int64)

    cases = [
        ("tanimoto", False, False),
        ("tversky", False, False),
        ("tversky_ref", False, False),
        ("tversky_fit", False, False),
        ("tanimoto", True, False),
        ("tversky", True, False),
        ("tanimoto", True, True),
        ("tversky_ref", True, True),
    ]

    for similarity, extended_points, only_extended in cases:
        legacy = get_overlap_pharm(
            ptype_1=types_1,
            ptype_2=types_2,
            anchors_1=anchors_1,
            anchors_2=anchors_2,
            vectors_1=vectors_1,
            vectors_2=vectors_2,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
        )

        batched = batch_pharm_similarity(
            anchors_1.unsqueeze(0),
            anchors_2.unsqueeze(0),
            vectors_1.unsqueeze(0),
            vectors_2.unsqueeze(0),
            types_1.unsqueeze(0),
            types_2.unsqueeze(0),
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            N_real=torch.tensor([N], dtype=torch.int32),
            M_real=torch.tensor([M], dtype=torch.int32),
        )[0]

        assert torch.allclose(legacy, batched, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_fast_pharm_batch_smoke_flags():
    _require_fast_cuda()

    from shepherd_score.alignment_utils.fast_pharm_se3 import fast_optimize_pharm_overlay_batch

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 2
    N = 6
    M = 5

    anchors_1 = torch.randn(B, N, 3, device=device, dtype=torch.float32) * 0.5
    anchors_2 = torch.randn(B, M, 3, device=device, dtype=torch.float32) * 0.5
    vectors_1 = torch.randn(B, N, 3, device=device, dtype=torch.float32)
    vectors_2 = torch.randn(B, M, 3, device=device, dtype=torch.float32)
    types_1 = torch.randint(0, 8, (B, N), device=device, dtype=torch.int64)
    types_2 = torch.randint(0, 8, (B, M), device=device, dtype=torch.int64)

    N_real = torch.tensor([N, N], device=device, dtype=torch.int32)
    M_real = torch.tensor([M, M], device=device, dtype=torch.int32)

    for similarity, extended_points, only_extended in [
        ("tanimoto", False, False),
        ("tversky_ref", True, False),
        ("tversky_fit", True, True),
    ]:
        _, _, _, _, scores = fast_optimize_pharm_overlay_batch(
            types_1,
            types_2,
            anchors_1,
            anchors_2,
            vectors_1,
            vectors_2,
            similarity=similarity,
            extended_points=extended_points,
            only_extended=only_extended,
            num_repeats=1,
            N_real=N_real,
            M_real=M_real,
            topk=2,
            steps_fine=2,
            lr=0.075,
        )
        out = scores.detach().cpu().tolist()
        assert all(math.isfinite(x) for x in out)

