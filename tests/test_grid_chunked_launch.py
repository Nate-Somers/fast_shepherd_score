"""A batch larger than one CUDA grid slice must score the same as one split across slices.

``drivers/terms._chunked`` slices a launch at the ``grid.z <= 65535`` hardware limit. The
coordinate tensors are per-MOLECULE and the pose tensors per POSE, and the real-point counts
``N_real`` / ``M_real`` are per-molecule too -- so they must be sliced with the molecules. They
were not: they were passed whole in ``kw``, so from the second grid slice on, every pose was
scored against a different molecule's atom count.

It is invisible below the limit. One slice needs no slicing, so every batch under 65,535 poses
is correct, which is every test in this suite, every CPU run (the numba kernels have no grid
limit and take one call) and every small GPU batch. It surfaced only in a 100,000-pair
benchmark cell: measured on an L40S, a 21,919-pair ``vol`` chunk (219,190 poses) returned 8,395
of 30,000 Tanimotos above 1, up to 7.3e3, and ``vol_esp`` up to 1.1e5.

These tests drive ``_chunked`` directly with a tiny stand-in limit, so they run on CPU and cost
milliseconds. The property is the one that matters: chunking must not change the answer.
"""
import numpy as np
import pytest

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

pytestmark = pytest.mark.skipif(not TORCH, reason="PyTorch required")


def _fake_kernel(A, B, q, t, *, alpha, N_real, M_real, NEED_GRAD=True, **extra):
    """Stand-in with the real kernels' contract: row i of the per-molecule args goes with
    ``N_real[i]``. Returns a value that DEPENDS on the count, so a misaligned slice shows up."""
    K = q.shape[0]
    S = K // A.shape[0]
    n = N_real.to(torch.float32).repeat_interleave(S)
    m = M_real.to(torch.float32).repeat_interleave(S)
    V = n * 1000.0 + m + A[:, 0, 0].repeat_interleave(S) * 1e6
    return V, torch.zeros_like(q), torch.zeros_like(t)


@pytest.mark.parametrize("S", [1, 4])
@pytest.mark.parametrize("n_mol", [10, 100, 257])
def test_chunked_launch_matches_a_single_launch(S, n_mol, monkeypatch):
    """Same inputs, one launch vs many: identical output. Fails if any per-molecule argument is
    not sliced with the molecules."""
    from shepherd_score.accel.drivers import terms as T

    g = torch.Generator().manual_seed(0)
    K = n_mol * S
    A = torch.rand(n_mol, 5, 3, generator=g)
    B = torch.rand(n_mol, 5, 3, generator=g)
    q = torch.rand(K, 4, generator=g)
    t = torch.rand(K, 3, generator=g)
    N_real = torch.randint(1, 6, (n_mol,), generator=g, dtype=torch.int32)
    M_real = torch.randint(1, 6, (n_mol,), generator=g, dtype=torch.int32)
    kw = dict(alpha=0.81, N_real=N_real, M_real=M_real, NEED_GRAD=True)

    monkeypatch.setattr(T, "_CHUNK", 10 ** 9)          # one launch
    whole, _, _ = T._chunked(_fake_kernel, K, S, (A, B), (q, t), kw, {})

    for chunk in (S, 3 * S, 7 * S):
        monkeypatch.setattr(T, "_CHUNK", max(chunk, 1))
        part, _, _ = T._chunked(_fake_kernel, K, S, (A, B), (q, t), kw, {})
        assert torch.equal(whole, part), (
            f"S={S} n_mol={n_mol}: chunking at {chunk} changed {int((whole != part).sum())} of "
            f"{K} values; a per-molecule argument is not being sliced with the molecules")


def test_chunked_keeps_each_seed_group_whole(monkeypatch):
    """A chunk boundary inside a molecule's seed group would make the kernel's own ``pid // S``
    address the wrong molecule, so the step is rounded down to a multiple of S."""
    from shepherd_score.accel.drivers import terms as T

    seen = []

    def spy(A, B, q, t, *, alpha, N_real, M_real, NEED_GRAD=True, **extra):
        seen.append((A.shape[0], q.shape[0]))
        return _fake_kernel(A, B, q, t, alpha=alpha, N_real=N_real, M_real=M_real)

    S, n_mol = 8, 50
    K = n_mol * S
    A = torch.zeros(n_mol, 4, 3); B = torch.zeros(n_mol, 4, 3)
    q = torch.zeros(K, 4); t = torch.zeros(K, 3)
    kw = dict(alpha=0.81, N_real=torch.full((n_mol,), 4, dtype=torch.int32),
              M_real=torch.full((n_mol,), 4, dtype=torch.int32), NEED_GRAD=True)
    monkeypatch.setattr(T, "_CHUNK", 30)               # not a multiple of S
    T._chunked(spy, K, S, (A, B), (q, t), kw, {})

    assert seen, "the kernel was never called"
    for n_rows, n_poses in seen:
        assert n_poses % S == 0, f"a chunk of {n_poses} poses splits a seed group of {S}"
        assert n_poses == n_rows * S, (
            f"{n_poses} poses against {n_rows} molecule rows: the two slices disagree")
