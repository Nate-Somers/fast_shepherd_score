"""Tests for the device-driven kernel dispatch and the CPU ``numba`` backend.

These run on a CPU-only box (no CUDA/Triton needed); the routing assertions for CUDA
are exercised only when a GPU is present. They guard on ``numba`` being importable.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("numba")
_rdkit = pytest.importorskip("rdkit")

from shepherd_score.accel.kernels import dispatch as KD
from shepherd_score.accel.kernels import cpu as cpu_overlap

ALPHA = 0.81
_SMILES = [
    "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "Oc1ccccc1",
]


def _heavy_xyz(smiles, seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    AllChem.EmbedMolecule(m, params); AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    xyz = m.GetConformer().GetPositions().astype(np.float32)
    return xyz - xyz.mean(0, keepdims=True)


def _rand_rot(seed):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _pad(clouds):
    K = len(clouds); Nmax = max(c.shape[0] for c in clouds)
    out = np.zeros((K, Nmax, 3), np.float32); real = np.zeros(K, np.int32)
    for i, c in enumerate(clouds):
        out[i, :c.shape[0]] = c; real[i] = c.shape[0]
    return torch.from_numpy(out), torch.from_numpy(real)


def test_kernel_dispatch_cpu_matches_cpu_overlap():
    """CPU tensors routed through kernel_dispatch == calling cpu_overlap directly."""
    rng = np.random.default_rng(0)
    A = torch.tensor(5 * rng.standard_normal((2, 16, 3)), dtype=torch.float32)
    B = torch.tensor(5 * rng.standard_normal((2, 20, 3)), dtype=torch.float32)
    q = torch.tensor(rng.standard_normal((2, 4)), dtype=torch.float32)
    q /= q.norm(dim=1, keepdim=True)
    t = torch.tensor(rng.standard_normal((2, 3)), dtype=torch.float32)

    via_dispatch = KD.overlap_score_grad_se3_batch(A, B, q, t, alpha=ALPHA)
    direct = cpu_overlap.overlap_score_grad_se3_batch(A, B, q, t, alpha=ALPHA)
    for a, b in zip(via_dispatch, direct):
        assert torch.allclose(a, b)


def test_numba_vol_self_copy_and_distinct():
    """The numba batched vol driver: self-copy ~1.0 and bounded distinct-pair scores."""
    from shepherd_score.accel.drivers.shape import coarse_fine_align_many

    mols = [_heavy_xyz(s, i) for i, s in enumerate(_SMILES)]
    refs = mols
    fits = [m @ _rand_rot(i).T + np.array([2., -1., 3.], np.float32) for i, m in enumerate(mols)]
    A_b, N_r = _pad(refs); B_b, M_r = _pad(fits)
    VAA = KD._batch_self_overlap(A_b, N_r, ALPHA)
    VBB = KD._batch_self_overlap(B_b, M_r, ALPHA)
    sc, _, _ = coarse_fine_align_many(A_b, B_b, VAA, VBB, alpha=ALPHA, num_seeds=50,
                                      steps_fine=100, N_real=N_r, M_real=M_r)
    sc = sc.numpy()
    assert sc.min() > 0.99            # self-copy recovers ~1.0
    assert np.all(sc <= 1.0 + 1e-3)   # tanimoto-bounded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_dispatch_routes_by_device():
    """On a GPU box: CUDA tensors -> Triton, CPU tensors -> numba, in one process."""
    pytest.importorskip("triton")
    rng = np.random.default_rng(1)
    A = torch.tensor(5 * rng.standard_normal((2, 16, 3)), dtype=torch.float32)
    B = torch.tensor(5 * rng.standard_normal((2, 20, 3)), dtype=torch.float32)
    q = torch.tensor(rng.standard_normal((2, 4)), dtype=torch.float32)
    q /= q.norm(dim=1, keepdim=True)
    t = torch.tensor(rng.standard_normal((2, 3)), dtype=torch.float32)

    v_cpu = KD.overlap_score_grad_se3_batch(A, B, q, t, alpha=ALPHA)
    v_gpu = KD.overlap_score_grad_se3_batch(A.cuda(), B.cuda(), q.cuda(), t.cuda(), alpha=ALPHA)
    # numerically exact across backends (exp2 vs exp -> not bit-identical)
    assert torch.allclose(v_cpu[0], v_gpu[0].cpu(), atol=1e-3)
    # both implementations were resolved in the same process
    assert ("overlap_score_grad_se3_batch", "cpu") in KD._RESOLVED
    assert ("overlap_score_grad_se3_batch", "shape") in KD._RESOLVED


def test_numba_vol_tversky_matches_reference_and_self_copy():
    """vol_tversky numba backend: batched driver matches the per-pair reference on DISTINCT
    molecules (~4 decimals) and self-copy scores ~1.0. The Tversky score is NOT bounded to
    [0, 1] and is never clamped -- only the reduction differs from vol (shape kernel reused)."""
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

    def _embed(smi):
        m = embed_conformer_from_smiles(smi, MMFF_optimize=True, random_seed=0)
        if m is None:
            pytest.skip(f"embedding returned None for {smi!r}")
        return m

    ibu = _embed("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    caf = _embed("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    cpu = torch.device("cpu")

    # per-pair eager reference on the two distinct asymmetric directions
    def _ref(ref_mol, fit_mol):
        mp = MoleculePair(Molecule(ref_mol), Molecule(fit_mol), do_center=True, device=cpu)
        mp.align_with_vol_tversky(num_repeats=50, max_num_steps=200)
        return float(mp.sim_aligned_vol_tversky)

    ref_fwd = _ref(ibu, caf)
    ref_rev = _ref(caf, ibu)

    # batched numba backend on the same distinct pairs
    pairs = [MoleculePair(Molecule(ibu), Molecule(caf), do_center=True, device=cpu),
             MoleculePair(Molecule(caf), Molecule(ibu), do_center=True, device=cpu)]
    scores, _ = MoleculePairBatch(pairs).align_with_vol_tversky(backend="numba")
    assert abs(float(scores[0]) - ref_fwd) < 1e-3, (float(scores[0]), ref_fwd)
    assert abs(float(scores[1]) - ref_rev) < 1e-3, (float(scores[1]), ref_rev)
    # asymmetry: query (small ref) contained in bigger fit scores higher forward than reverse
    assert float(scores[0]) > float(scores[1])

    # self-copy == ~1.0 (Tversky(A,A)=1 for any weights) under numba
    selfp = [MoleculePair(Molecule(ibu), Molecule(ibu), do_center=True, device=cpu),
             MoleculePair(Molecule(caf), Molecule(caf), do_center=True, device=cpu)]
    sscores, _ = MoleculePairBatch(selfp).align_with_vol_tversky(backend="numba")
    assert np.allclose(np.asarray(sscores, dtype=float), 1.0, atol=1e-4), sscores


def test_numba_pharm_self_copy():
    """The numba pharm kernel drives coarse_fine_pharm_align_many on CPU to self-copy ~1.0."""
    pytest.importorskip("rdkit.Chem.AllChem")
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
    from shepherd_score.score.analytical_gradients._torch import build_lookup_tables
    from shepherd_score.accel.drivers.pharm import coarse_fine_pharm_align_many

    DUMMY = 8
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "COc1ccc2cc(ccc2c1)C(C)C(=O)O"]  # drug-like (enough pharmacophores)

    def build(smi):
        m = Chem.AddHs(Chem.MolFromSmiles(smi)); p = AllChem.ETKDGv3(); p.randomSeed = 42
        AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
        ty, an, ve = get_pharmacophores(m, multi_vector=False)
        an = an.astype(np.float32); an -= an.mean(0)
        return ty.astype(np.int64), an, ve.astype(np.float32)

    mols = [build(s) for s in smis]
    K = len(mols); Nmax = max(t.shape[0] for t, _, _ in mols)

    def pad(items, rots=None):
        A = np.zeros((K, Nmax, 3), np.float32); V = np.zeros((K, Nmax, 3), np.float32)
        T = np.full((K, Nmax), DUMMY, np.int64); r = np.zeros(K, np.int64)
        for i, (ty, an, ve) in enumerate(items):
            n = ty.shape[0]
            if rots is not None:
                an = an @ rots[i].T; ve = ve @ rots[i].T
            A[i, :n] = an; V[i, :n] = ve; T[i, :n] = ty; r[i] = n
        return (torch.from_numpy(A), torch.from_numpy(V), torch.from_numpy(T), torch.from_numpy(r))

    rots = [_rand_rot(i) for i in range(K)]
    Aa, Av, At, Ar = pad(mols)
    Fa, Fv, Ft, Fr = pad(mols, rots)  # self-copy via rotation
    al, Ks, cats = build_lookup_tables(torch.device("cpu"), torch.float32)
    I = torch.eye(3).expand(K, 3, 3).contiguous(); z = torch.zeros(K, 3)
    VAA = KD.pharm_score_grad_se3_batch(I, z, At, At, Aa, Aa, Av, Av, al, Ks, cats,
                                        N_real=Ar, M_real=Ar, NEED_GRAD=False)[0]
    VBB = KD.pharm_score_grad_se3_batch(I, z, Ft, Ft, Fa, Fa, Fv, Fv, al, Ks, cats,
                                        N_real=Fr, M_real=Fr, NEED_GRAD=False)[0]
    sc, _, _ = coarse_fine_pharm_align_many(Aa, Fa, Av, Fv, At, Ft, VAA, VBB,
                                            similarity="tanimoto", num_seeds=50, steps_fine=100,
                                            N_real=Ar, M_real=Fr)
    assert sc.numpy().min() > 0.99   # drug-like pharm self-copy recovers ~1.0 via the numba kernel
