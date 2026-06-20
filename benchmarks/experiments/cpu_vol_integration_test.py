"""Validate + benchmark the CPU (numba) batched vol aligner wired into coarse_fine_align_many.

Three checks:
  1. fast_se3 imports on a CPU-only box (triton absent) via the cpu_overlap fallback.
  2. The numba overlap kernel's value+gradient match a torch autograd reference (<1e-4).
  3. The full coarse_fine_align_many driver, on REAL drug heavy-atom clouds:
       - self-copy score == 1.000  (correctness of the whole pipeline)
       - distinct-pair scores match the torch per-pair reference optimize_ROCS_overlay_analytical
     plus single-core throughput vs that reference.

Run: python -m benchmarks.experiments.cpu_vol_integration_test
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")  # single-core measurement
import sys
import time
from unittest.mock import MagicMock
sys.modules.setdefault("open3d", MagicMock())
import math
import numpy as np
import torch
torch.set_num_threads(1)
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.alignment.utils.cpu_overlap import overlap_score_grad_se3_batch
from shepherd_score.alignment.utils import fast_se3
from shepherd_score.alignment.utils.fast_se3 import coarse_fine_align_many, _HAS_TRITON
from shepherd_score.alignment.utils.cpu_overlap import _batch_self_overlap
from shepherd_score.alignment._torch_analytical import optimize_ROCS_overlay_analytical

ALPHA = 0.81
DRUGS = [
    ("benzene", "c1ccccc1"), ("phenol", "Oc1ccccc1"), ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"),
]


def heavy_xyz(smiles, seed=42):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    xyz = m.GetConformer().GetPositions().astype(np.float32)
    return xyz - xyz.mean(0, keepdims=True)


def rand_rot(seed):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def pad_batch(clouds):
    K = len(clouds); Nmax = max(c.shape[0] for c in clouds)
    out = np.zeros((K, Nmax, 3), np.float32); real = np.zeros(K, np.int32)
    for i, c in enumerate(clouds):
        out[i, :c.shape[0]] = c; real[i] = c.shape[0]
    return torch.from_numpy(out), torch.from_numpy(real)


def check_kernel_grad():
    """numba kernel value+grad vs torch autograd (same q=(w,x,y,z) convention)."""
    rng = np.random.default_rng(0)
    N, M = 20, 24
    A = torch.tensor(5 * rng.standard_normal((1, N, 3)), dtype=torch.float32)
    B = torch.tensor(5 * rng.standard_normal((1, M, 3)), dtype=torch.float32)
    qv = rng.standard_normal(4); qv /= np.linalg.norm(qv)
    q = torch.tensor(qv[None], dtype=torch.float32)
    t = torch.tensor(rng.standard_normal((1, 3)), dtype=torch.float32)

    qg = q.clone().double().requires_grad_(True); tg = t.clone().double().requires_grad_(True)
    w, x, y, z = qg[0, 0], qg[0, 1], qg[0, 2], qg[0, 3]
    R = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                     torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                     torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])
    Bp = B[0].double() @ R.T + tg[0]
    r2 = ((A[0].double()[:, None, :] - Bp[None, :, :]) ** 2).sum(-1)
    Kc = math.pi ** 1.5 / (2 * ALPHA) ** 1.5
    Vab = Kc * torch.exp(-ALPHA / 2 * r2).sum()
    Vab.backward()

    V, dQ, dT = overlap_score_grad_se3_batch(A, B, q, t, alpha=ALPHA)
    print(f"  VAB:  kernel {float(V[0]):.5f}  autograd {float(Vab):.5f}  rel={abs(float(V[0])-float(Vab))/abs(float(Vab)):.1e}")
    print(f"  dQ max|d|={float((dQ[0]-qg.grad[0]).abs().max()):.1e}   dT max|d|={float((dT[0]-tg.grad[0]).abs().max()):.1e}")


def score_ref(ref, fit, nr=50):
    out = optimize_ROCS_overlay_analytical(torch.from_numpy(ref), torch.from_numpy(fit),
                                           ALPHA, num_repeats=nr, max_num_steps=200, lr=0.1)
    s = out[-1]
    return float(s.item() if hasattr(s, "item") else np.asarray(s).reshape(-1)[-1])


if __name__ == "__main__":
    print(f"fast_se3 imported OK. _HAS_TRITON={_HAS_TRITON} (False => using cpu_overlap fallback)\n")
    print("1) numba kernel value+grad vs torch autograd:")
    check_kernel_grad()

    print("\n2) Building real conformers...")
    mols = {n: heavy_xyz(s) for n, s in DRUGS}
    names = list(mols)

    # self-copy batch
    refs = [mols[n] for n in names]
    fits = [mols[n] @ rand_rot(i).T + np.array([2., -1., 3.], np.float32) for i, n in enumerate(names)]
    A_b, N_r = pad_batch(refs); B_b, M_r = pad_batch(fits)
    VAA = _batch_self_overlap(A_b, N_r, ALPHA); VBB = _batch_self_overlap(B_b, M_r, ALPHA)
    sc, _, _ = coarse_fine_align_many(A_b, B_b, VAA, VBB, alpha=ALPHA, num_seeds=50,
                                      steps_fine=100, N_real=N_r, M_real=M_r)
    sc = sc.numpy()
    print(f"\n3) SELF-COPY (driver on real mols): min={sc.min():.4f} mean={sc.mean():.4f} (expect ~1.000)")

    # distinct pairs
    rng = np.random.default_rng(7)
    pairs = [tuple(rng.choice(len(names), 2, replace=False)) for _ in range(24)]
    refs_d = [mols[names[i]] for i, j in pairs]; fits_d = [mols[names[j]] for i, j in pairs]
    A_d, Nd = pad_batch(refs_d); B_d, Md = pad_batch(fits_d)
    VAAd = _batch_self_overlap(A_d, Nd, ALPHA); VBBd = _batch_self_overlap(B_d, Md, ALPHA)
    t0 = time.perf_counter()
    sc_d, _, _ = coarse_fine_align_many(A_d, B_d, VAAd, VBBd, alpha=ALPHA, num_seeds=50,
                                        steps_fine=100, N_real=Nd, M_real=Md)
    dt_cpu = time.perf_counter() - t0
    sc_d = sc_d.numpy()
    t0 = time.perf_counter()
    ref_sc = np.array([score_ref(refs_d[k], fits_d[k]) for k in range(len(pairs))])
    dt_ref = time.perf_counter() - t0
    diff = np.abs(sc_d - ref_sc)
    print(f"   DISTINCT vs torch per-pair ref: max|d|={diff.max():.2e} mean|d|={diff.mean():.2e}")
    print(f"   (driver scores >= ref on most pairs is expected: same seeds, take-the-max)")
    print(f"\n4) THROUGHPUT (single-core, {len(pairs)} distinct pairs):")
    print(f"   batched numba driver : {len(pairs)/dt_cpu:7.1f} pairs/s/core ({dt_cpu/len(pairs)*1e3:.1f} ms/pair)")
    print(f"   torch per-pair ref    : {len(pairs)/dt_ref:7.1f} pairs/s/core ({dt_ref/len(pairs)*1e3:.1f} ms/pair)")
    print(f"   speedup: {dt_ref/dt_cpu:.2f}x")
