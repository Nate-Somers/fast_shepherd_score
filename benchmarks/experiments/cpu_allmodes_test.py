"""Validate + benchmark the CPU (numba) batched aligner across ALL modes.

Confirms every mode's batched driver runs on a CPU-only box (no triton/jax) via the
cpu_overlap fallbacks, and is numerically correct:
  - vol      : self-copy=1.0, distinct vs torch ref (see cpu_vol_integration_test.py)
  - vol_esp  : self-copy=1.0 + single-core throughput        (real molecules + charges)
  - pharm    : self-copy~0.999 + single-core throughput      (real pharmacophores)
  - esp/surf : kernel value+grad vs autograd / torch reference (end-to-end needs open3d -> WSL2)

Single-core (NUMBA_NUM_THREADS=1). Run: python -m benchmarks.experiments.cpu_allmodes_test
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
import math
from unittest.mock import MagicMock
sys.modules.setdefault("open3d", MagicMock())
import numpy as np
import torch
torch.set_num_threads(1)
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.alignment.utils.fast_esp_se3 import coarse_fine_esp_align_many
from shepherd_score.alignment.utils.fast_pharm_se3 import coarse_fine_pharm_align_many
from shepherd_score.alignment.utils.cpu_overlap import (
    _batch_self_overlap_esp, pharm_score_grad_se3_batch, overlap_score_grad_esp_se3_batch)
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score.score.analytical_gradients._torch import (
    build_lookup_tables, compute_overlap_and_grad_pharm, _rotation_matrix_from_unit_quat)

ALPHA = 0.81
SMILES = [("phenol", "Oc1ccccc1"), ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
          ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"), ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
          ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
          ("indomethacin", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1")]


def rot(s):
    rng = np.random.default_rng(s)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3))); Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _embed(smi, seed=42):
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); p = AllChem.ETKDGv3(); p.randomSeed = seed
    AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
    return m


def vol_esp_check():
    LAM = 0.1
    def build(smi):
        m = _embed(smi); m = Chem.RemoveHs(m); AllChem.ComputeGasteigerCharges(m)
        xyz = m.GetConformer().GetPositions().astype(np.float32); xyz -= xyz.mean(0)
        ch = np.nan_to_num(np.array([float(a.GetProp("_GasteigerCharge")) for a in m.GetAtoms()], np.float32))
        return xyz, ch
    mols = [build(s) for _, s in SMILES]; K = len(mols)
    Nmax = max(c.shape[0] for c, _ in mols)
    def pad(cl, ch):
        P = np.zeros((K, Nmax, 3), np.float32); C = np.zeros((K, Nmax), np.float32); r = np.zeros(K, np.int32)
        for i, (c, q) in enumerate(zip(cl, ch)):
            P[i, :c.shape[0]] = c; C[i, :c.shape[0]] = q; r[i] = c.shape[0]
        return torch.from_numpy(P), torch.from_numpy(C), torch.from_numpy(r)
    refs = [m[0] for m in mols]; chs = [m[1] for m in mols]
    fits = [m[0] @ rot(i).T + np.array([2., -1., 3.], np.float32) for i, m in enumerate(mols)]
    A, CA, Nr = pad(refs, chs); B, CB, Mr = pad(fits, chs)
    VAA = _batch_self_overlap_esp(A, CA, Nr, ALPHA, LAM); VBB = _batch_self_overlap_esp(B, CB, Mr, ALPHA, LAM)
    t0 = time.perf_counter()
    sc, _, _ = coarse_fine_esp_align_many(A, B, CA, CB, VAA, VBB, alpha=ALPHA, lam=LAM,
                                          num_seeds=50, steps_fine=100, N_real=Nr, M_real=Mr)
    dt = time.perf_counter() - t0; sc = sc.numpy()
    print(f"  vol_esp : self-copy min={sc.min():.4f} mean={sc.mean():.4f}  | {K/dt:6.1f} pairs/s/core")


def pharm_check():
    DUMMY = 8
    def build(smi):
        m = _embed(smi); ty, an, ve = get_pharmacophores(m, multi_vector=False)
        an = an.astype(np.float32); an -= an.mean(0)
        return ty.astype(np.int64), an, ve.astype(np.float32)
    mols = [build(s) for _, s in SMILES]; K = len(mols); Nmax = max(t.shape[0] for t, _, _ in mols)
    def pad(items, rots=None):
        A = np.zeros((K, Nmax, 3), np.float32); V = np.zeros((K, Nmax, 3), np.float32)
        T = np.full((K, Nmax), DUMMY, np.int64); r = np.zeros(K, np.int64)
        for i, (ty, an, ve) in enumerate(items):
            n = ty.shape[0]
            if rots is not None:
                an = an @ rots[i].T; ve = ve @ rots[i].T
            A[i, :n] = an; V[i, :n] = ve; T[i, :n] = ty; r[i] = n
        return torch.from_numpy(A), torch.from_numpy(V), torch.from_numpy(T), torch.from_numpy(r)
    rots = [rot(i) for i in range(K)]
    Aa, Av, At, Ar = pad(mols); Fa, Fv, Ft, Fr = pad(mols, rots)
    al, Ks, cats = build_lookup_tables(torch.device("cpu"), torch.float32)
    I = torch.eye(3).expand(K, 3, 3).contiguous(); z = torch.zeros(K, 3)
    VAA = pharm_score_grad_se3_batch(I, z, At, At, Aa, Aa, Av, Av, al, Ks, cats, N_real=Ar, M_real=Ar, NEED_GRAD=False)[0]
    VBB = pharm_score_grad_se3_batch(I, z, Ft, Ft, Fa, Fa, Fv, Fv, al, Ks, cats, N_real=Fr, M_real=Fr, NEED_GRAD=False)[0]
    t0 = time.perf_counter()
    sc, _, _ = coarse_fine_pharm_align_many(Aa, Fa, Av, Fv, At, Ft, VAA, VBB, similarity="tanimoto",
                                            num_seeds=50, steps_fine=100, N_real=Ar, M_real=Fr)
    dt = time.perf_counter() - t0; sc = sc.numpy()
    print(f"  pharm   : self-copy min={sc.min():.4f} mean={sc.mean():.4f}  | {K/dt:6.1f} pairs/s/core")


def esp_kernel_parity():
    rng = np.random.default_rng(1); N, M, LAM = 18, 22, 0.3
    A = torch.tensor(5 * rng.standard_normal((1, N, 3)), dtype=torch.float32)
    B = torch.tensor(5 * rng.standard_normal((1, M, 3)), dtype=torch.float32)
    CA = torch.tensor(rng.standard_normal((1, N)), dtype=torch.float32)
    CB = torch.tensor(rng.standard_normal((1, M)), dtype=torch.float32)
    qv = rng.standard_normal(4); qv /= np.linalg.norm(qv)
    q = torch.tensor(qv[None], dtype=torch.float32); t = torch.tensor(rng.standard_normal((1, 3)), dtype=torch.float32)
    qg = q.clone().double().requires_grad_(True); tg = t.clone().double().requires_grad_(True)
    w, x, y, z = qg[0, 0], qg[0, 1], qg[0, 2], qg[0, 3]
    R = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                     torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                     torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])
    Bp = B[0].double() @ R.T + tg[0]
    r2 = ((A[0].double()[:, None, :] - Bp[None, :, :]) ** 2).sum(-1)
    dc = CA[0].double()[:, None] - CB[0].double()[None, :]
    Kc = math.pi ** 1.5 / (2 * ALPHA) ** 1.5
    V = (Kc * torch.exp(-ALPHA / 2 * r2 - dc * dc / LAM)).sum(); V.backward()
    Vk, dQk, dTk = overlap_score_grad_esp_se3_batch(A, B, CA, CB, q, t, alpha=ALPHA, lam=LAM)
    print(f"  esp/surf kernel vs autograd: VAB rel={abs(float(Vk[0])-float(V))/abs(float(V)):.1e} "
          f"dQ={float((dQk[0]-qg.grad[0]).abs().max()):.1e} dT={float((dTk[0]-tg.grad[0]).abs().max()):.1e}")


if __name__ == "__main__":
    print("CPU all-modes validation (single-core, numba fallbacks; vol in cpu_vol_integration_test.py):")
    vol_esp_check()
    pharm_check()
    esp_kernel_parity()
    print("  surf: same shape kernel as vol (import-guarded) -> end-to-end on WSL2 (open3d surfaces)")
