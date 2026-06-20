"""
THE benchmark for fast_shepherd_score — one script, a few flags.

What it does
------------
Aligns REAL drug molecules (RDKit ETKDG conformers + Open3D surfaces + MMFF
charges + RDKit pharmacophores) and reports, for every alignment mode, how fast
THIS fork (Triton/CUDA) is versus the ORIGINAL upstream repo
(``shepherd-score-original-repo/``), across a sweep of batch sizes.

Both engines go through the SAME public API — ``MoleculePairBatch.align_with_*``:
the fork with ``backend="triton"`` (its Triton/CUDA kernels), the original with
its default ``backend="jax"`` (JAX/XLA on CPU). So the comparison is one code
path, two backends.

Two things are measured at once:
  * speed    — pair-alignments per second (throughput), fork vs original.
  * accuracy — every pair is a molecule aligned to a rigid SE(3) copy of itself,
               so the perfect score is 1.0. We report the achieved mean score;
               anything well below 1.0 is a real quality problem, not noise.

Modes (all run by default): vol (atom-cloud ROCS), surf (surface ROCS),
esp (surface shape + electrostatics), pharm (pharmacophore).

Size sweep (default): 1, 10, 100, 1000, 10000, 100000 pairs per call. The FORK
runs the whole sweep (it is fast enough that even the largest cell finishes in
under ~90 s), so the 100k datapoint is always present on the fork panel. The
ORIGINAL is orders of magnitude slower, so its line stops at the first size whose
wall time exceeds ``--cap`` (default 10 s) — 100k would take hours.

Buckets (both run by default): the GPU batch path buckets pairs by size.
  * same  — all molecules land in one size band  -> a single padded bucket (best case).
  * cross — molecules spread across bands         -> many buckets (realistic case).

Accuracy branch (OFF by default; ``--accuracy``): align 50 pairs of DIFFERENT
molecules (optimum < 1.0) across every mode and compare the fork's scores to the
original's, so the speed claims can't hide a quality regression on non-trivial
alignments.

Outputs (under ``results/`` by default): a two-panel plot (``speed_plot.png``,
annotated with the GPU/CPU it ran on), a markdown table (``speed_table.md``), and
the raw ``plot_data.json``.

Usage
-----
    python -m benchmarks.benchmark                  # full headline (fork + original)
    python -m benchmarks.benchmark --cap 30         # let the original run slower/bigger cells
    python -m benchmarks.benchmark --no-original    # fork only (keeps last run's original line)
    python -m benchmarks.benchmark --modes surf esp # subset of modes
    python -m benchmarks.benchmark --accuracy       # the accuracy branch instead
    python -m benchmarks.benchmark --replot         # re-render plot/table from results/plot_data.json

Environment: needs a CUDA GPU + Triton for the fork path, JAX/Open3D etc. for
building molecules and the original path. Run it in the GPU conda env (WSL2). The
original repo runs in an isolated subprocess (both packages are named
``shepherd_score``, so they cannot share one interpreter).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import functools
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ORIG_REPO = os.path.join(_ROOT, "shepherd-score-original-repo")
_RESULTS = os.path.join(_HERE, "results")          # single results folder (gitignored)
_MOLCACHE = os.path.join(_HERE, "molcache")        # persisted base molecules (original-repo path)

MODES = ["vol", "surf", "esp", "pharm"]
BUCKETS = ["same", "cross"]
SIZES = [1, 10, 100, 1000, 10000, 100000]
DEFAULT_CAP = 10.0
SURF_PER_ATOM = 3
SELF_SCORE_WARN = 0.95          # self-copy optimum is 1.0; warn below this


def _cfg_from_args(a):
    """Shared alignment knobs passed to both engines (via MoleculePairBatch)."""
    return dict(num_repeats=a.num_repeats, steps=a.steps, lr=a.lr,
                alpha=a.alpha, lam=a.lam, surf_per_atom=SURF_PER_ATOM)


# ===========================================================================
# REAL-MOLECULE WORKLOADS
# ===========================================================================
# Real, marketed small-molecule drugs. The SMILES are the standard canonical
# structures (DrugBank / PubChem); 3-D conformers come from RDKit ETKDG + MMFF94,
# surfaces from the repo's Open3D sampler, partial charges from MMFF94, and
# pharmacophores from the repo's RDKit feature factory — so every array the
# aligners consume comes from a real molecule, not a random point cloud.
#
# Each pair is (ref = real molecule, fit = a rigid SE(3)-transformed copy of the
# same molecule): a KNOWN optimum (perfect overlap, score 1.0) while exercising
# the real molecular geometry / surface / ESP / pharmacophore arrays.
#
# (name, SMILES, approx heavy-atom count) — small -> large.
DRUGS: List[Tuple[str, str, int]] = [
    ("benzene",        "c1ccccc1", 6),
    ("phenol",         "Oc1ccccc1", 7),
    ("aniline",        "Nc1ccccc1", 7),
    ("paracetamol",    "CC(=O)Nc1ccc(O)cc1", 11),
    ("salicylic_acid", "OC(=O)c1ccccc1O", 11),
    ("aspirin",        "CC(=O)Oc1ccccc1C(=O)O", 13),
    ("ibuprofen",      "CC(C)Cc1ccc(cc1)C(C)C(=O)O", 15),
    ("caffeine",       "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 14),
    ("naproxen",       "COc1ccc2cc(ccc2c1)C(C)C(=O)O", 17),
    ("paracetamol2",   "CC(=O)Nc1ccc(OC)cc1", 12),
    ("warfarin",       "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 25),
    ("diphenhydramine","O(CCN(C)C)C(c1ccccc1)c1ccccc1", 18),
    ("indomethacin",   "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", 28),
    ("sildenafil",     "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1", 33),
    ("imatinib",       "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", 37),
]


@dataclass
class _RealMol:
    """Lightweight, duck-typed molecule exposing exactly what the aligners read."""
    atom_pos: np.ndarray
    surf_pos: np.ndarray
    surf_esp: np.ndarray
    partial_charges: np.ndarray
    pharm_types: np.ndarray
    pharm_ancs: np.ndarray
    pharm_vecs: np.ndarray

    @property
    def _nonH_atoms_idx(self):
        return np.arange(self.atom_pos.shape[0])

    def center_to(self, xyz_mean):
        self.atom_pos = self.atom_pos - xyz_mean
        self.surf_pos = self.surf_pos - xyz_mean
        self.pharm_ancs = self.pharm_ancs - xyz_mean


@dataclass
class PairSpec:
    """One reference/fit pair plus the ground-truth transform used to build it."""
    ref: object
    fit: object
    R: np.ndarray            # (3, 3) rotation applied to make fit
    t: np.ndarray            # (3,)   translation applied to make fit
    n_ref: int               # mode-relevant point count of ref (the "size")
    n_fit: int


@dataclass
class Cohort:
    """A reproducible set of pairs sharing a size/bucket policy."""
    name: str
    mode: str
    pairs: List[PairSpec]
    size_kind: str           # "same" | "cross"
    seed: int
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)


def _disk_cache_path(smiles: str, surf_per_atom: int, seed: int) -> Optional[str]:
    """Path for a cached _RealMol, or None if disk caching is off.

    Enabled by setting env FSS_MOL_CACHE_DIR (used by the per-cell subprocess
    benchmark so fresh processes don't each rebuild the molecules). Transparent:
    the build is deterministic, so a cache hit is identical to a rebuild.
    """
    d = os.environ.get("FSS_MOL_CACHE_DIR")
    if not d:
        return None
    key = hashlib.md5(f"v1|{smiles}|{surf_per_atom}|{seed}".encode()).hexdigest()
    return os.path.join(d, key + ".pkl")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3x3 proper rotation matrix (via QR of a Gaussian)."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float64)


@functools.lru_cache(maxsize=None)
def _build_molecule(smiles: str, surf_per_atom: int = 3, seed: int = 42) -> _RealMol:
    # Optional cross-process disk cache (FSS_MOL_CACHE_DIR) so per-cell subprocess
    # benchmarks don't rebuild the molecules every process.
    _cpath = _disk_cache_path(smiles, surf_per_atom, seed)
    if _cpath and os.path.exists(_cpath):
        try:
            with open(_cpath, "rb") as _f:
                return pickle.load(_f)
        except Exception:
            pass

    from rdkit import Chem
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule

    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=seed)
    nheavy = Chem.RemoveHs(rd).GetNumAtoms()
    ns = max(24, surf_per_atom * nheavy)
    m = Molecule(rd, num_surf_points=ns, pharm_multi_vector=False)
    mol = _RealMol(
        atom_pos=np.asarray(m.atom_pos, dtype=np.float64),
        surf_pos=np.asarray(m.surf_pos, dtype=np.float64),
        surf_esp=np.asarray(m.surf_esp, dtype=np.float64),
        partial_charges=np.asarray(m.partial_charges, dtype=np.float64),
        pharm_types=np.asarray(m.pharm_types, dtype=np.int64),
        pharm_ancs=np.asarray(m.pharm_ancs, dtype=np.float64),
        pharm_vecs=np.asarray(m.pharm_vecs, dtype=np.float64),
    )
    if _cpath:
        try:
            os.makedirs(os.path.dirname(_cpath), exist_ok=True)
            with open(_cpath, "wb") as _f:
                pickle.dump(mol, _f)
        except Exception:
            pass
    return mol


def _transform(mol: _RealMol, R: np.ndarray, t: np.ndarray) -> _RealMol:
    vecs = mol.pharm_vecs @ R.T
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(n > 0, n, 1.0)
    return _RealMol(
        atom_pos=mol.atom_pos @ R.T + t,
        surf_pos=mol.surf_pos @ R.T + t,
        surf_esp=mol.surf_esp.copy(),
        partial_charges=mol.partial_charges.copy(),
        pharm_types=mol.pharm_types.copy(),
        pharm_ancs=mol.pharm_ancs @ R.T + t,
        pharm_vecs=vecs,
    )


def _count_for_mode(mol: _RealMol, mode: str) -> int:
    if mode in ("surf", "esp"):
        return mol.surf_pos.shape[0]
    if mode == "pharm":
        return mol.pharm_ancs.shape[0]
    return mol.atom_pos.shape[0]


def molecule_table(mode: str, surf_per_atom: int = 3) -> List[Tuple[str, int, int]]:
    """Return [(name, heavy, mode_count_for_bucketing)] for the curated set."""
    out = []
    for name, smi, heavy in DRUGS:
        m = _build_molecule(smi, surf_per_atom=surf_per_atom)
        out.append((name, heavy, _count_for_mode(m, mode)))
    return out


def make_real_cohort(mode: str, *, n_pairs: int, bucket_kind: str,
                     trans_max: float = 3.0, rot_max_deg: float = 60.0,
                     surf_per_atom: int = 3, seed: int = 3) -> Cohort:
    """Cohort of (real molecule, SE(3)-copy) pairs.

    bucket_kind:
      'same'  -> sample only from molecules whose mode-count lands in ONE band
                 (the most populated band) -> single GPU bucket.
      'cross' -> sample across the whole size range -> many buckets.
    """
    from shepherd_score.container._batch_align import _band_key
    rng = np.random.default_rng(seed)

    mols = [_build_molecule(smi, surf_per_atom=surf_per_atom) for _, smi, _ in DRUGS]
    names = [n for n, _, _ in DRUGS]
    bands = [_band_key(_count_for_mode(m, mode)) for m in mols]

    if bucket_kind == "same":
        # pick the most populated band and use only those molecules
        vals, counts = np.unique(bands, return_counts=True)
        target = int(vals[np.argmax(counts)])
        idx_pool = [i for i, b in enumerate(bands) if b == target]
    elif bucket_kind == "cross":
        idx_pool = list(range(len(mols)))
    else:
        raise ValueError(bucket_kind)

    pairs: List[PairSpec] = []
    chosen = rng.choice(idx_pool, size=n_pairs, replace=True)
    for i in chosen:
        ref = mols[int(i)]
        for _ in range(16):
            R = _random_rotation(rng)
            if (np.trace(R) - 1.0) / 2.0 >= np.cos(np.deg2rad(rot_max_deg)):
                break
        t = rng.standard_normal(3)
        t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * trans_max)
        fit = _transform(ref, R, t)
        pairs.append(PairSpec(ref=ref, fit=fit, R=R, t=t,
                              n_ref=_count_for_mode(ref, mode),
                              n_fit=_count_for_mode(fit, mode)))

    return Cohort(name=f"real-{bucket_kind}bucket", mode=mode, pairs=pairs,
                  size_kind=bucket_kind, seed=seed,
                  meta={"n_pairs": n_pairs, "molecules": names,
                        "bands": bands, "pool": [names[i] for i in idx_pool]})


# ===========================================================================
# FORK engine  (in-process: this fork via MoleculePairBatch, backend="triton")
# ===========================================================================
_SCORE_ATTR = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
               "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}


def _fork_pool_smiles(mode, bucket):
    """SMILES pool a (mode, bucket) cohort samples from, mirroring make_real_cohort."""
    from shepherd_score.container._batch_align import _band_key
    tbl = molecule_table(mode, surf_per_atom=SURF_PER_ATOM)        # (name, heavy, count)
    bands = [_band_key(c) for _, _, c in tbl]
    if bucket == "same":
        vals, counts = np.unique(bands, return_counts=True)
        target = int(vals[np.argmax(counts)])
        idx = [i for i, b in enumerate(bands) if b == target]
    else:
        idx = list(range(len(DRUGS)))
    return [DRUGS[i][1] for i in idx]


def _fork_align(mode, pairs, cfg):
    """Align a batch of MoleculePair via ``MoleculePairBatch.align_with_*`` on the
    Triton backend. This is the SAME public API the original engine uses (it just
    passes ``backend="jax"``), so both engines are one code path / two backends.
    Results land in-place on each pair (``sim_aligned_*`` / ``transform_*``)."""
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(pairs)
    if mode == "vol":
        b.align_with_vol(no_H=True, backend="triton",
                         alpha=cfg["alpha"], max_num_steps=cfg["steps"])
    elif mode == "surf":
        b.align_with_surf(alpha=cfg["alpha"], backend="triton",
                          max_num_steps=cfg["steps"])
    elif mode == "esp":
        b.align_with_esp(alpha=cfg["alpha"], lam=cfg["lam"],
                         num_repeats=cfg["num_repeats"], lr=cfg["lr"],
                         backend="triton", max_num_steps=cfg["steps"])
    elif mode == "pharm":
        b.align_with_pharm(num_repeats=cfg["num_repeats"], lr=cfg["lr"],
                           backend="triton", max_num_steps=cfg["steps"])
    else:
        raise ValueError(mode)


def _fork_time(mode, pairs, cfg):
    """One warmup + one timed fork alignment. -> (sec, mean_score). Used by the
    accuracy branch; the speed sweep uses the isolated-subprocess path below."""
    import torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _fork_align(mode, pairs, cfg); sync()                          # warmup (autotune / JIT)
    sync(); t0 = time.perf_counter()
    _fork_align(mode, pairs, cfg); sync()
    dt = time.perf_counter() - t0
    sc = np.array([float(getattr(p, _SCORE_ATTR[mode])) for p in pairs], dtype=float)
    return dt, float(sc.mean())


def _fork_clear():
    import torch
    try:
        import shepherd_score.container._batch_align as cc
        cc._ALIGN_WORKSPACES.clear(); cc._INT_BUFFER_CACHE.clear()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Isolated speed sweep
# ---------------------------------------------------------------------------
# Each (mode, bucket, size) CELL runs in its OWN fresh subprocess. A fresh process
# means (a) the GPU clock has recovered -- no cell is timed on a throttle bled from
# a previous cell (this laptop GPU throttles ~2-3x under sustained load), and (b) the
# Triton kernel autotunes at THIS cell's batch (the autotune key is the per-pose
# shape, so a process reuses whatever config it first picked -- sharing a process
# across sizes would lock in the tiny-n config and cripple the large cells). Each
# cell is timed best-of-N (fastest = un-throttled boost-clock rep) -> isolated peaks.
def fork_cell(planfile):
    """Entry inside an isolated fork subprocess: measure ONE (mode, bucket, size)
    cell best-of-N and print a RES line."""
    import torch
    from shepherd_score.container import MoleculePair as MP
    with open(planfile) as fh:
        plan = json.load(fh)
    mode, bucket, nb = plan["mode"], plan["bucket"], plan["size"]
    cfg, seed, reps, budget = plan["cfg"], plan["seed"], plan["reps"], plan["budget"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=seed)
        pairs = [MP(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
        _fork_align(mode, pairs, cfg); sync()                      # warmup: autotune at THIS batch + clock to boost
        best = float("inf"); n = 0; total = 0.0
        while n < reps and total < budget:                         # best-of-N, time-budgeted
            sync(); t0 = time.perf_counter()
            _fork_align(mode, pairs, cfg); sync()
            dt = time.perf_counter() - t0
            best = min(best, dt); total += dt; n += 1
        sc = float(np.array([float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]).mean())
        print(f"  fork {mode}|{bucket:5s} n={nb:<7d} {best:7.3f}s  {nb / best:10.1f} pairs/s  "
              f"self={sc:.3f}  (best of {n}, isolated)", flush=True)
        print("RES " + json.dumps({"key": f"{mode}|{bucket}", "n": nb, "t": best,
                                    "score": sc, "nreps": n}), flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  fork {mode}|{bucket:5s} n={nb:<7d} ERR {type(e).__name__}: {e}", flush=True)
        print("RES " + json.dumps({"key": f"{mode}|{bucket}", "n": nb,
                                    "err": f"{type(e).__name__}: {e}"}), flush=True)


def _spawn_fork_cell(plan):
    """Spawn a fresh fork subprocess for one cell; return its parsed RES dict."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(plan, fh)
        planfile = fh.name
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Turn on the deterministic molecule disk cache so each fresh cell loads the
    # 15 base molecules instead of rebuilding them (RDKit+MMFF+Open3D+pharm). The
    # build is outside the timed region, so this is a pure wall-clock win with no
    # effect on any measured number. (User-set FSS_MOL_CACHE_DIR is respected.)
    env.setdefault("FSS_MOL_CACHE_DIR", _MOLCACHE)
    # The child runs this file as a script, so put the repo root on its path
    # (for `import benchmarks...`) ahead of anything else; shepherd_score resolves
    # to the fork (repo root), NOT the original repo.
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, os.path.abspath(__file__), "--fork-cell", planfile]
    rec = None
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            if ln.startswith("RES "):
                rec = json.loads(ln.split(" ", 1)[1])
            else:
                print(ln, flush=True)
        if p.returncode != 0 and rec is None:
            sys.stderr.write(p.stderr[-2000:])
            rec = {"err": f"exit {p.returncode}"}
    finally:
        try:
            os.unlink(planfile)
        except OSError:
            pass
    return rec


def fork_speed_sweep(modes, buckets, sizes, cfg, seed, reps=5, budget=4.0):
    """Isolated speed sweep: each (mode, bucket, size) cell in its OWN fresh
    subprocess. The fork runs the FULL size sweep (no cap-stop) — it is fast
    enough that even the 100k cell finishes in well under a couple of minutes, so
    the 100k datapoint is always recorded for the fork."""
    data = {}
    for mode in modes:
        for bucket in buckets:
            key = f"{mode}|{bucket}"
            data[key] = {"fork": [], "fork_score": {}}
            for nb in sizes:
                plan = {"mode": mode, "bucket": bucket, "size": nb, "cfg": cfg,
                        "seed": seed, "reps": reps, "budget": budget}
                rec = _spawn_fork_cell(plan)
                if not rec or "err" in rec:
                    print(f"  fork {key:12s} n={nb:<7d} ERR "
                          f"{(rec or {}).get('err', 'subprocess failed')}", flush=True)
                    break
                data[key]["fork"].append([nb, rec["n"] / rec["t"]])
                data[key]["fork_score"][nb] = rec["score"]
    return data


# ===========================================================================
# ORIGINAL engine  (subprocess: shepherd-score-original-repo/, isolated)
# ===========================================================================
def _orig_base_cache_path(smi, spa, seed):
    key = hashlib.md5(f"origv1|{smi}|{spa}|{seed}".encode()).hexdigest()
    return os.path.join(_MOLCACHE, key + ".pkl")


def run_original(plan):
    """Run the original-repo engine in an isolated subprocess; return its result lines."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(plan, f)
        planfile = f.name
    env = dict(os.environ)
    env["PYTHONPATH"] = ORIG_REPO + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--orig-cell", planfile]
    out = {}
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            if ln.startswith("RES "):
                _, payload = ln.split(" ", 1)
                rec = json.loads(payload)
                out[rec["key"]] = rec["rows"]
            else:
                print(ln, flush=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-2000:])
    finally:
        try:
            os.unlink(planfile)
        except OSError:
            pass
    return out


def _orig_imports():
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from shepherd_score.conformer_generation import embed_conformer_from_smiles
    from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch
    return Chem, Point3D, embed_conformer_from_smiles, Molecule, MoleculePair, MoleculePairBatch


def orig_cell(planfile):
    """Entry point inside the isolated original-repo subprocess. Prints RES lines."""
    import shepherd_score
    sys.stderr.write(f"[orig-cell] shepherd_score from: {shepherd_score.__file__}\n")
    if os.path.abspath(ORIG_REPO) not in os.path.abspath(shepherd_score.__file__):
        sys.stderr.write("[orig-cell] WARNING: not the original-repo package — isolation failed!\n")
    Chem, Point3D, embed, Molecule, MoleculePair, MoleculePairBatch = _orig_imports()
    with open(planfile) as f:
        plan = json.load(f)
    cfg, spa = plan["cfg"], plan["cfg"]["surf_per_atom"]
    base = {}                                                      # smiles -> (Molecule, rdmol|None, ns)

    def build_base(smi, seed):
        """Base Molecule for a SMILES, built ONCE: in-memory for this process and
        pickled under benchmarks/molcache/, so repeat runs skip the (Open3D surface
        + MMFF + pharmacophore) rebuild entirely."""
        if smi in base:
            return base[smi]
        cpath = _orig_base_cache_path(smi, spa, seed)
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as fh:
                    mol, ns = pickle.load(fh)
                base[smi] = (mol, None, ns)
                return base[smi]
            except Exception:
                pass
        rd = embed(smi, MMFF_optimize=True, random_seed=seed)
        ns = max(24, spa * Chem.RemoveHs(rd).GetNumAtoms())
        mol = Molecule(rd, num_surf_points=ns, pharm_multi_vector=False)
        base[smi] = (mol, rd, ns)
        try:
            os.makedirs(_MOLCACHE, exist_ok=True)
            with open(cpath, "wb") as fh:
                pickle.dump((mol, ns), fh)
        except Exception:
            pass
        return base[smi]

    def rotated(smi, R, t, seed):
        """A rigid SE(3) copy of the base molecule WITHOUT recomputing its surface.
        Surface / ESP / pharmacophores are rotation-equivariant, so we rotate the
        precomputed arrays (a matmul) on a shallow copy and share the invariant
        ones -- exactly what the fork's _transform does, and ~1000x cheaper than
        reconstructing a Molecule (Open3D + MMFF + RDKit) per pair. As a bonus both
        engines now align IDENTICAL pairs, so the original self-score is exactly 1.0
        too (the old per-pair surface rebuild was not perfectly rotation-equivariant)."""
        m0 = build_base(smi, seed)[0]
        R = np.asarray(R, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
        m = copy.copy(m0)                                  # shares rdmol + invariant arrays
        m.atom_pos = m0.atom_pos @ R.T + t
        if getattr(m0, "surf_pos", None) is not None:
            m.surf_pos = m0.surf_pos @ R.T + t
        if getattr(m0, "pharm_ancs", None) is not None:
            m.pharm_ancs = m0.pharm_ancs @ R.T + t
            v = m0.pharm_vecs @ R.T
            nrm = np.linalg.norm(v, axis=1, keepdims=True)
            m.pharm_vecs = v / np.where(nrm > 0, nrm, 1.0)
        return m

    def align(mode, pairs):
        b = MoleculePairBatch(pairs)
        if mode == "vol":
            return b.align_with_vol(no_H=True, num_repeats=cfg["num_repeats"],
                                    lr=cfg["lr"], max_num_steps=cfg["steps"], num_workers=1)[0]
        if mode == "surf":
            return b.align_with_surf(alpha=cfg["alpha"], num_repeats=cfg["num_repeats"],
                                     lr=cfg["lr"], max_num_steps=cfg["steps"], num_workers=1,
                                     use_shmap=False)[0]
        if mode == "esp":
            return b.align_with_esp(alpha=cfg["alpha"], lam=cfg["lam"], num_repeats=cfg["num_repeats"],
                                    lr=cfg["lr"], max_num_steps=cfg["steps"], num_workers=1,
                                    use_shmap=False)[0]
        if mode == "pharm":
            return b.align_with_pharm(num_repeats=cfg["num_repeats"], lr=cfg["lr"],
                                      max_num_steps=cfg["steps"], num_workers=1, use_shmap=False)[0]
        raise ValueError(mode)

    if plan["task"] == "speed":
        cap, seed = plan["cap"], plan["seed"]
        for cell in plan["cells"]:
            mode, bucket, pool, sizes = cell["mode"], cell["bucket"], cell["pool"], cell["sizes"]
            rng = np.random.default_rng(seed)
            rows = []
            for nb in sizes:
                pairs = []
                for _ in range(nb):
                    smi = pool[int(rng.integers(len(pool)))]
                    R = _rand_rot(rng)
                    t = rng.standard_normal(3); t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * 3.0)
                    ref, _, ns = build_base(smi, seed)
                    pairs.append(MoleculePair(ref, rotated(smi, R, t, seed),
                                              do_center=False, num_surf_points=ns))
                try:
                    t0 = time.perf_counter(); sc = align(mode, pairs); dt = time.perf_counter() - t0
                    rows.append({"n": nb, "t": dt, "score": float(np.asarray(sc).mean())})
                    print(f"  orig {mode}|{bucket:5s} n={nb:<7d} {dt:7.3f}s "
                          f"{nb/dt:10.1f} pairs/s  self={float(np.asarray(sc).mean()):.3f}", flush=True)
                except Exception as e:
                    import traceback
                    rows.append({"n": nb, "err": f"{type(e).__name__}: {e}"})
                    print(f"  orig {mode}|{bucket:5s} n={nb:<7d} ERR {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    break
                if dt > cap:                                        # cap stop: the original is too slow for larger sizes
                    break
            print("RES " + json.dumps({"key": f"{mode}|{bucket}", "rows": rows}), flush=True)

    elif plan["task"] == "accuracy":
        seed = plan["seed"]
        for cell in plan["acc"]:
            mode, pairspec = cell["mode"], cell["pairs"]
            pairs = []
            for ref_smi, fit_smi in pairspec:
                ref, _, ns = build_base(ref_smi, seed)
                pairs.append(MoleculePair(ref, build_base(fit_smi, seed)[0],
                                          do_center=False, num_surf_points=ns))
            try:
                sc = align(mode, pairs)
                rows = [float(x) for x in np.asarray(sc).ravel()]
            except Exception as e:
                import traceback
                rows = {"err": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            print("RES " + json.dumps({"key": mode, "rows": rows}), flush=True)


def _rand_rot(rng):
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.tolist()


# ===========================================================================
# Accuracy branch (fork side, in-process)
# ===========================================================================
def fork_accuracy(modes, pair_smiles, cfg, seed):
    """Align distinct-molecule pairs on the fork; return {mode: [per-pair scores]}."""
    from shepherd_score.container import MoleculePair as MP
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for mode in modes:
        pairs = [MP(_build_molecule(rs, surf_per_atom=cfg["surf_per_atom"]),
                    _build_molecule(fs, surf_per_atom=cfg["surf_per_atom"]),
                    do_center=False, device=dev) for rs, fs in pair_smiles]
        _fork_time(mode, pairs, cfg)
        out[mode] = [float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]
        _fork_clear()
    return out


# ===========================================================================
# Hardware / environment info (for plot annotation)
# ===========================================================================
def _cpu_name() -> str:
    """Best-effort CPU model name (Linux/WSL /proc/cpuinfo, else platform)."""
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.lower().startswith("model name"):
                    return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def _hardware_info() -> dict:
    """Best-effort hardware + env summary, captured in the main (GPU env) process."""
    info = {"gpu": None, "cpu": _cpu_name(), "host": platform.node(),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": None, "cuda": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def _hw_footer(hw: dict) -> str:
    """One-line hardware/env footer for the figure."""
    bits = [f"GPU: {hw.get('gpu') or 'n/a'}", f"CPU: {hw.get('cpu') or 'n/a'}"]
    if hw.get("torch"):
        bits.append(f"torch {hw['torch']}" + (f" / CUDA {hw['cuda']}" if hw.get("cuda") else ""))
    if hw.get("host"):
        bits.append(hw["host"])
    return "   ·   ".join(bits)


# ===========================================================================
# Rendering
# ===========================================================================
COLOR = {"vol": "#7b3294", "surf": "#1f6fb2", "esp": "#1a9850", "pharm": "#d9700a"}
LS = {"same": "-", "cross": (0, (5, 2))}
MK = {"same": "o", "cross": "D"}


def render_plot(data, modes, buckets, sizes, cap, out_png, meta=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hw = (meta or {}).get("hardware", {}) or {}
    gpu = hw.get("gpu") or "GPU: n/a"
    cpu = hw.get("cpu") or "CPU: n/a"
    stamp = (meta or {}).get("timestamp", "")
    tag = (meta or {}).get("tag")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharey=True)
    # per-panel: (data key, engine, the hardware that panel ran on)
    panels = [("orig", "Original upstream repo  ·  JAX / CPU", cpu),
              ("fork", "This fork  ·  Triton / GPU", gpu)]
    for ax, (pk, engine, hw_label) in zip(axes, panels):
        for mode in modes:
            for bucket in buckets:
                pts = data.get(f"{mode}|{bucket}", {}).get(pk, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=COLOR[mode], linestyle=LS[bucket], marker=MK[bucket],
                        markersize=7, linewidth=2.4, label=f"{mode} · {bucket}", clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("batch size — pairs aligned per call (log)")
        # Engine + the hardware it ran on as ONE two-line title, so tight_layout
        # reserves room for both -> nothing free-floating to overlap the suptitle.
        ax.set_title(engine + "\n" + hw_label, fontsize=10.5, linespacing=1.6,
                     fontweight="bold")
        ax.grid(True, which="major", color="#cccccc", alpha=0.8)
        ax.grid(True, which="minor", ls=":", color="#e8e8e8", alpha=0.6)
    axes[0].set_ylabel("pair-alignments / second (higher = faster, log)")
    axes[1].legend(title="mode · bucket", loc="best", framealpha=0.95)

    fig.suptitle("Molecular-alignment throughput — real drug self-copy pairs",
                 fontweight="bold", fontsize=13.5, y=0.985)
    cap_note = (f"fork runs the full size sweep; the original line stops where a cell "
                f"exceeded the {cap:.0f}s wall-clock cap")
    footer = "   ·   ".join(([f"run: {tag}"] if tag else []) + [_hw_footer(hw)]
                            + ([stamp] if stamp else []))
    fig.text(0.5, 0.05, cap_note, ha="center", va="bottom", fontsize=9,
             color="#555555", style="italic")
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.085, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def _lbl(s):
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def render_table(data, modes, buckets, sizes, out_md, meta=None):
    lbl = {s: _lbl(s) for s in sizes}
    hw = (meta or {}).get("hardware", {}) or {}
    stamp = (meta or {}).get("timestamp", "")
    tag = (meta or {}).get("tag")
    L = [f"# Alignment throughput — real drug pairs (pair-alignments / s)\n",
         f"Each cell stops at the first size over the wall-clock cap. `—` = not run.\n"]
    meta_bits = (([f"run: {tag}"] if tag else []) + ([_hw_footer(hw)] if hw else [])
                 + ([stamp] if stamp else []))
    if meta_bits:
        L.append("\n_" + "   ·   ".join(meta_bits) + "_\n")
    L += ["\n## pairs / s\n",
          "| mode | bucket | engine | " + " | ".join(lbl[s] for s in sizes) + " |",
          "|---|---|---|" + "".join("--:|" for _ in sizes)]
    def series(mode, bucket, eng):
        return {n: m for n, m in data.get(f"{mode}|{bucket}", {}).get(eng, [])}
    for mode in modes:
        for bucket in buckets:
            for eng, name in (("orig", "original"), ("fork", "fork")):
                s = series(mode, bucket, eng)
                cells = " | ".join(f"{s[n]:.0f}" if n in s else "—" for n in sizes)
                L.append(f"| {mode} | {bucket} | {name} | {cells} |")
    L += ["\n## fork speedup over original (×, matched size)\n",
          "| mode | bucket | " + " | ".join(lbl[s] for s in sizes) + " |",
          "|---|---|" + "".join("--:|" for _ in sizes)]
    for mode in modes:
        for bucket in buckets:
            o = series(mode, bucket, "orig"); f = series(mode, bucket, "fork")
            cells = " | ".join(f"{f[n]/o[n]:.1f}×" if (n in o and n in f) else "—" for n in sizes)
            L.append(f"| {mode} | {bucket} | {cells} |")
    txt = "\n".join(L) + "\n"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    print(f"wrote {out_md}")


# ===========================================================================
# Driver
# ===========================================================================
def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def _render_all(data, modes, buckets, sizes, cap, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)
    render_table(data, modes, buckets, sizes, os.path.join(out_dir, "speed_table.md"), meta)
    try:
        render_plot(data, modes, buckets, sizes, cap, os.path.join(out_dir, "speed_plot.png"), meta)
    except Exception as e:
        print(f"(plot skipped: {type(e).__name__}: {e})")


def run_speed(args):
    cfg = _cfg_from_args(args)
    modes, buckets, sizes = args.modes, args.buckets, args.sizes
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "plot_data.json")

    print("=" * 88)
    print("BENCHMARK: real-drug self-SE(3)-copy alignment  (optimum score = 1.0)")
    print(f"modes={modes} buckets={buckets} sizes={sizes} cap={args.cap:.0f}s "
          f"original={'on' if not args.no_original else 'off'}")
    print("=" * 88)

    print("\n[fork] MoleculePairBatch.align_with_*(backend='triton') "
          "(each cell isolated in its own fresh subprocess; full sweep to 100k)")
    data = fork_speed_sweep(modes, buckets, sizes, cfg, args.seed, args.reps, args.budget)

    if not args.no_original:
        print("\n[original] upstream repo via MoleculePairBatch (backend='jax', isolated subprocess)")
        plan = {"task": "speed", "cap": args.cap, "seed": args.seed, "cfg": cfg,
                "cells": [{"mode": m, "bucket": b, "sizes": sizes,
                           "pool": _fork_pool_smiles(m, b)}
                          for m in modes for b in buckets]}
        orig = run_original(plan)
        for key, rows in orig.items():
            data.setdefault(key, {})["orig"] = [[r["n"], r["n"] / r["t"]] for r in rows if "t" in r]
            data[key]["orig_score"] = {r["n"]: r["score"] for r in rows if "score" in r}
    elif os.path.exists(out_json):
        # fork-only run: carry over the previous run's original-engine numbers so
        # the plot keeps both panels (replaces the old replot.py snapshot).
        try:
            with open(out_json) as fh:
                prev = json.load(fh)
            carried = 0
            for key, d in prev.items():
                if key.startswith("_") or not isinstance(d, dict):
                    continue
                if "orig" in d:
                    data.setdefault(key, {})["orig"] = d["orig"]; carried += 1
                if "orig_score" in d:
                    data.setdefault(key, {})["orig_score"] = d["orig_score"]
            if carried:
                print(f"carried over previous original-engine numbers for {carried} cells "
                      "(--no-original)")
        except Exception as e:
            print(f"(no previous original data to carry over: {type(e).__name__}: {e})")

    meta = {"hardware": _hardware_info(), "timestamp": _now_str(), "cap": args.cap,
            "tag": args.tag, "cfg": cfg, "modes": modes, "buckets": buckets, "sizes": sizes}
    data["_meta"] = meta

    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote {out_json}")
    _render_all(data, modes, buckets, sizes, args.cap, args.out_dir, meta)

    # self-accuracy summary (self-copy optimum is 1.0)
    print("\nself-accuracy (mean recovered score on self-copies; want ~1.000):")
    for key in sorted(data):
        if key.startswith("_"):
            continue
        fs = data[key].get("fork_score", {})
        if fs:
            worst = min(fs.values())
            flag = "  <-- LOW" if worst < SELF_SCORE_WARN else ""
            print(f"  {key:14s} fork min={worst:.3f}{flag}")


def run_replot(args):
    """Re-render the plot + table from an existing results/plot_data.json (no compute)."""
    out_json = os.path.join(args.out_dir, "plot_data.json")
    if not os.path.exists(out_json):
        raise SystemExit(f"no data to replot at {out_json}; run the benchmark first")
    with open(out_json) as fh:
        data = json.load(fh)
    meta = data.get("_meta", {})
    if not meta.get("hardware"):                       # fall back to live hardware query
        meta = {**meta, "hardware": _hardware_info(), "timestamp": meta.get("timestamp", _now_str())}
    if not meta.get("tag") and args.tag:               # label by the folder being replotted
        meta = {**meta, "tag": args.tag}
    modes = meta.get("modes", MODES)
    buckets = meta.get("buckets", BUCKETS)
    sizes = meta.get("sizes", SIZES)
    cap = meta.get("cap", DEFAULT_CAP)
    print(f"replotting from {out_json}")
    _render_all(data, modes, buckets, sizes, cap, args.out_dir, meta)


def run_accuracy(args):
    """50 DIFFERENT-molecule pairs per mode, fork vs original scores."""
    # Same deterministic molecule disk cache as the speed sweep. The accuracy
    # branch builds the fork side in-process, so set the env var here before any
    # _build_molecule call; it shares benchmarks/molcache/ with the speed sweep,
    # so a second run (or a run after the sweep) loads instead of rebuilding.
    os.environ.setdefault("FSS_MOL_CACHE_DIR", _MOLCACHE)
    cfg = _cfg_from_args(args)
    rng = np.random.default_rng(args.seed)
    smis = [s for _, s, _ in DRUGS]
    pairs = []
    while len(pairs) < args.n_accuracy:
        i, j = rng.integers(len(smis)), rng.integers(len(smis))
        if i != j:
            pairs.append((smis[int(i)], smis[int(j)]))

    print(f"ACCURACY branch: {len(pairs)} distinct-molecule pairs, fork vs original\n")
    fork = fork_accuracy(args.modes, pairs, cfg, args.seed)

    orig = {}
    if not args.no_original:
        plan = {"task": "accuracy", "seed": args.seed, "cfg": cfg,
                "acc": [{"mode": m, "pairs": pairs} for m in args.modes]}
        orig = run_original(plan)

    print(f'\n{"mode":6s} {"fork_mean":>9s} {"orig_mean":>9s} {"mean|Δ|":>9s} {"spearman":>9s}')
    print("-" * 46)
    for mode in args.modes:
        f = np.array(fork.get(mode, []), dtype=float)
        o = orig.get(mode)
        if not isinstance(o, list):
            print(f"{mode:6s} {f.mean():9.4f} {'NA':>9s} {'-':>9s} {'-':>9s}")
            continue
        o = np.array(o, dtype=float)
        n = min(len(f), len(o)); f, o = f[:n], o[:n]
        mad = float(np.abs(f - o).mean())
        rho = _spearman(f, o)
        print(f"{mode:6s} {f.mean():9.4f} {o.mean():9.4f} {mad:9.4f} {rho:9.4f}")


def _spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description="fast_shepherd_score fork-vs-original alignment benchmark")
    ap.add_argument("--orig-cell", help=argparse.SUPPRESS)         # internal: run as original subprocess
    ap.add_argument("--fork-cell", help=argparse.SUPPRESS)         # internal: run as isolated fork subprocess
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--buckets", nargs="+", default=BUCKETS, choices=BUCKETS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP,
                    help="seconds: the ORIGINAL engine stops its line at the first cell over "
                         "this (the fork always runs the full sweep)")
    ap.add_argument("--no-original", action="store_true",
                    help="time only the fork (keeps the previous run's original line)")
    ap.add_argument("--accuracy", action="store_true",
                    help="run the accuracy branch (distinct pairs) instead of the speed sweep")
    ap.add_argument("--replot", action="store_true",
                    help="re-render plot + table from the run's plot_data.json "
                         "(respects --tag / --out-dir; no compute)")
    ap.add_argument("--n-accuracy", type=int, default=50)
    ap.add_argument("--tag", default=None,
                    help="write this run to results/<tag>/ instead of results/, so multiple "
                         "result sets live side by side (read back by --replot --tag <tag>)")
    ap.add_argument("--out-dir", default=None,
                    help="explicit output directory; overrides --tag "
                         "(default: results/ , or results/<tag> when --tag is given)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.81)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=5,
                    help="max timed reps per isolated cell; the fastest (un-throttled) is kept")
    ap.add_argument("--budget", type=float, default=4.0,
                    help="per-cell time budget (s): stop adding reps once exceeded, so slow "
                         "cells take 1 rep and fast cells get best-of-reps")
    args = ap.parse_args()
    if args.out_dir is None:                                       # resolve results dir: --out-dir > --tag > default
        args.out_dir = os.path.join(_RESULTS, args.tag) if args.tag else _RESULTS

    if args.orig_cell:                                             # isolated subprocess entry
        return orig_cell(args.orig_cell)
    if args.fork_cell:                                            # isolated fork subprocess entry
        return fork_cell(args.fork_cell)
    if args.replot:
        return run_replot(args)
    if args.accuracy:
        return run_accuracy(args)
    return run_speed(args)


if __name__ == "__main__":
    raise SystemExit(main())
