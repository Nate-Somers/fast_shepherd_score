"""
Headline benchmark for fast_shepherd_score — ONE script, a few flags.

What it does
------------
Aligns REAL drug molecules (RDKit ETKDG conformers + Open3D surfaces + MMFF
charges + RDKit pharmacophores) and reports, for every alignment mode, how fast
THIS fork (Triton/CUDA batch path) is versus the ORIGINAL upstream repo
(``shepherd-score-original-repo/``, its own in-process batch path), across a
sweep of batch sizes.

Two things are measured at once:
  * speed   — pair-alignments per second (throughput), fork vs original.
  * accuracy — every pair is a molecule aligned to a rigid SE(3) copy of itself,
               so the perfect score is 1.0. We report the achieved mean score;
               anything well below 1.0 is a real quality problem, not noise.

Modes (all run by default): vol (atom-cloud ROCS), surf (surface ROCS),
esp (surface shape + electrostatics), pharm (pharmacophore).

Size sweep (default): 1, 10, 100, 1000, 10000, 100000 pairs per call. For each
(mode, bucket, engine) line the sweep stops at the first size whose wall time
exceeds the cap (default 10 s) — larger sizes are not run. Raise the cap with
``--cap``.

Buckets (both run by default): the GPU batch path buckets pairs by size.
  * same  — all molecules land in one size band  -> a single padded bucket (best case).
  * cross — molecules spread across bands         -> many buckets (realistic case).

Accuracy branch (OFF by default; ``--accuracy``): align 50 pairs of DIFFERENT
molecules (optimum < 1.0) across every mode and compare the fork's scores to the
original's, so the speed claims can't hide a quality regression on non-trivial
alignments.

Outputs: a markdown table (``speed_table.md``) and a two-panel plot
(``speed_plot.png``), plus the raw ``plot_data.json``.

Usage
-----
    python -m benchmarks.headline                 # full headline (fork + original)
    python -m benchmarks.headline --cap 30        # allow slower/bigger cells
    python -m benchmarks.headline --no-original    # fork only (fast iteration)
    python -m benchmarks.headline --modes surf esp # subset of modes
    python -m benchmarks.headline --accuracy       # the accuracy branch instead

Environment: needs a CUDA GPU + Triton for the fork path, JAX/Open3D etc. for
building molecules and the original path. Run it in the GPU conda env (WSL2).
The original repo runs in an isolated subprocess (both packages are named
``shepherd_score``, so they cannot share one interpreter).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
ORIG_REPO = os.path.join(_ROOT, "shepherd-score-original-repo")
_MOLCACHE = os.path.join(_HERE, "molcache")     # persisted base molecules (original-repo path)


def _orig_base_cache_path(smi, spa, seed):
    key = hashlib.md5(f"origv1|{smi}|{spa}|{seed}".encode()).hexdigest()
    return os.path.join(_MOLCACHE, key + ".pkl")

MODES = ["vol", "surf", "esp", "pharm"]
BUCKETS = ["same", "cross"]
SIZES = [1, 10, 100, 1000, 10000, 100000]
DEFAULT_CAP = 10.0
SURF_PER_ATOM = 3
SELF_SCORE_WARN = 0.95          # self-copy optimum is 1.0; warn below this


def _cfg_from_args(a):
    """Shared alignment knobs passed to both engines."""
    return dict(num_repeats=a.num_repeats, steps=a.steps, lr=a.lr,
                alpha=a.alpha, lam=a.lam, topk=a.topk, surf_per_atom=SURF_PER_ATOM)


# ===========================================================================
# FORK engine  (in-process: this fork's Triton/CUDA batch path)
# ===========================================================================
def _fork_pool_smiles(mode, bucket):
    """SMILES pool a (mode, bucket) cohort samples from, mirroring make_real_cohort."""
    from benchmarks.real_workloads import DRUGS, molecule_table
    from shepherd_score.container._core import _band_key
    tbl = molecule_table(mode, surf_per_atom=SURF_PER_ATOM)        # (name, heavy, count)
    bands = [_band_key(c) for _, _, c in tbl]
    if bucket == "same":
        vals, counts = np.unique(bands, return_counts=True)
        target = int(vals[np.argmax(counts)])
        idx = [i for i, b in enumerate(bands) if b == target]
    else:
        idx = list(range(len(DRUGS)))
    return [DRUGS[i][1] for i in idx]


def _fork_time(mode, pairs, cfg):
    """One timed fork alignment of an already-built list of MoleculePair. -> (sec, mean_score)."""
    import torch
    from shepherd_score.container import MoleculePair as MP

    def run():
        if mode == "vol":
            MP.align_batch_vol(pairs, alpha=cfg["alpha"], steps_fine=cfg["steps"])
        elif mode == "surf":
            MP.align_batch_surf(pairs, alpha=cfg["alpha"], steps_fine=cfg["steps"])
        elif mode == "esp":
            MP.align_batch_esp(pairs, alpha=cfg["alpha"], lam=cfg["lam"],
                               num_repeats=cfg["num_repeats"], topk=cfg["topk"],
                               steps_fine=cfg["steps"], lr=cfg["lr"])
        elif mode == "pharm":
            MP.align_batch_pharm(pairs, num_repeats=cfg["num_repeats"], topk=cfg["topk"],
                                 steps_fine=cfg["steps"], lr=cfg["lr"])

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    run(); sync()                                                  # warmup (Triton autotune / JIT)
    sync(); t0 = time.perf_counter(); run(); sync()
    dt = time.perf_counter() - t0
    attr = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
            "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}[mode]
    sc = np.array([float(getattr(p, attr)) for p in pairs], dtype=float)
    return dt, float(sc.mean())


def _fork_clear():
    import torch
    try:
        import shepherd_score.container._core as cc
        cc._ALIGN_WORKSPACES.clear(); cc._INT_BUFFER_CACHE.clear()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fork_speed_sweep(modes, buckets, sizes, cap, cfg, seed):
    """Run the fork over every (mode, bucket) line with the cap stop rule."""
    from benchmarks.real_workloads import make_real_cohort
    from shepherd_score.container import MoleculePair as MP
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {}
    for mode in modes:
        for bucket in buckets:
            key = f"{mode}|{bucket}"
            data[key] = {"fork": [], "fork_score": {}}
            for nb in sizes:
                co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=seed)
                pairs = [MP(p.ref, p.fit, do_center=False, device=dev) for p in co.pairs]
                try:
                    dt, score = _fork_time(mode, pairs, cfg)
                except Exception as e:
                    print(f"  fork {key:12s} n={nb:<7d} ERR {type(e).__name__}", flush=True)
                    _fork_clear(); break
                mps = nb / dt
                data[key]["fork"].append([nb, mps])
                data[key]["fork_score"][nb] = score
                print(f"  fork {key:12s} n={nb:<7d} {dt:7.3f}s  {mps:10.1f} pairs/s  self={score:.3f}",
                      flush=True)
                _fork_clear()
                if dt > cap:                                       # cap stop: don't run larger sizes
                    break
    return data


# ===========================================================================
# ORIGINAL engine  (subprocess: shepherd-score-original-repo/, isolated)
# ===========================================================================
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
                if dt > cap:
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
    from benchmarks.real_workloads import _build_molecule
    from shepherd_score.container import MoleculePair as MP
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attr = {"vol": "sim_aligned_vol_noH", "surf": "sim_aligned_surf",
            "esp": "sim_aligned_esp", "pharm": "sim_aligned_pharm"}
    out = {}
    for mode in modes:
        pairs = [MP(_build_molecule(rs, surf_per_atom=cfg["surf_per_atom"]),
                    _build_molecule(fs, surf_per_atom=cfg["surf_per_atom"]),
                    do_center=False, device=dev) for rs, fs in pair_smiles]
        _fork_time(mode, pairs, cfg)
        out[mode] = [float(getattr(p, attr[mode])) for p in pairs]
        _fork_clear()
    return out


# ===========================================================================
# Rendering
# ===========================================================================
COLOR = {"vol": "#7b3294", "surf": "#1f6fb2", "esp": "#1a9850", "pharm": "#d9700a"}
LS = {"same": "-", "cross": (0, (5, 2))}
MK = {"same": "o", "cross": "D"}


def render_plot(data, modes, buckets, sizes, cap, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
    for ax, (pk, title) in zip(axes, [("orig", "Original upstream repo"), ("fork", "This fork (Triton · GPU)")]):
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
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="major", color="#cccccc", alpha=0.8)
        ax.grid(True, which="minor", ls=":", color="#e8e8e8", alpha=0.6)
    axes[0].set_ylabel("pair-alignments / second (higher = faster, log)")
    axes[1].legend(title="mode · bucket", loc="best", framealpha=0.95)
    fig.suptitle("Molecular-alignment throughput — real drug self-copy pairs\n"
                 f"each line stops where a cell exceeded the {cap:.0f}s wall-clock cap",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_png}")


def _lbl(s):
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def render_table(data, modes, buckets, sizes, out_md):
    lbl = {s: _lbl(s) for s in sizes}
    L = [f"# Alignment throughput — real drug pairs (pair-alignments / s)\n",
         f"Each cell stops at the first size over the wall-clock cap. `—` = not run.\n",
         "\n## pairs / s\n",
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
    with open(out_md, "w") as fh:
        fh.write(txt)
    print(txt)
    print(f"wrote {out_md}")


# ===========================================================================
# Driver
# ===========================================================================
def run_speed(args):
    cfg = _cfg_from_args(args)
    modes, buckets, sizes = args.modes, args.buckets, args.sizes

    print("=" * 88)
    print("HEADLINE: real-drug self-SE(3)-copy alignment  (optimum score = 1.0)")
    print(f"modes={modes} buckets={buckets} sizes={sizes} cap={args.cap:.0f}s "
          f"original={'on' if not args.no_original else 'off'}")
    print("=" * 88)

    print("\n[fork] Triton/CUDA batch path")
    data = fork_speed_sweep(modes, buckets, sizes, args.cap, cfg, args.seed)

    if not args.no_original:
        print("\n[original] upstream repo (isolated subprocess)")
        plan = {"task": "speed", "cap": args.cap, "seed": args.seed, "cfg": cfg,
                "cells": [{"mode": m, "bucket": b, "sizes": sizes,
                           "pool": _fork_pool_smiles(m, b)}
                          for m in modes for b in buckets]}
        orig = run_original(plan)
        for key, rows in orig.items():
            data.setdefault(key, {})["orig"] = [[r["n"], r["n"] / r["t"]] for r in rows if "t" in r]
            data[key]["orig_score"] = {r["n"]: r["score"] for r in rows if "score" in r}

    out_json = os.path.join(args.out_dir, "plot_data.json")
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote {out_json}")
    render_table(data, modes, buckets, sizes, os.path.join(args.out_dir, "speed_table.md"))
    try:
        render_plot(data, modes, buckets, sizes, args.cap, os.path.join(args.out_dir, "speed_plot.png"))
    except Exception as e:
        print(f"(plot skipped: {type(e).__name__}: {e})")

    # self-accuracy summary (self-copy optimum is 1.0)
    print("\nself-accuracy (mean recovered score on self-copies; want ~1.000):")
    for key in sorted(data):
        fs = data[key].get("fork_score", {})
        if fs:
            worst = min(fs.values())
            flag = "  <-- LOW" if worst < SELF_SCORE_WARN else ""
            print(f"  {key:14s} fork min={worst:.3f}{flag}")


def run_accuracy(args):
    """50 DIFFERENT-molecule pairs per mode, fork vs original scores."""
    from benchmarks.real_workloads import DRUGS
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
    ap = argparse.ArgumentParser(description="Headline fork-vs-original alignment benchmark")
    ap.add_argument("--orig-cell", help=argparse.SUPPRESS)         # internal: run as original subprocess
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--buckets", nargs="+", default=BUCKETS, choices=BUCKETS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP,
                    help="seconds: a cell over this ends its line; larger sizes not run")
    ap.add_argument("--no-original", action="store_true", help="time only the fork")
    ap.add_argument("--accuracy", action="store_true",
                    help="run the accuracy branch (distinct pairs) instead of the speed sweep")
    ap.add_argument("--n-accuracy", type=int, default=50)
    ap.add_argument("--out-dir", default=_HERE)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.81)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    if args.orig_cell:                                             # isolated subprocess entry
        return orig_cell(args.orig_cell)
    if args.accuracy:
        return run_accuracy(args)
    return run_speed(args)


if __name__ == "__main__":
    raise SystemExit(main())
