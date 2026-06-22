"""
THE CPU benchmark for fast_shepherd_score — original JAX vs fork numba, one script.

The GPU benchmark (``benchmark.py``) pits the fork's Triton/CUDA path against the
original repo's JAX path. This is its CPU-only counterpart: the original repo's
**default JAX/XLA** path versus the fork's **numba** path, both on the CPU, through
the SAME public API — ``MoleculePairBatch.align_with_*``. The original passes no
``backend`` (its only path is JAX); the fork passes ``backend="numba"`` (the same
batched coarse-to-fine driver the Triton path uses, with numba CPU kernels). One
code shape, two CPU engines.

Why a separate script: the two engines parallelise on *different* axes, so the only
fair common knob is a **core budget**.
  * JAX scales by **processes** — ``num_workers=N`` (size-sorted multiprocessing;
    each worker re-imports JAX and re-JITs per call).
  * numba scales by **threads** — ``@njit(parallel=True)`` prange over poses, plus
    torch's CPU intra-op threads; one process.
So "test multiproc" cannot mean "match the mechanism" — it means **match the cores**.
Every cell pins both engines to the same core budget (numba: torch+numba threads;
JAX: ``num_workers`` × 1 thread/worker) and sweeps it: ``--procs 1 N``.
  * **1 core** — per-core kernel throughput (the load-bearing CPU metric).
  * **N cores** — aggregate throughput (the multiproc comparison).

Fairness, otherwise identical to the GPU benchmark: same molecules (shared molcache),
same seed, same alignment cfg; every pair is a real drug aligned to a rigid SE(3)
copy of itself, so the optimum score is 1.0 and each cell self-reports accuracy;
warmup (numba/JAX JIT) is excluded; each cell is best-of-N, time-budgeted, and runs
in its OWN isolated subprocess (both packages are named ``shepherd_score``, so they
cannot share an interpreter). The numba cell sets ``CUDA_VISIBLE_DEVICES=""`` so it
is provably CPU-only.

Modes: vol, surf, esp, pharm. (``esp_combo`` has no numba path — excluded.)

Outputs (under ``results_cpu/`` by default): ``speed_plot_cpu.png`` (one panel per
core budget, JAX vs numba), ``speed_table_cpu.md``, ``plot_data_cpu.json``.

Usage
-----
    python -m benchmarks.benchmark_cpu                 # full: JAX vs numba, 1 & N cores
    python -m benchmarks.benchmark_cpu --procs 1 8     # 1-core and 8-core budgets
    python -m benchmarks.benchmark_cpu --no-original   # numba only (keep last JAX line)
    python -m benchmarks.benchmark_cpu --modes vol pharm
    python -m benchmarks.benchmark_cpu --accuracy      # numba-vs-JAX distinct-pair parity
    python -m benchmarks.benchmark_cpu --replot

Environment: needs the original repo's JAX extra + open3d (surfaces) and the fork's
numba extra. Run in the same conda env the GPU benchmark uses (WSL2 ``SimModelEnv``);
the JAX path and surface builder do not exist on the CPU-only Windows box.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

# Reuse the GPU benchmark's machinery verbatim — curated drug set, deterministic
# molecule builder + disk cache, real-cohort sampler, the original-repo imports and
# cache path, hardware probe, render colours. No molecule logic is duplicated here;
# importing benchmark.py is safe (it imports no shepherd_score at module level).
from benchmarks.benchmark import (
    DRUGS, SURF_PER_ATOM, _MOLCACHE, ORIG_REPO, _ROOT,
    _build_molecule, make_real_cohort, _fork_pool_smiles, _count_for_mode,
    _rand_rot, _orig_base_cache_path, _orig_imports, _SCORE_ATTR,
    _hardware_info, _hw_footer, _now_str, _spearman, COLOR,
)

MODES = ["vol", "surf", "esp", "pharm"]            # esp_combo has no numba path
SIZES = [10, 100, 1000, 10000]                     # upstream timings.md range; CPU is slow
BUCKETS = ["cross"]                                # realistic varied-size case (also: "same")
DEFAULT_CAP = 20.0                                 # s: stop growing a series past this wall time
SELF_SCORE_WARN = 0.95
_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cpu")
ENGINES = ["jax", "numba"]                         # original (jax) vs fork (numba)

# The original repo's DOCUMENTED default multi-core path per mode (its `use_shmap`
# defaults in `MoleculePairBatch.align_with_*`). This is "as the documentation
# intends": vol/pharm parallelise via jax.shard_map across virtual CPU devices
# (needs XLA_FLAGS=--xla_force_host_platform_device_count=N set before JAX import and
# JAX >= 0.9.0); surf/esp via size-sorted multiprocessing (num_workers workers, one
# thread each). At a 1-core budget num_workers=1 runs the sequential path for all.
SHMAP_DEFAULT = {"vol": True, "surf": False, "esp": False, "pharm": True}


def _cfg_from_args(a):
    return dict(num_repeats=a.num_repeats, steps=a.steps, lr=a.lr,
                alpha=a.alpha, lam=a.lam, surf_per_atom=SURF_PER_ATOM)


def _default_procs():
    n = os.cpu_count() or 1
    return [1] if n <= 1 else [1, n]


# ===========================================================================
# FORK engine — numba (backend="numba"), one process, scales by threads
# ===========================================================================
def _numba_align(mode, pairs, cfg, num_workers=1):
    """Align a batch via ``MoleculePairBatch.align_with_*(backend="numba")`` — the
    same public API and same batched driver the Triton path uses, with numba CPU
    kernels. ``num_workers>1`` shards the pairs across the persistent CPU process pool
    (``_cpu_pool``). Results land in-place (``sim_aligned_*`` / ``transform_*``)."""
    from shepherd_score.container import MoleculePairBatch
    b = MoleculePairBatch(pairs)
    if mode == "vol":
        b.align_with_vol(no_H=True, backend="numba", num_workers=num_workers,
                         alpha=cfg["alpha"], max_num_steps=cfg["steps"])
    elif mode == "surf":
        b.align_with_surf(alpha=cfg["alpha"], backend="numba", num_workers=num_workers,
                          max_num_steps=cfg["steps"])
    elif mode == "esp":
        b.align_with_esp(alpha=cfg["alpha"], lam=cfg["lam"], num_workers=num_workers,
                         num_repeats=cfg["num_repeats"], lr=cfg["lr"],
                         backend="numba", max_num_steps=cfg["steps"])
    elif mode == "pharm":
        b.align_with_pharm(num_repeats=cfg["num_repeats"], lr=cfg["lr"], num_workers=num_workers,
                           backend="numba", max_num_steps=cfg["steps"])
    else:
        raise ValueError(mode)


def _torch_threads_for(mode, procs):
    """Optimal torch thread count for the numba path (measured; see
    cpu_numba_scaling_probe.py + the sweep in README_cpu). The heavy work is the numba
    `prange` kernel (gets `procs` threads); the per-step torch Adam/where bookkeeping is
    the only torch work. A big torch pool oversubscribes the numba pool, so:
      * 1-core budget -> torch=1 (keep it a true single core).
      * dense shape modes (vol/surf/esp): kernel dominates -> torch=1 (torch=N is
        ~30-40% slower at 8 cores).
      * pharm: the type-sparse kernel is light, so its torch bookkeeping is a larger
        fraction and benefits from a small pool -> torch=2 (torch=1 costs it ~21%)."""
    env = os.environ.get("FSS_TORCH_THREADS")
    if env:
        return int(env)
    if procs == 1:
        return 1
    return 2 if mode == "pharm" else 1


def _pin_threads(numba_t, torch_t):
    import torch
    torch.set_num_threads(int(torch_t))
    try:
        import numba
        numba.set_num_threads(int(numba_t))
    except Exception:
        pass


def numba_cell(planfile):
    """Isolated numba subprocess: time ONE (mode, bucket, size, procs) cell."""
    import torch
    from shepherd_score.container import MoleculePair as MP
    with open(planfile) as fh:
        plan = json.load(fh)
    mode, bucket, nb, procs = plan["mode"], plan["bucket"], plan["size"], plan["procs"]
    cfg, seed, reps, budget = plan["cfg"], plan["seed"], plan["reps"], plan["budget"]
    nmode = plan.get("numba_mode", "threads")                      # "threads" | "pool"
    nw = procs if nmode == "pool" else 1
    if nmode == "pool":
        _pin_threads(1, 1)            # parent single-threaded; the pool workers carry the budget
    else:
        _pin_threads(procs, _torch_threads_for(mode, procs))
    cpu = torch.device("cpu")
    try:
        co = make_real_cohort(mode, n_pairs=nb, bucket_kind=bucket, seed=seed)
        pairs = [MP(p.ref, p.fit, do_center=False, device=cpu) for p in co.pairs]
        _numba_align(mode, pairs, cfg, num_workers=nw)             # warmup: JIT + spawn pool
        best = float("inf"); n = 0; total = 0.0
        while n < reps and total < budget:
            t0 = time.perf_counter()
            _numba_align(mode, pairs, cfg, num_workers=nw)
            dt = time.perf_counter() - t0
            best = min(best, dt); total += dt; n += 1
        sc = float(np.array([float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]).mean())
        print(f"  numba {mode}|{bucket:5s} p={procs:<3d} {nmode:7s} n={nb:<7d} {best:8.3f}s "
              f"{nb / best:10.1f} pairs/s  self={sc:.3f}  (best of {n})", flush=True)
        print("RES " + json.dumps({"n": nb, "t": best, "score": sc, "nreps": n}), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("RES " + json.dumps({"n": nb, "err": f"{type(e).__name__}: {e}"}), flush=True)


# ===========================================================================
# ORIGINAL engine — JAX (no backend arg), scales by num_workers (multiprocessing)
# ===========================================================================
def orig_cpu_cell(planfile):
    """Isolated original-repo subprocess: time one mode's cells at a core budget.

    Runs with PYTHONPATH=shepherd-score-original-repo, so ``shepherd_score`` is the
    upstream package (pure JAX). Uses the mode's DOCUMENTED default multi-core path
    (``SHMAP_DEFAULT``): vol/pharm via ``jax.shard_map`` (the parent set
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before this process imported
    JAX, so ``len(jax.devices()) == N``); surf/esp via multiprocessing with
    ``OMP_NUM_THREADS=1`` per worker. At a 1-core budget ``num_workers=1`` is the
    sequential path for every mode."""
    import shepherd_score
    sys.stderr.write(f"[orig-cpu] shepherd_score from: {shepherd_score.__file__}\n")
    if os.path.abspath(ORIG_REPO) not in os.path.abspath(shepherd_score.__file__):
        sys.stderr.write("[orig-cpu] WARNING: not the original-repo package — isolation failed!\n")
    Chem, Point3D, embed, Molecule, MoleculePair, MoleculePairBatch = _orig_imports()
    with open(planfile) as fh:
        plan = json.load(fh)
    cfg, spa, W = plan["cfg"], plan["cfg"]["surf_per_atom"], plan["procs"]
    cap, seed, reps, budget = plan["cap"], plan["seed"], plan["reps"], plan["budget"]
    base = {}

    def build_base(smi):
        """Base Molecule for a SMILES (in-memory + benchmarks/molcache disk cache)."""
        if smi in base:
            return base[smi]
        cpath = _orig_base_cache_path(smi, spa, seed)
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as fh:
                    mol, ns = pickle.load(fh)
                base[smi] = (mol, ns); return base[smi]
            except Exception:
                pass
        rd = embed(smi, MMFF_optimize=True, random_seed=seed)
        ns = max(24, spa * Chem.RemoveHs(rd).GetNumAtoms())
        mol = Molecule(rd, num_surf_points=ns, pharm_multi_vector=False)
        base[smi] = (mol, ns)
        try:
            os.makedirs(_MOLCACHE, exist_ok=True)
            with open(cpath, "wb") as fh:
                pickle.dump((mol, ns), fh)
        except Exception:
            pass
        return base[smi]

    def rotated(smi, R, t):
        """Rigid SE(3) copy of the base molecule by rotating its precomputed arrays
        (surface/ESP/pharm are rotation-equivariant) — identical to the GPU bench."""
        m0 = build_base(smi)[0]
        R = np.asarray(R, np.float64); t = np.asarray(t, np.float64)
        m = copy.copy(m0)
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
        # The mode's documented default path: shmap (vol/pharm) or multiprocessing
        # (surf/esp). At W=1 both reduce to the sequential path inside align_with_*.
        kw = dict(num_repeats=cfg["num_repeats"], lr=cfg["lr"],
                  max_num_steps=cfg["steps"], num_workers=W, use_shmap=SHMAP_DEFAULT[mode])
        if mode == "vol":
            return b.align_with_vol(no_H=True, **kw)[0]
        if mode == "surf":
            return b.align_with_surf(alpha=cfg["alpha"], **kw)[0]
        if mode == "esp":
            return b.align_with_esp(alpha=cfg["alpha"], lam=cfg["lam"], **kw)[0]
        if mode == "pharm":
            return b.align_with_pharm(**kw)[0]
        raise ValueError(mode)

    for cell in plan["cells"]:
        mode, bucket, pool = cell["mode"], cell["bucket"], cell["pool"]
        path = "seq" if W == 1 else ("shmap" if SHMAP_DEFAULT[mode] else "mp")
        rng = np.random.default_rng(seed)
        rows = []
        for nb in cell["sizes"]:
            pairs = []
            for _ in range(nb):
                smi = pool[int(rng.integers(len(pool)))]
                R = _rand_rot(rng)
                t = rng.standard_normal(3); t = t / (np.linalg.norm(t) + 1e-9) * (rng.random() * 3.0)
                ref, ns = build_base(smi)
                pairs.append(MoleculePair(ref, rotated(smi, R, t), do_center=False, num_surf_points=ns))
            try:
                align(mode, pairs)                                  # warmup: JAX JIT
                best = float("inf"); n = 0; total = 0.0
                while n < reps and total < budget:
                    t0 = time.perf_counter(); sc = align(mode, pairs); dt = time.perf_counter() - t0
                    best = min(best, dt); total += dt; n += 1
                m = float(np.asarray(sc).mean())
                rows.append({"n": nb, "t": best, "score": m})
                print(f"  jax   {mode}|{bucket:5s} p={W:<3d} {path:5s} n={nb:<7d} {best:8.3f}s "
                      f"{nb/best:10.1f} pairs/s  self={m:.3f}  (best of {n})", flush=True)
            except Exception as e:
                import traceback
                rows.append({"n": nb, "err": f"{type(e).__name__}: {e}"})
                print(f"  jax   {mode}|{bucket:5s} p={W:<3d} {path:5s} n={nb:<7d} ERR {type(e).__name__}: {e}", flush=True)
                traceback.print_exc(); break
            if best > cap:                                          # cap: too slow for larger sizes
                break
        print("RES " + json.dumps({"key": f"{mode}|{bucket}", "rows": rows}), flush=True)


# ===========================================================================
# Subprocess spawn helpers
# ===========================================================================
def _spawn(cmd_flag, planfile, env, python=None):
    cmd = [python or sys.executable, os.path.abspath(__file__), cmd_flag, planfile]
    env = {**os.environ, **env, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def _spawn_numba_cell(plan):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(plan, fh); planfile = fh.name
    n = str(plan["procs"])
    env = {"PYTHONPATH": _ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "FSS_MOL_CACHE_DIR": _MOLCACHE, "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
           "NUMEXPR_NUM_THREADS": n, "NUMBA_NUM_THREADS": n}
    rec = None
    try:
        p = _spawn("--numba-cell", planfile, env)
        for ln in p.stdout.splitlines():
            if ln.startswith("RES "):
                rec = json.loads(ln.split(" ", 1)[1])
            else:
                print(ln, flush=True)
        if p.returncode != 0 and rec is None:
            sys.stderr.write(p.stderr[-2000:]); rec = {"err": f"exit {p.returncode}"}
    finally:
        try: os.unlink(planfile)
        except OSError: pass
    return rec


def _spawn_orig_cell(plan, mode, procs, orig_python):
    """One original-repo subprocess for one mode at a core budget, using that mode's
    documented multi-core path. shmap modes (vol/pharm) get
    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (set here, before the child
    imports JAX) and let XLA's threadpool use the machine; multiproc modes (surf/esp)
    and the 1-core case pin ``OMP=1`` so workers don't oversubscribe."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(plan, fh); planfile = fh.name
    # ORIG_REPO first so `shepherd_score` is the upstream clone; _ROOT after it so this
    # script's `from benchmarks.benchmark import ...` still resolves (the clone has no
    # `benchmarks` package, so there is no collision).
    env = {"PYTHONPATH": os.pathsep.join([ORIG_REPO, _ROOT, os.environ.get("PYTHONPATH", "")])}
    if SHMAP_DEFAULT[mode] and procs > 1:        # shard_map: N virtual CPU devices
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={procs}"
    else:                                        # multiprocessing / sequential: 1 thread/worker
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    out = {}
    try:
        p = _spawn("--orig-cpu-cell", planfile, env, python=orig_python)
        for ln in p.stdout.splitlines():
            if ln.startswith("RES "):
                rec = json.loads(ln.split(" ", 1)[1]); out[rec["key"]] = rec["rows"]
            else:
                print(ln, flush=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-2000:])
    finally:
        try: os.unlink(planfile)
        except OSError: pass
    return out


# ===========================================================================
# Speed sweep — for each core budget, both engines across modes/buckets/sizes
# ===========================================================================
def _series_key(engine, mode, bucket, procs):
    return f"{engine}|{mode}|{bucket}|p{procs}"


def run_speed(args):
    cfg = _cfg_from_args(args)
    modes, buckets, sizes, procs = args.modes, args.buckets, args.sizes, args.procs
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "plot_data_cpu.json")

    print("=" * 90)
    print("CPU BENCHMARK: original JAX vs fork numba  —  real-drug self-SE(3)-copy (optimum=1.0)")
    print(f"modes={modes} buckets={buckets} sizes={sizes} procs(cores)={procs} "
          f"cap={args.cap:.0f}s original={'off' if args.no_original else 'on'}")
    print("=" * 90)

    data = {}
    # --- fork numba: one isolated subprocess per cell ----------------------
    print(f"\n[numba] MoleculePairBatch.align_with_*(backend='numba')  "
          f"(multi-core via {args.numba_mode}; isolated cells)")
    for p in procs:
        for mode in modes:
            for bucket in buckets:
                key = _series_key("numba", mode, bucket, p)
                pts, scores = [], {}
                for nb in sizes:
                    plan = {"mode": mode, "bucket": bucket, "size": nb, "procs": p,
                            "cfg": cfg, "seed": args.seed, "reps": args.reps,
                            "budget": args.budget, "numba_mode": args.numba_mode}
                    rec = _spawn_numba_cell(plan)
                    if not rec or "err" in rec:
                        print(f"  numba {key} n={nb} ERR {(rec or {}).get('err', 'failed')}", flush=True)
                        break
                    pts.append([nb, rec["n"] / rec["t"]]); scores[nb] = rec["score"]
                    if rec["t"] > args.cap:
                        break
                data[key] = {"pts": pts, "score": scores}

    # --- original jax: one isolated subprocess per core budget -------------
    if not args.no_original:
        print(f"\n[jax] original repo via MoleculePairBatch — intended path: shmap "
              f"(vol/pharm) + multiproc (surf/esp); orig python={args.orig_python or sys.executable}")
        for p in procs:
            for mode in modes:
                plan = {"procs": p, "cap": args.cap, "seed": args.seed, "cfg": cfg,
                        "reps": args.reps, "budget": args.budget,
                        "cells": [{"mode": mode, "bucket": b, "sizes": sizes,
                                   "pool": _fork_pool_smiles(mode, b)} for b in buckets]}
                orig = _spawn_orig_cell(plan, mode, p, args.orig_python)
                for mb, rows in orig.items():
                    m2, bucket = mb.split("|")
                    key = _series_key("jax", m2, bucket, p)
                    data[key] = {"pts": [[r["n"], r["n"] / r["t"]] for r in rows if "t" in r],
                                 "score": {r["n"]: r["score"] for r in rows if "score" in r}}
    elif os.path.exists(out_json):
        try:                                                        # carry over previous JAX lines
            with open(out_json) as fh:
                prev = json.load(fh)
            carried = 0
            for k, d in prev.items():
                if k.startswith("jax|") and k not in data:
                    data[k] = d; carried += 1
            if carried:
                print(f"carried over previous JAX numbers for {carried} series (--no-original)")
        except Exception as e:
            print(f"(no previous JAX data to carry over: {type(e).__name__}: {e})")

    meta = {"hardware": _hardware_info(), "timestamp": _now_str(), "cap": args.cap,
            "tag": args.tag, "cfg": cfg, "modes": modes, "buckets": buckets,
            "sizes": sizes, "procs": procs, "numba_mode": args.numba_mode}
    data["_meta"] = meta
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote {out_json}")
    _render_all(data, meta, args.out_dir)

    print("\nself-accuracy (mean recovered score on self-copies; want ~1.000):")
    for k in sorted(data):
        if k.startswith("_"):
            continue
        sc = data[k].get("score", {})
        if sc:
            worst = min(sc.values())
            flag = "  <-- LOW" if worst < SELF_SCORE_WARN else ""
            print(f"  {k:28s} min={worst:.3f}{flag}")


# ===========================================================================
# Rendering — one panel per core budget; colour=mode, linestyle/marker=engine
# ===========================================================================
LS = {"jax": (0, (5, 2)), "numba": "-"}
MK = {"jax": "D", "numba": "o"}
ENG_LABEL = {"jax": "original · JAX", "numba": "fork · numba"}


def render_plot(data, meta, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes, buckets = meta["modes"], meta["buckets"]
    procs, sizes, cap = meta["procs"], meta["sizes"], meta["cap"]
    hw = meta.get("hardware", {}) or {}
    tag, stamp = meta.get("tag"), meta.get("timestamp", "")

    n = len(procs)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 7.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, procs):
        for mode in modes:
            for bucket in buckets:
                for eng in ENGINES:
                    pts = data.get(_series_key(eng, mode, bucket, p), {}).get("pts", [])
                    if not pts:
                        continue
                    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
                    lbl = f"{mode} · {ENG_LABEL[eng]}" + (f" · {bucket}" if len(buckets) > 1 else "")
                    ax.plot(xs, ys, color=COLOR[mode], linestyle=LS[eng], marker=MK[eng],
                            markersize=7, linewidth=2.4, label=lbl, clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("batch size — pairs aligned per call (log)")
        ax.set_title(f"{p} CPU core{'s' if p != 1 else ''}", fontsize=11.5, fontweight="bold")
        ax.grid(True, which="major", color="#cccccc", alpha=0.8)
        ax.grid(True, which="minor", ls=":", color="#e8e8e8", alpha=0.6)
    axes[0].set_ylabel("pair-alignments / second (higher = faster, log)")
    axes[-1].legend(title="mode · engine", loc="best", framealpha=0.95, fontsize=8.5)

    fig.suptitle("CPU molecular-alignment throughput — original JAX vs fork numba",
                 fontweight="bold", fontsize=13.5, y=0.985)
    note = (f"both engines pinned to the same core budget per panel; each series stops "
            f"where a cell exceeded the {cap:.0f}s wall-clock cap")
    footer = "   ·   ".join(([f"run: {tag}"] if tag else []) + [_hw_footer(hw)]
                            + ([stamp] if stamp else []))
    fig.text(0.5, 0.045, note, ha="center", va="bottom", fontsize=9, color="#555555", style="italic")
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.075, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def _lbl(s):
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def render_table(data, meta, out_md):
    modes, buckets, procs, sizes = meta["modes"], meta["buckets"], meta["procs"], meta["sizes"]
    hw, tag, stamp = meta.get("hardware", {}), meta.get("tag"), meta.get("timestamp", "")
    lbl = {s: _lbl(s) for s in sizes}

    def series(eng, mode, bucket, p):
        return {n: m for n, m in data.get(_series_key(eng, mode, bucket, p), {}).get("pts", [])}

    L = ["# CPU alignment throughput — original JAX vs fork numba (pair-alignments / s)\n",
         "Real drug self-SE(3)-copy pairs (optimum 1.0). `—` = over the wall-clock cap / not run.\n"]
    bits = (([f"run: {tag}"] if tag else []) + ([_hw_footer(hw)] if hw else [])
            + ([stamp] if stamp else []))
    if bits:
        L.append("\n_" + "   ·   ".join(bits) + "_\n")
    for p in procs:
        L += [f"\n## {p} CPU core{'s' if p != 1 else ''} — pairs / s\n",
              "| mode | bucket | engine | " + " | ".join(lbl[s] for s in sizes) + " |",
              "|---|---|---|" + "".join("--:|" for _ in sizes)]
        for mode in modes:
            for bucket in buckets:
                for eng in ENGINES:
                    s = series(eng, mode, bucket, p)
                    cells = " | ".join(f"{s[n]:.0f}" if n in s else "—" for n in sizes)
                    L.append(f"| {mode} | {bucket} | {eng} | {cells} |")
        L += [f"\n### {p}-core numba speedup over JAX (×, matched size)\n",
              "| mode | bucket | " + " | ".join(lbl[s] for s in sizes) + " |",
              "|---|---|" + "".join("--:|" for _ in sizes)]
        for mode in modes:
            for bucket in buckets:
                j = series("jax", mode, bucket, p); nu = series("numba", mode, bucket, p)
                cells = " | ".join(f"{nu[n]/j[n]:.1f}×" if (n in j and n in nu) else "—" for n in sizes)
                L.append(f"| {mode} | {bucket} | {cells} |")
    txt = "\n".join(L) + "\n"
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt); print(f"wrote {out_md}")


def _render_all(data, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_table(data, meta, os.path.join(out_dir, "speed_table_cpu.md"))
    try:
        render_plot(data, meta, os.path.join(out_dir, "speed_plot_cpu.png"))
    except Exception as e:
        print(f"(plot skipped: {type(e).__name__}: {e})")


def run_replot(args):
    out_json = os.path.join(args.out_dir, "plot_data_cpu.json")
    if not os.path.exists(out_json):
        raise SystemExit(f"no data to replot at {out_json}; run the benchmark first")
    with open(out_json) as fh:
        data = json.load(fh)
    meta = data.get("_meta", {})
    if not meta.get("hardware"):
        meta = {**meta, "hardware": _hardware_info(), "timestamp": meta.get("timestamp", _now_str())}
    if not meta.get("tag") and args.tag:
        meta = {**meta, "tag": args.tag}
    print(f"replotting from {out_json}")
    _render_all(data, meta, args.out_dir)


# ===========================================================================
# Accuracy branch — numba vs JAX on DISTINCT-molecule pairs (optimum < 1.0)
# ===========================================================================
def numba_accuracy(modes, pair_smiles, cfg):
    """Align distinct-molecule pairs on the numba backend (in-process, CPU)."""
    from shepherd_score.container import MoleculePair as MP
    import torch
    cpu = torch.device("cpu")
    out = {}
    for mode in modes:
        pairs = [MP(_build_molecule(rs, surf_per_atom=cfg["surf_per_atom"]),
                    _build_molecule(fs, surf_per_atom=cfg["surf_per_atom"]),
                    do_center=False, device=cpu) for rs, fs in pair_smiles]
        _numba_align(mode, pairs, cfg)
        out[mode] = [float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]
    return out


def run_accuracy(args):
    os.environ.setdefault("FSS_MOL_CACHE_DIR", _MOLCACHE)
    cfg = _cfg_from_args(args)
    rng = np.random.default_rng(args.seed)
    smis = [s for _, s, _ in DRUGS]
    pairs = []
    while len(pairs) < args.n_accuracy:
        i, j = rng.integers(len(smis)), rng.integers(len(smis))
        if i != j:
            pairs.append((smis[int(i)], smis[int(j)]))

    print(f"ACCURACY: {len(pairs)} distinct-molecule pairs, numba vs JAX (CPU)\n")
    numba = numba_accuracy(args.modes, pairs, cfg)

    orig = {}
    if not args.no_original:
        import tempfile
        plan = {"task": "accuracy", "seed": args.seed, "cfg": cfg,
                "acc": [{"mode": m, "pairs": pairs} for m in args.modes]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(plan, fh); pf = fh.name
        env = {"PYTHONPATH": os.pathsep.join([ORIG_REPO, _ROOT, os.environ.get("PYTHONPATH", "")]),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        # Reuse the GPU benchmark's original-repo accuracy entry point (orig_cell).
        # Distinct-pair accuracy uses num_workers=1 (sequential), so any JAX works;
        # run it on the same interpreter as the speed sweep's JAX side.
        cmd = [args.orig_python or sys.executable, "-m", "benchmarks.benchmark", "--orig-cell", pf]
        p = subprocess.run(cmd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                           capture_output=True, text=True)
        for ln in p.stdout.splitlines():
            if ln.startswith("RES "):
                rec = json.loads(ln.split(" ", 1)[1]); orig[rec["key"]] = rec["rows"]
        try: os.unlink(pf)
        except OSError: pass

    print(f'\n{"mode":6s} {"numba_mean":>10s} {"jax_mean":>9s} {"mean|Δ|":>9s} {"spearman":>9s}')
    print("-" * 48)
    for mode in args.modes:
        f = np.array(numba.get(mode, []), dtype=float)
        o = orig.get(mode)
        if not isinstance(o, list):
            print(f"{mode:6s} {f.mean():10.4f} {'NA':>9s} {'-':>9s} {'-':>9s}"); continue
        o = np.array(o, dtype=float)
        k = min(len(f), len(o)); f, o = f[:k], o[:k]
        print(f"{mode:6s} {f.mean():10.4f} {o.mean():9.4f} "
              f"{float(np.abs(f - o).mean()):9.4f} {_spearman(f, o):9.4f}")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="fast_shepherd_score CPU benchmark — JAX vs numba")
    ap.add_argument("--numba-cell", help=argparse.SUPPRESS)
    ap.add_argument("--orig-cpu-cell", help=argparse.SUPPRESS)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--buckets", nargs="+", default=BUCKETS, choices=["same", "cross"])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--procs", type=int, nargs="+", default=None,
                    help="CPU core budgets to sweep (numba: torch+numba threads; "
                         "JAX: num_workers / shmap device count). Default: 1 and all cores.")
    ap.add_argument("--orig-python", default=None,
                    help="Python interpreter for the original-repo (JAX) subprocesses. "
                         "Use a JAX>=0.9 env for the intended shard_map path; numba cells "
                         "still use this script's interpreter. Default: same interpreter.")
    ap.add_argument("--numba-mode", choices=["threads", "pool"], default="threads",
                    help="how the numba path uses the core budget: 'threads' (one process, "
                         "numba prange) or 'pool' (shard pairs across a persistent "
                         "single-threaded process pool, _cpu_pool). Default: threads.")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP,
                    help="seconds: a series stops at the first cell over this wall time")
    ap.add_argument("--no-original", action="store_true",
                    help="time only numba (keeps the previous run's JAX lines)")
    ap.add_argument("--accuracy", action="store_true",
                    help="run the numba-vs-JAX distinct-pair parity branch instead")
    ap.add_argument("--replot", action="store_true", help="re-render from plot_data_cpu.json")
    ap.add_argument("--n-accuracy", type=int, default=50)
    ap.add_argument("--tag", default=None, help="write to results_cpu/<tag>/ instead of results_cpu/")
    ap.add_argument("--out-dir", default=None, help="explicit output dir; overrides --tag")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--num-repeats", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.81)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=5,
                    help="max timed reps per cell; the fastest is kept (best-of-N)")
    ap.add_argument("--budget", type=float, default=8.0,
                    help="per-cell time budget (s): stop adding reps once exceeded")
    args = ap.parse_args()
    if args.procs is None:
        args.procs = _default_procs()
    args.procs = sorted(set(args.procs))
    if args.out_dir is None:
        args.out_dir = os.path.join(_RESULTS, args.tag) if args.tag else _RESULTS

    if args.numba_cell:
        return numba_cell(args.numba_cell)
    if args.orig_cpu_cell:
        return orig_cpu_cell(args.orig_cpu_cell)
    if args.replot:
        return run_replot(args)
    if args.accuracy:
        return run_accuracy(args)
    return run_speed(args)


if __name__ == "__main__":
    raise SystemExit(main())
