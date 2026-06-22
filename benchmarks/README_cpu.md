# CPU benchmark — original JAX vs fork numba

The CPU counterpart to [`README.md`](README.md). Same idea, two **CPU** engines through
the same public API (`MoleculePairBatch.align_with_*`):

* **original** — the upstream repo's default **JAX/XLA** path (no `backend` arg).
* **fork** — `backend="numba"`: the *same* batched coarse-to-fine driver the Triton
  path uses, with numba CPU kernels.

```bash
# numba runs in this interpreter; the JAX side runs in --orig-python (a JAX>=0.9 env,
# needed for the intended shard_map path — see "What makes it fair").
PY=/path/to/SimModelEnv/bin/python                       # numba + orchestrator (open3d, numba)
JX=/path/to/SimModelEnvJax09/bin/python                  # original JAX side (jax>=0.9)

$PY -m benchmarks.benchmark_cpu --orig-python $JX                 # full: 1-core and all-cores
$PY -m benchmarks.benchmark_cpu --orig-python $JX --procs 1 8     # specific core budgets
$PY -m benchmarks.benchmark_cpu --no-original                     # numba only (keeps last JAX line)
$PY -m benchmarks.benchmark_cpu --orig-python $JX --modes vol pharm
$PY -m benchmarks.benchmark_cpu --orig-python $JX --accuracy      # numba-vs-JAX distinct-pair parity
$PY -m benchmarks.benchmark_cpu --numba-mode pool --no-original   # numba via the process pool (see scaling)
$PY -m benchmarks.benchmark_cpu --replot                          # re-render from plot_data_cpu.json
```

Run it in the **GPU conda env (WSL2 `SimModelEnv`)** — the JAX path and the Open3D
surface builder don't exist on the CPU-only Windows box. It's CPU-only at run time
(the numba cell sets `CUDA_VISIBLE_DEVICES=""`).

## What makes it fair

Every pair is a **real drug aligned to a rigid SE(3) copy of itself** (optimum
score = 1.0), so each cell self-reports accuracy. Both engines share the molcache,
seed, and alignment cfg; warmup (numba/JAX JIT) is excluded; each cell is best-of-N,
time-budgeted, and runs in its **own isolated subprocess** (both packages are named
`shepherd_score`). `esp_combo` is excluded — it has no numba path.

**The multiproc problem.** The two engines don't parallelise the same way, and the
JAX path uses **the mechanism the original docs intend** — each mode's documented
default (`MoleculePairBatch.align_with_*`):

| engine | mode | multi-core path | knob |
|---|---|---|---|
| JAX | vol, pharm | **`jax.shard_map`** across N virtual CPU devices (one process) | `use_shmap=True`, `XLA_FLAGS=--xla_force_host_platform_device_count=N` |
| JAX | surf, esp | size-sorted **multiprocessing** (1 thread/worker) | `use_shmap=False`, `num_workers=N` |
| numba | all | **threads** (`@njit(parallel=True)` prange + torch CPU threads, one process) | `torch`/`numba` thread count |

So "test multiproc" can't mean *match the mechanism* — it means **match the cores**.
Each cell pins both engines to the same core budget and `--procs` sweeps it:

* **1 core** — per-core kernel throughput (the load-bearing CPU metric; see
  [`../SPEED_EXPERIMENTS_CPU.md`](../SPEED_EXPERIMENTS_CPU.md)). All engines run their
  sequential path (`num_workers=1`, single thread).
* **N cores** — aggregate throughput (the multiproc comparison): numba at N threads;
  JAX via shard_map (N devices) for vol/pharm or N worker processes for surf/esp.

> **shard_map needs JAX ≥ 0.9.0** (its `_jax_parallel.py` uses the new `jax.shard_map`
> API). SimModelEnv ships JAX 0.7.1, so the JAX side runs in a **separate `--orig-python`
> env** (a clone with `jax[cpu]>=0.9` installed); numba stays in SimModelEnv. The parent
> sets `XLA_FLAGS` before that subprocess imports JAX, so `len(jax.devices()) == N`.
> The shard_map path uses `lax.scan` (fixed `max_num_steps`, no early stop) — a real
> property of the intended path; self-copy accuracy stays ~1.0.

## numba multi-core scaling (why ~5–6×, not N×)

The numba path parallelises by **threads**: the `@njit(parallel=True)` `prange` kernel
over K = pairs×seeds poses does the heavy O(K·N·M) overlap+grad; the per-step torch
Adam/`where` bookkeeping is the only other work. [`experiments/cpu_numba_scaling_probe.py`](experiments/cpu_numba_scaling_probe.py)
isolates where scaling is lost (Ultra 9 185H, 6 P-cores + 8 E-cores):

- **The kernel itself caps at ~5×** (end-to-end tracks kernel-only, so the torch
  bookkeeping is *not* the bottleneck). The ceiling is physical: ~6 fast P-cores. Because
  `prange` static-schedules evenly, a slow E-core thread straggles each per-call barrier —
  so **6 threads can beat 8**.
- **Don't oversubscribe torch.** numba and torch each spinning N threads fight: torch=N is
  ~30–40% slower than torch=1 at 8 cores for the dense shape modes. The benchmark now sets
  **torch=1 for vol/surf/esp, torch=2 for pharm** (its type-sparse kernel is lighter, so its
  torch fraction is bigger). Net vs the old torch=N: **vol +13%, surf +30%, esp +62%**, pharm ~flat.
- **Process-sharding** is implemented as a persistent single-threaded worker pool
  ([`shepherd_score/container/_cpu_pool.py`](../shepherd_score/container/_cpu_pool.py)),
  opt-in via `align_with_*(backend="numba", num_workers=N)` (mirrors the JAX path) or
  `benchmark_cpu.py --numba-mode pool`. It shards the *pairs* across N processes, each
  running the unchanged batched aligner single-threaded — no per-step barrier, so it wins
  where the thread path is barrier/overhead-bound. Measured at 8 cores (apples-to-apples,
  both via `align_with_*`): **pharm +53–60%, vol +9–10%**, surf/esp a wash (already
  compute-bound near the physical ceiling). Bit-identical at the optimum (self-copy);
  within the driver's batch-global early-stop tolerance otherwise (see
  [`tests/test_cpu_pool.py`](../tests/test_cpu_pool.py)).

On a homogeneous many-core server, expect scaling much closer to N×; this laptop's P/E split
is the cap (the pool removes the *barrier* penalty but not the physical-core ceiling).

## Outputs

Written to `--out-dir` (default `benchmarks/results_cpu/`, or `results_cpu/<tag>/`):

* `speed_plot_cpu.png` — **one panel per core budget**; colour = mode, solid = numba
  (fork), dashed = JAX (original); annotated with the CPU it ran on.
* `speed_table_cpu.md` — pairs/s per cell + numba-over-JAX speedup, per core budget.
* `plot_data_cpu.json` — raw numbers + a `_meta` block, read back by `--replot`.

Modes (default all): `vol`, `surf`, `esp`, `pharm`. Sizes (default): 10, 100, 1000,
10000 — a series stops where a cell exceeds `--cap` (default 20 s).
