# Alignment speed + accuracy benchmark

A fair, reproducible harness that measures **how fast** and **how accurately**
each alignment *mode* runs across every compute *configuration* shepherd_score
offers.

```
python -m benchmarks.run_alignment_benchmark --quick                 # smoke
python -m benchmarks.run_alignment_benchmark --out-md bench.md --out-json bench.json
python -m benchmarks.run_alignment_benchmark --mixed-range 8 120     # stress bucketing
```

It runs on CPU-only boxes (GPU rows auto-skip) and on GPU boxes (everything
runs except multi-GPU, which needs >1 device).

---

## The matrix

**Modes** (the objective being optimised): `vol` (atom-cloud ROCS shape),
`surf` (surface-cloud ROCS shape), `esp` (surface shape + electrostatics),
`pharm` (pharmacophore).  `esp_combo` is a documented extension point (it needs
~10 correlated arrays; see *Extending* below).

**Configurations** (device × granularity):

| backend | device | granularity | engine |
|---|---|---|---|
| `cpu_single_torch` | CPU | single (1 pair/call) | torch autograd, 1 thread — **accuracy reference** |
| `cpu_single_analytical` | CPU | single | analytical-gradient optimiser, 1 thread |
| `cpu_multi_torch` | CPU | multi (pair-level) | thread pool of single-pair aligns (1 thread each) |
| `cpu_multi_intraop` | CPU | multi (op-level) | sequential pairs, N intra-op BLAS threads |
| `gpu_single_torch` | GPU | single | torch autograd on CUDA |
| `gpu_single_fast` | GPU | single | Triton fast optimiser (`fast_optimize_*`) |
| `gpu_multi_batch` | GPU | multi (batched) | container `align_batch_*` with size bucketing |
| `gpu_multi_jax_shmap` | GPU×N | multi-device | JAX `shard_map` across devices — **auto-skips unless >1 device** |

"single" = one pair per call (latency); "multi" = many pairs processed together
(throughput).  Both are summarised as **pairs/s** so they are directly
comparable, and also as **ms/pair** latency.

There are deliberately **two** CPU-multi backends because CPU parallelism has two
distinct regimes, and reporting only one would be misleading:

* `cpu_multi_torch` parallelises *across pairs* (thread pool).  Torch's C++ ops
  release the GIL so the heavy linear algebra runs in parallel, but the Python
  optimisation loop is serialised — so its speedup is best when there are many
  pairs and saturates below #cores.
* `cpu_multi_intraop` parallelises *within* each align (multi-threaded BLAS).
  Its speedup grows with molecule *size* (bigger matmuls), not pair count.

For tiny molecules both can be *slower* than single-threaded because of
threading overhead — that is real and the harness shows it rather than hiding it.

---

## Why these workloads (fair & honest by construction)

* **Known ground truth.** Each pair's *fit* is a rigid SE(3) copy of its *ref*
  (optionally + Gaussian noise).  With `--noise 0` the global optimum is a
  perfect overlap, so the best attainable Tanimoto score is exactly **1.0** for
  every mode.  Accuracy is then "how close to 1.0 did this backend get"
  (`gap_to_ideal`) — a single, backend-neutral, interpretable number.  We do not
  test bit-for-bit equality of two implementations; we test *delivered alignment
  quality*, which is what actually matters downstream.

* **Size control → bucketing fairness.** Every cohort comes in two flavours:
  * **uniform** — all molecules the same size ⇒ the GPU batch path collapses to a
    **single padded bucket** (its best case).
  * **mixed** — sizes spread over a range ⇒ the batch path spreads across **many
    buckets** with more padding waste (its realistic case).

  The report prints the **bucket count** for every GPU-multi cell, so the batch
  path's throughput advantage is always shown *next to* the size homogeneity that
  produced it. Run both before quoting a speedup; quoting only the uniform number
  is the classic way to flatter a bucketed kernel.

* **Backend-neutral storage.** Workloads are plain numpy.  Each backend converts
  to its own representation (cpu/cuda tensors, padded batches, int vs float
  pharm types) **inside its timed `prepare` step**, so device-transfer and
  padding costs are charged to the backend that incurs them — never hidden inside
  the compute number.

The synthetic generator is decoupled from RDKit/xTB/Open3D so the suite is fully
reproducible and runs anywhere. The arrays it produces are the *exact* arrays the
kernels consume in production — only the source of the numbers is synthetic. A
real-molecule loader can be dropped in behind the `Cohort` interface (see below).

---

## What the timings mean (the fairness ledger)

| column | meaning |
|---|---|
| `prepare ms` | one-time data prep: host→device transfer, batch padding, MoleculePair construction. **Excluded** from steady-state compute. |
| `cold ms` | `prepare` + the **first** compute call — the true first-call latency, including Triton autotune / JIT. |
| `pairs/s`, `ms/pair` | **median** over post-warmup repeats, with `torch.cuda.synchronize()` bracketing every measured region (CUDA kernels are async — timing without a sync measures the launch, not the work). |
| `mean`, `gap→ideal` | delivered overlap; `gap→ideal = 1 − score` (noiseless only). |
| `vs-ref |Δ|`, `spearman` | per-pair agreement vs `cpu_single_torch`: mean abs score diff and rank correlation (rank correlation is what matters for screening/triage). |
| `buckets` | number of padded size buckets the GPU batch path used for this cohort. |

Honesty notes baked in:

* The Triton fast/batch optimiser and the torch reference use **genuinely
  different algorithms** (coarse grid → top-k → fine-tune vs random-restart
  autograd). We never pretend they should match to machine precision — we report
  achieved score **and** time so the speed/accuracy trade-off is explicit.
* On **noiseless** cohorts every backend reaches ≈1.0, which makes per-pair
  *ranking* (`spearman`) degenerate — there is nothing to rank. Use `--noise`
  (e.g. `--noise 0.3`) to create a spread of achievable scores and get a
  meaningful ranking comparison.
* `cpu_multi_*` thread counts and `gpu_multi_batch` bucket counts are recorded in
  the JSON `extra`/`n_buckets` fields.

---

## Accuracy assertions (CI)

`tests/test_alignment_backends.py` expresses the accuracy half as pytest
assertions (markers `slow`, `cuda`):

* every available backend recovers ≥ 0.90 overlap on a noiseless cohort;
* the GPU fast/batch paths agree with the torch reference within tolerance;
* bucket counts are correct (uniform → 1, mixed → ≥ 2).

```
pytest tests/test_alignment_backends.py -m "slow"          # CPU
pytest tests/test_alignment_backends.py -m "slow and cuda" # +GPU
```

No timing is asserted (machine dependent) — speed lives in the benchmark report.

---

## Extending

* **`esp_combo` mode.** Add an entry to `MODES`, generate the extra arrays
  (atom centers w/ H, partial charges, radii, surface ESP) in
  `workloads._make_molecule`, and wire `optimize_esp_combo_score_overlay` /
  `fast_optimize_esp_combo_score_overlay` / `align_batch_esp_combo` into
  `backends.py`.
* **Real molecules.** Implement an alternative `make_cohort` that builds
  `shepherd_score.container.Molecule` objects from SMILES (via
  `conformer_generation` + xTB + Open3D surfaces) and bins them by heavy-atom
  count into uniform/mixed cohorts. The rest of the harness is unchanged.
* **Multi-GPU** is implemented (`gpu_multi_jax_shmap`, in
  `alignment_bench/jax_shmap.py`) on top of
  `shepherd_score.alignment._jax_parallel`'s `shard_map` kernels. It shards the
  pair axis across `jax.devices()` (the GPUs on a multi-GPU host) and supports
  `vol`/`surf`/`esp`/`pharm`. It auto-skips when only one device is visible.

  **Run it on real GPUs:** nothing special — JAX shards across every visible
  GPU; restrict with `CUDA_VISIBLE_DEVICES`. The backend disables JAX VRAM
  preallocation so it coexists with the torch/Triton backends.

  **Validate it without a multi-GPU box:** JAX can fork N virtual CPU devices,
  exercising the identical shard_map code path. The env vars must be set before
  JAX imports, so it runs as its own entry point:

  ```
  SIM_DEVICES=4 python -m benchmarks.validate_multigpu     # simulate 4 devices
  python  -m benchmarks.validate_multigpu --real           # use real devices
  ```

  `tests/test_alignment_backends.py::test_multidevice_shmap_simulated` runs this
  in a subprocess as part of CI. On a real multi-GPU host, drop
  `gpu_multi_jax_shmap` into a normal `run_alignment_benchmark` run and it will
  participate automatically.

## Layout

```
benchmarks/
  run_alignment_benchmark.py     # CLI
  alignment_bench/
    workloads.py                 # size-controlled cohorts + ground truth
    backends.py                  # prepare/run adapters per (mode, config)
    timing.py                    # warmup + cuda-sync + robust stats
    metrics.py                   # accuracy vs reference / vs ideal
    runner.py                    # matrix + markdown/JSON report
tests/test_alignment_backends.py # accuracy assertions (slow/cuda)
```
