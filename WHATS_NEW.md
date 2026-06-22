# What's New — GPU-Accelerated Batch Alignment

This fork adds a **Triton GPU engine** for molecular alignment on top of upstream
[`shepherd-score`](https://github.com/coleygroup/shepherd-score). It keeps the entire
original API and behavior, and adds a single opt-in seam.

- **One new public knob:** a `backend=` argument on the existing
  `MoleculePairBatch.align_with_*` methods — `"triton"` for the GPU path, or `"numba"`
  for an explicit CPU path. Nothing else in the public batch API changed.
- **Backward compatible:** the default backend is still the original JAX/CPU path, so
  existing code behaves exactly as before. Triton is an **optional** dependency — if it
  (or a GPU) isn't present, everything falls back transparently.
- **CPU too:** the *same* batched driver runs on a Triton-free **numba** kernel
  (`cpu_overlap.py`) for CPU tensors — selected **per call by tensor device**
  (`kernel_dispatch.py`), so it runs whether or not Triton is installed (CUDA tensors →
  Triton, CPU tensors → numba, in one process). Numerically exact and **~25× faster than
  the original per-pair CPU path** on `vol` (all batched modes except `esp_combo`).

---

## 1. Organization — where the code lives

The acceleration is layered: hand-written GPU kernels at the bottom, batched optimizers
in the middle, and a thin integration seam at the top. Only the top layer is public.
**Genuinely new files are marked `NEW`; everything else is an existing upstream file that
was modified in place.**

```
shepherd_score/
├── score/                              # ── Layer 1: Triton kernels (the raw compute)
│   ├── gaussian_overlap_triton.py          # NEW: fused value+gradient shape (ROCS) overlap
│   ├── gaussian_overlap_esp_triton.py      # NEW: + electrostatic-potential (ESP) weighting
│   ├── pharmacophore_overlap_triton.py     # NEW: typed/directional pharmacophore overlap
│   │                                       #   (pure PyTorch despite the _triton filename)
│   └── pharmacophore_grad_triton.py        # NEW: the Triton pharmacophore value+SE(3) gradient kernel
│
├── alignment/utils/                    # ── Layer 2: batched coarse-to-fine optimizers
│   ├── fast_common.py                      # NEW: batched SE(3) seed generation, quaternion ops
│   ├── fast_se3.py                         # NEW: volumetric (atom-cloud) alignment driver
│   ├── fast_surface_se3.py                 # NEW: surface-point alignment driver
│   ├── fast_esp_se3.py                     # NEW: ESP-weighted alignment driver
│   ├── fast_pharm_se3.py                   # NEW: pharmacophore alignment driver
│   ├── fast_esp_combo_se3.py               # NEW: ShaEP-style combo alignment driver
│   ├── cpu_overlap.py                      # NEW: numba CPU kernels for the batched fast_* drivers
│   ├── kernel_dispatch.py                  # NEW: per-call device routing (Triton on CUDA, numba on CPU)
│   └── se3.py                              # modified: batched (q,t)→SE(3) helper added
│
└── container/                          # ── Layer 3: integration (the public seam)
    ├── _batch_align.py  (NEW)              # NEW: all batch orchestration — size bucketing,
    │                                       #   sub-batching, scatter-fill, _esp_bucketed_align,
    │                                       #   _align_batch_* drivers, multi-GPU sharding
    ├── multi_gpu.py     (NEW)              # NEW: explicit one-process-per-GPU data-parallel
    │                                       #   driver (align_multi_gpu / MultiGPUAligner)
    ├── _batch.py        → MoleculePairBatch  # modified: PUBLIC seam align_with_*(backend="triton")
    └── _core.py         → MoleculePair       # modified: binds the _align_batch_* statics from
                                            #   _batch_align.py; opt-in per-pair fast path (use_fast)
```

**Why this shape.** The speedup is a *batch* phenomenon — it comes from optimizing many
pairs at once in a single GPU dispatch, which amortizes the per-pair Python/launch
overhead. So the public entry point is the **batch** container (`MoleculePairBatch`); the
per-pair `MoleculePair` is a public class whose batch-orchestration helpers
(`_align_batch_*`) are private internals re-exported from the new `_batch_align.py`. The
kernels and optimizers are pure internals, gated behind a `try/except ImportError` so the
package imports fine without Triton, and each batched driver additionally falls back to a
**numba CPU** implementation (`cpu_overlap.py`) when Triton is unavailable. All five `fast_*`
modules import on a CPU-only box; the validated CPU aligners are `vol`/`vol_esp`/`surf`/`esp`/`pharm`,
while `esp_combo` reuses the same numba shape kernel but stays GPU-targeted (its CPU path is
not tuned or validated).

### Changes to existing upstream files

The 14 modules above are **new**. Beyond them, the fork touches only **6 existing upstream
files**; the table below accounts for every one (Δlines = real diff vs
[`coleygroup/shepherd-score`](https://github.com/coleygroup/shepherd-score), ignoring
line-ending noise). Everything else in the package is byte-identical to upstream, and **no
upstream file was deleted**.

| File | Δlines | What changed |
|---|--:|---|
| `container/_batch.py` | +218 | **The public seam.** Adds `_TRITON_BACKENDS` / `_NUMBA_BACKENDS`, the `_triton_align()` router and a `_prepare_numba()` guard, plus a `backend="jax"` (default) argument on every `align_with_*` method (and a brand-new `align_with_esp_combo` batch method — the other five already existed upstream). `backend` in `{"triton","cuda","gpu"}` routes to the batched `MoleculePair._align_batch_*` GPU path; `{"numba","cpu"}` runs that **same** batched driver on CPU with the numba kernels (forces CPU; works on **any** box — kernels are selected per call by tensor device via `kernel_dispatch.py`, so it runs even on a GPU box; excludes `esp_combo`); `"jax"` runs the original path unchanged; any other value raises. |
| `container/_core.py` | +276 | Binds the `_align_batch_*` static methods (defined in the new `_batch_align.py`) onto `MoleculePair`; adds an **opt-in** `use_fast=False` kwarg gating the per-pair Triton fast path on `align_with_esp` / `align_with_esp_combo` / `align_with_pharm` (default preserves the original torch/analytical behavior and honors `use_analytical`); adds a `score_with_vol()` helper. |
| `alignment/utils/se3.py` | +84 | Adds batched SE(3) builders `quaternion_to_SE3` / `quaternions_to_SE3_batch` (the GPU write-back uses the batched form); reworks `apply_SE3_transform` to a single fused `baddbmm` that collapses a singleton batch — the upstream parameter name (`SE3_transform`) and the three shape-validation checks are retained. |
| `generate_point_cloud.py` | +17 | Makes the Open3D import **lazy** (`from __future__ import annotations` + a `_LazyOpen3D` proxy). Open3D is a ~30 s cold import and is fork-hostile (it breaks a later `fork`+CUDA), so deferring it keeps `import shepherd_score` fast and lets the fork-based multi-GPU pool work. Behavior is identical on first real surface use. |
| `protonation/protonate.py` | +2 | Adds `from __future__ import annotations` (deferred annotation evaluation; no behavior change). |
| `alignment/_torch.py` | +8 | Cosmetic only — three trailing commas, one blank line, one comment. (The previously-added dead `VAA_const` parameter was removed, restoring the exact upstream `objective_ROCS_overlay` signature.) |

---

## 2. Format & usage

### Quick start

```python
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

# Build a batch of pairs (each pair = a reference + a fit molecule)
pairs = [MoleculePair(ref, fit) for ref, fit in my_pairs]
batch = MoleculePairBatch(pairs)

# The ONLY thing that changes between engines is the `backend=` argument.
scores, aligned = batch.align_with_surf(alpha=0.81)                     # default: original JAX/CPU
scores, aligned = batch.align_with_surf(alpha=0.81, backend="triton")   # GPU (Triton)  — aliases "cuda"/"gpu"
scores, aligned = batch.align_with_surf(alpha=0.81, backend="numba")    # fast CPU (numba) — alias "cpu"
```

**Which backend?**

| You want… | Use | What it does |
|---|---|---|
| Exactly the upstream behavior (no new deps) | *(omit `backend`)* or `backend="jax"` | Original per-pair JAX/XLA CPU path. |
| Maximum throughput on a CUDA GPU | `backend="triton"` | Batched coarse-to-fine drivers on the Triton GPU kernels. |
| A fast CPU run (no GPU, or to keep the GPU free) | `backend="numba"` | The **same** batched drivers, run on the numba CPU kernels. ~25× over the original CPU path on `vol`. |

`backend="numba"` forces every pair onto CPU and **works on any machine** — even a GPU box with Triton installed — because the kernel is chosen *per call by tensor device* (CPU→numba, CUDA→Triton). So you don't have to uninstall Triton to get a deterministic CPU pass. (`esp_combo` is the one mode with no numba path — it raises `NotImplementedError`; use `"triton"` or `"jax"` for it.)

- `scores` → `np.ndarray` of shape `(N,)`.
- Results are also written **in place** on each pair (e.g. `pair.sim_aligned_surf`,
  `pair.transform_surf`), exactly as in the original API.
- `backend` accepts `"jax"` (default), `"triton"` (GPU; aliases `"cuda"`/`"gpu"`), or
  `"numba"` (explicit CPU; alias `"cpu"`); the alias sets are
  `_TRITON_BACKENDS = ("triton", "cuda", "gpu")` and `_NUMBA_BACKENDS = ("numba", "cpu")`.
- `return_aligned=True` (Triton path) also returns the transformed coordinates; it's
  `False` by default to skip that work when you only need scores + transforms.

### Multi-GPU

Two paths exist, with different trade-offs:

- **Automatic in-library sharding** (no extra arguments — the default). If several CUDA
  devices are visible, large batches are auto-sharded across them on a worker-thread path.
  Sharding only engages above **~4,096 pairs per device**, so small/mid batches deliberately
  stay on a single GPU. Because the alignment is *host-bound*, this thread path is
  GIL-limited and yields only roughly **~1.0–2.3×** on 4 GPUs in these runs — it helps the
  heavier modes (surf/esp/pharm) at very large batch but does not reach ~N×, and `vol`
  (which already saturates one card) is unchanged. **This automatic path is what produced
  the 4-GPU column in the results table below.**
- **Explicit data-parallel driver** (`shepherd_score.container.multi_gpu` —
  `align_multi_gpu` / `MultiGPUAligner`). A separate, opt-in path that launches **one OS
  process per GPU**, each owning its shard end-to-end (build + align, data resident on its
  GPU) with CPU threads capped to `cores/ndev` to avoid oversubscription. By sidestepping
  the GIL it scales closer to linear (**~3.5–3.9× on 4×L40S** in separate experiments). It
  is a one-shot launcher (pays a fixed spawn cost once), so it is aimed at large screens. It
  is **not** what the table below reflects, and it is not yet exported from
  `container/__init__.py` (reachable via the deep `shepherd_score.container.multi_gpu` path).

### The backend matrix

Every alignment mode is reachable from the batch API across all three backends:

| Mode (`MoleculePairBatch` method) | `backend="jax"` (default) | `backend="triton"` (GPU) | `backend="numba"` (CPU) |
|---|:--:|:--:|:--:|
| `align_with_vol`       (shape, heavy-atom)        | ✓ | ✓ | ✓ |
| `align_with_vol_esp`   (shape + ESP, heavy-atom)  | ✓ | ✓ | ✓ |
| `align_with_surf`      (surface shape)            | ✓ | ✓ | ✓ |
| `align_with_esp`       (surface + ESP)            | ✓ | ✓ | ✓ |
| `align_with_pharm`     (pharmacophore)            | ✓ | ✓ | ✓ |
| `align_with_esp_combo` (ShaEP-style combo)        | ✓ | ✓ | ✗ |

Notes:
- **`backend="numba"`** (alias `"cpu"`) runs the *same* batched coarse-to-fine drivers
  as the Triton path, but with the numba CPU kernels instead of Triton. It forces every
  pair onto CPU and **works on any box** — including a GPU box where Triton is installed —
  because kernel selection is **per call, by tensor device** (`alignment/utils/kernel_dispatch.py`):
  CUDA tensors dispatch to Triton, CPU tensors to numba, in the same process. This lets you
  reserve the GPU for another task or run a deterministic CPU pass without uninstalling
  Triton. (Numerically exact vs the Triton path: GPU-vs-CPU agreement ~1e-3, self-copy
  stays 1.000.) All five modes (`vol`/`vol_esp`/`surf`/`esp`/`pharm`) run the numba batched
  kernel on CPU; `pharm` falls back to the per-pair legacy optimizer only when numba is
  absent. (The batched `pharm` aligner now **centers each pair's clouds** before the
  coarse-to-fine optimization — a translation-invariant fix that recovers self-copy to
  ~1.0 even for molecules with very few pharmacophores, on both backends, at no throughput
  cost; this also removes a `numba`-vs-`triton` basin divergence in that regime.)
  `esp_combo` is **excluded** (its CPU path is not tuned/validated) and raises
  `NotImplementedError`.
- The Triton/numba `vol`/`vol_esp` backends align **heavy atoms only** (`no_H=True`); passing
  `no_H=False` raises `NotImplementedError`.
- `max_num_steps` maps to the Triton optimizer's fine-step count.
- The `esp` `lam` weighting is scaled identically across backends; `vol_esp` uses raw
  `lam`, matching the per-pair `MoleculePair.align_with_vol_esp`.
- **Per-pair container API:** the per-pair `MoleculePair.align_with_esp/esp_combo/pharm`
  default to the original torch path and honor `use_analytical`. The per-pair Triton fast
  kernel is **opt-in** via `use_fast=True` — it is no longer auto-selected just because a
  CUDA device is visible (this keeps the per-pair default behavior identical to upstream).
  The batch `backend=` argument above is the primary, fully backward-compatible GPU entry
  point.

### Requirements

The GPU path needs a CUDA device plus [`triton`](https://github.com/triton-lang/triton)
and a CUDA build of PyTorch; the CPU fallback for the batched drivers uses
[`numba`](https://numba.pydata.org/). Both are **optional extras** declared in
`pyproject.toml`:

```bash
pip install shepherd-score[gpu]   # triton  — GPU Triton kernels
pip install shepherd-score[cpu]   # numba   — CPU fallback for the batched drivers
```

`triton` is guarded behind a `try/except ImportError` and `numba` is imported lazily (only
when the CPU fallback actually runs), so with neither installed `import shepherd_score` still
succeeds and the original per-pair JAX/CPU path runs exactly as before.

---

## 3. Strategies Used

The per-step speedups below were measured on a single RTX 4050 laptop GPU (best-of-N,
paired timing); the cross-GPU peaks are in **Net result** at the end. Throughput is in
aligned **pairs per second**; "bit-identical" means scores match the reference to the
last decimal.

### A. The engine — batched optimization (GPU + CPU)
1. **Triton value+gradient kernels** for shape, ESP, and pharmacophore
   overlap (`gaussian_overlap_triton.py`, `gaussian_overlap_esp_triton.py`,
   `pharmacophore_overlap_triton.py`, `pharmacophore_grad_triton.py`). One CTA per pose
   computes the overlap *and* its SE(3) gradient in a single fused pass — replacing the
   per-pair CPU/JAX optimization loop.
2. **Batched coarse-to-fine SE(3) search.** Seed generation, the Adam fine-tuning loop,
   and per-pair selection all run over the whole batch on-GPU, so a 4,096-pair batch is one
   sequence of kernel launches instead of 4,096 independent optimizations.
3. **Two-tier multi-GPU.** Automatic in-library thread sharding (the default, and what the
   4-GPU column below uses) for convenience; plus a separate, opt-in one-process-per-GPU
   driver (`multi_gpu.py`) that sidesteps the GIL for closer-to-N× scaling on large screens
   (see **D** for the robustness work behind the automatic path).
4. **CPU engine (numba) — the same batched driver, on CPU.** The batched drivers (all
   modes except `esp_combo`) have a numba (`@njit(parallel=True)`) value+SE(3)-gradient
   kernel (`cpu_overlap.py`) that replicates the Triton kernel operation-for-operation.
   Kernel choice is **per call, by tensor device** (`kernel_dispatch.py`): CUDA tensors run
   the Triton kernels, CPU tensors the numba ones — so the numba path runs whenever the data
   is on CPU, whether or not Triton is installed (e.g. `backend="numba"` on a GPU box). It is **numerically exact** (computes the true
   overlap+gradient; not bit-identical, since `math.exp` ≠ Triton's `exp2`), so self-copy
   stays 1.000 and distinct-pair scores match. This is a real speedup of the CPU path, not
   just a safety net: on `vol` it reaches **~161 pairs/s/core, ~25× over the original torch
   per-pair CPU path** (and ~5–15× over JAX-batch single-core); `vol_esp` ~162–182/core,
   `pharm` **~237/core** (the numba pharm kernel, ~20× over the previous per-pair-legacy CPU
   fallback). The compute-bound surface modes (`surf`/`esp`) stay slow — they are
   FLOP-bound on `exp`. The GPU path is untouched: for CUDA tensors the dispatcher always
   selects the Triton kernels (the previous behavior). (A push for >2,000 pairs/s/core was explored and found
   physically out of reach for `surf`/`esp`; full log in
   [`SPEED_EXPERIMENTS_CPU.md`](SPEED_EXPERIMENTS_CPU.md).)

### B. Kill the per-pair overhead
5. Profiling showed the alignment was **overhead-bound, not kernel-bound**: per-call setup
   dominated while the kernel was <20% of the time. The fix was to **vectorize the
   per-pair Python loops** — a single host→device fill for real-atom counts and a
   **batched SE(3) write-back** (`quaternions_to_SE3_batch`) instead of one matrix-build
   call per pair. This is pure data movement, so it is **bit-identical**:

   | mode | before | after | speedup |
   |---|--:|--:|:--:|
   | vol  | 5,260 | **18,367** | 3.5× |
   | surf | 2,545 | **16,342** | 6.5× |
   | esp  | 1,673 | **6,199**  | 3.7× |

6. **Batched pad-fill** of the workspace tensors (instead of per-pair GPU
   copies) — also bit-identical: vol **18.4k → 26k**, surf **16k → 18.2k**.
7. **F3 — finish the job (`_scatter_fill`).** Profiling showed step 6's pad-fill
   was *itself* the single largest cost — *bigger than the kernel* for vol. Replacing every
   per-pair fill with one batched `torch.cat` + a vectorized scatter (launch count O(1) in the
   batch size instead of O(K)), plus a single `tolist()` score write-back, is pure data movement
   and **bit-identical**:

   | mode | before | after | speedup |
   |---|--:|--:|:--:|
   | vol   | 23,700 | **54,800** | 2.3× |
   | surf  | 16,500 | **28,900** | 1.76× |
   | esp   | 5,700  | **8,000**  | 1.4× |
   | pharm | 6,200  | **11,800** | 1.9× |

   This is what made the align **GPU-bound** — host down to ~4 µs/pair (under the 10 µs/pair =
   100k-pairs/s target), lifting the host ceiling that had been capping GPU scaling.

### C. Accuracy-safe kernel & loop tuning
8. **Early-stop trim** (patience 5→2) for the shape/surface modes, which converge fast —
   bit-identical, ~1.15× (vol) / ~1.28× (surf). (ESP/pharm keep patience 5; they converge
   slower.) This is exposed as a tunable (`FINE_ES_PATIENCE`) used by the speed-lab harness.
9. **Kernel occupancy & schedule tuning.** Small tiles (one warp per CTA) suit the tiny
   per-pose problems; the BLOCK size and kernel schedule (`num_warps`, `num_stages`) are now
   selected per problem shape via `@triton.autotune` and validated to be bit-identical.

### D. Multi-GPU & very large batches (cluster / L40S work)
10. **Robust multi-GPU sharding.** Automatic sharding engages only above **~4,096 pairs per
    device** (so on a 4-GPU node it's a ≥~16k-pair feature; smaller batches stay on one GPU).
    Making it hold up on a real 4×L40S node took several fixes: a single-threaded **per-device
    cuSOLVER warm-up** (PyTorch's lazy handle init isn't thread-safe), a **device-keyed
    CUDA-graph cache** with thread-local capture, and **migrating** the cached
    `vol_esp`/`esp_combo`/`pharm` workspace tensors onto the active device. CUDA-graphs are
    turned off on the sharded path (capture races across worker threads), so it runs the eager
    fine loop — **same scores**, just without the single-GPU graph speedup. This automatic
    thread path is what the 4-GPU column below uses; being GIL-limited it gains only
    ~1.0–2.3× there. The separate `multi_gpu.py` one-process-per-GPU driver is the path that
    reaches closer-to-N× scaling, but it is **not** reflected in that table.
11. **100k-pairs-per-call batches.** cuSOLVER's batched `eigh` fails past ~8k problems, so the
    SE(3) seed solve is **chunked to ≤4,096** — numerically identical, but it's what lets a
    single call align **100,000 pairs** without crashing. This is what backs the flat
    large-batch throughput on the L40S below.

### E. Note
12. The headline benchmark runs **each `(mode, size)` cell in its own fresh subprocess**.
    A laptop GPU throttles under sustained load (a ~2–3× artifact across a long run) and
    Triton autotune keys on the per-pose shape (so a process that warms up on a tiny batch
    would lock in the wrong config). Per-cell isolation gives each measurement a recovered
    clock and a correctly-autotuned kernel.

### Net result
Peak aligned **pairs/second** by mode and GPU — the *same* fork code (Triton/GPU)
on six devices, taken from the most recent benchmark run on each (real drug
self-copy pairs, isolated best-of-N; the 4-GPU column is the automatic in-library
thread-sharding path — the same `MoleculePairBatch.align_with_*(backend="triton")` call run
on a 4-GPU node, not the explicit `multi_gpu.py` driver; **bold** = fastest per mode):

| mode | RTX 4050 laptop | L40S · 1 GPU | L40S · 4 GPU | RTX PRO 6000 Blackwell | H100 NVL | H200 |
|---|--:|--:|--:|--:|--:|--:|
| vol   | 54,200 | 160,500 | 160,200     | 174,400 | **177,600** | 165,700 |
| surf  | 28,800 | 81,700  | **125,800** | 108,800 | 65,700      | 70,600  |
| esp   | 8,500  | 32,100  | **70,500**  | 48,900  | 28,500      | 31,800  |
| pharm | 23,300 | 57,300  | **83,900**  | 64,500  | 67,800      | 74,800  |

![Molecular-alignment throughput across GPUs](benchmarks/results/speed_all_hardware.png)

Every datacenter card clears **160k+ vol pairs/s** — roughly **3× the laptop** —
because the post-F3 host path (~4 µs/pair) finally *feeds* a fast GPU instead of
starving it; on the laptop the small GPU is the ceiling, so it sits at ~54k. A
second-through-fourth L40S GPU mostly helps the heavier modes at large batch (surf/esp/pharm
at 100k, where automatic sharding engages above ~4,096 pairs/device — though, being
host/GIL-bound, it gains only ~1.0–2.3× there, not ~4×); `vol` already saturates one card,
so 1- and 4-GPU `vol` match.

*(laptop: RTX 4050 · Core Ultra 9 185H · torch 2.5.1 / CUDA 12.4. Cluster cards
(`pi_melkin` nodes) all on torch 2.11 / CUDA 12.8: L40S 1-/4-GPU · Xeon Gold 6542Y;
RTX PRO 6000 Blackwell · EPYC 9135; H100 NVL · EPYC 9474F; H200 · Xeon Platinum 8580.
Molecule cache on; all runs 2026-06-19.)*

Full experiment log, including rejected ideas, is in [`SPEED_EXPERIMENTS.md`](SPEED_EXPERIMENTS.md);
the CPU-fallback experiments are in [`SPEED_EXPERIMENTS_CPU.md`](SPEED_EXPERIMENTS_CPU.md).
Reference benchmark outputs (RTX 4050 laptop, L40S 1-/4-GPU, RTX PRO 6000 Blackwell,
H100 NVL, H200) ship under [`benchmarks/results/`](benchmarks/results/) — one folder per
device plus a combined [`speed_all_hardware.png`](benchmarks/results/speed_all_hardware.png)
across all six (regenerate it with `python benchmarks/plot_all_hardware.py`). Re-running a benchmark builds a
deterministic molecule cache once (`FSS_MOL_CACHE_DIR`, default `benchmarks/molcache/`) so
repeat runs start fast.