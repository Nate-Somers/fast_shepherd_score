# What's New — GPU-Accelerated Batch Alignment

This fork adds a **Triton GPU engine** for molecular alignment on top of upstream
[`shepherd-score`](https://github.com/coleygroup/shepherd-score). It keeps the entire
original API and behavior, and adds a single opt-in seam.

- **One new public knob:** a `backend="triton"` argument on the existing
  `MoleculePairBatch.align_with_*` methods. Nothing else in the public API changed.
- **Backward compatible:** the default backend is still the original JAX/CPU path, so
  existing code behaves exactly as before. Triton is an **optional** dependency — if it
  (or a GPU) isn't present, everything falls back transparently.

---

## 1. Organization — where the new code lives

The acceleration is layered: hand-written GPU kernels at the bottom, a batched optimizer
in the middle, and a thin integration seam at the top. Only the top layer is public.

```
shepherd_score/
├── score/                              # ── Layer 1: Triton kernels (the raw compute)
│   ├── gaussian_overlap_triton.py          # fused value+gradient shape (ROCS) overlap
│   ├── gaussian_overlap_esp_triton.py      # + electrostatic-potential weighting
│   ├── pharmacophore_overlap_triton.py     # typed/directional pharmacophore overlap
│   └── pharmacophore_grad_triton.py        # pharmacophore value+SE(3) gradient
│
├── alignment/utils/                    # ── Layer 2: batched coarse-to-fine optimizers
│   ├── fast_common.py                      # batched SE(3) seed generation, quaternion ops
│   ├── fast_se3.py                         # volumetric (atom-cloud) alignment driver
│   ├── fast_surface_se3.py                 # surface-point alignment driver
│   ├── fast_esp_se3.py                     # ESP-weighted alignment driver
│   ├── fast_pharm_se3.py                   # pharmacophore alignment driver
│   └── fast_esp_combo_se3.py               # ShaEP-style combo alignment driver
│
└── container/                          # ── Layer 3: integration (the public seam)
    ├── _batch.py    → MoleculePairBatch    # PUBLIC: align_with_*(backend="triton")
    └── _core.py     → MoleculePair         # PRIVATE: _align_batch_* orchestration
                                            #   (bucketing, multi-GPU sharding, sub-batching,
                                            #    device-tensor caching) + _esp_bucketed_align
```

**Why this shape.** The speedup is a *batch* phenomenon — it comes from optimizing many
pairs at once in a single GPU dispatch, which amortizes the per-pair Python/launch
overhead. So the public entry point is the **batch** container (`MoleculePairBatch`), not
the per-pair `MoleculePair`. The kernels and optimizers are pure internals, gated behind a
`try/except ImportError` so the package imports fine without Triton.

---

## 2. Format & usage

### Quick start

```python
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

# Build a batch of pairs (each pair = a reference + a fit molecule)
pairs = [MoleculePair(ref, fit) for ref, fit in my_pairs]
batch = MoleculePairBatch(pairs)

# Original behavior (unchanged): JAX/CPU
scores, aligned = batch.align_with_surf(alpha=0.81)

# NEW: same call, GPU-accelerated
scores, aligned = batch.align_with_surf(alpha=0.81, backend="triton")
```

- `scores` → `np.ndarray` of shape `(N,)`.
- Results are also written **in place** on each pair (e.g. `pair.sim_aligned_surf`,
  `pair.transform_surf`), exactly as in the original API.
- `backend` accepts `"jax"` (default) or `"triton"` (aliases `"cuda"`/`"gpu"`).
- **Multi-GPU is automatic:** if several CUDA devices are visible, the batch is sharded
  across them — no extra arguments.
- `return_aligned=True` (Triton path) also returns the transformed coordinates; it's
  `False` by default to skip that work when you only need scores + transforms.

### The backend matrix

Every alignment mode is reachable from the batch API with both backends:

| Mode (`MoleculePairBatch` method) | `backend="jax"` (default) | `backend="triton"` (GPU) |
|---|:--:|:--:|
| `align_with_vol`       (shape, heavy-atom)        | ✓ | ✓ |
| `align_with_vol_esp`   (shape + ESP, heavy-atom)  | ✓ | ✓ |
| `align_with_surf`      (surface shape)            | ✓ | ✓ |
| `align_with_esp`       (surface + ESP)            | ✓ | ✓ |
| `align_with_pharm`     (pharmacophore)            | ✓ | ✓ |
| `align_with_esp_combo` (ShaEP-style combo)        | ✓ | ✓ |

Notes:
- The Triton `vol`/`vol_esp` backends align **heavy atoms only** (`no_H=True`); passing
  `no_H=False` raises `NotImplementedError`.
- `max_num_steps` maps to the Triton optimizer's fine-step count.
- The `esp` `lam` weighting is scaled identically across backends; `vol_esp` uses raw
  `lam`, matching the per-pair `MoleculePair.align_with_vol_esp`.

### Requirements

The GPU path needs a CUDA device plus [`triton`](https://github.com/triton-lang/triton)
and a CUDA build of PyTorch. With neither installed, import still succeeds and the JAX/CPU
path runs as before.

---

## 3. Strategies Used

Measured on a single RTX 4050 laptop GPU (best-of-N, paired timing). Throughput is in
aligned **pairs per second**; "bit-identical" means scores match the reference to the
last decimal.

### A. The engine — batched GPU optimization
1. **Triton value+gradient kernels** for shape, ESP, and pharmacophore
   overlap. One CTA per pose computes the overlap *and* its SE(3) gradient in a single
   fused pass — replacing the per-pair CPU/JAX optimization loop.
2. **Batched coarse-to-fine SE(3) search.** Seed generation, the Adam fine-tuning loop,
   and top-k selection all run over the whole batch on-GPU, so a 4,096-pair batch is one
   sequence of kernel launches instead of 4,096 independent optimizations.
3. **Multi-GPU auto-sharding** — batches are split across all visible CUDA devices with no
   code change.

### B. kill the per-pair overhead 
4. Profiling showed the alignment was **overhead-bound, not kernel-bound**: per-call setup
   dominated while the kernel was <20% of the time. The fix was to **vectorize the
   per-pair Python loops** — a single host→device fill for real-atom counts and a
   **batched SE(3) write-back** (`quaternions_to_SE3_batch`) instead of one
   `quaternion_to_SE3` call per pair. This is pure data movement, so it is **bit-identical**:

   | mode | before | after | speedup |
   |---|--:|--:|:--:|
   | vol  | 5,260 | **18,367** | 3.5× |
   | surf | 2,545 | **16,342** | 6.5× |
   | esp  | 1,673 | **6,199**  | 3.7× |

5. **Batched pad-fill** of the workspace tensors (`pad_sequence` instead of per-pair GPU
   copies) — also bit-identical: vol **18.4k → 26k**, surf **16k → 18.2k**.

### C. Accuracy-safe kernel & loop tuning
6. **Early-stop trim** (patience 5→2) for the shape/surface modes, which converge fast —
   bit-identical, ~1.15× (vol) / ~1.28× (surf). (ESP/pharm keep patience 5; they converge
   slower.)
7. **Kernel occupancy fix** — small tiles (`BLOCK=16`, one warp per CTA) suited to the tiny
   per-pose problems; ~3× at large batch over the naive configuration.
8. **Autotuned kernel schedules** (`num_stages`, `maxnreg`) selected per problem shape and
   validated to be bit-identical — a free ~1.09× on the kernel.

### D. Note
9. The headline benchmark runs **each `(mode, size)` cell in its own fresh subprocess**.
   A laptop GPU throttles under sustained load (a ~2–3× artifact across a long run) and
   Triton autotune keys on the per-pose shape (so a process that warms up on a tiny batch
   would lock in the wrong config). Per-cell isolation gives each measurement a recovered
   clock and a correctly-autotuned kernel.

### Net result
True per-mode peak throughput (isolated, single GPU): **vol ≈ 25k**, **surf ≈ 16k**,
**esp ≈ 6k**, **pharm ≈ 9k** pairs/s — with the shape and surface modes producing results
that are **bit-identical** to the original implementation. Full experiment log, including
rejected ideas, is in [`SPEED_EXPERIMENTS.md`](SPEED_EXPERIMENTS.md).
