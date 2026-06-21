# Potential Future Work

> Working notes on **profiled-but-unimplemented** ideas for `fast_shepherd_score`. Each entry is
> **IDEA / not implemented**: a write-up of measurements + a proposed change, kept so the analysis
> isn't re-derived. Any implementation must stay **backwards compatible with the default set to the
> original behavior** (gate every new lever OFF by default; see each idea's constraints). The speed
> levers that were *tried and rejected/accepted* live in `SPEED_EXPERIMENTS.md`; this file is for
> larger ideas that are measured but not yet built.
>
> Ideas:
> 1. Cutting `Molecule` construction overhead (mesh-free surface)
> 2. Per-pair early-stop for `esp` distinct-pair alignment

---

# Idea 1 — Cutting `Molecule` construction overhead (mesh-free surface)

**Status: IDEA / not implemented.** This is a write-up of profiling + a proposed opt-in
fast surface path. Nothing here is wired into the code yet. Any implementation must be
**backwards compatible with the default set to the original** behavior (see *Constraints*).

### TL;DR
- Building a `Molecule` costs **~58 ms** (range 25–107 ms over the real drug set). **~90% of
  that is Open3D surface generation, and ~87% is a single `create_from_point_cloud_ball_pivoting`
  call** (50.8 ms). Everything else (MMFF charges, ESP, pharmacophores, vdW radii) is <4 ms combined.
- The ball-pivoting mesh is built only to **Poisson-disk-resample evenly-spaced surface points**;
  the mesh itself is thrown away.
- A **mesh-free** surface (mask the fibonacci sphere samples, then subsample) is **~62× faster**
  at surface generation. With **farthest-point sampling (FPS)** the subsample, distinct-molecule
  alignment scores/rankings are **indistinguishable from the current method's own run-to-run noise**.
- **BUT** mesh-free points lie *exactly on atom spheres* → they **leak atom positions**, and FPS is
  **deterministic**. That makes mesh-free/FPS great for **scoring/screening** but **wrong for the
  generative-modeling pipeline**, which depends on the original surface's stochasticity + seam-smoothing.
- **Proposal:** add the fast surfacer as an **opt-in** path (default unchanged), used only at
  scoring/screening call sites. Leave the generative pipeline on the original surface.

### 1. Measured overhead
Profiled in WSL `SimModelEnv` (RDKit 2025.03.6 / Open3D 0.19), benchmark drug set, `surf_per_atom=3`,
best-of-N wall-clock + cProfile attribution.

| step (inside `Molecule.__init__`) | cost / build | share |
|---|--:|--:|
| Open3D surface gen (`get_pc`) | ~54 ms | ~90% |
| └─ `create_from_point_cloud_ball_pivoting` | **50.8 ms** | **~87%** |
| `get_pharmacophore` | 1.68 ms | ~3% |
| cdist + fibonacci sampling + vdW radii | ~1.9 ms | ~3% |
| MMFF `get_partial_charges` + ESP | 0.40 ms | <1% |
| **total `Molecule.__init__`** | **~58 ms (mean)** | 100% |

Separately, conformer embedding (`embed_conformer_from_smiles`, the **precursor** — not part of
`Molecule.__init__`) is ~18 ms mean (up to 59 ms for imatinib).

**Where it does / doesn't matter.** One build (~58 ms) ≈ 100–1000× a single GPU pair-alignment, so
the surface build dominates wall-clock only in an **embed-free / xTB-free pure-geometry screen** (and
in `benchmarks/`). In the real pipelines (`objective.py`, `evaluations/*`) each construction sits
behind ETKDG embedding and often an **xTB** single-point/relax (seconds–minutes), which dominates;
there, conformer/representation caching beats optimizing the 58 ms surface build. Also note: **ESP
alignment uses MMFF charges by default** (~0.14 ms) — xTB charges are an opt-in quality upgrade passed
in via `partial_charges=`, not a requirement.

### 2. The surface is non-deterministic
Two builds of the *same* molecule with identical args differ by up to ~1 Å — Open3D's
`sample_points_poisson_disk` is unseeded. Consequences: caching a surface *freezes one random draw*
(a behavior change, not a transparent optimization); parallel builds are statistically equivalent,
**not** bit-identical; and the array-reuse SE(3)-copy trick is in fact the *only* reproducible build.

### 3. Mitigations, ranked

1. **Skip surface when unused (`num_surf_points=None`) — already supported, free.** Drops ~90% of the
   build for vol/pharm-only alignment. Exact, but only for scorers that don't read surf/ESP.
2. **Reuse precomputed arrays / SE(3)-copy — already supported, exact.** For rigid copies, matrix-rotate
   the equivariant arrays (`atom_pos`/`surf_pos`/`pharm_ancs`/`pharm_vecs`) and pass them back via
   `surface_points=`/`electrostatics=`/`partial_charges=`/`pharm_*=`. ~1000× cheaper than rebuilding
   (already done in `benchmarks/benchmark.py::_transform`/`rotated`).
3. **Mesh-free surface — ~62×, but subsample choice is the whole ballgame (see §4–5).**
4. **Parallel construction — 3.4× / 5.2× / 7.0× at 4/8/16 workers**, but only with `multiprocessing`
   **fork + `OMP_NUM_THREADS=1`**. Naive spawn + unpinned BLAS/Open3D threads was **0.6× (slower)**.
   Windows is spawn-only; the real heavy pipeline runs in WSL2 (fork available).
5. Skipping the sub-ms steps (charges/ESP/pharm): negligible; not worth it for speed alone.

### 4. Mesh-free accuracy on DISTINCT molecules
14 drugs → 182 ordered distinct pairs, `num_surf_points=200`, `alpha=0.81`, surf + esp. Deltas are vs
the current `mesh` baseline; **`mesh2` = an independent rebuild of the SAME method = the intrinsic
noise floor** a mesh-free change must beat.

| variant | mean \|Δ\| | max \|Δ\| | rank ρ | top-1 match |
|---|--:|--:|--:|--:|
| `mesh2` (same method, rebuilt) — *noise floor* | 0.010 | 0.04 | 0.996 | 14/14 |
| mesh-free, **random** subsample | **0.066** | 0.22 | 0.961 | **8–9/14** |
| mesh-free, **FPS** subsample | **0.010** | 0.05 | 0.997 | **14/14** |

(surf and esp gave nearly identical numbers; scores spanned ~0.23–0.98, not saturated.)

**Random subsampling is genuinely different** (~6× the noise floor, reorders ~⅓ of best matches).
**FPS is not different in any way that matters** — it lands on the noise floor, preserves every
ranking, and is deterministic (so *more* stable than today's unseeded resample).

### 5. The catch: mesh-free leaks atom positions (matters for generative modeling)
The original mesh+poisson surface is **smooth + stochastic** by design, so a network can't trivially
read atom centers off the surface. Mesh-free breaks both properties.

**Leak metric** — per surface point, residual `|dist-to-nearest-atom − that atom's (vdW+probe) radius|`.
~0 ⇒ the point sits exactly on an atom's sphere ⇒ atom center is directly recoverable.

| method | median residual | % within 0.05 Å of a sphere | stochastic? |
|---|--:|--:|:--:|
| mesh+poisson (current) | 0.010 Å | 86–98% | yes (unseeded) |
| mesh-free, random | **0.000 Å** | 98–100% | yes (seedable) |
| mesh-free, FPS | **0.000 Å** | 98–100% | **no (deterministic)** |

Both mesh-free variants are a subset of the on-sphere fibonacci points → every point lies exactly on
an atom sphere → atom positions leak. Mesh+poisson samples the *triangle faces* bridging the sphere
patches, so its points sit ~0.01–0.026 Å off the spheres and the seam crevices are filled by facets —
that's the seam-smoothing the generative pipeline relies on. (Honest caveat from cross-section plots:
all three trace the *same* fundamentally scalloped solvent-accessible envelope; the mesh's anti-leak
advantage is real but modest in shape and concentrated at the concave seams — its bigger
generative-relevant property is the off-sphere jitter + stochasticity, both of which FPS discards.)

**Conclusion: this splits by use case.**
- **Scoring / alignment** (this fork's job, no atom-recovery concern): FPS is ideal — fast, accurate,
  *deterministic is a feature*.
- **Generative modeling** (needs stochasticity + non-leakage): FPS is the wrong tool, and even
  mesh-free-random fails (still on-sphere). Keep the original mesh+poisson there.

### 6. Proposal
Add an **opt-in** surface method, default = original, used only where explicitly requested.

- `get_molecular_surface(..., method="mesh")` default (today's ball-pivoting + Poisson, **bit-identical**),
  plus `method="fps"` (and later `method="metaball"`).
- Thread it through `Molecule(..., surface_method="mesh")` (default).
- Wire the fast path in only at scoring/screening call sites; leave the generative pipeline untouched.
- Pattern mirrors the repo's existing opt-ins (`backend="triton"`, `FSS_MGPU_BACKEND`).

**Possible future smooth-stochastic surfacer (for the generative side, if speed is wanted there):**
a genuinely smooth *implicit* surface — Gaussian "metaball" / SDF iso-surface via marching cubes,
sampled stochastically — or a cheap Laplacian-smoothing pass on the masked points. Both keep
stochasticity, push points off the exact spheres (reduce leak), and may still beat ball-pivoting.
Each needs the §5 leak metric + §4 accuracy check before becoming anything.

### 7. Constraints (hard requirements)
- **Backwards compatible; default = original.** When the opt-in flag isn't set, behavior must be
  **bit-identical** to today. The generative pipeline and all existing results depend on the current
  surface.
- **`ALPHA` calibration.** `score/constants.py` defines `ALPHA = interp1d([50…400], …)` with
  `bounds_error=True` → it **crashes** for point counts outside `[50,400]`, and it's calibrated to the
  even-Poisson distribution. A mesh-free path must keep the count in range and ideally re-confirm
  scores under the calibrated `ALPHA(num_surf_points)`, not just a fixed `alpha`.

### 8. Before anything ships
- Validate FPS scoring on a **larger, more diverse** set (not 14 drugs), with retrospective
  **enrichment / AUROC**, under calibrated `ALPHA`.
- Decide the opt-in surface API (`method=` arg vs env flag) and confirm default path is bit-identical.
- If pursuing the generative speedup, prototype the metaball/marching-cubes (or Laplacian-smoothing)
  surfacer and run the leak metric + accuracy + speed comparison.

### Reproduction notes
- Env: WSL2 `SimModelEnv` (`/home/nsomers/miniconda3/envs/SimModelEnv`), CUDA GPU; the Windows box
  has no Open3D wheel for Python 3.13, so surface work runs in WSL.
- All numbers above came from throwaway scripts run in that env (profiler, distinct-pair A/B, leak
  metric + cross-section renders). They were not committed; the methodology is described inline so
  they can be reconstructed.

---

# Idea 2 — Per-pair early-stop for `esp` distinct-pair alignment

**Status: IDEA / not implemented.** Measured this session (RTX 4050 laptop GPU, WSL2 `SimModelEnv`,
benchmark-faithful isolated cells). The cheap partial win (`FINE_ES_PATIENCE=2`) and the
already-landed seed-plumbing fix are noted at the end; the structural lever (L1, per-pair early-stop)
is written up here with its downsides so the cost/benefit isn't re-derived.

### Problem (measured)
On **distinct** (different-molecule) pairs — the real screening workload — `esp` alignment is the
clear outlier. Self-copy-vs-distinct wall-time penalty (self time ÷ distinct time):

| mode | n=100 | n=1000 | n=10000 |
|---|--:|--:|--:|
| vol | 2.9× | 2.1× | 2.6× |
| surf | 3.3× | 2.3× | 1.13× (→ parity) |
| **esp** | **8.6×** | **12.1×** | **8.2×** |
| pharm | 4.3× | 2.0× | 2.1× |

Absolute distinct throughput at n=10000: vol 16570, surf 5133, pharm 3560, **esp ~1000 pairs/s** —
esp is ~16× slower than vol and ~5× slower than surf/pharm on real pairs. vol/surf/pharm converge
toward ~1–2.6× at scale (their penalty is padding/launch overhead, which amortizes); **esp stays
8–12× at every batch size** because its penalty is structural.

### Root cause (verified in code)
The esp fine loop (`shepherd_score/alignment/utils/fast_esp_se3.py:256-289`) uses a **batch-GLOBAL**
early-stop: `best_score.max().item()` over the flattened `BATCH×P` tensor (`:275`). On a heterogeneous
distinct batch *some* pose is always still improving, so the global plateau test almost never fires →
every pair runs to/near `steps_fine=100`. Self-copies plateau globally and break in ~10–25 steps. Cost
= **steps × poses × the heavy `esp_combo` kernel** (shape Gaussian overlap + the electrostatic term —
~1.3–1.5× heavier per pose-step than `surf`, irreducible).

**esp distinct is STEP-bound, not pose-bound** (measured): reducing `num_seeds` 50→24→16 sped up
**self-copies** but left **distinct flat** (914→882→867 pairs/s, scores unchanged) — fewer poses just
means more steps to converge. Cutting **steps** is what helps.

### Proposal (L1): per-pair early-stop with compaction
Replace the batch-global stop with a **per-pair** stop: each distinct pair freezes on *its own*
plateau (like self-copies do), and every ~10 steps `index_select` the still-active poses into dense
buffers so the kernel grid actually shrinks. **Expected ~2–3.5× on esp distinct; self unchanged.**
Structurally accuracy-safe (per-pair stop is *strictly more conservative* than the global test — it
can only stop an already-plateaued pair, never a still-improving one).

### Downsides / why it's deferred
1. **Implementation & correctness risk (the real cost).** The "done" decision is per-**pair** but the
   compacted tensors are per-**pose** (`BATCH×P` rows). When a pair plateaus, remove all `P` of its
   poses together, snapshot its per-pair argmax, and maintain a `(BATCH×P ↔ compacted)` index map so
   the final gather (`:294-300`) scatters each frozen pair's `best_q/best_t` to the right output slot.
   ~16 tensors sliced in lockstep — including the esp-only charge tensors `CA_k/CB_k` that the vol/surf
   prune block (`fast_se3.py:444-456`) does **not** handle. Not a copy-paste port; the index-map is
   where silent wrong-pose bugs hide (only visible as elevated mean|Δ|).
2. **Tail-bound & overhead-floored → ~2–3.5×, not the naive step ratio.** Each step still runs until
   the *slowest live pair* plateaus; compaction shrinks grid *width*, not *length*, and a few hard
   pairs drag the loop. The mask-only variant (no compaction) is worthless (**<1.2×**: `grid=(K,)`
   launches every CTA regardless). Compaction itself adds ~11 gather launches per event; below some
   active-pose count it's net-negative (needs a stop-compacting threshold + the 1-pair-left edge case).
3. **Largely subsumes — doesn't stack with — the free patience win.** Both attack steps; with a true
   per-pair stop the global patience knob is nearly irrelevant. The *marginal* gain over just shipping
   `FINE_ES_PATIENCE=2` is ~1.5–2.6×, shrinking the ROI on the M–L effort.
4. **Accuracy risk small but esp-specific.** esp landscapes have long, shallow ESP-driven improvements
   (exactly why esp keeps patience 5 while surf/vol use 2, `fast_se3.py:312-316`). A slow climber can
   trip the plateau test and freeze just before a late gain → a minority of borderline pairs settle
   slightly lower. Must re-gate, not assume safe.
5. **Nondeterminism complicates validation.** `esp_combo` is already nondeterministic; compaction
   changes pose grouping/reduction order, so you can't bit-compare — distinguishing a real ~5e-3
   regression from drift needs more pairs/repeats.
6. **Forecloses the CUDA-graph path for esp** (graph capture needs static shapes) and diverges the esp
   loop further from surf/vol (two early-stop designs to maintain).

### Cheaper alternative available now (no code)
`FINE_ES_PATIENCE=2` tightens the *global* stop → **1.33× on esp distinct**, zero code, parity-safe on
the gate (mean|Δ| 0.0123 vs 0.0128 baseline, spearman 0.9884, self stays 1.000). The codebase chose
patience 5 deliberately, so validate a **fuller sweep + the self-copy floor across the pseudo-symmetric
drugs** before changing the default.

### Related (already landed — not future work)
The **seed-plumbing fix** is in the working tree: `fast_optimize_ROCS_esp_overlay_batch` gained a
`num_seeds` param that it now passes through, and `_esp_bucketed_align` wires `_NUM_SEEDS` in + fixes
the hardcoded-`50` sub-batch cache key (`container/_batch_align.py`). esp now respects
`num_repeats`/`FINE_NUM_SEEDS` (**default 50, a no-op at default**), fixing a latent bug (esp silently
ran 50 poses and ignored the knob; the cache key mis-sized the footprint when seeds change). It speeds
the **self-copy headline benchmark** but **not** distinct pairs (step-bound). `FINE_NUM_SEEDS=16` was
parity-safe on the 50-pair gate but needs a pseudo-symmetric/larger sweep before lowering the default.

### Validation procedure (any esp change)
- **Accuracy gate:** self-copy esp ~1.000; distinct upstream **mean|Δ| ≲ 0.014, spearman ≳ 0.98** over
  50 pairs — `python -m benchmarks.benchmark --accuracy --modes esp --n-accuracy 50`.
- **Timing:** measure **distinct** pairs with a distinct-pair harness — the stock speed sweep times
  **self-copies only**, so it won't show a per-pair-stop benefit. Use isolated per-cell subprocesses
  (recovered GPU clock + autotune at batch) + best-of-N; the laptop GPU throttles, so trust *ratios*
  measured back-to-back over absolutes.
- Run in WSL2 `SimModelEnv`. Gate every new lever **OFF by default** behind a `speedlab.py` knob.
- See `SPEED_EXPERIMENTS.md` for the broader speed-lever ledger (incl. the rejected fixed-step-cap
  lever, which is L1's mechanism done bluntly).
