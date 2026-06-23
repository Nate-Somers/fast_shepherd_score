# Handoff: the molecule-construction bottleneck at library scale

> **Audience: a separate agent picking up the "make `Molecule` construction
> scale to a billion-molecule library" workstream.** This is a self-contained
> briefing — the problem, the measured numbers, where the cost lives in the code,
> the known fast paths, and a concrete ask. It is the *populate-the-store* half of
> the streaming design in [`STREAMING_DESIGN.md`](./STREAMING_DESIGN.md). The
> profiling write-up it builds on is `Shepherd-Score-Paper/POTENTIAL_FUTURE_WORK.md`
> **Idea 1** (mesh-free surface) — read that for the deep detail; this doc frames
> it for the billion-scale precompute pipeline and adds the I/O dimension.

## 1. TL;DR

For a billion-molecule screen, **building the interaction profiles — not aligning
them — is the dominant cost**, and it is a *one-time, precompute-once* job whose
output feeds the on-disk `ProfileStore`. Two levers decide whether it is feasible:

1. **Surface generation** dominates `Molecule.__init__`: **~58 ms/molecule, ~87%
   of it a single Open3D `create_from_point_cloud_ball_pivoting` call** (~50.8 ms).
   At 1e9 that is **~670 single-core CPU-days** (~1.8 CPU-years) for surfaces alone.
2. **Charges**: MMFF (the **default**) is ~0.14 ms and negligible; **xTB is opt-in
   and costs seconds–minutes/molecule**, which would make *it* the bottleneck
   (years→millennia of CPU) if a high-quality ESP screen demands it.

Alignment, by contrast, is hours on one GPU for the whole billion (see
`STREAMING_DESIGN.md` §1). So the construction pipeline is the thing to optimize
and parallelize; the aligner is already fast.

## 2. Measured cost breakdown (one `Molecule`, benchmark drug set)

From `POTENTIAL_FUTURE_WORK.md` Idea 1 (WSL `SimModelEnv`, RDKit 2025.03.6 /
Open3D 0.19, `surf_per_atom=3`):

| step (inside `Molecule.__init__`) | cost/build | share |
|---|--:|--:|
| Open3D surface gen (`get_pc`) | ~54 ms | ~90% |
| └─ `create_from_point_cloud_ball_pivoting` | **50.8 ms** | **~87%** |
| `get_pharmacophore` | 1.68 ms | ~3% |
| cdist + fibonacci sampling + vdW radii | ~1.9 ms | ~3% |
| MMFF `get_partial_charges` + ESP | 0.40 ms | <1% |
| **total `Molecule.__init__`** | **~58 ms (mean, 25–107 ms range)** | 100% |

Precursor (separate, before `Molecule`): ETKDG `embed_conformer_from_smiles`
~18 ms mean (up to 59 ms); optional **xTB** single-point/relax **seconds–minutes**.

Scaling that to a billion:

| stage | per-molecule | × 1e9, single-core |
|---|--:|--:|
| ETKDG embed | ~18 ms | ~208 days |
| `Molecule.__init__` (MMFF, w/ surface) | ~58 ms | ~670 days |
| └ if surface skipped (`num_surf_points=None`) | ~4 ms | ~46 days |
| xTB charges (if used) | ~1–60 s | **~31 CPU-yr to millennia** |

These are embarrassingly parallel, so wall-clock = CPU-time / cores. The job is a
cluster batch job, but the per-molecule constant is what decides the cluster bill.

## 3. Where the cost lives in the code

- `shepherd_score/container/_core.py` — `Molecule.__init__` →
  `get_pc()` (surface), `get_electrostatic_potential()` (ESP), `get_pharmacophore()`.
- `shepherd_score/generate_point_cloud.py` — `get_molecular_surface()`: the Open3D
  ball-pivoting mesh + Poisson-disk resample. **This is the 87% line.** The mesh is
  built only to evenly resample surface points, then thrown away.
- `shepherd_score/conformer_generation.py` — `embed_conformer_from_smiles`
  (ETKDG) and `charges_from_single_point_conformer_with_xtb` (the opt-in xTB path).
- `shepherd_score/score/constants.py` — `ALPHA = interp1d([50…400], …)` with
  `bounds_error=True`: any surface-point count outside `[50, 400]` **crashes**, and
  `ALPHA` is calibrated to the even-Poisson distribution. Any new surfacer must
  keep the count in range and re-validate scores under calibrated `ALPHA`.

## 4. Known fast paths (already measured)

Ranked, from `POTENTIAL_FUTURE_WORK.md` Idea 1 §3:

1. **Skip the surface when unused** (`num_surf_points=None`) — already supported,
   exact, drops ~90% of the build. Use it for `vol`/`pharm`-only stores.
2. **Reuse precomputed arrays / SE(3)-copy** — already supported, exact, ~1000×
   cheaper than rebuilding for rigid copies (rotate the equivariant
   `atom_pos`/`surf_pos`/`pharm_*` arrays, pass back via `surface_points=` /
   `electrostatics=` / `partial_charges=` / `pharm_*=`).
3. **Mesh-free surface — ~62× faster.** Mask the fibonacci-sphere samples then
   subsample. With **farthest-point sampling (FPS)** the screening
   scores/rankings are **indistinguishable from the method's own run-to-run noise**
   (mean |Δ| 0.010 = the rebuild noise floor; top-1 14/14; ρ 0.997).
   **Catch:** mesh-free points lie exactly on atom spheres → they **leak atom
   positions**, and FPS is deterministic. That is **fine for scoring/screening**
   (this pipeline's job) but **wrong for the generative-modeling pipeline**, which
   depends on the original surface's off-sphere jitter + stochasticity. So it must
   ship **opt-in, default = original**, wired in only at screening call sites.
4. **Parallel construction** — 3.4×/5.2×/7.0× at 4/8/16 workers, **but only** with
   `multiprocessing` **fork + `OMP_NUM_THREADS=1`**; naive spawn + unpinned
   BLAS/Open3D threads measured **0.6× (slower)**. (Windows is spawn-only; the heavy
   pipeline runs in WSL2/Linux where fork is available.)

## 5. The ask (what this workstream should deliver)

Two coupled deliverables; both must keep existing behavior **bit-identical when
the opt-in flag is off**.

1. **An opt-in fast surfacer.** Add `get_molecular_surface(..., method="mesh")`
   (default, today's ball-piviting, bit-identical) plus `method="fps"`; thread it
   through `Molecule(..., surface_method="mesh")`. Wire the fast path in only at
   scoring/screening call sites; leave the generative pipeline on `mesh`. Pattern
   mirrors the existing `backend="triton"` / `FSS_*` opt-ins. Validate FPS with
   retrospective **enrichment / AUROC** on a **larger, diverse** set (not the 14
   drugs) under calibrated `ALPHA` before changing any default.

2. **A parallel "build-the-store" pipeline** that writes the `ProfileStore` from
   `STREAMING_DESIGN.md`: SMILES/3D-input shards → (ETKDG → optional xTB → profiles)
   → `ProfileStore.add()` → independent shard files, one process per core with
   **fork + pinned BLAS/Open3D threads**, resumable per shard. Make the
   **MMFF-vs-xTB charge decision explicit and configurable** — it is the single
   biggest cost lever (ms vs seconds/molecule); default MMFF, xTB opt-in with a
   clear cost warning. Record `num_surf_points`, `surface_method`, and charge
   source in the store manifest so screens stay reproducible and `ALPHA`-correct.

## 6. Constraints & validation (do not skip)

- **Backwards compatible; default = original.** The generative pipeline and all
  existing results depend on the current ball-pivoting+Poisson surface.
- **`ALPHA` range `[50, 400]`**, calibrated to even-Poisson; re-confirm scores
  under `ALPHA(num_surf_points)`, not a fixed `alpha`.
- **Mesh-free leaks atom positions + is deterministic** → screening-only; never
  the generative pipeline. Gate behind the opt-in.
- **Parallelism only helps with fork + `OMP_NUM_THREADS=1`** (and Open3D thread
  pinning); spawn + unpinned threads is a measured regression.
- Validate any surfacer change with the leak metric (§5 of Idea 1) + accuracy
  (mean |Δ| vs the rebuild noise floor) + enrichment on a diverse set.
