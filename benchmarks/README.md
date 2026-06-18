# Headline benchmark

One script, a few flags. It measures **how fast** and **how accurately** this
fork aligns real drug molecules, versus the original upstream repo.

```bash
python -m benchmarks.headline                 # the headline: fork vs original, all modes
python -m benchmarks.headline --no-original    # fork only (fast iteration)
python -m benchmarks.headline --cap 30         # allow slower / bigger cells
python -m benchmarks.headline --modes surf esp # a subset of modes
python -m benchmarks.headline --accuracy       # the accuracy branch (off by default)
```

Run it in the GPU conda env (needs CUDA + Triton for the fork; JAX/Open3D/RDKit
for building molecules and the original path). On this machine that is the
`SimModelEnv` conda env under WSL2.

## What it does

Every pair is a **real drug molecule aligned to a rigid SE(3) copy of itself**,
so the perfect score is exactly **1.0** — accuracy is just "how close to 1.0 did
we get". Molecules are real marketed drugs (RDKit ETKDG conformers, Open3D
surfaces, MMFF charges, RDKit pharmacophores); see `real_workloads.py`.

**Modes** (all run by default): `vol` (atom-cloud ROCS), `surf` (surface ROCS),
`esp` (surface shape + electrostatics), `pharm` (pharmacophore).

**Size sweep** (default `1 10 100 1000 10000 100000` pairs/call): for each
(mode, bucket, engine) line the sweep stops at the first size whose wall time
exceeds the **cap** (default 10 s) — larger sizes are not run. Raise it with
`--cap`.

**Buckets** (both run by default): the GPU batch path buckets pairs by size.
* `same`  — molecules in one size band → a single padded bucket (best case).
* `cross` — molecules spread across bands → many buckets (realistic case).

**Original repo** (`shepherd-score-original-repo/`) runs by default in an
**isolated subprocess** — both packages are named `shepherd_score`, so they
can't share one interpreter. It uses the original's in-process batch path
(`MoleculePairBatch.align_with_*`, `num_workers=1`). Disable with
`--no-original`.

## Outputs

Written to `--out-dir` (default `benchmarks/`):
* `speed_table.md` — pairs/s per cell + fork-over-original speedup, per size.
* `speed_plot.png` — two-panel throughput-vs-size plot (original | fork).
* `plot_data.json` — the raw numbers.

A self-accuracy summary prints at the end (self-copy optimum is 1.0; a value
well below 1.0 is a real quality bug).

## The accuracy branch (`--accuracy`, off by default)

Aligns 50 pairs of **different** molecules (optimum < 1.0) across every mode and
compares the fork's per-pair scores to the original's: `mean|Δ|` and `spearman`
rank correlation. This is the check that the speed numbers aren't hiding a
quality regression on non-trivial alignments. Tune the count with `--n-accuracy`.

## Files

```
benchmarks/
  headline.py        # the whole suite (driver, fork timer, original subprocess,
                     #   accuracy branch, plot + table renderers)
  real_workloads.py  # real-drug molecule builder + same/cross bucket cohorts
```
