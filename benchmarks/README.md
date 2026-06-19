# Benchmark

One script, a few flags. It measures **how fast** and **how accurately** this
fork aligns real drug molecules, versus the original upstream repo.

```bash
python -m benchmarks.benchmark                  # the headline: fork vs original, all modes
python -m benchmarks.benchmark --no-original    # fork only (keeps last run's original line)
python -m benchmarks.benchmark --cap 30         # let the original run slower / bigger cells
python -m benchmarks.benchmark --modes surf esp # a subset of modes
python -m benchmarks.benchmark --accuracy       # the accuracy branch (off by default)
python -m benchmarks.benchmark --tag triton-v2  # write a NEW result set to results/triton-v2/
python -m benchmarks.benchmark --replot         # re-render plot/table from results/, no compute
python -m benchmarks.benchmark --replot --tag triton-v2   # ...or re-render a specific set
```

Run it in the GPU conda env (needs CUDA + Triton for the fork; JAX/Open3D/RDKit
for building molecules and the original path). On this machine that is the
`SimModelEnv` conda env under WSL2.

## What it does

Every pair is a **real drug molecule aligned to a rigid SE(3) copy of itself**,
so the perfect score is exactly **1.0** — accuracy is just "how close to 1.0 did
we get". Molecules are real marketed drugs (RDKit ETKDG conformers, Open3D
surfaces, MMFF charges, RDKit pharmacophores); the builder lives in the same
script.

Both engines go through the **same public API** — `MoleculePairBatch.align_with_*`:
the fork with `backend="triton"` (its Triton/CUDA kernels), the original with its
default `backend="jax"` (JAX/XLA on CPU). One code path, two backends.

**Modes** (all run by default): `vol` (atom-cloud ROCS), `surf` (surface ROCS),
`esp` (surface shape + electrostatics), `pharm` (pharmacophore).

**Size sweep** (default `1 10 100 1000 10000 100000` pairs/call): the **fork runs
the whole sweep**, so the 100k datapoint is always on the fork panel. The
**original** is far slower, so its line stops at the first size whose wall time
exceeds the **cap** (default 10 s) — 100k would take hours. Raise it with `--cap`.

**Buckets** (both run by default): the GPU batch path buckets pairs by size.
* `same`  — molecules in one size band → a single padded bucket (best case).
* `cross` — molecules spread across bands → many buckets (realistic case).

**Original repo** (`shepherd-score-original-repo/`) runs in an **isolated
subprocess** — both packages are named `shepherd_score`, so they can't share one
interpreter. Disable with `--no-original` (which keeps the previous run's original
numbers so both panels still render).

Each `(mode, bucket, size)` fork cell runs in its **own fresh subprocess**,
best-of-N time-budgeted — a recovered GPU clock per cell (this laptop GPU
throttles 2–3× under sustained load) and a kernel autotuned at that cell's batch.

## Outputs

Written to `--out-dir` (default `benchmarks/results/`):
* `speed_plot.png` — two-panel throughput-vs-size plot (original | fork),
  **annotated with the GPU and CPU it ran on** (per-panel + a footer line).
* `speed_table.md` — pairs/s per cell + fork-over-original speedup, per size.
* `plot_data.json` — the raw numbers plus a `_meta` block (hardware, timestamp,
  config) used by `--replot`.

### Keeping multiple result sets

By default each run overwrites `results/`. To keep a run without overwriting the
current one, give it a `--tag`: the outputs go to `results/<tag>/` (the tag is
also stamped onto the plot footer, table, and `plot_data.json`, so sets are
self-labelling). Re-render any set later with `--replot --tag <tag>`. For a path
outside `results/`, use `--out-dir <path>` (it overrides `--tag`).

```bash
python -m benchmarks.benchmark --tag baseline      # -> results/baseline/
python -m benchmarks.benchmark --tag triton-v2     # -> results/triton-v2/  (baseline untouched)
python -m benchmarks.benchmark --replot --tag baseline   # re-render results/baseline/
```

A self-accuracy summary prints at the end (self-copy optimum is 1.0; a value
well below 1.0 is a real quality bug).

## The accuracy branch (`--accuracy`, off by default)

Aligns 50 pairs of **different** molecules (optimum < 1.0) across every mode and
compares the fork's per-pair scores to the original's: `mean|Δ|` and `spearman`
rank correlation. This is the check that the speed numbers aren't hiding a
quality regression on non-trivial alignments. Tune the count with `--n-accuracy`.

## Layout

```
benchmarks/
  benchmark.py        # the whole suite — molecule builder, fork timer, original
                      #   subprocess, accuracy branch, replot, plot + table renderers
  results/            # generated artifacts (gitignored): speed_plot.png, speed_table.md, plot_data.json
  molcache/           # persisted base molecules for the original path (gitignored)
  experiments/        # tuning tools (see ../SPEED_EXPERIMENTS.md), NOT the headline benchmark
    speedlab.py       #   in-process paired A/B lab
    kernelbench.py    #   kernel-level microbench
    parity_scores.py  #   deterministic bit-exact score gate
```
