# Figure 5 — head-to-head vs ROSHAMBO2 (the missing benchmark)

**Claim defended:** on identical molecules and identical hardware,
`fast_shepherd_score` is competitive with / faster than ROSHAMBO2 — the closest
open-source GPU comparator (Gaussian shape + pharmacophore "color", GPL-3.0).
**No such apples-to-apples benchmark exists in the literature** (ROSHAMBO2's
headline ">200×" is vs its own v1, not vs an external tool), so this is a genuine
contribution.

## Status: ready-to-run, NOT executed here
ROSHAMBO wraps **PAPER's CUDA kernels** and must be compiled with the CUDA
**toolkit**. This machine's env (`SimModelEnv`) has the CUDA *runtime* (torch+cu124)
but **no `nvcc` and no `cmake`**, and the only GPU here is a 6 GB RTX 4050 laptop —
unsuitable for a fair "fast GPU" throughput claim. So we provide a complete,
documented harness + build script rather than a number measured on the wrong setup.

Run on a CUDA-toolkit machine (ideally a datacenter GPU, to match the Fig 2 L40S/
H100 numbers):
```bash
bash paper/fig5_roshambo_headtohead/setup.sh SimModelEnv   # installs nvcc+cmake, builds roshambo
PYTHONPATH=. python paper/fig5_roshambo_headtohead/run.py  # writes results.json
```

## Design (fair)
- **Same molecules:** the repo's curated drug set (or `--sdf`), each as a rigid
  SE(3) self-copy (known optimum = perfect overlap), written to `query.sdf` +
  `dataset.sdf`.
- **Same hardware, same library**, both timed end-to-end:
  - fss: `align_with_surf(backend="triton")`, batched → pairs/s + recovered self-sim.
  - ROSHAMBO: `get_similarity_scores(...)` on the same SDFs → mols/s + ShapeTanimoto.
- Compare **throughput** (the headline) and **recovered self-overlap** (both should
  recover ~1.0 on self-copies — a quality check).
- Shape mode is the apples-to-apples axis (both do Gaussian shape). fss additionally
  offers ESP overlay, which ROSHAMBO does not (that's the capability gap, Fig 3/4).

## Fairness notes to honor
- Match `num_repeats`/optimization effort as closely as the two APIs allow, and
  report it — fss's multi-start gradient optimization and ROSHAMBO's optimizer are
  not identical, so report both throughput **and** recovered overlap quality.
- Verify the `roshambo.api` call signature against the installed version (it has
  changed across releases); the harness marks the call site.
- Run on the SAME GPU you cite for fss (re-run `benchmarks/benchmark.py` there too).

## Verify the ROSHAMBO API
After building, check the real signature:
`python -c "import roshambo.api, inspect; print(inspect.signature(roshambo.api.get_similarity_scores))"`
and adjust `run.py` accordingly.
