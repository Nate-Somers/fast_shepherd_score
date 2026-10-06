# Guide to the accelerated API

This checkout extends the original ShEPhERD scoring and alignment interface with
CPU/GPU batch alignment, additional scoring objectives, reusable disk stores and
parallel screening. It still imports as `shepherd_score`. Install this checkout;
an older PyPI package or the upstream documentation may not expose these APIs.

## What the accelerated implementation adds

| Capability | Interface | How it works |
|---|---|---|
| Batched rigid alignment | `MoleculePairBatch.align_with_<mode>` | Groups pairs by feature size and optimizes multiple starting poses with Numba CPU or Triton CUDA kernels |
| Additional objectives | Mode-specific alignment methods | Combines shape with typed features or scalar fields; supports asymmetric Tversky variants |
| Reusable library profiles | `ProfileStore.create`, `add`, `close`, `open` | Saves the channels needed by the selected modes in disk shards |
| Single-query screening | `screen` | Streams prepared shards and returns ranked hits |
| Multi-query screening | `screen_many` | Reads each shard once and reuses it across a query panel |
| Multiple GPUs | `screen(..., ndev=...)`, `align_multi_gpu`, `MultiGPUAligner` | Divides screening or in-memory pair alignment across devices |
| CPU worker processes | `screen_parallel`, `screen_parallel_close` | Screens an in-memory library using a persistent pool of forked workers |

The [usage guide](usage.rst) gives a short introduction. The
[API reference](api/index.rst) supplies signatures and parameter descriptions;
[scoring theory](theory.md) describes the objectives. This page connects the
public entry points rather than documenting internal kernel functions.

## Prepare molecules once

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

def prepare(smiles):
    rd = embed_conformer_from_smiles(smiles, MMFF_optimize=True)
    return Molecule(rd, num_surf_points=200, pharm_multi_vector=False,
                    charge_model="mmff")

ref = prepare("CCOc1ccccc1")
fit = prepare("CCNc1ccccc1")
```

`Molecule` holds a conformer and its features. Leave `num_surf_points=None` when
surfaces are unnecessary. Pharmacophores require `pharm_multi_vector` to be
specified or explicit feature arrays. Charges are lazy; requesting surface ESP
also requests charges. The default charge model is xTB, with a warning and MMFF
fallback when xTB is unavailable or fails. Select a model explicitly for comparisons.
The default xTB charge call uses total charge zero; supply explicit charges
when a different total charge is required. MMFF94 follows molecular formal
charge. Fukui fields use the formal charge and its adjacent charge states,
require xTB, and do not fall back to MMFF94.

MMFF-relaxed conformers are sanitized to restore aromaticity. Surface sampling
is stochastic: save and reuse its coordinates and ESP for reproducible
comparisons. A conformer seed alone does not freeze its surface features.
See [Molecule](api/container/molecule.rst) for feature arguments and
[profile containers](api/container/profiles.rst) for structured surfaces,
pharmacophores and surface diagnostics.

## Align pairs and choose an objective

```python
pair = MoleculePair(ref, fit)
scores, aligned = MoleculePairBatch([pair]).align_with_vol(
    backend="numba", return_aligned=True)
```

Use `backend="triton"` for CUDA batches. Alignment writes mode-specific score
and transform attributes on each pair; `return_aligned` controls whether the
transformed arrays are returned as well. For parameter and return details, see
[MoleculePairBatch](api/container/molecule_pair_batch.rst).

| Mode family | Representation / objective |
|---|---|
| `vol`, `surf` | Gaussian overlap of heavy-atom or surface point clouds |
| `vol_esp`, `surf_esp` | Overlap weighted by atomic charges or surface potentials |
| `pharm` | Typed, directional pharmacophore overlap |
| `vol_color` | Shape plus directionless pharmacophore color |
| `vol_lipo` | Shape plus lipophilicity features |
| `vol_mr` | Shape plus per-atom Crippen molar refractivity |
| `vol_fukui` | Shape plus condensed Fukui reactivity field |
| `vol_atomtype` | Shape plus element-type features |
| `vol_pharm` | Shape plus directional pharmacophore features |
| `vol_and_surf_esp` | Shape plus masked surface-potential agreement |
| `vol_avoid` | Shape minus an excluded-volume penalty; requires `avoid_points` |

Tversky variants are available for `vol`, `vol_esp`, `surf`, `surf_esp`,
`vol_color`, `vol_lipo`, `pharm` and `vol_and_surf_esp`, with the
`_tversky` suffix. They change normalization, not the input representation. The default Tversky
weights are 0.95 on the fixed reference self-overlap and 0.05 on the fitted
molecule self-overlap. Pharmacophore Tversky is clamped at one; other
Gaussian-overlap Tversky ratios can exceed one.
For `vol_and_surf_esp_tversky`, the change applies to the shape term.
`align_with_esp` and `align_with_esp_combo` remain compatibility aliases for
`surf_esp` and `vol_and_surf_esp`.

`vol_color` ignores direction vectors, whereas `vol_pharm` and `pharm` use them.
These modes read the pharmacophores already present on each molecule; choosing
an alignment mode does not regenerate features or switch SMARTS definitions.
`Molecule` defaults to `feature_set="shepherd"`; `feature_set="rdkit_base"` is
an explicit alternative during preparation.

`vol_and_surf_esp` and `vol_and_surf_esp_tversky` differentiate both the shape
term and the ESP agreement term, so the optimizer follows the gradient of the
combined score it reports.
Mode defaults, channel requirements and optimization budgets are defined in
[`accel/_modes.py`](../shepherd_score/accel/_modes.py). A step budget is a ceiling,
not a fixed amount of work. Where early stopping is enabled, the CPU, GPU eager
and GPU CUDA-graph loops use the same patience and check for improvement every
five steps. Each internal optimization sub-batch stops only after all its pairs fail to
improve beyond the tolerance for the required consecutive checks. The CUDA-graph paths
for `vol_and_surf_esp` and `vol_and_surf_esp_tversky` always run their full step
budget. CPU/GPU rounding and batch composition can still affect the stopping
point and the selected pose.

## Build a library and screen it

```python
from shepherd_score.screen import ProfileStore, screen, screen_many

store = ProfileStore.create("example-library.fss", num_surf_points=200,
                            modes=["vol", "surf_esp"])
store.add(fit, id="compound-1-conformer-1")
store.close()

library = ProfileStore.open("example-library.fss")
hits = screen(ref, library, mode="vol", backend="numba", top_k=1)
panel_hits = screen_many([ref, fit], library, mode="vol",
                         backend="numba", top_k=1)
for hit in hits:
    print(hit.id, hit.score, hit.transform)
```

Declare every intended mode when creating a store: it retains only the required
channels. Surface queries and stores must have consistent point counts.
Canonical stores rotate supported atom-based profiles into principal frames;
returned transforms are composed back into the input coordinate frame.
Hits are conformer-level results. Compound-level enrichment requires aggregation
over conformers; the paper benchmark uses the maximum score.

`screen_many` returns one ranked hit list per query. `scores_out` can retain
all scores in caller-provided arrays on the single-process path. See
[screening](api/screening.rst) for its shape requirements and supported options.

## Parallel execution

For disk-backed GPU screening, pass `backend="triton", ndev=2` to `screen` or
`screen_many` on a supported fast path. Workers start on the first call and are
reused. `close_multigpu_pool()` releases them. For in-memory pairs,
`shepherd_score.accel.multi_gpu.align_multi_gpu` handles a single batch;
`MultiGPUAligner` keeps workers for repeated alignment.

`shepherd_score.accel.screen_parallel.screen_parallel` is a different interface:
it takes a query and a RAM-resident molecule list, plus a mode and worker count.
It uses forked CPU workers and retains them for subsequent calls against that
library. Call `screen_parallel_close()` when finished. Startup belongs outside
a steady-state throughput measurement but inside first-call latency.
Use a script entry-point guard (`if __name__ == "__main__":`) for spawned GPU
workers; the fork-based CPU interface requires a platform supporting fork.

## Further reference

- [Installation](installation.rst): optional backends and external executables.
- [Batch alignment](api/container/molecule_pair_batch.rst): backend selection and methods.
- [Screening and parallel APIs](api/screening.rst): stores, hits, queries and workers.
- [Analytical gradients](api/alignment/analytical_gradients.rst): lower-level gradient interface.
- [Scoring theory](theory.md): formulas and interpretation of the representations.
- [Measured throughput](performance/timings.md): performance measurements and qualifications.
- [Paper benchmark procedures](https://github.com/Nate-Somers/Shepherd-Score-Paper/blob/main/benchmarks/README.md): dataset selection, timing boundaries and metrics.

The existing tutorial notebooks also cover the original preparation and scoring
workflows; they are not a complete walkthrough of the accelerated API.

## Reproducible GPU launch configurations

Triton normally selects kernel configurations by timing candidates and caching
its choices. Those choices can vary between installations and affect throughput.
To record the choices used by a representative warm-up:

```python
from shepherd_score.accel.kernels.tuning import export_configurations
# Run the intended workloads once, outside timing.
export_configurations("triton-configurations.json")
```

Start subsequent processes with `FSS_TRITON_CONFIGS=/absolute/path/triton-configurations.json`
set before importing the kernels. They reuse the recorded configurations,
including in spawned screening workers. A missing kernel/shape entry or a
GPU-model, architecture, or Triton-version mismatch raises an error rather than
silently tuning. Cover every workload shape before freezing the profile; the
export includes only configurations used in the current process. Profiles from
different runs must agree on overlapping entries before they are combined.

This freezes launch choices, not wall-clock performance, and does not establish
that the recorded choices are optimal. Keep the profile alongside the engine
commit, software versions, inputs, and timing results. The default remains
ordinary Triton autotuning when the environment variable is unset.

For multi-GPU screening, `screen(..., ndev=4, worker_threads=1)` fixes host compute threads per worker. The default divides the allocated host CPUs among GPUs. Read-ahead threads are separate from this compute-thread limit.
