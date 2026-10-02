# ShEPhERD-score

Molecular conformer preparation, 3D similarity scoring, rigid alignment, and
virtual screening with PyTorch, Numba CPU, and Triton GPU implementations.
Scoring includes shape, electrostatics, pharmacophores, and composite objectives.

## What this checkout adds

The accelerated implementation adds batched rigid alignment on Numba CPUs and
Triton GPUs, feature-based and Tversky scoring modes, disk-backed library stores,
multi-query screening, and CPU-worker and multi-GPU execution. Stores reuse
prepared features across queries; batch kernels optimize multiple starting poses
without requiring a separate Python alignment call for each pair.

Combined volume-and-surface ESP alignment optimizes both terms of its reported
score. CPU and GPU use the same early-stopping patience where enabled; the
combined-ESP CUDA-graph paths run the full step budget. See the API guide for
mode-specific settings and CPU/GPU numerical differences.

Start with the [accelerated API guide](docs/accelerated_api.md) for a walkthrough
of preparation, mode selection, batch alignment, stores, screening and parallel
execution. The [usage guide](docs/usage.rst) is a shorter introduction;
[scoring theory](docs/theory.md) explains the objectives, and the
[API reference](docs/api/index.rst) documents signatures and parameters.
The API guide includes a mode table and explains feature requirements and
CPU/GPU differences.

## Installation

Install this checkout to use the accelerated implementations described here:

```bash
python -m pip install -e .
python -m pip install -e ".[gpu]"   # optional; CUDA PyTorch and Triton
```

The package imports as `shepherd_score`. Python and dependency requirements are
declared in [pyproject.toml](pyproject.toml). Extras: `jax`, `docking`, `dev`, and
`docs`. Numba is a core dependency. Do not assume an older PyPI release contains
the features in this checkout.

For the CPU benchmark environment, use [environment.yml](environment.yml).
The published CPU rates use Intel SVML; check it with
`python -c "import numba.core.config as c; print(c.USING_SVML)"`.
Other Numba builds are supported but may be slower and differ within numerical
tolerances. xTB is required for xTB charges and Fukui descriptors; MMFF94 charges
are available without xTB via `charge_model="mmff"`. The default xTB charge
path falls back to MMFF94 with a warning if xTB is unavailable. GPU kernels require a supported CUDA/Triton system.

## Align conformers

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

ref = Molecule(embed_conformer_from_smiles("CCOc1ccccc1", MMFF_optimize=True))
fit = Molecule(embed_conformer_from_smiles("CCNc1ccccc1", MMFF_optimize=True))
pair = MoleculePair(ref, fit)
scores, _ = MoleculePairBatch([pair]).align_with_vol(backend="numba")
print(scores)
```

MMFF-relaxed conformers are sanitized before being returned, restoring RDKit
aromaticity for pharmacophore and Crippen feature assignment.

Use `backend="triton"` for GPU batches. Each mode has an `align_with_<mode>`
method. Mode-specific features must be prepared before alignment: surface modes
need surface points, pharmacophore modes need pharmacophores, and field modes
need their corresponding atomic descriptors. See [usage](docs/usage.rst),
[scoring theory](docs/theory.md), and [API reference](docs/api/index.rst).

## Screen a library

```python
from shepherd_score.screen import ProfileStore, screen

store = ProfileStore.create("library.fss", num_surf_points=0, modes=["vol"])
store.add(fit, id="compound-1")
store.close()
hits = screen(ref, ProfileStore.open("library.fss"), mode="vol", backend="numba", top_k=1)
print(hits)
```

Stores cache prepared conformers on disk. Select all required modes at creation;
the store retains only the channels those modes need. Atom-seeded modes use
canonical frames by default. See [screening](docs/api/screening.rst) for batched
queries, CPU workers, and multiple GPUs.

## Evaluations and documentation

- [Generated-molecule evaluations](scripts/README.md)
- [Tutorial notebooks](examples/)
- [Measured throughput](docs/performance/timings.md)
- [Paper benchmark scripts and results](https://github.com/Nate-Somers/Shepherd-Score-Paper)

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest tests/ -m "not jax"
python -m sphinx -b html docs docs/_build/html
```

## Citation and license

The original representations and generative-model evaluations are described in
Adams et al., *ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for
bioisosteric drug design*, ICLR 2025. The updated alignment benchmarks and
manuscript are in the paper repository linked above.
This package is distributed under the [MIT license](LICENSE).

For reproducible GPU measurements, [record and replay Triton launch configurations](docs/accelerated_api.md#reproducible-gpu-launch-configurations).
