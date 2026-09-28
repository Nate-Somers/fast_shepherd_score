# ShEPhERD-score

Molecular conformer preparation, 3D similarity scoring, rigid alignment, and
virtual screening with PyTorch, Numba CPU, and Triton GPU implementations.
Scoring includes shape, electrostatics, pharmacophores, and composite objectives.

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
are available without xTB. GPU kernels require a supported CUDA/Triton system.

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
