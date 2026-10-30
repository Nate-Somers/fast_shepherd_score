# ShEPhERD-score

Conformer preparation, 3D similarity scoring, rigid alignment and virtual
screening for molecules. Scores cover shape, electrostatics, pharmacophores and
blends of shape with other atomic features. Each objective has a PyTorch
reference implementation, and batched alignment and screening run on Numba
(CPU) and Triton (GPU) kernels.

Beyond the original ShEPhERD scoring code, this version adds:

- batched rigid alignment of many pairs at once (`MoleculePairBatch`)
- additional objectives (`vol_color`, `vol_lipo`, `vol_mr`, `vol_fukui`,
  `vol_atomtype`, `vol_pharm`, `vol_avoid`) and a `similarity=` argument that
  selects Tanimoto or Tversky normalization for any overlap mode
- disk-backed library stores and single- or multi-query screening
- multi-GPU and multi-process CPU execution

These features are not in older PyPI releases; install from this repository.

## Installation

```bash
python -m pip install -e .
python -m pip install -e ".[gpu]"   # optional: CUDA PyTorch and Triton
```

The package imports as `shepherd_score`. Other extras are `jax`, `docking`,
`dev` and `docs`. See [installation](docs/installation.rst) for the CPU
benchmark environment and external executables (xTB, Vina, ChemAxon).

## Align a pair

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

ref = Molecule(embed_conformer_from_smiles("CCOc1ccccc1", MMFF_optimize=True))
fit = Molecule(embed_conformer_from_smiles("CCNc1ccccc1", MMFF_optimize=True))
pair = MoleculePair(ref, fit)

result = pair.align("vol", similarity="tversky")               # PyTorch, one pair
scores, _ = MoleculePairBatch([pair]).align_with_vol(backend="numba")  # batched
```

`MoleculePair.align(mode, **params)` runs the PyTorch reference optimizer for a
named mode and returns an `AlignmentResult`. `MoleculePairBatch` runs the same
modes on the Numba (`backend="numba"`) or Triton (`backend="triton"`) kernels.
Surface modes need `num_surf_points`, pharmacophore modes need
`pharm_multi_vector`, and field modes need their atomic descriptors on each
`Molecule`.

## Screen a library

```python
from shepherd_score.screen import ProfileStore, screen

with ProfileStore.create("library.fss", num_surf_points=0, modes=["vol"]) as store:
    store.add(fit, id="compound-1")
hits = screen(ref, ProfileStore.open("library.fss"), mode="vol", top_k=1)
```

A store saves only the arrays needed by the modes it was created for. See
[screening](docs/api/screening.rst) for multiple queries, all-score output and
multiple GPUs.

## Documentation

- [Accelerated API guide](docs/accelerated_api.md): preparation, modes,
  batched alignment, stores, screening and parallel execution
- [Usage](docs/usage.rst): a shorter introduction
- [Scoring theory](docs/theory.md): objectives and analytic gradients
- [API reference](docs/api/index.rst)
- [Measured throughput](docs/performance/timings.md)
- [Scoring-mode development skills](docs/scoring_mode_skills.md)
- [Generated-molecule evaluations](scripts/README.md)
- [Tutorial notebooks](examples/)
- [Paper benchmark scripts and results](https://github.com/Nate-Somers/Shepherd-Score-Paper)

Tests and documentation build:

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest tests/ -m "not jax"
python -m sphinx -b html docs docs/_build/html
```

## Citation and license

The representations and generative-model evaluations are described in Adams et
al., *ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for
bioisosteric drug design*, ICLR 2025. The alignment benchmarks for this version
are in the paper repository linked above. Distributed under the
[MIT license](LICENSE).
