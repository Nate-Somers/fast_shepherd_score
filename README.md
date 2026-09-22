# *ShEPhERD* Scoring Functions

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/shepherd-score.svg)](https://pypi.org/project/shepherd-score/)
[![Python versions](https://img.shields.io/pypi/pyversions/shepherd-score.svg)](https://pypi.org/project/shepherd-score/)
[![Documentation Status](https://readthedocs.org/projects/shepherd-score/badge/?version=latest)](https://shepherd-score.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/coleygroup/shepherd-score)


📄 **[Paper](https://arxiv.org/abs/2411.04130)** | 📚 **[Documentation](https://shepherd-score.readthedocs.io/en/latest/)** | 📦 **[PyPI](https://pypi.org/project/shepherd-score/)**

</div>


This repository contains the code for **generating/optimizing conformers**, **extracting interaction profiles**, **aligning interaction profiles**, and **differentiably scoring 3D similarity**. It also contains modules to evaluate conformers generated with *ShEPhERD*<sup>1</sup> and other generative models.

The formulation of the interaction profile representation, scoring, alignment, and evaluations are found in our preprint [*ShEPhERD*: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://arxiv.org/abs/2411.04130). The diffusion model itself is found in a *separate* repository: [https://github.com/coleygroup/shepherd](https://github.com/coleygroup/shepherd).


<p align="center">
  <img width="200" src="./logo.svg">
</p>

<sub><sup>1</sup> *ShEPhERD*: **S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion</sub>

## Table of Contents
1. [Documentation](#documentation)
2. [File Structure](#file-structure)
3. [Installation](#installation)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Scoring and Alignment Examples](#scoring-and-alignment-examples)
7. [Evaluation Examples and Scripts](#evaluation-examples-and-scripts)
8. [Data](#data)
9. [License](#license)
10. [Citation](#citation)

## Documentation

Full documentation is available at [shepherd-score.readthedocs.io](https://shepherd-score.readthedocs.io/en/latest/).


## File Structure
```
.
├── shepherd_score/
│   ├── accel/                            # Batched alignment kernels (Triton GPU, numba CPU)
│   │   ├── kernels/                      # Triton kernels and their numba CPU mirrors
│   │   ├── drivers/                      # Per-mode coarse-to-fine SE(3) optimizers
│   │   ├── batch/                        # Padding, bucketing, and batched dispatch
│   │   ├── _modes.py                     # Registry describing every alignment mode as data
│   │   ├── channels.py                   # Per-molecule inputs the modes read
│   │   ├── cpu_pool.py                   # Worker pool for the CPU backend
│   │   ├── multi_gpu.py                  # Data-parallel alignment across several GPUs
│   │   └── screen_parallel.py            # Shard-parallel CPU screening
│   ├── alignment/                        # Alignment package with PyTorch, JAX, and utilities
│   │   ├── utils/                        # SE(3) and PCA utilities (torch, numpy, jax)
│   │   │   ├── se3*.py                   # SE(3) transformations (torch, numpy, jax)
│   │   │   └── pca*.py                   # Principal component alignment (torch, numpy, jax)
│   │   ├── _torch.py                     # PyTorch alignment algorithms (autograd)
│   │   ├── _torch_analytical.py          # PyTorch alignment algorithms (analytical gradients)
│   │   ├── _jax.py                       # JAX alignment algorithms
│   │   └── _jax_parallel.py              # JAX parallel alignment algorithms
│   ├── alignment_jax.py                  # Backwards compatibility shim
│   ├── evaluations/                      # Evaluation suite
│   │   ├── utils/                        # Converting data types and others
│   │   ├── docking/
│   │   │   ├── pdbs/                     # PDBQT files used in *ShEPhERD* manuscript
│   │   │   ├── docking.py                # Docking classes
│   │   │   └── pipelines.py              # Docking evaluation pipelines
│   │   └── evaluate/                     # Generated conformer evaluation pipelines
│   │       ├── evals.py                  # Individual evaluation classes
│   │       └── pipelines.py              # Evaluation pipeline classes
│   ├── pharm_utils/                      # Pharmacophore definitions
│   ├── protonation/                      # Functions for protonation
│   ├── score/                            # Scoring related functions and constants
│   │   ├── analytical_gradients/         # Analytical gradient implementations
│   │   │   └── _torch.py                 # PyTorch analytical gradients for shape, ESP, pharmacophore
│   │   ├── atomtype_scoring.py           # Atom-identity (element-matched) overlap scoring
│   │   ├── constants.py
│   │   ├── electrostatic_scoring.py
│   │   ├── gaussian_overlap.py
│   │   └── pharmacophore_scoring.py
│   ├── conformer_generation.py           # RDKit and xtb related functions for conformers
│   ├── container/                        # Molecule, MoleculePair, MoleculePairBatch classes
│   │   └── profiles.py                   # Surface and Pharmacophore containers
│   ├── extract_profiles.py               # Functions to extract interaction profiles
│   ├── generate_point_cloud.py
│   ├── objective.py                      # Objective function used for REINVENT
│   ├── screen.py                         # ProfileStore and streaming virtual screening
│   ├── surface_diagnostics.py            # Quality checks for generated surface point clouds
│   └── visualize.py                      # Visualization tools
├── scripts/                              # Scripts for running evaluations
├── examples/                             # Jupyter notebook tutorials/examples
├── docs/                                 # Sphinx documentation sources
├── tests/
└── README.md
```


## Installation

### Via PyPI
```bash
pip install shepherd-score
```

#### Install xTB
xTB will need to be installed manually since there are no PyPi bindings. This can be done in a conda environment, but since this approach has been reported to lead to conflicts, we suggest installing 
from [source](https://xtb-docs.readthedocs.io/en/latest/setup.html) and adding it to `PATH`.
### With optional dependencies
```bash
# GPU kernels (Triton)
pip install "shepherd-score[gpu]"

# JAX implementations of scoring and alignment
pip install "shepherd-score[jax]"

# Include docking evaluation tools
pip install "shepherd-score[docking]"

# Everything
pip install "shepherd-score[all]"
```
The `gpu` extra installs `triton>=3.6` and needs `torch>=2.6` built against CUDA. The CPU
kernels need no extra: `numba` is a core dependency.


### Fast CPU alignment (numba kernels)
The CPU backend (`MoleculePairBatch.align_with_*(backend="numba")` and
`screen(..., backend="numba")`) reaches full speed only with an SVML-enabled numba build, which
lets numba emit Intel SVML vector `exp` calls in the overlap kernels. That build is
`numba<=0.59` together with `icc_rt`; the conda [`environment.yml`](environment.yml) provides it.
Without SVML the kernels give the same results, run slower, and emit one `RuntimeWarning` the
first time they are used, so the slower regime is never silent. Check which build is active with:

```bash
python -c "import numba.core.config as c; print(c.USING_SVML)"
```

Measured CPU and GPU throughput per mode is in
[`docs/performance/timings.md`](docs/performance/timings.md).


### For local development
```bash
git clone https://github.com/coleygroup/shepherd-score.git
cd shepherd-score
pip install -e ".[all]"
```

## Requirements

This package works where PyTorch, Open3D, RDKit, and xTB can be installed for Python >=3.9. **If you are coming from the *ShEPhERD* repository, you can use the same environment as described there.**
Core dependencies (installed automatically) which enables interaction profile extraction, scoring/alignment, and evaluations are listed below.
```
python>=3.9
numpy
torch>=1.12
open3d>=0.18
rdkit>=2023.03
pandas>=2.0
scipy>=1.10
py3Dmol
molscrub
numba
tqdm
threadpoolctl
```


> **Note**: If using `torch<=2.4`, ensure that `mkl==2024.0` with conda since there is a known [issue](https://github.com/pytorch/pytorch/issues/123097) that prevents importing torch.

#### Docking with Autodock Vina
Installing `shepherd-score[docking]` will automatically install the python bindings for Autodock Vina for the python interface. However, a manual installation of the executable of Autodock Vina v1.2.5 is required and can be found here: [https://vina.scripps.edu/downloads/](https://vina.scripps.edu/downloads/).

## Usage
The package has convenience wrappers and base functions. Most users should reach for the convenience wrappers below; the base functions underneath are there if you need lower-level control. Scoring can be done with either NumPy or Torch, but alignment requires Torch. There are also Jax implementations for both scoring and alignment of gaussian overlap, ESP similarity, and pharmacophore similarity.

**Update 8/20/25**: Applicable xTB functions and evaluation pipeline evaluations are now parallelizable through the `num_workers` argument in the `.evaluate` method.

### Convenience wrappers
- `Molecule` class
    - `shepherd_score.container.Molecule` accepts an RDKit `Mol` object (with an associated conformer) and generates its interaction profiles, exposed via two properties:
      - `.surface` &rarr; `Surface(positions, esp, probe_radius)`
      - `.pharmacophore` &rarr; `Pharmacophore(types, positions, vectors, atom_ids, labels)`
    - Pass `return_atom_ids=True` and/or `priority_atoms=[...]` to `Molecule.get_pharmacophore()` to populate `.pharmacophore.atom_ids`/`.labels`, e.g. for priority-weighted pharmacophore scoring
- `MoleculePair` class
    - `shepherd_score.container.MoleculePair` operates on two `Molecule` objects and prepares their `Surface`/`Pharmacophore` profiles for scoring and alignment
- `MoleculePairBatch` class
    - `shepherd_score.container.MoleculePairBatch` operates on a list of `MoleculePair` objects and enables accelerated alignment by padding all profile arrays to a common shape so a single compiled kernel is reused across every pair. Supports optional multi-CPU parallelism.

### Alignment modes
Every mode below is reachable as `MoleculePair.align_with_<mode>`, as
`MoleculePairBatch.align_with_<mode>`, and as the `mode=` argument of `screen`. Each mode carries
its own default number of SO(3) seeds and optimizer steps (`MODE_SEEDS` and `MODE_STEPS` in
`shepherd_score/accel/_modes.py`); pass `max_num_steps` for a different budget. A `_tversky` mode
is its parent with the symmetric Tanimoto replaced by an asymmetric Tversky reduction, weighted
toward the reference by default (`tversky_alpha=0.95`, `tversky_beta=0.05`).

| Mode | What it scores |
| :------- | :------- |
| `vol` | Gaussian volume overlap of the heavy-atom clouds |
| `vol_esp` | Heavy-atom volume overlap weighted by the atomic partial charges |
| `surf` | Gaussian overlap of the molecular surface point clouds |
| `surf_esp` | Surface overlap weighted by the surface electrostatic potential |
| `vol_and_surf_esp` | Volumetric or surface shape overlap (selected by `alpha`) blended with the ShaEP surface-ESP agreement |
| `pharm` | Directional pharmacophore overlap over anchors, vectors, and types |
| `vol_color` | Shape overlap blended with directionless pharmacophore ("color") overlap |
| `vol_tversky` | `vol` with a Tversky reduction |
| `vol_lipo` | Shape overlap blended with the per-atom Crippen logP field |
| `vol_esp_tversky` | `vol_esp` with a Tversky reduction |
| `vol_mr` | Shape overlap blended with the per-atom Crippen molar-refractivity field |
| `surf_tversky` | `surf` with a Tversky reduction |
| `surf_esp_tversky` | `surf_esp` with a Tversky reduction |
| `vol_lipo_tversky` | `vol_lipo` with a Tversky reduction |
| `vol_color_tversky` | `vol_color` with a Tversky reduction |
| `vol_atomtype` | Shape overlap blended with element-matched atom overlap |
| `vol_pharm` | Shape overlap blended with directional pharmacophore overlap |
| `pharm_tversky` | `pharm` with the asymmetric pharmacophore similarity |
| `vol_and_surf_esp_tversky` | `vol_and_surf_esp` with a Tversky reduction |
| `vol_fukui` | Shape overlap blended with the per-atom condensed Fukui field |
| `vol_avoid` | Shape overlap minus an excluded-volume penalty against a fixed avoid cloud |

The mode formerly called `esp` is now `surf_esp` and `esp_combo` is now `vol_and_surf_esp`; both
old names are still accepted.

`MoleculePairBatch.align_with_*` and `screen` take a `backend=` argument whose default (`None`)
resolves per device: the Triton kernels on CUDA, the numba kernels on CPU, with `backend="jax"`
selecting the JAX implementation when the `[jax]` extra is installed.

### Base functions
#### Conformer generation
Useful conformer generation functions are found in the `shepherd_score.conformer_generation` module.

#### Interaction profile extraction
| Interaction profile | Function | Returns |
| :------- | :------- | :------- |
| shape | `shepherd_score.extract_profiles.get_molecular_surface()` | `np.ndarray` (M,3) surface positions |
| electrostatics | `shepherd_score.extract_profiles.get_electrostatic_potential()` | `np.ndarray` (M,) ESP per surface point |
| pharmacophores | `shepherd_score.extract_profiles.get_pharmacophores()` | `Pharmacophore` |

`shepherd_score.pharm_utils.pharmacophore.Pharmacophore` is a lightweight dataclass; it also unpacks as `types, positions, vectors = get_pharmacophores(mol)`. The surface position/ESP arrays are assembled into a matching `shepherd_score.container.profiles.Surface` dataclass by `Molecule` (above). Most users won't call these extraction functions or construct the containers directly.

#### Scoring
```shepherd_score.score``` contains the base scoring functions with seperate modules for those dependent on PyTorch (`*.py`), NumPy (`*_np.py`), and Jax (`*_jax.py`).

| Similarity | Function |
| :------- | :------- |
| shape | `shepherd_score.score.gaussian_overlap.get_overlap()` |
| electrostatics | `shepherd_score.score.electrostatic_scoring.get_overlap_esp()` |
| pharmacophores | `shepherd_score.score.pharmacophore_scoring.get_overlap_pharm()` |


## Scoring and Alignment Examples

Full jupyter notebook tutorials/examples for extraction, scoring, and alignments are found in the [`examples`](./examples/) folder. Some minimal examples are below.

### Extraction
Extraction of interaction profiles via the `Molecule` convenience wrapper.

```python
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.conformer_generation import charges_from_single_point_conformer_with_xtb
from shepherd_score.container import Molecule

# Embed conformer with RDKit and partial charges from xTB
ref_mol = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
partial_charges = charges_from_single_point_conformer_with_xtb(ref_mol)

# `Molecule` extracts and owns all interaction profiles
ref_molec = Molecule(
    ref_mol,
    # optional options otherwise .surface/.pharmacophore stay empty
    num_surf_points=200,
    partial_charges=partial_charges, # If None, computed lazily with gfn2-xTB
    pharm_multi_vector=False # recommended
    # e.g., carbonyls get one HBA vector rather than two
)

# Surface point cloud + electrostatic potential
surf_pos = ref_molec.surface.positions # np.array (200,3)
esp = ref_molec.surface.esp # np.array (200,)

# Pharmacophores as a `Pharmacophore` container, unpacks as (types, positions, vectors)
# ref_molec.pharmacophore.types: np.array (P,)
# ref_molec.pharmacophore.{positions/vectors}: np.array (P,3)
pharm = ref_molec.pharmacophore
```

### 3D similarity scoring
An example of scoring the similarity of two different molecules using 3D surface, ESP, and pharmacophore similarity metrics.

```python
from shepherd_score.score.constants import ALPHA
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.conformer_generation import optimize_conformer_with_xtb
from shepherd_score.container import Molecule, MoleculePair

# Embed a random conformer with RDKit
ref_mol_rdkit = embed_conformer_from_smiles('Oc1ccc(CC=C)cc1', MMFF_optimize=True)
fit_mol_rdkit = embed_conformer_from_smiles('O=CCc1ccccc1', MMFF_optimize=True)
# Local relaxation with xTB
ref_mol, _, ref_charges = optimize_conformer_with_xtb(ref_mol_rdkit)
fit_mol, _, fit_charges = optimize_conformer_with_xtb(fit_mol_rdkit)

# Extract interaction profiles
ref_molec = Molecule(ref_mol,
                     num_surf_points=200,
                     partial_charges=ref_charges,
                     pharm_multi_vector=False)
fit_molec = Molecule(fit_mol,
                     num_surf_points=200,
                     partial_charges=fit_charges,
                     pharm_multi_vector=False)

# Centers the two molecules' COM's to the origin
mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

# Compute the similarity score for each interaction profile
shape_score = mp.score_with_surf(ALPHA(mp.num_surf_points))
esp_score = mp.score_with_esp(ALPHA(mp.num_surf_points), lam=0.3)
pharm_score = mp.score_with_pharm()
```

### Alignment
Next we show alignment using the same `MoleculePair` class.

```python
# Centers the two molecules' COM's to the origin
mp = MoleculePair(ref_molec, fit_molec, num_surf_points=200, do_center=True)

# Align fit_molec to ref_molec with your preferred objective function
# By default we use automatic differentiation via pytorch
surf_points_aligned = mp.align_with_surf(ALPHA(mp.num_surf_points),
                                         num_repeats=50)
surf_points_esp_aligned = mp.align_with_surf_esp(ALPHA(mp.num_surf_points),
                                                 lam=0.3,
                                                 num_repeats=50)
pharm_pos_aligned, pharm_vec_aligned = mp.align_with_pharm(num_repeats=50)

# Optimal scores and SE(3) transformation matrices are stored as attributes
# mp.sim_aligned_surf, mp.sim_aligned_surf_esp, mp.sim_aligned_pharm
# mp.transform_surf, mp.transform_surf_esp, mp.transform_pharm

# Get a copy of the optimally aligned fit Molecule object
transformed_fit_molec = mp.get_transformed_molecule(
    se3_transform=mp.transform_surf
)
```

Alignment of many `MoleculePair` objects at once is accelerated by `MoleculePairBatch`, which pads
every pair's arrays to a common shape so one compiled kernel is reused across the batch.

```python
from shepherd_score.container import MoleculePairBatch

batch = MoleculePairBatch(pairs)  # `pairs` is a list of MoleculePair objects

# backend=None (the default) picks Triton on a CUDA device and numba on CPU
scores, aligned = batch.align_with_vol(return_aligned=True)
scores, aligned = batch.align_with_vol_esp(lam=0.1, return_aligned=True)
scores, aligned = batch.align_with_surf(ALPHA(200), return_aligned=True)

# The JAX path stays available when the [jax] extra is installed. Its multi-CPU shard_map
# mode requires XLA_FLAGS to be set *before* JAX is imported.
scores, aligned = batch.align_with_vol(backend="jax", num_workers=4, num_buckets=4,
                                       use_shmap=True)
```


### Virtual screening (`ProfileStore` + `screen`)
For one query against a large library, featurise the library once into an on-disk `ProfileStore`
and stream it past the query with `screen`. The store keeps only the arrays the requested modes
need (float16 by default, about 200 bytes per molecule for `vol`) in independent shards, so the
library is never held in RAM.

```python
import numpy as np
from shepherd_score.screen import ProfileStore, screen

# Build once. Each library entry is a featurised Molecule (conformer, charges, surface,
# pharmacophores -- see "Extraction" above). One store serves every mode listed at creation.
store = ProfileStore.create("library.fss", num_surf_points=200, modes=("vol", "vol_esp"))
for molecule in library_molecules:     # any iterable; nothing accumulates in RAM
    store.add(molecule)
store.close()

# Screen as often as you like. backend=None resolves to Triton on a CUDA machine, numba otherwise.
store = ProfileStore.open("library.fss")
hits = screen(query_molecule, store, mode="vol", top_k=1000)      # [Hit(score, id, transform), ...]
hits = screen(query_molecule, store, mode="vol_esp", top_k=1000, lam=0.1)

# Every score in library order (e.g. for an enrichment analysis), not only the top-K
scores = np.empty(len(store), dtype=np.float32)
screen(query_molecule, store, mode="vol", top_k=1, scores_out=scores)

# Several GPUs on one node: one worker process per device, spawned on the first call and
# kept for later screens (close_multigpu_pool() releases them)
hits = screen(query_molecule, store, mode="vol", ndev=4)
```

- `num_surf_points` and the mode-specific keyword arguments (`alpha` for the surface modes, `lam`
  for the ESP modes) must match how the query was built; `screen` raises if a required one is
  missing.
- A store that serves a mode seeded from the heavy-atom cloud, such as `vol`, is canonical by
  default: each molecule is stored rotated into its own principal-axis frame, which lets the screen
  run one constant seed set instead of a per-molecule eigensolve. Returned transforms are composed
  back to the centred frame, so hits read the same either way. Scores from a canonical store move
  at the 1e-3 level against a non-canonical one; pass `canonical=True` or `canonical=False` to
  `ProfileStore.create` to decide explicitly.
- A store directory is single-writer. For a parallel build, give each worker its own store and
  screen the parts in turn.
- On CPU, `screen(..., backend="numba")` reaches full speed only with the SVML build described
  under [Fast CPU alignment](#fast-cpu-alignment-numba-kernels) above.


## Evaluation Examples and Scripts

We implement three evaluations of generated 3D conformers. Evaluations can be done on an individual basis or in a pipeline. Here we show the most basic use case in the unconditional setting.


- `ConfEval`
    - Checks validity, pre-/post-xTB relaxation
    - Calculates 2D graph properties
- `ConsistencyEval`
    - Inherits from `ConfEval` and evaluates the consistency of the molecule's jointly generated interaction profiles with the true interaction profiles using 3D similarity scoring functions
- `ConditionalEval`
    - Inherits from `ConfEval` and evaluates the 3D similarity between generated molecules and the target molecule

**Note**: Evaluations can be run from any molecule's atomic numbers and positions with explicit hydrogens (i.e., straight from an xyz file).

### Examples

Full jupyter notebook tutorials/examples for evaluations are found in the [`examples`](./examples/) folder. Some minimal examples are below.

```python
from shepherd_score.evaluations.evaluate import ConfEval
from shepherd_score.evaluations.evaluate import UnconditionalEvalPipeline

# ConfEval evaluates the validity of a given molecule, optimizes it with xTB,
#   and also computes various 2D graph properties
# `atom_array` np.ndarray (N,) atomic numbers of the molecule (with explicit H)
# `position_array` np.ndarray (N,3) atom coordinates for the molecule
conf_eval = ConfEval(atoms=atom_array, positions=position_array)

# Alternatively, if you have a list of molecules you want to test:
uncond_pipe = UnconditionalEvalPipeline(
    generated_mols = [(a, p) for a, p in zip(atom_arrays, position_arrays)]
)
uncond_pipe.evaluate(num_workers=4)

# Properties are stored as attributes and can be converted into pandas df's
global_series, sample_df = uncond_pipe.to_pandas()
```

### Scripts
Scripts to evaluate *ShEPhERD*-generated samples can be found in the `scripts` directory.

## Data
We provide the data used for model training, benchmarking, and all *ShEPhERD*-generated samples reported in the paper at this [Dropbox link](https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0). There are comprehensive READMEs in the Dropbox describing the different folders.

## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
If you use or adapt `shepherd_score` or [*ShEPhERD*](https://github.com/coleygroup/shepherd) in your work, please cite us:

```bibtex
@inproceedings{
adams2025shepherd,
title={Sh{EP}h{ERD}: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design},
author={Keir Adams and Kento Abeywardane and Jenna Fromer and Connor W. Coley},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=KSLkFYHlYg}
}
```
