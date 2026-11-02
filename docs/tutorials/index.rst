Tutorials
=========

Jupyter notebooks covering profile extraction, scoring and evaluation. They use
the original single-pair API; see :doc:`../accelerated_api` for batched alignment and
screening.

.. toctree::
   :maxdepth: 2

   01_profiles
   02_scoring
   03_evaluation

Tutorial Overview
-----------------

1. **Interaction Profiles** (``01_profiles.ipynb``): extracting shape,
   electrostatic and pharmacophore profiles from molecules.

2. **Scoring and Alignment** (``02_scoring.ipynb``): computing 3D similarity scores
   and aligning molecules by their interaction profiles.

3. **Evaluation Pipelines** (``03_evaluation.ipynb``): evaluating generated
   conformers.

Running the Tutorials
---------------------

From the repository checkout, install the package and open ``examples/``:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install jupyter
   cd examples
   jupyter notebook

See :doc:`../installation` for optional GPU and xTB dependencies.
