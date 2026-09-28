Tutorials
=========

These tutorials provide step-by-step guides for using shepherd-score. They are Jupyter 
notebooks that you can also download and run interactively.

.. toctree::
   :maxdepth: 2

   01_profiles
   02_scoring
   03_evaluation

Tutorial Overview
-----------------

1. **Interaction Profiles** (``01_profiles.ipynb``): Learn how to extract shape, 
   electrostatic, and pharmacophore profiles from molecules.

2. **Scoring and Alignment** (``02_scoring.ipynb``): Compute 3D similarity scores 
   and align molecules based on their interaction profiles.

3. **Evaluation Pipelines** (``03_evaluation.ipynb``): Use the evaluation framework 
   to assess generated conformers.

Running the Tutorials
---------------------

From the repository checkout, install the package and open ``examples/``:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install jupyter
   cd examples
   jupyter notebook

See :doc:`../installation` for optional GPU and xTB dependencies.
