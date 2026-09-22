Installation
============

Via PyPI
--------

The simplest way to install shepherd-score is via pip:

.. code-block:: bash

   pip install shepherd-score

Install xTB
~~~~~~~~~~~

xTB will need to be installed manually since there are no PyPI bindings. This can be done in a conda
environment, but since this approach has been reported to lead to conflicts, we suggest installing
from `source <https://xtb-docs.readthedocs.io/en/latest/setup.html>`_ and adding it to ``PATH``.

Optional Dependencies
---------------------

GPU Kernels
~~~~~~~~~~~

The Triton kernels behind ``backend="triton"``:

.. code-block:: bash

   pip install "shepherd-score[gpu]"

The extra installs ``triton>=3.6`` and needs ``torch>=2.6`` built against CUDA. The CPU kernels
need no extra, since ``numba`` is a core dependency.

JAX Support
~~~~~~~~~~~

For the JAX implementations of scoring and alignment (``backend="jax"``):

.. code-block:: bash

   pip install "shepherd-score[jax]"

Docking Evaluation Tools
~~~~~~~~~~~~~~~~~~~~~~~~

Include docking evaluation tools:

.. code-block:: bash

   pip install "shepherd-score[docking]"

.. note::

   Installing ``shepherd-score[docking]`` will automatically install the Python bindings for
   Autodock Vina. However, a manual installation of the executable of Autodock Vina v1.2.5
   is required and can be found at: https://vina.scripps.edu/downloads/

All Dependencies
~~~~~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   pip install "shepherd-score[all]"

Fast CPU Alignment
------------------

The CPU backend reaches full speed only with an SVML-enabled numba build, which lets numba emit
Intel SVML vector ``exp`` calls in the overlap kernels. That build is ``numba<=0.59`` together
with ``icc_rt``; the conda ``environment.yml`` in the repository root provides one. Without SVML
the kernels produce the same results, run slower, and emit a single ``RuntimeWarning`` the first
time they are used. Check which build is active with:

.. code-block:: bash

   python -c "import numba.core.config as c; print(c.USING_SVML)"

Local Development
-----------------

For local development:

.. code-block:: bash

   git clone https://github.com/coleygroup/shepherd-score.git
   cd shepherd-score
   pip install -e ".[all]"

Requirements
------------

This package works where PyTorch, Open3D, RDKit, and xTB can be installed for Python >=3.9.
If you are coming from the *ShEPhERD* repository, you can use the same environment.

Core dependencies (installed automatically):

* ``python>=3.9``
* ``numpy``
* ``torch>=1.12``
* ``open3d>=0.18``
* ``rdkit>=2023.03``
* ``pandas>=2.0``
* ``scipy>=1.10``
* ``py3Dmol``
* ``molscrub``
* ``numba``
* ``tqdm``
* ``threadpoolctl``

Optional extras:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Adds
   * - ``gpu``
     - ``triton>=3.6`` and ``torch>=2.6`` for the CUDA kernels
   * - ``jax``
     - ``jax``, ``jaxlib``, ``optax`` and ``scikit-learn``
   * - ``cpu``
     - ``numba``; kept for compatibility, as numba is already a core dependency
   * - ``docking``
     - ``meeko``, ``vina>=1.2.5``, ``openbabel``, ``prolif`` and ``biopython``
   * - ``docs``
     - the Sphinx toolchain used to build this documentation
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``ruff`` and ``mypy``
   * - ``all``
     - every extra above

.. note::

   If using ``torch<=2.4``, ensure that ``mkl==2024.0`` with conda since there is a known
   `issue <https://github.com/pytorch/pytorch/issues/123097>`_ that prevents importing torch.
