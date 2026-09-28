Installation
============

Install the checkout to use its accelerated scoring and screening features::

   python -m pip install -e .

Numba CPU kernels are included. GPU kernels require a CUDA build of PyTorch and
Triton on a supported platform::

   python -m pip install -e ".[gpu]"

Other extras are ``jax``, ``docking``, ``dev``, and ``docs``. Install only those
needed for the intended workflow. The Python and dependency bounds are defined
in ``pyproject.toml``. Older PyPI releases may not include this checkout's API.

CPU benchmark environment
-------------------------

``environment.yml`` defines the conda environment used for the SVML CPU path.
Check whether Numba loaded SVML::

   python -c "import numba.core.config as c; print(c.USING_SVML)"

Numba builds without SVML also work, but performance and floating-point rounding
can differ. Published throughput should be compared using the recorded software
versions, hardware, and settings in the paper's result files.

External executables
--------------------

xTB must be on ``PATH`` for xTB charges, quantum-chemical relaxation, and Fukui
descriptors. ``Molecule(..., charge_model="mmff")`` uses RDKit MMFF94 charges.
Docking needs the ``docking`` extra and the Vina executable. Protonation tools
and visualization dependencies are documented in their API pages.

Documentation and tests
-----------------------

From the repository root::

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests/ -m "not jax"
   python -m sphinx -b html docs docs/_build/html
