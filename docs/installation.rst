Installation
============

Install from the repository root::

   python -m pip install -e .

Numba CPU kernels are included. GPU kernels need a CUDA build of PyTorch and
Triton::

   python -m pip install -e ".[gpu]"

Other extras are ``jax``, ``docking``, ``dev`` and ``docs``. Python and
dependency versions are set in ``pyproject.toml``. Older PyPI releases do not
include the batched alignment and screening APIs described here.

CPU benchmark environment
-------------------------

``environment.yml`` defines the conda environment used for the published CPU
timings, which use Intel SVML. Check whether Numba loaded it::

   python -c "import numba.core.config as c; print(c.USING_SVML)"

Without SVML the kernels still run, but more slowly, and results can differ
within floating-point tolerance.

External executables
--------------------

- **xTB** must be on ``PATH`` for xTB charges, xTB relaxation and Fukui
  descriptors. ``Molecule`` uses xTB in implicit water for charges by default
  and falls back to MMFF94, with a warning, when xTB is missing. Pass
  ``charge_model="mmff"`` to use MMFF94 directly.
- **Vina**: docking uses the Vina Python API from the ``docking`` extra.
- **ChemAxon**: the ChemAxon protonation path needs a licensed ``cxcalc``.
  The MolScrub path needs no extra install.

Documentation and tests
-----------------------

From the repository root::

   python -m pip install -e ".[dev,docs]"
   python -m pytest tests/ -m "not jax"
   python -m sphinx -b html docs docs/_build/html
