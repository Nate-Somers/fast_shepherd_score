MoleculePairBatch
=================

``MoleculePairBatch`` aligns a list of ``MoleculePair`` objects. The accelerated
backends group pairs by feature size, pad and mask arrays, and optimize multiple
starting poses. The class writes each pair's score and transform in place.

Backends
--------

``backend="numba"`` selects CPU kernels; ``backend="triton"`` selects CUDA kernels.
The default resolves from the available device. ``backend="jax"`` selects the
optional JAX implementation for supported original modes. Options specific to JAX,
such as ``use_shmap`` and ``num_buckets``, do not select JAX automatically.

.. code-block:: python

   from shepherd_score.container import MoleculePairBatch

   scores, aligned = MoleculePairBatch(pairs).align_with_vol(
       backend="numba", return_aligned=True)

Every registered mode has an ``align_with_<mode>`` method. ``align_with_esp`` and
``align_with_esp_combo`` remain compatibility aliases for ``surf_esp`` and
``vol_and_surf_esp``. Default seed and step counts are mode-specific; see
``shepherd_score/accel/_modes.py``. Returned aligned arrays are built only when
``return_aligned=True``.

For JAX parallel alignment see :doc:`../alignment/jax_parallel`.
For streaming libraries see :doc:`../screening`.

Class reference
---------------

.. autoclass:: shepherd_score.container._batch.MoleculePairBatch
   :members:
   :undoc-members:
   :show-inheritance:
