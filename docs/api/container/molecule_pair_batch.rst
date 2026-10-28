MoleculePairBatch
=================

``MoleculePairBatch`` aligns a list of ``MoleculePair`` objects. The accelerated
backends group pairs by feature size, pad and mask arrays, and optimize multiple
starting poses. The class writes each pair's score and transform in place.

Backends
--------

``backend="numba"`` selects CPU kernels; ``backend="triton"`` selects CUDA kernels.
The default resolves from the available device. ``backend="jax"`` selects the
optional JAX implementation for ``vol``, ``vol_esp``, ``surf``, ``surf_esp`` and
``pharm``; other modes warn and run per-pair PyTorch. Options specific to JAX,
such as ``use_shmap`` and ``num_buckets``, do not select JAX automatically.

.. code-block:: python

   from shepherd_score.container import MoleculePairBatch

   scores, aligned = MoleculePairBatch(pairs).align_with_vol(
       backend="numba")

Every registered mode has an ``align_with_<mode>`` method. ``align_with_esp`` and
``align_with_esp_combo`` remain compatibility aliases for ``surf_esp`` and
``vol_and_surf_esp``. ``num_repeats``, ``max_num_steps`` and ``lr`` default to
``None``, which takes the mode's registry values in ``shepherd_score/accel/_modes.py``;
explicit values override them on every backend. ``verbose`` only affects the JAX path. Aligned arrays are returned by default;
pass ``return_aligned=False`` to skip building them.

For JAX parallel alignment see :doc:`../alignment/jax_parallel`.
For streaming libraries see :doc:`../screening`.

Class reference
---------------

.. autoclass:: shepherd_score.container._batch.MoleculePairBatch
   :members:
   :undoc-members:
   :show-inheritance:
