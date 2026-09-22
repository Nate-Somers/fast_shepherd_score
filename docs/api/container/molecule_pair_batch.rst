MoleculePairBatch
=================

:class:`~shepherd_score.container.MoleculePairBatch` accepts a list of
:class:`~shepherd_score.container.MoleculePair` objects and aligns them
efficiently by padding all atom/pharmacophore arrays to a common maximum
length.  Because every call shares the same padded array shape, one compiled
kernel is reused for every pair in the batch, avoiding the per-pair
recompilation overhead that occurs when array shapes differ.

Backends
--------

Every ``align_with_*`` method takes a ``backend`` argument. The default, ``None``, resolves per
device: the Triton GPU kernels when CUDA is available, the numba CPU kernels otherwise.
``backend="jax"`` selects the JAX implementation, which requires the ``jax`` extra and is no
longer the default. The sections below describe the JAX backend.

.. note::

   While ``shard_map`` is recommended for all JAX-backend cases, it requires
   ``jax>=0.9.0`` and thus ``python>=3.11``. An alternative is to use
   ``multiprocessing`` with ``'spawn'`` context by setting
   ``use_shmap=False``. However, this is known to NOT work on Linux HPC
   environments and has only been tested on M-series Macs.

Masking strategy
----------------

``_pad_arrays()`` pads each coordinate (or charge) array to ``max_len`` and
produces a binary ``float32`` mask (``1.0`` = real atom, ``0.0`` = padding).
An outer-product pair mask

.. code-block:: python

   pair_mask = mask_fit[:, None] * mask_ref[None, :]

zeroes out all padding-atom contributions to the Gaussian overlap sum,
self-overlaps, and gradients.  Since padded shapes are fixed across all pairs,
the compiled kernel is reused without recompilation.

Parallel volumetric alignment via ``jax.shard_map``
----------------------------------------------------

.. note::

   ``XLA_FLAGS`` **must** be set before any JAX import.  Place the
   environment variable assignments at the very top of your script, before
   any ``import jax`` or ``from shepherd_score ...`` statements.

:meth:`~shepherd_score.container.MoleculePairBatch.align_with_vol` supports
a ``use_shmap=True`` path that distributes pairs across virtual CPU devices
using :func:`~shepherd_score.alignment._jax_parallel.optimize_ROCS_overlay_jax_vol_shmap`
instead of Python ``multiprocessing``:

.. code-block:: python

   import os
   os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
   os.environ['JAX_PLATFORMS'] = 'cpu'

   from shepherd_score.container import MoleculePairBatch

   batch = MoleculePairBatch(pairs)

   # Sequential (default)
   scores, aligned = batch.align_with_vol(num_workers=1)

   # Parallel via shard_map
   scores, aligned = batch.align_with_vol(num_workers=4, use_shmap=True)

Why ``shard_map`` instead of ``multiprocessing`` on HPC?

* ``multiprocessing`` with ``'spawn'`` context can be unreliable on Linux HPC
  (JAX initialisation in subprocesses, resource limits).
* ``shard_map`` distributes work across virtual CPU devices **within a single
  process** without forking/spawning.

Bucketing for heterogeneous molecule sets
-----------------------------------------

By default (``n_buckets=1``) all pairs are padded to the global atom-count
maximum and processed in a single ``shard_map`` call (one JIT compilation,
lowest overhead).  For large heterogeneous sets use ``n_buckets > 1``, which
sorts pairs by ``(max(ref, fit), min(ref, fit))`` via ``np.lexsort`` and
processes each bucket with its own local padding maximum — reducing wasted
computation at the cost of multiple sequential ``shard_map`` calls:

.. code-block:: python

   # Default: single pass, one compilation
   scores, aligned = batch.align_with_vol(num_workers=4, use_shmap=True)

   # Bucketed: useful for datasets with diverse molecule sizes
   scores, aligned = batch.align_with_vol(num_workers=4, use_shmap=True, num_buckets=4)

.. note::

   ``use_shmap=True`` uses ``lax.scan`` (fixed steps, no convergence-based
   early stopping).  ``max_num_steps=200`` is the default.

Available batch alignment methods
----------------------------------

Every alignment mode in the registry has a matching ``align_with_<mode>`` method, under the same
name it carries on :class:`~shepherd_score.container.MoleculePair`: ``align_with_vol``,
``align_with_vol_esp``, ``align_with_surf``, ``align_with_surf_esp``,
``align_with_vol_and_surf_esp``, ``align_with_pharm``, ``align_with_vol_color``,
``align_with_vol_tversky``, ``align_with_vol_lipo``, ``align_with_vol_esp_tversky``,
``align_with_vol_mr``, ``align_with_surf_tversky``, ``align_with_surf_esp_tversky``,
``align_with_vol_lipo_tversky``, ``align_with_vol_color_tversky``, ``align_with_vol_atomtype``,
``align_with_vol_pharm``, ``align_with_pharm_tversky``,
``align_with_vol_and_surf_esp_tversky``, ``align_with_vol_fukui`` and ``align_with_vol_avoid``.

``align_with_esp`` and ``align_with_esp_combo`` remain as aliases for ``align_with_surf_esp``
and ``align_with_vol_and_surf_esp``.

Each method returns the per-pair scores followed by the aligned arrays (built only when
``return_aligned=True``) and writes the mode's transform and score attributes back onto every
:class:`~shepherd_score.container.MoleculePair` in the batch, for example
``transform_vol_noH`` and ``sim_aligned_vol_noH`` for ``vol``. The per-mode defaults for the
number of SO(3) seeds and optimizer steps are in ``shepherd_score/accel/_modes.py``.

For the low-level parallel kernel see :doc:`../alignment/jax_parallel`.
For the full scoring and alignment theory see :doc:`../../theory`.

Class reference
---------------

.. autoclass:: shepherd_score.container._batch.MoleculePairBatch
   :members:
   :undoc-members:
   :show-inheritance:
