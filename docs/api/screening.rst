Virtual Screening
=================

Stream an on-disk library of featurised molecules past one query (or a panel of
queries) and return the top-K hits, without holding the library in RAM. The
library is featurised once into a :class:`~shepherd_score.screen.ProfileStore`
and screened any number of times; see the README's *Virtual screening* section
for the end-to-end example.

.. autoclass:: shepherd_score.screen.ProfileStore
   :members: create, open, add, close, canonical
   :undoc-members:

.. autofunction:: shepherd_score.screen.screen

.. autofunction:: shepherd_score.screen.screen_many

.. autofunction:: shepherd_score.screen.close_multigpu_pool

Shard-parallel CPU screening
----------------------------

A RAM-resident list of molecules screened across forked worker processes, one
core each; the pool is kept for later calls against the same library.

.. autofunction:: shepherd_score.accel.screen_parallel.screen_parallel

.. autofunction:: shepherd_score.accel.screen_parallel.screen_parallel_close
