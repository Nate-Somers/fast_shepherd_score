Usage
=====

Molecules and profiles
----------------------

``Molecule`` wraps an RDKit molecule with a 3D conformer. Features are generated
or cached as requested. Set ``num_surf_points`` for surface modes and
``pharm_multi_vector=False`` for a single vector per directional pharmacophore.
Pass partial charges explicitly when a particular charge model is required.

.. code-block:: python

   from shepherd_score.conformer_generation import embed_conformer_from_smiles
   from shepherd_score.container import Molecule, MoleculePair, MoleculePairBatch

   rd = embed_conformer_from_smiles("CCOc1ccccc1", MMFF_optimize=True)
   ref = Molecule(rd, num_surf_points=200, pharm_multi_vector=False,
                  charge_model="mmff")
   fit = Molecule(embed_conformer_from_smiles("CCNc1ccccc1", MMFF_optimize=True),
                  num_surf_points=200, pharm_multi_vector=False, charge_model="mmff")
   pair = MoleculePair(ref, fit)
   scores, _ = MoleculePairBatch([pair]).align_with_vol(backend="numba")

The alignment writes the score and transform onto each pair. ``return_aligned``
controls whether transformed arrays are also returned. Explicitly choose
``backend="triton"`` for GPU or ``backend="numba"`` for CPU; the default selects
according to the available device. JAX is available for the original modes.

Scoring modes
-------------

``vol`` and ``surf`` compare heavy-atom and surface point clouds. ``vol_esp`` and
``surf_esp`` weight overlaps by partial charge and surface potential.
``pharm`` compares typed, directional pharmacophores. ``vol_color``, ``vol_lipo``,
``vol_mr``, ``vol_fukui``, ``vol_atomtype``, and ``vol_pharm`` combine shape with
additional molecular features. Tversky variants use asymmetric normalization.
``vol_avoid`` adds an excluded-volume penalty.

``vol_and_surf_esp`` combines shape with masked surface-potential agreement.
Its accelerated optimizer uses shape gradients to generate poses and the
combined score to select poses; it does not differentiate the ESP agreement term.

The registry in ``shepherd_score/accel/_modes.py`` specifies supported modes and
their default seeds and step budgets. These budgets differ by mode. CPU and GPU
early stopping and floating-point behavior can produce different local optima.

Screening
---------

.. code-block:: python

   from shepherd_score.screen import ProfileStore, screen

   store = ProfileStore.create("library.fss", num_surf_points=200,
                               modes=["vol", "surf_esp"])
   store.add(fit, id="compound-1")
   store.close()
   hits = screen(ref, ProfileStore.open("library.fss"), mode="vol",
                 backend="numba", top_k=1)

A store retains the channels required by its declared modes. Surface modes
require a consistent surface-point count. Atom-seeded modes use canonical frames
by default; returned transforms are mapped back to the input frame.
See :doc:`api/screening` for multi-query and parallel screening.

Evaluation
----------

The scripts and command examples in ``scripts/README.md`` evaluate generated
ShEPhERD samples. The paper repository contains the throughput and retrieval
benchmarks. The tutorials cover lower-level scoring and preparation workflows;
their xTB examples require the executable to be installed.
