Usage
=====

Molecules and profiles
----------------------

``Molecule`` wraps an RDKit molecule with a 3D conformer. Features are generated
or cached as requested. Set ``num_surf_points`` for surface modes and
``pharm_multi_vector=False`` for a single vector per directional pharmacophore.
Set ``charge_model="mmff"`` for MMFF94 charges or pass partial charges explicitly.
The default is xTB, with a warning and MMFF94 fallback when xTB is unavailable.
Charges are computed on demand; constructing a surface also computes its ESP.
Pharmacophores are omitted unless ``pharm_multi_vector`` is set or their arrays
are supplied. Mesh surface sampling is stochastic; reuse saved surface points
and ESP arrays for comparisons that require identical features.

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
``vol_avoid`` adds an excluded-volume penalty and requires an explicit
``avoid_points`` cloud in addition to the reference and fit molecules.

``vol_and_surf_esp`` combines shape with masked surface-potential agreement.
Its accelerated optimizer follows the gradient of the combined score, including
the ESP agreement term.

The registry in ``shepherd_score/accel/_modes.py`` specifies supported modes and
their default seeds and step budgets. Early stopping uses the same patience on
CPU and GPU where enabled. The CUDA-graph paths for ``vol_and_surf_esp`` and
``vol_and_surf_esp_tversky`` run the full step budget. Floating-point arithmetic
and batch composition can affect the stopping point and the selected pose.

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
