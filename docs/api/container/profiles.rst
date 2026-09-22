Profile Containers
==================

Structured containers for extracted interaction profiles, used internally by
:class:`~shepherd_score.container.Molecule` and available for serialization and
visualization.

.. autoclass:: shepherd_score.container.profiles.Surface
   :members:
   :exclude-members: positions, esp, probe_radius
   :undoc-members:
   :show-inheritance:

``Pharmacophore`` is defined in :mod:`shepherd_score.pharm_utils.pharmacophore`
and re-exported from :mod:`shepherd_score.container` for convenience. See
:doc:`../pharmacophore` for priority-label semantics.

Surface diagnostics
-------------------

:mod:`shepherd_score.surface_diagnostics` measures how faithfully a surface point cloud
reproduces the atom spheres it was sampled from: the shell residual of every point, the
atom-border ("crimp") points, and how well atom centres can be recovered from the cloud alone.
It needs only NumPy and SciPy, so it can gate any surface generator.

.. automodule:: shepherd_score.surface_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
