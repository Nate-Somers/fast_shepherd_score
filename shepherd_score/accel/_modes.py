"""Single source of truth for the alignment *modes*.

Pure data, zero heavy imports (no torch / numpy / container), so every layer can import
this freely without import-cycle risk. Before this module the same facts were copied across
``aligners.py`` (seed/step counts), ``screen.py`` / ``multi_gpu.py`` (result-attribute maps),
and ``cpu_pool.py`` / ``multi_gpu.py`` / ``screen.py`` (legacy-alias dicts). Consumers now
derive from here; ``tests/test_mode_registry.py`` pins the invariants (incl. agreement with the
torch-typed ``accel/batch/_dispatch.py:_MODE_SPEC``, which stays the authority for the *process*
path's per-mode tensor plumbing).

To add a new mode, register it here (attrs + seeds + steps, and PROCESS_MODES if it gets a
``_MODE_SPEC`` entry); the accel layers and the screen front-end pick it up from these tables.
"""

# Canonical mode id -> (transform_attr, score_attr) written in-place on a MoleculePair by
# ``MoleculePairBatch.align_with_<mode>``. This is the full set of the 10 canonical modes; the
# tuple order is the public mode order (preserved from screen.py's historical _VALID_MODES).
MODE_ATTRS = {
    "vol":              ("transform_vol_noH",          "sim_aligned_vol_noH"),
    "vol_esp":          ("transform_vol_esp_noH",      "sim_aligned_vol_esp_noH"),
    "surf":             ("transform_surf",             "sim_aligned_surf"),
    "surf_esp":         ("transform_surf_esp",         "sim_aligned_surf_esp"),
    "vol_and_surf_esp": ("transform_vol_and_surf_esp", "sim_aligned_vol_and_surf_esp"),
    "pharm":            ("transform_pharm",            "sim_aligned_pharm"),
    "vol_color":        ("transform_vol_color",        "sim_aligned_vol_color"),
    "vol_tversky":      ("transform_vol_tversky",      "sim_aligned_vol_tversky"),
    "vol_lipo":         ("transform_vol_lipo",         "sim_aligned_vol_lipo"),
    "vol_esp_tversky":  ("transform_vol_esp_tversky",  "sim_aligned_vol_esp_tversky"),
    # SI experimental modes (accelerated; reuse existing kernels/drivers or a small new one).
    "vol_mr":                   ("transform_vol_mr",                   "sim_aligned_vol_mr"),
    "surf_tversky":             ("transform_surf_tversky",             "sim_aligned_surf_tversky"),
    "surf_esp_tversky":         ("transform_surf_esp_tversky",         "sim_aligned_surf_esp_tversky"),
    "vol_lipo_tversky":         ("transform_vol_lipo_tversky",         "sim_aligned_vol_lipo_tversky"),
    "vol_color_tversky":        ("transform_vol_color_tversky",        "sim_aligned_vol_color_tversky"),
    "vol_atomtype":             ("transform_vol_atomtype",             "sim_aligned_vol_atomtype"),
    "vol_pharm":                ("transform_vol_pharm",                "sim_aligned_vol_pharm"),
    "pharm_tversky":            ("transform_pharm_tversky",            "sim_aligned_pharm_tversky"),
    "vol_and_surf_esp_tversky": ("transform_vol_and_surf_esp_tversky", "sim_aligned_vol_and_surf_esp_tversky"),
}

# The 10 canonical mode ids, in public order.
CANONICAL_MODES = tuple(MODE_ATTRS)

# Legacy (pre-rename) mode names -> canonical. The old public API keeps working through this:
# MoleculePair.align_with_esp / align_with_esp_combo, MoleculePairBatch likewise, the screen
# ``mode=`` arg, the cpu_pool / multi_gpu mode strings, and old pickles. Normalized at every
# public entry point via ``canonical()``.
LEGACY_MODE_ALIASES = {"esp": "surf_esp", "esp_combo": "vol_and_surf_esp"}


def canonical(mode: str) -> str:
    """Resolve a (possibly legacy) mode name to its canonical form; unknown names pass through."""
    return LEGACY_MODE_ALIASES.get(mode, mode)


# Modes that have an ``accel/batch/_dispatch.py:_MODE_SPEC`` entry, i.e. a process-per-GPU
# (multi_gpu) and CPU-pool (cpu_pool) path. The other modes run single-GPU only. ``_MODE_SPEC``
# (torch-typed extract/tensors/out plumbing) remains the authority; the registry test asserts
# ``tuple(_MODE_SPEC) == PROCESS_MODES`` so this list can never silently drift.
PROCESS_MODES = ("vol", "surf", "surf_esp", "pharm")

# Per-mode defaults: (SO(3) multi-start seed count, fine-optimizer step count). Read by
# ``aligners._seeds_for`` / ``_steps_for`` and used by both backends (triton/numba) and both
# workloads (pairwise ``MoleculePairBatch.align_with_*`` and the streaming screen).
#
# These are BALANCED defaults, chosen at the accuracy/throughput knee rather than for maximum
# accuracy: seed count is cheap in retrospective-screening ROC-AUC but expensive in throughput,
# and ROC-AUC plateaus at low seed counts for every mode. Relative to an accuracy-maximizing
# choice these cost <=0.006 ROC-AUC while recovering >=99% of the best achievable overlap.
# Callers who want more accuracy pass ``num_repeats`` / ``max_num_steps`` explicitly.
MODE_SEEDS = {"vol": 10, "surf": 8, "surf_esp": 8, "vol_esp": 16, "vol_and_surf_esp": 8,
              "pharm": 32, "vol_color": 16, "vol_tversky": 10, "vol_lipo": 16,
              "vol_esp_tversky": 16,
              "vol_mr": 16, "surf_tversky": 8, "surf_esp_tversky": 8, "vol_lipo_tversky": 16,
              "vol_color_tversky": 16, "vol_atomtype": 16, "vol_pharm": 32, "pharm_tversky": 32,
              "vol_and_surf_esp_tversky": 8}
MODE_STEPS = {"vol": 30, "surf": 40, "surf_esp": 40, "vol_esp": 50, "vol_and_surf_esp": 60,
              "pharm": 50, "vol_color": 40, "vol_tversky": 40, "vol_lipo": 50,
              "vol_esp_tversky": 50,
              "vol_mr": 50, "surf_tversky": 40, "surf_esp_tversky": 40, "vol_lipo_tversky": 50,
              "vol_color_tversky": 40, "vol_atomtype": 40, "vol_pharm": 50, "pharm_tversky": 50,
              "vol_and_surf_esp_tversky": 60}
