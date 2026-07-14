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
# ``MoleculePairBatch.align_with_<mode>``. This is the full set of the 7 canonical modes; the
# tuple order is the public mode order (preserved from screen.py's historical _VALID_MODES).
MODE_ATTRS = {
    "vol":              ("transform_vol_noH",          "sim_aligned_vol_noH"),
    "vol_esp":          ("transform_vol_esp_noH",      "sim_aligned_vol_esp_noH"),
    "surf":             ("transform_surf",             "sim_aligned_surf"),
    "surf_esp":         ("transform_surf_esp",         "sim_aligned_surf_esp"),
    "vol_and_surf_esp": ("transform_vol_and_surf_esp", "sim_aligned_vol_and_surf_esp"),
    "pharm":            ("transform_pharm",            "sim_aligned_pharm"),
    "vol_color":        ("transform_vol_color",        "sim_aligned_vol_color"),
}

# The 7 canonical mode ids, in public order.
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

# Per-mode default (seed count, fine-step count) -- read by aligners._seeds_for/_steps_for and
# used by BOTH backends (triton/numba) and BOTH workloads (pairwise MoleculePairBatch.align_with_*
# and the streaming screen path). FINE_NUM_SEEDS / FINE_NUM_STEPS env vars override every mode.
#
# These are BALANCED defaults: tuned for throughput-per-unit-accuracy, not for maximum accuracy.
# Benchmarks must NOT override them -- they exist so that "how fast can I align molecules with this
# package" and "how well does it enrich" are both answered by the shipped configuration.
#
# Tuned 2026-07-13 against three metrics measured together (2026-07 sweeps):
#   (1) ROC-AUC   -- full 41-target DUDE-Z retrospective screen (8 query actives vs ~3k decoys),
#                    an 11 seeds {1..64} x 3 steps {40,70,200} grid = 9,471 screened cells. This is
#                    the authoritative screening signal.
#   (2) mean/ceil + tail% -- synthetic cross-pair overlap recovery vs a 64x200 ceiling, on a drug-like
#                    and a ZINC-diverse set.
#   (3) pairwise aligns/sec -- pure alignment throughput (the "how fast can I align" number).
#
# The key finding: seed count is CHEAP in ROC-AUC but EXPENSIVE in throughput. An accuracy-maximizing
# pick (e.g. vol/vol_color at 48 seeds) buys only +0.002-0.006 ROC-AUC while costing 1.2-4.9x pairwise
# throughput. ROC-AUC plateaus at low seed counts for every mode (the Pareto knee is sharp), so these
# defaults sit at the knee rather than past it. Measured cost of the balanced pick vs an accuracy-maxed
# one, in ROC-AUC / pairwise speedup:
#   vol               10x50  -0.0021 / 4.9x faster   (vol ROC is flat in BOTH seeds and steps -- 100
#                    steps bought nothing measurable, so the step count was halved to 50)
#   surf              8x40   -0.0005 / 1.5x faster   (n.s.)
#   surf_esp          8x40   -0.0005 / 1.9x faster
#   vol_and_surf_esp  8x60   -0.0001 / ~2.3x faster  (ROC statistically IDENTICAL to 24x70)
#   vol_esp          16x50   -0.0015 / ~1.2x faster  (ESP-bound: leaning buys little speed)
#   pharm            32x50   ~-0.001 / ~1.4x faster  (steps 50 recovers most of the 40-step loss)
#   vol_color        16x40   -0.0064 / 3.1x faster   (the sharpest trade -- vol_color is the one mode
#                    whose ROC climbs monotonically with seeds; 16 is the deliberate speed-side choice)
# mean/ceil lands at 0.990-0.996 (vs 0.997-0.999 for the accuracy-maxed picks): still >=99% recovery of
# the best-achievable overlap. FINE_NUM_SEEDS / FINE_NUM_STEPS env vars still override every mode.
MODE_SEEDS = {"vol": 10, "surf": 8, "surf_esp": 8, "vol_esp": 16, "vol_and_surf_esp": 8,
              "pharm": 32, "vol_color": 16}
MODE_STEPS = {"vol": 50, "surf": 40, "surf_esp": 40, "vol_esp": 50, "vol_and_surf_esp": 60,
              "pharm": 50, "vol_color": 40}
