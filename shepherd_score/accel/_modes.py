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
# These are the cheapest (seeds, steps) per mode that still hold all three: (a) MEAN cross-overlap
# >= 99.7% of the per-pair ceiling, (b) a <= 8% per-pair tail (fraction of pairs > 1% below the
# ceiling -- the tightest tail surf can reach), and (c) self-copy recovery 1.0. Re-validated
# 2026-07 on THREE diverse cross-pair sweeps (drugs.smi tuning set, library.smi 460-mol diverse
# ZINC-style, molecules.smi drug-like) via investigate_seeds_acc.py + a tail-aware pick.
# surf was massively over-seeded: 16 holds 99.8-100% mean/ceiling with tail <= 3% on all three
# (4x fewer than the legacy 64, essentially free). pharm is the one genuinely multi-basin mode --
# even 64 only reaches ~99.4% mean/ceiling on the diverse library, so 48 seeds are indistinguishable
# from 64 (<= 0.27% loss, tail <= 4.3%) while saving 1.33x. The fast-converging shape / ESP channels
# need far fewer still. vol_and_surf_esp seeds from its VOLUME centers (esp_combo._VOL_SEEDS, default
# on) so 24 seeds match the legacy 64 surface seeds (99.7% mean, ~1.96x). Steps sit at the 40/70 knee.
MODE_SEEDS = {"vol": 16, "surf": 16, "surf_esp": 12, "vol_esp": 8, "vol_and_surf_esp": 24,
              "pharm": 48, "vol_color": 20}
MODE_STEPS = {"vol": 40, "surf": 40, "surf_esp": 70, "vol_esp": 40, "vol_and_surf_esp": 70,
              "pharm": 70, "vol_color": 40}
