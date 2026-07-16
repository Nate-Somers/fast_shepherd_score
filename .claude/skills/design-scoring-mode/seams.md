# Reference-layer seams

The files a new mode touches, bottom-up. Everything here is additive: you add functions and one
registry row per table. You do not modify the front-end (`screen.py`, `accel/multi_gpu.py`,
`accel/cpu_pool.py`) — those derive the mode set from `accel/_modes.py`.

| Layer | File | What you add |
|---|---|---|
| Channel math | `score/<family>_scoring.py` (+ `_np`, optional `_jax`) | The pure overlap function: inputs in, scalar out, no optimization. Reuse a family (`gaussian_overlap`, `electrostatic_scoring`, `pharmacophore_scoring`) or add one. |
| Eager optimizer | `alignment/_torch.py` | `objective_<mode>_overlay` (single-pose value) and `optimize_<mode>_overlay` (multi-seed SO(3) + Adam → best pose+score). **The oracle.** |
| Per-pair API | `container/_core.py` | `MoleculePair.align_with_<mode>(...)`; defaults `num_repeats`/`max_num_steps` to `None` → `_default_seeds`/`_default_steps`; writes `transform_<mode>` / `sim_aligned_<mode>`. |
| Registry (identity) | `accel/_modes.py` | One row each in `MODE_ATTRS`, `MODE_SEEDS`, `MODE_STEPS`. Nothing else. |
| Exports | `score/__init__.py`, `alignment/__init__.py`, `container/__init__.py` | Public function names, in the existing style. |
| Tests | `tests/test_<mode>.py` | From `template_test.py`. Plus `tests/test_mode_registry.py` must still pass. |

## The registry (`accel/_modes.py`)

`accel/_modes.py` is the single source of truth for the mode set — pure data, no heavy imports,
so every layer reads it without an import cycle. The tables you extend:

- `MODE_ATTRS[<mode>] = ("transform_<mode>", "sim_aligned_<mode>")` — where results are written
  on a `MoleculePair`. Adding here is what makes the mode "exist" to the front-end.
- `MODE_SEEDS[<mode>]` — SO(3) multi-start seed count (default for `num_repeats`).
- `MODE_STEPS[<mode>]` — fine-optimizer step count (default for `max_num_steps`).

Leave `PROCESS_MODES` and (in `accel/batch/_dispatch.py`) `_MODE_SPEC` alone. They are the
accel skill's territory, and a registry test asserts `tuple(_MODE_SPEC) == PROCESS_MODES`.

## Eager-function naming: use the canonical id

The eager optimizers in `alignment/_torch.py` are historically named after the physics, not the
mode id. New modes should **not** copy that — name after the canonical id:

| Canonical mode id | Existing eager optimizer | Convention to follow |
|---|---|---|
| `vol` | `optimize_ROCS_overlay` / `optimize_ROCS_overlay_analytical` | legacy physics name |
| ESP family (`surf_esp`, `vol_esp`, `vol_and_surf_esp`) | `optimize_ROCS_esp_overlay`, `optimize_esp_combo_score_overlay` | legacy physics names |
| `pharm` | `optimize_pharm_overlay` | id-based (good) |
| `vol_color` | `optimize_vol_color_overlay` | id-based (good — copy this pattern) |

For a new mode `foo`, write `objective_foo_overlay` and `optimize_foo_overlay`. To learn the
mechanics, read whichever existing method in `container/_core.py` calls the eager optimizer of
the **family nearest your target** and follow its structure:

- shape-only target → read `align_with_vol` (calls `optimize_ROCS_overlay`).
- ESP target → read `align_with_surf_esp` / `align_with_vol_and_surf_esp`.
- pharmacophore / combined target → read `align_with_pharm` / `align_with_vol_color`.

## Return signature (keep it uniform)

`optimize_<mode>_overlay(...) -> (aligned_fit_points, se3_transform, score)`, all torch tensors
on the input device. `align_with_<mode>` converts them to NumPy, stores transform+score, and
returns the aligned points. Match the existing methods exactly so downstream code is uniform.
