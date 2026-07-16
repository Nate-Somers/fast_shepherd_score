# Reference-layer seams

The files a new mode touches, bottom-up. Everything here is additive: you add functions and one
line to the `_ALIGN_KEYS` tuple. You do **not** touch `accel/_modes.py` or the front-end
(`screen.py`, `accel/multi_gpu.py`, `accel/cpu_pool.py`) — those are the *canonical* registry and
its consumers, which the accel skill wires up once a batched path exists.

| Layer | File | What you add |
|---|---|---|
| Channel math | `score/<family>_scoring.py` (+ `_np`, optional `_jax`) | The pure overlap function: inputs in, scalar out, no optimization. Reuse a family (`gaussian_overlap`, `electrostatic_scoring`, `pharmacophore_scoring`) or add one. |
| Eager optimizer | `alignment/_torch.py` | `objective_<mode>_overlay` (single-pose value) and `optimize_<mode>_overlay` (multi-seed SO(3) + Adam → best pose+score). **The oracle.** |
| Per-pair API | `container/_core.py` | `MoleculePair.align_with_<mode>(...)`; give `num_repeats`/`max_num_steps` **literal** defaults (not `_default_seeds`/`_default_steps` — see below); writes `transform_<mode>` / `sim_aligned_<mode>`. |
| Result slots | `container/_core.py` | Add the mode id to the `_ALIGN_KEYS` tuple — generates the `transform_<mode>` / `sim_aligned_<mode>` accessors. Do **not** add to `accel/_modes.py` (see below). |
| Exports | `score/__init__.py`, `alignment/__init__.py`, `container/__init__.py` | Public function names, in the existing style. |
| Tests | `tests/test_<mode>.py` | From `template_test.py`. Plus `tests/test_mode_registry.py` must still pass. |

## Two registries: result slots vs canonical modes

There are two separate places a mode can be "registered", and telling them apart is the one
non-obvious part of this skill:

- **`_ALIGN_KEYS`** (in `container/_core.py`) — the list of result slots. Adding your mode id here
  generates its `transform_<mode>` / `sim_aligned_<mode>` accessors. **This is the only
  registration a reference-only mode does.** Purely additive and safe.
- **`accel/_modes.py`** (`MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`) — the *canonical* (screening)
  registry. **Do not touch it in this skill.** It is not additive: `MODE_ATTRS` feeds
  `CANONICAL_MODES`, which `@_bind_batch_aligners` walks at import time calling
  `getattr(accel.batch, "_align_batch_<mode>")`. That aligner does not exist until the accel skill
  builds it, so adding your mode here makes `import shepherd_score.container` raise. On top of that,
  `tests/test_mode_registry.py` pins `len(CANONICAL_MODES) == 7`, `set(MODE_SEEDS) ==
  set(CANONICAL_MODES)`, `set(MODE_STEPS) == set(CANONICAL_MODES)`, and a batch-bind for every
  canonical mode — all of which fail the moment you add a mode with no aligner.

Because your mode is not in `MODE_SEEDS` / `MODE_STEPS`, you also cannot use `_default_seeds` /
`_default_steps` for its defaults (they read those tables) — give `align_with_<mode>` literal
defaults instead. The accel skill promotes the mode into `accel/_modes.py` and moves the defaults
there once the batched aligner exists. `PROCESS_MODES` and (in `accel/batch/_dispatch.py`)
`_MODE_SPEC` are likewise accel-skill territory.

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
