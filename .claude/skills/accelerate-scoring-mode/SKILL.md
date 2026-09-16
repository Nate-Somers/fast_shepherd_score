---
name: accelerate-scoring-mode
description: >-
  Take a correct but slow reference alignment mode in shepherd_score (an eager optimizer produced
  by `design-scoring-mode`) and build the fast backend: matched Triton (GPU) and numba (CPU)
  value+gradient kernels where new math is genuinely needed, a batched coarse-to-fine driver, the
  `MoleculePairBatch` API, and the array-native screening path — validated against the reference.
  Use when a mode runs correctly per-pair but needs to screen at 10k-100k alignments/second.
---

# Accelerate a scoring mode

You are given a working reference mode: an eager optimizer in `alignment/_torch.py` (or a reuse of
an existing one), its test, and its result slots registered in `_ALIGN_KEYS`. It is **not** yet in
`accel/_modes.py` — promoting it to a canonical screening mode is part of your job.

## What "fast" means here

The reference optimizer is autograd over one pair at a time. The accel layer instead:

- computes **value and gradient in a single hand-written kernel** (Triton on CUDA, numba on CPU),
- emits the gradient directly in **unit-quaternion space** (`dO/dq`) in-register, so there is no
  host-side chain-rule tail,
- runs a **batched coarse-to-fine** loop over many pairs, with a shared CUDA-graph fine loop,
- exposes it through `MoleculePairBatch.align_with_<mode>(backend=...)`,
- and, for a screening mode, aligns straight from the store's contiguous arrays.

Despite the "coarse-to-fine" naming there is **no coarse grid on the default path**: every seed
goes into the fine loop and the per-pair maximum is taken. The coarse grid runs only when
`trans_init=True`. Ranking seeds on raw un-optimized overlap repeatedly discarded the true basin
for pseudo-symmetric molecules.

## The oracle

The reference optimizer is your ground truth. Never "fix" a parity failure by changing the
reference — the reference is correct by construction; the kernel is what is under test.

Be precise about what parity means here, because the honest tolerances are looser than they look
(`parity_gates.md`): the accelerated seeder uses a **different seed set** from the reference by
design, so agreement is at the *basin* level, not bitwise.

## Progress checklist

```
Mode acceleration progress:
- [ ] 1. Read the gradient structure; separate driver-analog from kernel-analog
- [ ] 2. Do you even need a new kernel? (usually NO — 17 of 21 modes reuse one)
- [ ] 3. numba CPU kernel      | only if step 2 says yes
- [ ] 4. Triton GPU twin       | only if step 2 says yes
- [ ] 5. Dispatch wrapper      | only if step 2 says yes
- [ ] 6. Batched driver (reuse the shared CUDA-graph fine loop)
- [ ] 7. _align_batch_<mode> in aligners.py + re-export from batch/__init__.py
- [ ] 8. Promote to canonical (accel/_modes.py) + bump the registry count 21 -> 22
- [ ] 9. MoleculePairBatch.align_with_<mode>(backend=None)
- [ ] 10. (Optional) worker-process path: paired _MODE_SPEC + PROCESS_MODES
- [ ] 11. Wire into screen — store data, then the ARRAY path — see screen_wiring.md
- [ ] Gates: 1 kernel≡ref · 2 Triton≡numba · 3 batched≡per-pair · 4 self=1.000 · 5 screen≡batch

EARLY EXIT: if steps 3-5 are all "reuse" AND an existing driver already serves your mode
unchanged, you are adding ONE aligner, three registry rows, one API method and the screen
wiring. Skip kernel_anatomy.md except its graph-budget table.
```

## Steps

### 1. Read the reference's gradient structure

Understand how the objective's SE(3) gradient decomposes: which channels contribute, and what each
channel's `dO/dq` looks like. A combined mode is a weighted sum of per-channel gradients in the
same quaternion space. Reuse the shape channel's `dR/dq` tail — it is validated and every mode's
positional gradient shares it.

**Separate the two "nearest modes".** For a blended mode, the mode whose **driver / combining
structure** you copy is usually not the one whose **field kernel** you reuse. A mode that is
`(1−w)·shape + w·<scalar field>` has the driver shape of a two-channel combined-gradient mode, but
its field kernel is the ESP kernel (a signed scalar field over atoms), not a pharmacophore kernel.
Do not assume "looks like `vol_esp`" means "reuse `vol_esp`": `vol_esp` optimizes the field
**alone**, so its driver cannot produce a shape+field blend. Identify the two independently.

### 2. Do you even need a new kernel? Almost certainly not

Twenty-one modes are served by **four** Triton kernel modules. Before writing anything, check whether an
existing dispatched kernel already emits your channel's value and gradient:

| Channel | Kernel | Module |
|---|---|---|
| shape (Gaussian volume) | `overlap_score_grad_se3_batch` | `kernels/shape_triton.py` + `kernels/cpu.py` |
| signed scalar field over atoms | `overlap_score_grad_esp_se3_batch` | `kernels/esp_triton.py` + `kernels/cpu.py` |
| ShaEP surface-ESP comparison (value only) | `esp_comparison_batch` | same |
| pharmacophore, directional | `pharm_grad_dq_se3_batch` | `kernels/pharm_triton.py` + `kernels/cpu.py` |
| pharmacophore, directionless ("color") | `pharm_color_score_grad_se3_batch` | same |
| hard-sphere excluded volume | `overlap_score_grad_avoid_se3_batch` | `kernels/avoid_triton.py` + `kernels/cpu.py` |

**Feeding a new per-atom scalar where the ESP kernel expects charges is a reuse, not a new
kernel.** That single move covers `vol_lipo`, `vol_mr` and `vol_fukui`. A Tversky variant is a
host-side reduction over a parent's kernel output — also not a new kernel.

Only one mode in recent history needed genuinely new channel math: `vol_avoid`, whose relu-hinge
penalty no Gaussian kernel computes. If you conclude you need a new kernel, state which existing
kernel you rejected and why. Then do steps 3–5; otherwise skip straight to the driver.

### 3. Write the numba CPU kernel first *(new math only)*

CPU is easier to debug than Triton. Add the value+grad kernel to `accel/kernels/cpu.py` as an
`@njit(parallel=True, fastmath=True, cache=True)` inner kernel plus a thin torch-facing wrapper.
It must return the same value the reference computes **and** the analytic `dO/dq`. Validate it
against the reference immediately (gate 1) before writing any Triton.

### 4. Write the Triton GPU twin *(new math only)*

Add the matching kernel to `accel/kernels/<family>_triton.py` with an **identical call signature**
to the numba wrapper. Match the surrounding idiom: `@triton.autotune` over the family's config
list, `tl.exp2(x * inv_ln2)` rather than `tl.exp`, and the shared `_quat_to_rotmat` /
`_quat_grad_tail` helpers from `shape_triton.py`. Validate against numba (gate 2).

### 5. Register the dispatch wrapper *(new math only)*

Add the routing wrapper in `accel/kernels/dispatch.py` via its `_make(name, triton_tag)` factory.
Routing is **per call, by the device of the first tensor argument** — never frozen at import — and
Triton source modules are imported lazily, so a CPU-only box never touches them. The wrapper
exports 11 names today; yours makes 12.

### 6. Write the batched driver

Add `accel/drivers/<mode>.py`, modeled on the nearest family. Fourteen driver modules serve 21 modes,
so check first whether an existing driver already takes your mode as a parameterisation —
`drivers/vol_lipo.py` serves `vol_lipo`, `vol_mr` **and** `vol_fukui` unchanged, and
`drivers/vol_tversky.py` serves both `vol_tversky` and `surf_tversky`.

Use the shared CUDA-graph fine loop in `drivers/_graphed.py` (`run_graphed(make, key, inputs, *,
es_patience, es_tol, es_seeds)`) and the helpers in `drivers/_common.py`. Do not reimplement the
graph loop. Every mode but `vol_avoid` uses it; `vol_avoid` opts out because its objective is
piecewise-linear and kinked while the graph's early-stop schedule is tuned for smooth Gaussians.

### 7. Wire the batched aligner

Add `_align_batch_<mode>(pairs, ...)` to `accel/batch/aligners.py`: upload per-pair inputs with
`_batch_upload`, bucket and pad, call the driver, write `transform_<mode>` / `sim_aligned_<mode>`
back onto each pair. Then add an explicit re-export line to `accel/batch/__init__.py`.

You do **not** bind it onto `MoleculePair` by hand: the `@_bind_batch_aligners` decorator in
`container/_core.py` walks `CANONICAL_MODES` at import and binds every one as a static method. That
is also why step 8 must come after this step and not before.

### 8. Promote the mode to canonical

Add one row each to `MODE_ATTRS`, `MODE_SEEDS` and `MODE_STEPS` in `accel/_modes.py`, choosing
seed/step defaults at the accuracy/throughput knee. Then bump the hardcoded count in
`tests/test_mode_registry.py` from 21 to 22 and update its trailing comment.

This is exactly the step the reference skill could not do: adding to `MODE_ATTRS` before the
aligner exists makes `import shepherd_score.container` raise, because the decorator walks
`CANONICAL_MODES` calling `getattr(accel.batch, "_align_batch_<mode>")`.

> **Say where your seed/step numbers came from.** `_modes.py` records that only one of the 21
> entries has measured data behind it; the rest were inherited from a sibling mode. Inheriting is
> a reasonable starting point, but state in the commit that you did, rather than presenting an
> unmeasured constant as a measured knee.

Now that the registry carries the defaults, switch the reference `align_with_<mode>`'s literal
seed/step defaults to `_default_seeds` / `_default_steps` so the per-pair and batched paths share
one source. Eleven shipped modes never got this treatment and their eager and batched defaults
disagree; do not add a twelfth.

### 9. Public batched API

Add `MoleculePairBatch.align_with_<mode>(backend=None, return_aligned=False)` in
`container/_batch.py`. Copy the thin-wrapper template — `align_with_vol_mr` is the cleanest
instance. Resolve `num_repeats` / `max_num_steps` from the registry, then call
`_run_fast_or_fallthrough(...)`, falling back to `_delegate_alignment` for the JAX/unknown path.
`backend=None` resolves device-aware (Triton on CUDA, else numba); do not hard-default.

### 10. (Optional) worker-process path

Only if the mode should run across multiple GPUs via `align_multi_gpu` / `MultiGPUAligner` or the
CPU process pool: add a `_MODE_SPEC` entry in `accel/batch/_dispatch.py` (declaring `extract`,
`tensors` and `out`) **and** add the mode to `PROCESS_MODES` in `accel/_modes.py`. A registry test
asserts `tuple(_MODE_SPEC) == PROCESS_MODES`, so these are a pair — never one without the other.

Only four modes have this today (`vol`, `surf`, `surf_esp`, `pharm`). Skipping it is the normal
outcome; the mode then runs single-GPU or in-process, which is fine. Note that `screen(ndev>1)` is
a **separate** multi-GPU path and is *not* restricted to `PROCESS_MODES`.

### 11. Wire the mode into `screen` — see `screen_wiring.md`

The registry makes `screen` **dispatch** your mode; it does not teach the on-disk store what
per-molecule arrays your mode *reads*, and it does not give you the fast path. This is the step
that is easy to miss: the in-memory path works, the mode imports, tests pass, and
`screen(..., mode="<yours>")` still raises or feeds zeros.

Three things, all detailed with exact edit lists in **`screen_wiring.md`** — read it before
starting:

- **Store data.** *Tier A* reuses arrays the store already holds: one `_store_supports` line.
  *Tier B* needs new per-molecule data: schema flag, `MoleculeProfile` slots and accessors,
  serialization with its own offset table, and — easy to forget — **rotation in the canonical
  block**, since a store is canonical by default whenever it serves `vol`.
- **The array-native path**, which is what `screen` actually runs. Five edits plus one aligner in
  `accel/batch/_arrays.py`. A mode left off it still screens correctly through the object path, but
  that path is 1.3–5× slower on the *identical* GPU kernel, so benchmarking it against the array
  modes understates it. Wire it, or do not plot its screening throughput on the same axis.
- **Nine of the 21 modes are deliberately not screen-wired.** If yours is one of them — a mode
  taking a third non-molecule input cannot be, since the store models molecules — say so with an
  explicit documented `return False` in `_store_supports` rather than letting it fall through.

### 12. Validate against the gates

See `parity_gates.md`. Gates 1–4 must pass before you declare the mode accelerated, plus gate 5
(the screen round-trip) if the mode screens.

## Minimality discipline

- **Derive mode *routing*; add mode *data*.** `_bind_batch_aligners`, `_TRANSFORM_ATTR`,
  `_SCORE_ATTR`, `_seeds_for` / `_steps_for` and `screen_parallel`'s `_ALIGN_ATTR` all derive from
  `accel/_modes.py` — never hardcode a list to decide which modes exist or where they dispatch.
  This does **not** forbid the per-mode data plumbing of step 11, which is inherently
  mode-specific. The tell: editing a mode-name list to decide *dispatch* is wrong; adding a
  `_store_supports` branch or a `MoleculeProfile` field is right.
- **One dispatch table, not two.** `screen.py`'s `_array_dispatch` is the single selector for both
  the in-process driver and the multi-GPU worker. A second table is what made `screen(ndev>1)`
  return `vol`'s answers under every mode's name.
- **Two kernels, identical signatures.** The dispatch wrapper cannot adapt between calling
  conventions.
- **Prefer parameterising an existing driver over a near-copy.** One shared
  `_build_fit_arrays_scalar_field` serves `vol_lipo` and `vol_fukui`; a third such mode should be a
  third line, not a third function.
- **Small diff.** Typically: no new kernel, one driver (or a parameterisation), one aligner, three
  registry rows plus the count bump, one API method, and the step-11 screen wiring. If the diff is
  much larger than that, question it.
- **Keep the tests derived.** Drive new parity tests off `_ARRAY_MODES` / `CANONICAL_MODES` rather
  than a fresh hardcoded list. Three hardcoded lists once let six modes ship with zero coverage,
  and the negative control had silently gone false.

See `seams.md` for the file map, `kernel_anatomy.md` for kernel/dispatch/graph mechanics,
`screen_wiring.md` for step 11, and `parity_gates.md` for the validation contract. `evals/` holds grading
rubrics — they state expected answers, so they are for reviewing work, not for doing it.
