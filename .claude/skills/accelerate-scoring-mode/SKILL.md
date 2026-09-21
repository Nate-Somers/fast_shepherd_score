---
name: accelerate-scoring-mode
description: >-
  Take a correct but slow reference alignment mode in shepherd_score (an eager optimizer produced
  by `design-scoring-mode`) and give it the fast backend: a ModeSpec in the mode registry, a
  Channel row if it reads per-molecule data the library does not carry, and matched Triton/numba
  value+gradient kernels only where the math is genuinely new — validated against the reference.
  Use when a mode runs correctly per-pair but needs to screen at 10k-100k alignments/second.
---

# Accelerate a scoring mode

You are given a working reference mode: an eager optimizer in `alignment/_torch.py` (or a reuse of
an existing one), its test, and its result slots registered in `_ALIGN_KEYS`. It is **not** yet in
`accel/_modes.py` — promoting it is your job.

## The one thing to understand first

**A mode is DATA, not code.** `accel/_modes.py` holds a `ModeSpec` per mode: the per-molecule
channels it reads, the objective terms it optimises, how each term reduces and how they blend, and
its optimiser schedule. Everything mode-shaped downstream reads that spec:

| Layer | File | Per-mode code you write |
|---|---|---|
| batched pairwise aligner | `accel/batch/aligners.py` | **none** — `_align_batch_<mode>` is generated |
| fine loop (eager, CUDA-graph, fused CPU) | `accel/drivers/engine.py` | **none** |
| array-native screen aligner | `accel/batch/_arrays.py` | **none** |
| store schema / profile / shard I/O | `shepherd_score/screen.py` | **none** |
| query + fit tensor plumbing | `shepherd_score/screen.py` | **none** |
| process-per-GPU / CPU-pool spec | `accel/batch/_dispatch.py` | **none** |

So a mode that reuses existing kernels and existing per-molecule data is **one `ModeSpec` plus one
API method**. It inherits, automatically and on the day it lands: the batched aligner, adaptive
bucketing, memory-safe sub-batching, the CUDA-graph fine loop, the fused numba CPU fine loop, the
array-native screen path, the multi-GPU screen, the process pool, and a canonical store's constant
seeds where its seed channel allows.

If you find yourself writing a per-mode driver, a per-mode aligner, a per-mode array builder or a
per-mode `screen.py` branch, **stop**: that is the shape the refactor removed, and reintroducing it
takes the mode back off the derived path.

## What "fast" means here

The reference optimizer is autograd over one pair at a time. The accel layer instead:

- computes **value and gradient in a single hand-written kernel** (Triton on CUDA, numba on CPU),
- emits the gradient directly in **unit-quaternion space** (`dO/dq`) in-register, so there is no
  host-side chain-rule tail,
- runs one **batched coarse-to-fine** loop over many pairs, with a shared CUDA-graph fine loop and
  a fused numba CPU twin,
- exposes it through `MoleculePairBatch.align_with_<mode>(backend=...)`,
- and aligns straight from the store's contiguous arrays when screening.

Despite the "coarse-to-fine" naming there is **no coarse grid on the default path**: every seed
goes into the fine loop and the per-pair maximum is taken. The coarse grid runs only when
`trans_init=True`. Ranking seeds on raw un-optimized overlap repeatedly discarded the true basin
for pseudo-symmetric molecules.

## The oracle

The reference optimizer is your ground truth. Never "fix" a parity failure by changing the
reference — the reference is correct by construction; the new wiring is what is under test.

Be precise about what parity means, because the honest tolerances are looser than they look
(`parity_gates.md`): the accelerated seeder uses a **different seed set** from the reference by
design, so agreement is at the *basin* level, not bitwise.

## Progress checklist

```
Mode acceleration progress:
- [ ] 1. Read the gradient structure; name each channel and each term
- [ ] 2. Do you need a new KERNEL?  (usually NO — 21 modes run on 6 kernels)
- [ ] 3. numba CPU kernel      | only if step 2 says yes
- [ ] 4. Triton GPU twin       | only if step 2 says yes
- [ ] 5. Dispatch wrapper      | only if step 2 says yes
- [ ] 6. Do you need a new CHANNEL? (only if the mode reads per-molecule data nobody stores)
- [ ] 7. The ModeSpec in accel/_modes.py  <- the actual work
- [ ] 8. MoleculePairBatch.align_with_<mode>(backend=None)
- [ ] 9. Bump the registry count 21 -> 22 in tests/test_mode_registry.py
- [ ] Gates: 1 kernel≡ref · 2 Triton≡numba · 3 batched≡per-pair · 4 self=1.000 · 5 screen≡batch
```

`EARLY EXIT`: if step 2 and step 6 are both "reuse", you are writing **one spec, one API method
and one test-count bump**. That is the normal outcome.

## Steps

### 1. Read the reference's gradient structure

Decompose the objective into TERMS. Each term is one kernel launch per fine step over one pair of
channels, with a reduction (`tanimoto`, `tversky`, `raw`, `agreement`, `pharm_sim`) and a blend
weight. A two-channel blend is two terms whose weights are `w` and its complement; a penalty is a
term with a negative weight and the `raw` reduction.

**Separate the two "nearest modes".** For a blended mode, the mode whose **term structure** you copy
is usually not the one whose **field kernel** you reuse. A mode that is `(1−w)·shape + w·<scalar
field>` has two terms, and its field term runs on the **ESP kernel** (a signed scalar field over
atoms), not a pharmacophore kernel. Do not assume "looks like `vol_esp`" means "reuse `vol_esp`":
`vol_esp` is a single ESP term with no shape term at all.

### 2. Do you need a new kernel? Almost certainly not

Twenty-one modes are served by **six** kernels. Check whether one already emits your channel's
value and gradient:

| Channel | `Term.kernel` | Module |
|---|---|---|
| shape (Gaussian volume) | `shape` | `kernels/shape_triton.py` + `kernels/cpu.py` |
| signed scalar field over points | `esp` | `kernels/esp_triton.py` + `kernels/cpu.py` |
| ShaEP surface-ESP agreement (value only) | `esp_cmp` | same |
| pharmacophore, directional | `pharm` | `kernels/pharm_triton.py` + `kernels/cpu.py` |
| pharmacophore/element, directionless | `color` | same |
| hard-sphere excluded volume | `avoid` | `kernels/avoid_triton.py` + `kernels/cpu.py` |

**Feeding a new per-atom scalar where the ESP kernel expects charges is a reuse.** That one move
covers `vol_lipo`, `vol_mr` and `vol_fukui`. **A Tversky variant is a `reduction=` string**, not a
kernel and not a driver. **An element-identity channel is the `color` kernel with element tables**
(`Term(tables="element")`).

Only one mode in recent history needed genuinely new channel math: `vol_avoid`, whose relu-hinge
penalty no Gaussian kernel computes. If you conclude you need a kernel, state which existing kernel
you rejected and why, then do steps 3–5. Otherwise skip to step 6.

### 3–5. The kernel trio *(new math only)*

Write the **numba CPU kernel first** (`accel/kernels/cpu.py`), because CPU is easier to debug;
validate it against the reference (gate 1). Then the **Triton twin** with an *identical call
signature* (`accel/kernels/<family>_triton.py`), validated against numba (gate 2). Then the
**dispatch wrapper** (`accel/kernels/dispatch.py`, one `_make(name, triton_tag)` line): routing is
per call by the device of the first tensor argument, never frozen at import.

Then teach `drivers/terms.py` to call it: one `if term.kernel == "<yours>":` branch in `evaluate`,
one in `self_overlap` if the reduction needs a self-overlap, and one closure in
`kernels/cpu_fused.py::_term_closure` so the fused CPU loop can run it. Three small branches — see
`kernel_anatomy.md`.

### 6. Do you need a new channel?

A **channel** is one named per-molecule array: `accel/channels.py` holds the table. Add a row only
if your mode reads data no channel already carries. The row states how to read it off a `Molecule`,
which pair tensor attribute carries it, how the store persists it (array key, offset table or
dense, schema flag), whether it rotates/translates under the canonical frame, and what a padded
slot holds.

```python
_reg(Channel("tpsa_pos", "points", "tpsa", "tpsa_pos",
             _read_accessor("get_tpsa_positions", "TPSA centres"), "tpsa_pos", flag="tpsa"))
_reg(Channel("tpsa", "scalar", "tpsa", "tpsa",
             _read_accessor_noH("get_tpsa", "TPSA"), "tpsa", flag="tpsa"))
```

Then three small additions, all in one place each:

- a **basis** entry in `channels.BASES` (its offset-table key) if the data is not 1:1 with an
  existing basis. **Give a per-atom field its own table**: `atom_pos` is the `Chem.RemoveHs` set and
  is longer than the true-heavy set whenever RemoveHs retained an isotope-labelled H, so a shared
  table desyncs the field from its positions on exactly the molecules hardest to notice;
- the flag in `channels.SCHEMA_FLAGS`;
- a row in `screen.py::_BASIS_TABLE` (`basis, flag, offset key, [(profile attr, store key)]`) and
  the matching `MoleculeProfile` slots + `get_<data>()` accessors **named exactly like the
  `Molecule` methods**, so a profile duck-types into the aligner identically.

Extraction, pre-centring, the canonical rotation, `_concat`, `_reconstruct`, the query tensors, the
fit tensors and the array builder are then all derived from those rows. In particular you do **not**
hand-write the canonical rotation any more — `_profile_from_schema` rotates every channel whose
`kind` is `points` or `vectors`. That was the single most dangerous manual step: getting it wrong
left scores right and `Hit.transform` silently wrong.

### 7. Write the ModeSpec — the actual work

```python
_reg(ModeSpec("vol_tpsa", ("transform_vol_tpsa", "sim_aligned_vol_tpsa"), 16, 50, 2,
              seed_channel="atoms", channels=("atoms", "tpsa_pos", "tpsa"), bucket=("atoms",),
              terms=_field_blend("tpsa_pos", "tpsa", "tpsa_weight"),
              params={"alpha": 0.81, "lam": 0.1, "tpsa_weight": 0.5, "lr": 0.075},
              graph_budget=30_000_000, screen_lr=0.1))
```

Read the dataclass docstrings in `accel/_modes.py` before filling these in; the fields that decide
behaviour rather than describing it are `seed_channel` (which decides whether a canonical store's
constant seeds apply), `bucket` + `work` (the cost model), `graph_budget`, `cpu_fused`,
`lam_scaling`, `screen_lr` and `honors_num_repeats`.

> **Say where your seed/step numbers came from.** `_modes.py` records that only one of the 21
> entries has measured data behind it; the rest were inherited from a sibling. Inheriting is a
> reasonable starting point — state in the commit that you did, rather than presenting an
> unmeasured constant as a measured knee.

### 8. Public batched API

Add `MoleculePairBatch.align_with_<mode>(backend=None, return_aligned=False)` in
`container/_batch.py` — copy `align_with_vol_mr`, the cleanest instance. Resolve `num_repeats` /
`max_num_steps` from the registry via `_default_seeds` / `_default_steps`, then call
`_run_fast_or_fallthrough(...)`, falling back to `_delegate_alignment` for the JAX/unknown path.
`backend=None` resolves device-aware; do not hard-default.

You do **not** bind `_align_batch_<mode>` onto `MoleculePair` by hand: the `@_bind_batch_aligners`
decorator in `container/_core.py` walks `CANONICAL_MODES` at import. That is also why step 7 must
come after the aligner exists — and it always does now, because the aligner is generated from the
spec in the same module import.

Now switch the reference `MoleculePair.align_with_<mode>`'s literal seed/step defaults to
`_default_seeds` / `_default_steps` so the per-pair and batched paths share one source.

### 9. The one hardcoded thing left

`tests/test_mode_registry.py` pins `len(CANONICAL_MODES)` as a literal. Bump it and update the
trailing comment. That is deliberate: it is the tripwire that makes adding a mode a visible act.

Everything else in the test suite is derived — the parity tests parametrize over
`CANONICAL_MODES` / `screen._ARRAY_MODES`, the store fixture is built for the whole tuple, the
required-kwarg table is read off the specs, and the self-copy exclusion is read off the terms. Your
mode is covered on the next run without touching them. If you find yourself editing a mode-name
list in a test, check first whether it should be derived.

### 10. Validate against the gates

See `parity_gates.md`. Gates 1–4 before you declare the mode accelerated, plus gate 5 if it screens.

## Deliberately NOT screen-wired

Every registry mode screens now, including `vol_avoid`, whose avoid cloud is carried with the query
(`screen(..., avoid_points=...)`) rather than stored per molecule. If your mode genuinely cannot
screen, make `_store_supports` say so explicitly with a comment rather than letting it fall
through — a reader otherwise cannot tell "deliberately pairwise-only" from "someone forgot".

## Minimality discipline

- **Derive routing; add data.** Never hardcode a mode-name list to decide which modes exist, where
  they dispatch, what the store keeps or which tensors a mode reads. The tell: editing a list of
  mode names is almost always wrong now; adding a `ModeSpec` or a `Channel` row is right.
- **One dispatch table, not two.** `screen.py::_array_dispatch` is the single selector for both the
  in-process driver and the multi-GPU worker. A second table is what made `screen(ndev>1)` return
  `vol`'s answers under every mode's name.
- **Two kernels, identical signatures.** The dispatch wrapper cannot adapt between calling
  conventions.
- **Small diff.** Typically: no kernel, no channel, one spec, one API method, one count bump. If the
  diff is much larger, question it.
- **Keep the tests derived.** Drive new tests off `CANONICAL_MODES` / `_ARRAY_MODES`, never a fresh
  hardcoded list. Three hardcoded lists once let six modes ship with zero coverage.

See `seams.md` for the file map, `kernel_anatomy.md` for kernel/engine/graph mechanics,
`screen_wiring.md` for what the store still needs from you, and `parity_gates.md` for the validation
contract. `evals/` holds grading rubrics — they state expected answers, so they are for reviewing
work, not for doing it.
