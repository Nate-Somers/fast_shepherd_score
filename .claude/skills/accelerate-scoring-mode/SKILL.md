---
name: accelerate-scoring-mode
description: >-
  Take a correct but slow reference alignment mode in shepherd_score (an eager
  `optimize_<mode>_overlay` produced by `design-scoring-mode`) and build the fast backend:
  matched Triton (GPU) and numba (CPU) value+gradient kernels, a batched coarse-to-fine driver,
  and the `MoleculePairBatch` API — validated bit-for-bit against the reference. Use when a mode
  runs correctly per-pair but needs to screen at 10k-100k alignments/second.
---

# Accelerate a scoring mode

You are given a working reference mode: an eager `optimize_<mode>_overlay` in
`alignment/_torch.py`, its test, and its result slots registered via `_ALIGN_KEYS` in
`container/_core.py`. It is **not** yet in `accel/_modes.py` — promoting it to a canonical
screening mode is part of your job. Make it fast — write the GPU and CPU kernels, wire them into
the batched path, and register the mode — while keeping the diff clean and minimal and reproducing
the reference's numbers exactly.

## What "fast" means here

The reference optimizer is autograd over one pair at a time. The accel layer instead:
- computes **value and gradient in a single hand-written kernel** (Triton on CUDA, numba on CPU),
- emits the gradient directly in **unit-quaternion space** (`dO/dq`) in-register, so there is no
  host-side chain-rule/projection tail,
- runs a **batched coarse-to-fine** loop over many pairs at once, with a CUDA-graph fine loop,
- exposes it through `MoleculePairBatch.align_with_<mode>(backend=...)`.

## The oracle

The reference `optimize_<mode>_overlay` is your ground truth. You are done when your kernels
reproduce it to tolerance (see `parity_gates.md`). Never "fix" a parity failure by changing the
reference — the reference is correct by construction; the kernel is what is under test.

## Steps

### 1. Read the reference's gradient structure
Before writing a kernel, understand how the reference objective's SE(3) gradient decomposes:
which channels contribute, and for each channel what the `dO/dq` term looks like. A combined mode
is a weighted sum of per-channel gradients, each in the same quaternion space. Reuse the shape
channel's `dR/dq` tail — it is already validated and every mode's positional gradient shares it.

**Separate the two "nearest modes".** For a *blended* mode, the existing mode whose **driver /
combining structure** you copy is usually NOT the one whose **field physics / kernel** you reuse.
A mode that is `(1-w)·shape + w·<scalar-field>` has the *driver* shape of a two-channel
combined-gradient mode (e.g. `vol_color`), but its field *kernel* is the ESP kernel (a signed
scalar field over atoms), not `vol_color`'s pharmacophore kernel. Do **not** assume "looks like
`vol_esp`" means "reuse `vol_esp`": `vol_esp` optimizes the field **alone**, so its driver cannot
produce a shape+field blend. Identify the driver-analog and the kernel-analog independently.

**Do you even need a new kernel?** If every channel's value+gradient is already computed by an
existing kernel — shape via `overlap_score_grad_se3_batch`, a signed scalar field via the ESP
kernel `overlap_score_grad_esp_se3_batch`, pharmacophores via the pharm kernel (all already
device-dispatched) — then **reuse them and write NO new kernel**: skip steps 2–4 entirely and go
straight to the driver (step 5), which blends the per-channel `dQ`s
(`g_q = (1-w)·(-scale_a·dQ_a) + w·(-scale_b·dQ_b)`). Feeding a new per-atom scalar (e.g.
lipophilicity) in where the ESP kernel expects charges is a *reuse*, not a new kernel. Steps 2–4
below are only for a mode that introduces genuinely new channel math.

### 2. Write the numba CPU kernel first *(only if the mode needs new channel math — see step 1)*
CPU is easier to debug than Triton, so start there. Add the value+grad kernel to
`accel/kernels/cpu.py` (or `cpu_fused.py` / `cpu_soa.py` if it fits an existing fused/SoA path).
It must return the same value the reference computes **and** the analytic `dO/dq`. Validate it
against autograd on the reference immediately (parity gate 1) before moving on — do not write the
Triton kernel against an unvalidated CPU kernel.

### 3. Write the Triton GPU twin
Add the matching kernel to `accel/kernels/<family>_triton.py` with an **identical call
signature** to the numba kernel. Use `tl.exp2` (not `tl.exp`) and `@triton.autotune` to match
the surrounding kernels. Validate Triton against numba (parity gate 2).

### 4. Register the dispatch wrapper
Add the thin routing wrapper in `accel/kernels/dispatch.py`. It picks Triton vs numba **by the
device of the tensor arguments, per call** — never frozen at import time, so one process can run
both paths. Import the Triton source module lazily (only when a CUDA tensor is first dispatched)
so a CPU-only box never touches it.

### 5. Write the batched driver
Add `accel/drivers/<mode>.py`, modeled on the existing driver of the nearest family (`shape.py`,
`esp.py`, `esp_combo.py`, `pharm.py`). It runs the coarse-to-fine schedule, calls your kernel via
the dispatch wrapper, and uses the shared CUDA-graph fine loop in `drivers/_graphed.py` and the
helpers in `drivers/_common.py`. Do not reimplement the graph loop.

### 6. Wire the batched aligner
Add `_align_batch_<mode>(pairs, ...)` to `accel/batch/aligners.py`: pad the per-pair inputs into
batch tensors, call the driver, and write `transform_<mode>` / `sim_aligned_<mode>` back onto each
pair. Bind it onto `MoleculePair` following the binding block at the bottom of that module, and
export it from `accel/batch/__init__.py`.

### 7. Promote the mode to canonical (`accel/_modes.py`)
Now that `_align_batch_<mode>` exists, register the mode in the canonical registry — this is what
makes `screen` / `multi_gpu` / `cpu_pool` pick it up. Add one row each to `MODE_ATTRS` (the
`(transform_<mode>, sim_aligned_<mode>)` pair), `MODE_SEEDS`, and `MODE_STEPS`, choosing balanced
seed/step defaults at the accuracy/throughput knee. Then bump the hardcoded `len(CANONICAL_MODES)`
count in `tests/test_mode_registry.py`; the `@_bind_batch_aligners` walk now resolves because your
aligner exists.

This is exactly the step the reference skill could not do — adding a mode to `MODE_ATTRS` before
its aligner exists makes `import shepherd_score.container` raise (the decorator walks
`CANONICAL_MODES` calling `getattr(accel.batch, "_align_batch_<mode>")`). Do it here, after step 6,
and the import stays green. Now that `MODE_SEEDS` / `MODE_STEPS` carry the defaults, you may switch
the reference `align_with_<mode>`'s literal seed/step defaults over to `_default_seeds` /
`_default_steps` so the per-pair and batched paths share one source.

### 8. (Optional) multi-GPU / CPU-pool path
Only if the mode should run across multiple GPUs or the CPU process pool: add a `_MODE_SPEC`
entry in `accel/batch/_dispatch.py` (declaring how to extract inputs as numpy, rebuild device
tensors in a worker, and read results back) **and** add the mode to `PROCESS_MODES` in
`accel/_modes.py`. A registry test asserts `tuple(_MODE_SPEC) == PROCESS_MODES`, so these two
edits are a pair — never do one without the other. If the mode does not need this, skip both; it
runs single-GPU/in-process and that is fine.

### 9. Public batched API
Add `MoleculePairBatch.align_with_<mode>(backend=...)` in `container/_batch.py`. Default
`backend=None` and resolve it device-aware (Triton on CUDA else numba) via the existing resolver
— do not hard-default to a single backend.

### 10. Wire the mode into `screen` (the out-of-core store) — REQUIRED, do not skip
The registry (`accel/_modes.py`) makes `screen` / `multi_gpu` / `cpu_pool` **dispatch** your mode:
resolve its name, result attributes, and seed/step defaults. It does **not** teach the on-disk
store what per-molecule arrays your mode *reads*. `screen` streams a library through
`MoleculeProfile` (an RDKit-free, arrays-only stand-in for `Molecule`) and `ProfileStore` (sharded
`npz`). If your mode needs data the store does not persist, `screen` cannot serve it — the pairwise
`MoleculePairBatch` API is not enough. This is the step that is easy to miss: the in-memory path
works, the mode imports, tests pass, and yet `screen(..., mode="<yours>")` fails or feeds zeros.
Decide your mode's tier:

**Tier A — reuses data the store already holds.** A reduction over an existing channel (e.g.
`vol_tversky` is a different reduction of the same heavy-atom shape overlap; `atom_pos` is *always*
stored). Edit **one** function: `_store_supports(schema, mode)` in `screen.py`, returning the right
predicate (`return True` for a shape-only mode). Nothing else — it routes through the `fast=False`
`MoleculePairBatch` path automatically.

**Tier B — needs new per-molecule DATA the store doesn't carry** (e.g. `esp_field`'s
variable-length signed ESP field points). Wire it end to end in `screen.py`, mirroring the
`pharm` / `field_points` plumbing already there:
  1. **`MoleculeProfile`** — add the array(s) to `__slots__`; accept them in `__init__` (store via
     `_f32`); shift the positional ones in `center_to` (they move rigidly with the molecule); add a
     `get_<data>()` accessor **named exactly like the `Molecule` method the batched driver calls**
     (`_batch_upload(pairs, ..., lambda p: p.<side>_molec.get_<data>()[k], ...)`), so a profile
     duck-types into the aligner identically to a `Molecule`.
  2. **`_schema_from_modes`** — add a boolean schema flag set when your mode is requested.
  3. **`_store_supports`** — return `schema.get("<flag>", False)`.
  4. **`_profile_from_schema`** — extract the data off the `Molecule`, pre-center it alongside
     `atom_pos` when `pre_center`, and pass it into the returned `MoleculeProfile(...)`.
  5. **`ProfileStore._concat`** (pack to `npz`) — fixed-width array → `np.stack`; a **variable-length**
     set (per-molecule count differs) → write an **offset table** (`<name>_off = offsets(lens)`)
     plus the concatenated points, mirroring the `pharm_off` block exactly.
  6. **`ProfileStore._reconstruct`** (unpack) — read the offset table and slice each molecule's
     segment back out, mirroring the pharm unpack.
  7. If the data is **cached on the `Molecule`** (e.g. a lazy field), mirror the `center_to` shift in
     `container/_core.py` too, so a query centered by `screen` never carries a stale copy.

**Fast-engine registration is OPTIONAL and orthogonal.** `_FAST_MODES` in `screen.py` lists the
modes that get the direct array→kernel fast path (build fit tensors once per shard, bypass
`MoleculePair`). That is a **throughput** optimization, not a correctness requirement. A mode left
out of `_FAST_MODES` still screens correctly through the `fast=False` object path. Add it (with a
`_build_fit_fast_pairs` / `_ref_tensors_from_arrays` branch) only once the mode is proven and you
want the streaming speed — never as part of first wiring.

**Verify (screen analog of parity gate 3).** Build a `ProfileStore` for the mode, `screen()` a
query, and assert the scores match `MoleculePairBatch.align_with_<mode>` on the same centered
molecules. Add the test to `tests/test_screen.py` (models: `test_vol_tversky_stream_matches_object`
for Tier A, `test_esp_field_stream_matches_object` for Tier B — the latter also asserts the
variable-length arrays reached disk).

`multi_gpu` / `cpu_pool` remain registry-driven for *routing* — never hardcode a mode-name list in
their dispatch. The per-mode DATA plumbing above is inherently mode-specific and cannot be derived;
that is the one place `screen.py` legitimately grows per mode.

### 11. Validate against all four parity gates
See `parity_gates.md`. All four must pass before you declare the mode accelerated:
numba ≡ autograd reference, Triton ≡ numba, batched ≡ per-pair, self-copy = 1.000. Plus the
step-10 screen round-trip (streamed ≡ `MoleculePairBatch`) if the mode is meant to screen.

## Minimality discipline
- **Derive mode *routing*, never hardcode a mode-name list.** If you write `["vol", "surf", ...]`
  to decide *which modes exist / dispatch where*, you are doing it wrong — that comes from
  `accel/_modes.py`. This does **not** forbid the per-mode DATA plumbing in step 10: a mode's
  `_store_supports` branch and its `MoleculeProfile` / serialization fields are inherently
  mode-specific and legitimately name the mode.
- **Two kernels, identical signatures.** The dispatch wrapper only works if the numba and Triton
  kernels are drop-in interchangeable.
- **Match the surrounding kernel idiom** — `tl.exp2`, autotune, the shared `dR/dq` tail, the
  `_graphed` loop. A new mode should read like the modes already there.
- **Small diff.** You are adding one kernel pair, one driver, one aligner, the canonical registry
  rows (`MODE_ATTRS` / `MODE_SEEDS` / `MODE_STEPS`) plus the pinned-count bump, one API method, the
  step-10 `screen` wiring (one `_store_supports` line for Tier A; the `MoleculeProfile` + schema +
  serialization fields for Tier B), and at most one `_MODE_SPEC`/`PROCESS_MODES` pair. If the diff
  is larger than that, question it.

See `seams.md` for the file map, `kernel_anatomy.md` for the kernel/dispatch/graph mechanics, and
`parity_gates.md` for the validation contract.
