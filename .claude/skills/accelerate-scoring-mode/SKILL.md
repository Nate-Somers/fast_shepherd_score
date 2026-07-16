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
`alignment/_torch.py`, its test, and a registry entry in `accel/_modes.py`. Your job is to make
it fast — write the GPU and CPU kernels and wire them into the batched screening path — while
keeping the diff clean and minimal and reproducing the reference's numbers exactly.

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

### 2. Write the numba CPU kernel first
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

### 7. (Optional) multi-GPU / CPU-pool path
Only if the mode should run across multiple GPUs or the CPU process pool: add a `_MODE_SPEC`
entry in `accel/batch/_dispatch.py` (declaring how to extract inputs as numpy, rebuild device
tensors in a worker, and read results back) **and** add the mode to `PROCESS_MODES` in
`accel/_modes.py`. A registry test asserts `tuple(_MODE_SPEC) == PROCESS_MODES`, so these two
edits are a pair — never do one without the other. If the mode does not need this, skip both; it
runs single-GPU/in-process and that is fine.

### 8. Public batched API
Add `MoleculePairBatch.align_with_<mode>(backend=...)` in `container/_batch.py`. Default
`backend=None` and resolve it device-aware (Triton on CUDA else numba) via the existing resolver
— do not hard-default to a single backend.

### 9. Do not touch the front-end
`screen.py`, `accel/multi_gpu.py`, and `accel/cpu_pool.py` derive the mode set from `accel/_modes.py`. If you
find yourself editing their mode logic, stop — you have missed the registry-driven path. The only
registry edits you make are `PROCESS_MODES` (step 7, if applicable).

### 10. Validate against all four parity gates
See `parity_gates.md`. All four must pass before you declare the mode accelerated:
numba ≡ autograd reference, Triton ≡ numba, batched ≡ per-pair, self-copy = 1.000.

## Minimality discipline
- **Derive, never hardcode** a mode list. If you write `["vol", "surf", ...]` anywhere outside
  `accel/_modes.py`, you are doing it wrong.
- **Two kernels, identical signatures.** The dispatch wrapper only works if the numba and Triton
  kernels are drop-in interchangeable.
- **Match the surrounding kernel idiom** — `tl.exp2`, autotune, the shared `dR/dq` tail, the
  `_graphed` loop. A new mode should read like the modes already there.
- **Small diff.** You are adding one kernel pair, one driver, one aligner, one API method, and at
  most one `_MODE_SPEC`/`PROCESS_MODES` pair. If the diff is larger than that, question it.

See `seams.md` for the file map, `kernel_anatomy.md` for the kernel/dispatch/graph mechanics, and
`parity_gates.md` for the validation contract.
