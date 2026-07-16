# Kernel anatomy

How the accel kernels, dispatch, and driver fit together. Read this alongside an existing family's
kernel + driver before writing your own.

## The value+gradient kernel

Each mode has one fused kernel that computes, in a single launch, both the overlap value and its
gradient with respect to the SE(3) pose. It does not return an intermediate the host has to
differentiate — the analytic gradient is computed in-register.

- **Inputs**: padded batch tensors (reference and fit positions / charges / pharm types+anchors+
  vectors, as the mode needs), the current unit quaternion `q` and translation, and the real
  per-pair counts used to mask padding.
- **Outputs**: the scalar overlap per pair and `dO/dq` (gradient in unit-quaternion space) per
  pair, plus the translation gradient.

### Emit `dO/dq` in-register (no host projection tail)
Older code computed a gradient in rotation-matrix space and then projected it back to the
quaternion on the host (`R → q` chain rule, normalization Jacobian). Do not do that. Compute
`dO/dq` directly inside the kernel by reusing the shape channel's validated `dR/dq` tail:

- **Positional term**: force ⊗ fit-anchor, fed through the `dR/dq` tail.
- **Directional term** (if the mode weights orientation vectors): Σ over the vector pairs of the
  per-feature coefficient times (ref-vec ⊗ fit-vec), through the same tail.

`q` is kept unit each optimizer step (Adam renormalizes), so pass it as a unit quaternion and the
kernel does not need to carry the full normalization Jacobian in the hot loop.

### Combining channels
A combined mode (e.g. shape + color) is a weighted sum in the same quaternion space:
`dO/dq = (1 - w) * (scale_shape * dQ_shape) + w * (scale_color * dQ_color)`.
Because projection is linear, scaling each channel's `dO/dq` and summing is identical to
projecting the combined rotation-space gradient — but cheaper and done in one place. If both
channels are computed in one kernel, flip the gradient flag so value and gradient come out of the
same launch rather than two.

## Dispatch (`accel/kernels/dispatch.py`)

Each kernel name is exported as a thin wrapper that routes a *call* to the implementation matching
the **device of its tensor arguments**:
- CUDA tensors → the Triton kernel; CPU tensors → the numba kernel.
- The choice is per-call, never frozen at import. One process must be able to run both (that is
  what lets `backend="numba"` run CPU tensors on a box where Triton also imports).
- Triton source modules are imported lazily — only the first time a CUDA tensor is dispatched — so
  importing `dispatch` on a CPU-only box (no Triton installed) never touches them.

The numba and Triton kernels must therefore have **identical signatures**; the wrapper cannot
adapt between two different calling conventions.

## The driver and the CUDA-graph fine loop

The batched driver (`accel/drivers/<mode>.py`) runs a coarse pass over all SO(3) seeds, prunes to
survivors, then a fine pass. The fine loop is shared across modes in `drivers/_graphed.py`:
capture the per-step kernel+optimizer update once as a CUDA graph, then replay it. Reuse it;
do not write a per-mode graph loop.

- **Blocked early-stop**: match the reference's step schedule (plus a small margin) rather than
  stopping per-pair, so the graphed loop does not over- or under-run relative to the eager
  reference — that keeps gate 3 (batched ≡ per-pair) tight.
- Use the shared helpers in `drivers/_common.py` for seed generation, padding, and result
  extraction so your driver stays small.

## numba specifics
- Call `torch.set_num_threads(1)` around the numba path — numba parallelizes the kernel itself, and
  leaving Torch's thread pool active oversubscribes cores and slows it down.
- An SoA / fp32 kernel path (`cpu_soa.py`) exists for throughput; it changes precision, so gate it
  and validate it against the same reference. The plain `cpu.py` kernel stays the correctness
  baseline.

## Padding and masking
Pad per-pair inputs to a common width for batching, but mask padded slots by the **real count**,
not by a sentinel type value. Masking by a magic type index is fragile: it breaks the moment the
type table is reordered. Carry `N_real` / `M_real` and mask on those.
