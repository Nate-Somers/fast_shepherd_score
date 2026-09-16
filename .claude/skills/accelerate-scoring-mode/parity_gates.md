# Parity gates

A mode is accelerated when gates 1–4 pass, plus gate 5 if it screens. Each gate isolates one link
in the chain reference → CPU kernel → GPU kernel → batched driver → out-of-core store. Run them in
order: a failure at gate N is almost always a bug introduced at step N, not earlier.

**Read this section before quoting a tolerance.** The honest tolerances are looser than a naive
reading suggests, for one structural reason: the accelerated seeder uses a **different seed set**
from the reference — identity, 4 PCA quaternions, up to 6 structured ±90° axis swaps, then a
Fibonacci fill, where the reference has no structured seeds. So the two do not optimize from the
same starting points and cannot agree bitwise. What you are testing is that they land in the
**same basin**, and the numbers below reflect that.

## Gate 1 — kernel ≡ reference

The kernel's value and gradient must match the eager reference on the same inputs at the same pose.

**If you wrote a new kernel**, compare the kernel's value and `dO/dq` directly against the
reference objective, on several distinct pairs, not just self-pairs. `vol_avoid` — the most recent
new kernel — reached value agreement at 1e-14 and gradient-vs-finite-difference at 1e-9 in double
precision. That is the standard for a genuinely new kernel, and it is achievable because kernel
and reference compute the same math two ways.

**Compare the gradient in the tangent space of the unit quaternion, not raw.** The kernels emit a
**raw** `dO/dq` and do not project out the component parallel to `q`, because the optimizer
renormalizes `q` each step and discards that radial part. Autograd taken through the repo's
*normalizing* quaternion→rotation map is already tangent-projected. A naive raw-vs-autograd
comparison therefore **fails at ~0.4–0.8 even for a correct kernel** — the entire mismatch is the
radial component. Subtract the `q`-parallel part from both before comparing, or differentiate a
non-normalizing map. The tangent component is the physical part that drives the step.

**If you reused existing kernels** — the usual case — there is no new kernel math to validate, and
this gate instead checks that your driver's blend and scaling are right. In practice that is done
at the score level, through gate 3.

**What the suite enforces today, so you can say accurately what you ran.** The standing tests are
the reference-layer autograd-vs-finite-difference check that every mode ships (float32,
`eps=1e-3`, `atol=2e-3`, at a non-identity pose), the tangent-space projection check on the
reference layer's `project_grad_R_to_quaternion` (`atol=rtol=1e-8`, float64, in
`tests/test_analytical_gradients.py`), and the score-level gates below. A numba-kernel-gradient
comparison at the double-precision tolerance above is run when a new kernel is written, not kept
as a standing test. So: if you reused kernels, state that gate 1 holds by inheritance rather than
implying you ran a gradient comparison; if you wrote one, add the comparison as a test.

Kernel-dispatch sanity, which is cheap and is tested: call through `kernels/dispatch.py` and
directly against the concrete module and assert the full returned tuple matches at `torch.allclose`
defaults, then assert both `(name, tag)` pairs landed in `dispatch._RESOLVED` in one process.

## Gate 2 — Triton ≡ numba

The GPU twin must match the CPU kernel.

- **What to compare**: value and `dO/dq` on identical inputs, or end-to-end scores for a
  kernel-reuse mode.
- **Tolerance**: the two are not bit-identical — Triton uses `tl.exp2` where the numba path uses
  `exp` — so expect fp32 agreement. Shipped assertions: `1e-3` on a raw kernel value, `5e-3`
  end-to-end on `vol_and_surf_esp`, `1e-2` end-to-end on `vol_fukui`. Tighter is better; do not
  claim bit-identity.
- **Verify in-process.** Run both in the same Python process and compare live. Triton autotuning
  can pick different configurations across processes, so a cached result from a separate run is not
  a reliable baseline.
- **Guard it.** Mark the test `@pytest.mark.cuda` *and* give it its own
  `skipif(not torch.cuda.is_available())` — there is no `conftest.py`, so the marker alone skips
  nothing. Triton needs its own `importorskip`. CI has no GPU, so this gate runs only where you
  run it; if you could not reach a CUDA box, say the gate is pending rather than implying it passed.

## Gate 3 — batched driver ≡ per-pair reference

This is the gate that actually catches driver bugs, and for a kernel-reuse mode it is the primary
correctness evidence.

- **What to compare**: `MoleculePairBatch.align_with_<mode>(backend="numba")` against looping
  `MoleculePair.align_with_<mode>()` over the same pairs, on **distinct** molecules, at the shipped
  `(MODE_SEEDS[mode], MODE_STEPS[mode])` budget. Import those from the registry rather than
  hardcoding them.
- **Tolerance**: `1e-3` for a single-channel or reduction mode; `1e-2` for a multi-channel,
  basin-sensitive mode. Both are in the tree. The gap is the different seed sets, not fp noise.
- This is also where padding and masking bugs surface: pad slots must be masked by a real count,
  not left to contaminate the overlap.

## Gate 4 — self-copy = 1.000

A molecule aligned to a copy of itself scores 1.000, on **both** backends. Assert at `atol=1e-4`.
This is the cheap end-to-end smoke test that kernel → driver → API is wired correctly.

Two modes legitimately fail a naive version of this: `vol_and_surf_esp` (its strided ESP term) and
any Tversky mode with `ta ≠ tb` on a non-identical pair. The screen array tests keep an explicit
`_SELF_SCORE_NOT_ONE` exclusion set rather than loosening the assertion for everyone; follow that
pattern if your mode belongs there.

For a surface mode, build the self-pair from a **shared surface** — two independently sampled
surfaces of the same molecule are not the same point cloud and will not score 1.000.

## Gate 5 — streamed screen ≡ per-pair `MoleculePairBatch` *(only if the mode screens)*

The out-of-core path must reproduce the in-memory result. Full detail in `screen_wiring.md`.

- **What to compare**: `screen(query, store, mode=<mode>)` against
  `MoleculePairBatch.align_with_<mode>()` on the *same centered molecules* — deep-copy and
  `center_to` each library molecule to its own heavy-atom COM to match a `pre_centered` store.
- **Tolerance**: `1e-4`, effectively bit-identical, for a single-channel or reduction mode, because
  the array and object paths feed identical inputs to one driver. `1e-2` for a basin-sensitive
  multi-channel mode, where the query's independent re-centering differs by ~2e-7 and a rugged
  multi-start objective can let that flip a near-tie. Do not loosen past `1e-2`: a real wiring bug
  moves scores by far more.
- **Array path vs object path is a separate, tighter comparison: BIT-IDENTICAL.** The array path is
  a front-end refactor with no arithmetic in it, so the standard is `0` of `N` scores moved,
  `max|Δ|` exactly `0.000e+00`, identical ids and identical transform elements. Eleven modes were
  converted against that gate. If yours shows a tolerance-level difference, it is feeding different
  inputs — find that rather than widening the gate.
- **Also assert the data reached disk** for a Tier-B mode, reading through `store.read_shard(0)[1]`
  rather than globbing a filename (the default shard format is `.npy`, one file per array). Check
  the new arrays *and* the offset table. A mode that silently stores nothing still passes a score
  comparison when every molecule's data is empty.
- **Also assert the pose** on a canonical store. A missing canonical rotation leaves the score
  untouched while the returned transform re-scores far below it. Re-score the returned transform
  against the molecule's own centered coordinates.

  Two things will bite you writing that check. **`get_overlap` and `get_overlap_esp` already
  return the Tanimoto**, not the raw overlap — wrapping them in `n / (saa + sbb - n)` gives a
  wrong answer that is *exactly right on a self-pair*, because a Tanimoto of 1.0 survives the
  second application, so the error hides wherever you are most likely to look first. And on a
  `pre_centered` store the 4×4 maps **centred onto centred**, so the reference is the centred
  query, not the raw one. Get either wrong and the harness will condemn correct code. Always
  re-score a known-good mode as a control, and when the control fails, fix the harness.

## The rest of the suite

| Suite | What it covers |
|---|---|
| `tests/test_mode_registry.py` | the registry invariants and the hardcoded count |
| `tests/test_new_modes_accel.py` | gates 1(self-copy), 3 and 2 for nine modes off one `MODES` list |
| `tests/test_<mode>_accel.py` | the same three gates for a mode with its own fixture needs |
| `tests/test_numba_backend.py` | dispatch routing and CPU end-to-end |
| `tests/test_fast_batch_alignment.py` | batch-vs-single, padding masks, numba-vs-Triton |
| `tests/test_screen.py` | gate 5, store round-trip |
| `tests/test_screen_arrays.py` | array-vs-object bit-identity, driven off `_ARRAY_MODES` |
| `tests/test_screen_ndev_dispatch.py` | the one-dispatch-table rule for the multi-GPU worker |

Add your mode to the shared harnesses where they fit. **Drive new tests off `_ARRAY_MODES` /
`CANONICAL_MODES`, not a fresh hardcoded list** — three hardcoded lists plus a five-mode store
fixture once let six modes ship with zero coverage, and the negative control had gone false by
asserting two modes were not array modes after they became so.

## Do not weaken a tolerance to make a gate pass

If a gate fails, the kernel or the wiring is wrong. The one legitimate reason a tolerance here is
loose is the seed-set difference described at the top, and that reason is already priced in to the
numbers above. Anything beyond them is a bug.

Equally: do not report a gate as passing when you could not run it. Gate 2 needs a CUDA device and
gate 5 needs a built store; "pending a CUDA box, and the Triton kernel is a mechanical clone of the
validated shape kernel" is an acceptable thing to write in a commit. Silence is not.
