# Parity gates

A mode is accelerated when gates 1–4 pass, plus gate 5 if it screens. Each gate isolates one link
in the chain reference → CPU kernel → GPU kernel → engine → out-of-core store. Run them in order: a
failure at gate N is almost always a bug introduced at step N, not earlier.

**Read this before quoting a tolerance.** The honest tolerances are looser than a naive reading
suggests, for one structural reason: the accelerated seeder uses a **different seed set** from the
reference — identity, 4 PCA quaternions, up to 6 structured ±90° axis swaps, then a Fibonacci fill,
where the reference has no structured seeds. The two do not optimize from the same starting points
and cannot agree bitwise. What you are testing is that they land in the **same basin**.

## Gate 1 — kernel ≡ reference

The kernel's value and gradient must match the eager reference on the same inputs at the same pose.

**If you wrote a new kernel**, compare its value and `dO/dq` directly against the reference
objective, on several distinct pairs, not just self-pairs. `vol_avoid` — the most recent new kernel
— reached value agreement at 1e-14 and gradient-vs-finite-difference at 1e-9 in double precision.
That is the standard for a genuinely new kernel, and it is achievable because kernel and reference
compute the same math two ways.

**Compare the gradient in the tangent space of the unit quaternion, not raw.** The kernels emit a
**raw** `dO/dq` and do not project out the component parallel to `q`, because the optimizer
renormalizes `q` each step and discards that radial part. Autograd taken through the repo's
*normalizing* quaternion→rotation map is already tangent-projected. A naive raw-vs-autograd
comparison therefore **fails at ~0.4–0.8 even for a correct kernel** — the entire mismatch is the
radial component. Subtract the `q`-parallel part from both before comparing, or differentiate a
non-normalizing map.

**If you reused existing kernels** — the usual case — there is no new kernel math to validate, and
this gate instead checks that your spec's terms, reductions and weights are right. In practice that
is done at the score level, through gate 3. Say so: state that gate 1 holds by inheritance rather
than implying you ran a gradient comparison.

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
  can pick different configurations across processes.
- **Guard it.** Mark the test `@pytest.mark.cuda` *and* give it its own
  `skipif(not torch.cuda.is_available())` — there is no `conftest.py`, so the marker alone skips
  nothing. Triton needs its own `importorskip`. CI has no GPU, so this gate runs only where you run
  it; if you could not reach a CUDA box, say the gate is pending rather than implying it passed.

## Gate 3 — batched engine ≡ per-pair reference

This is the gate that catches spec bugs, and for a kernel-reuse mode it is the primary correctness
evidence. A wrong `weight`, a wrong `reduction`, a channel pair in the wrong argument order or a
missing `guard` all surface here and nowhere earlier.

- **What to compare**: `MoleculePairBatch.align_with_<mode>(backend="numba")` against looping
  `MoleculePair.align_with_<mode>()` over the same pairs, on **distinct** molecules, at the shipped
  `(MODE_SEEDS[mode], MODE_STEPS[mode])` budget. Import those from the registry.
- **Tolerance**: `1e-3` for a single-channel or reduction mode; `1e-2` for a multi-channel,
  basin-sensitive mode. The gap is the different seed sets, not fp noise.
- This is also where padding and masking bugs surface: pad slots must be masked by a real count.

## Gate 3b — the three fine loops agree *(new; the engine has three)*

The eager loop, the CUDA-graph replay and the fused numba CPU loop are three implementations of one
step. Run your mode through each and compare:

- **graph vs eager** (CUDA): expect **bit-identical** unless your mode has a value-only term
  evaluated on a stride. The two combo modes are the documented exception — the eager loop applies
  `_ESP_STRIDE` and the captured step does not, so the graph scores every step and returns uniformly
  higher values. Measured on the 6-molecule fixture: 3.354e-03 (1.55%) for `vol_and_surf_esp` and
  5.291e-04 (0.063%) for its Tversky twin, every other mode exactly 0. If YOUR mode shows a
  difference and has no strided term, that is a bug in your spec or your term evaluator.
- **fused CPU vs eager CPU**: expect fp32 agreement, but measure it **with and without SVML** —
  the fused loop swaps in the fp32 SoA kernels (`kernels/cpu_soa.py`) only when `USING_SVML`, and
  the gap widens when it does. Across all 21 modes: 0.0004% without SVML, 0.0375% with it for
  `surf_esp` and 0.0005% for everything else. A tolerance fitted on one environment will fail in
  the other; that already happened once. A basin-level difference means the mode is multi-basin
  enough that the float32 tail's rounding flips which seed wins — set `cpu_fused=False` on the spec, as the pharmacophore family does, and say what you
  measured. Do not ship a mode whose default CPU path disagrees with its own eager loop.

The CPU half is now a TEST, not a manual check: `tests/test_cpu_fine_loops_agree.py` parametrizes
over `CANONICAL_MODES`, runs each mode twice -- once normally, once with `cpu_fused.run_fused`
forced to raise so the engine's own fallback takes the eager loop -- and bounds the gap at 1e-5
absolute / 0.01% relative. Your mode joins it by being in the registry, so there is nothing to add;
just read the failure if it fires. Measured across all 21 modes on that fixture: worst 1.699e-06
(0.0004%, `surf_esp`). It also asserts that forcing the fused loop off really reaches the eager
loop, so the comparison cannot pass vacuously.

`tests/_es_fixture.py` remains the standing harness for the step-count side: run
`python tests/_es_fixture.py` and every fine loop should execute its full configured budget
(30/30, 40/40, 50/50 today) with deficits at the fp32 noise floor.

## Gate 4 — self-copy = 1.000

A molecule aligned to a copy of itself scores 1.000, on **both** backends. Assert at `atol=1e-4`.
This is the cheap end-to-end smoke test that kernel → engine → API is wired correctly.

Two structures legitimately fail a naive version of this, and both are readable off the spec rather
than listed by name: a mode with a value-only `agreement` term (the ShaEP surface-ESP channel), and
a mode with a subtracted penalty. `tests/test_screen_arrays.py` derives its exclusion set from
exactly those two properties; follow that rather than adding a name. A Tversky mode with `ta ≠ tb`
also need not RANK the self-copy first, though it still scores it 1.000.

For a surface mode, build the self-pair from a **shared surface** — two independently sampled
surfaces of the same molecule are not the same point cloud and will not score 1.000.

## Gate 4b — the `trans_init` path *(cheap, and it has caught a real bug)*

`trans_init=True` swaps the SO(3) multi-start for a coarse grid of poses built around the
reference molecule's atom positions. Every mode reaches it through the same engine, so a new mode
gets it for free and nobody thinks to check it — which is how a 1.24% score move on
`vol_and_surf_esp` survived a full green parity run. Add your mode to
`tests/test_trans_init_accel.py`: one line in `GRID_CLOUD` naming the cloud the grid is built from
(your seed cloud unless you set `coarse_channel`), and its keywords in `KW`. The spy test then
pins the cloud and the self-copy test covers the path end to end.

Only 7 of the 21 modes expose `trans_init` at all, and two of those (`vol`, `surf`) accept it and
**ignore** it — the accelerated shape path re-derives its own seeds. If your mode's wrapper does not
take the keyword, it has nothing to add here. Expect the self-copy to reach ~1.0 if it does act on
it; if it does not, check the base before treating it as your bug, because `vol_and_surf_esp` has
always scored ~0.54 there.

## Gate 5 — streamed screen ≡ per-pair `MoleculePairBatch`

The out-of-core path must reproduce the in-memory result. Full detail in `screen_wiring.md`.

- **What to compare**: `screen(query, store, mode=<mode>)` against
  `MoleculePairBatch.align_with_<mode>()` on the *same centred molecules*.
- **Tolerance**: `1e-4`, effectively bit-identical, for a single-channel or reduction mode. `1e-2`
  for a basin-sensitive multi-channel mode, where the query's independent re-centring differs by
  ~2e-7 and a rugged multi-start objective can flip a near-tie. Do not loosen past `1e-2`.
- **Array path vs object path is a separate, tighter comparison: BIT-IDENTICAL.** The array path is
  a front-end refactor with no arithmetic in it, so the standard is `0` of `N` scores moved,
  `max|Δ|` exactly `0.000e+00`, identical ids and identical transform elements. All 21 modes were
  converted against that gate and hold it. If yours shows a tolerance-level difference, it is
  feeding different inputs — find that rather than widening the gate.
- **Also assert the data reached disk** for a Tier-B mode, reading through `store.read_shard(0)[1]`
  rather than globbing a filename. Check the new arrays *and* the offset table.
- **Also assert the pose** on a canonical store. Re-score the returned transform against the
  molecule's own centred coordinates.

  Two things will bite you writing that check. **`get_overlap` and `get_overlap_esp` already return
  the Tanimoto**, not the raw overlap — wrapping them in `n / (saa + sbb - n)` gives a wrong answer
  that is *exactly right on a self-pair*, because a Tanimoto of 1.0 survives the second application,
  so the error hides wherever you are most likely to look first. And on a `pre_centered` store the
  4×4 maps **centred onto centred**, so the reference is the centred query, not the raw one. Always
  re-score a known-good mode as a control, and when the control fails, fix the harness.

## Gate 6 — the registry still holds

`tests/test_mode_registry.py` must pass with only its count bumped. It asserts that `MODE_ATTRS`,
`MODE_SEEDS`, `MODE_STEPS`, `PROCESS_MODES` and `_MODE_SPEC` all agree, that every canonical mode
has a bound `_align_batch_<mode>`, and that `aligners._MODE_SEEDS is M.MODE_SEEDS` by identity. All
of those are derived from `SPECS` now, so a failure here means your spec is malformed rather than
that you forgot a table.

## The rest of the suite

| Suite | What it covers |
|---|---|
| `tests/test_mode_registry.py` | the registry invariants and the hardcoded count |
| `tests/test_new_modes_accel.py` | gates 4, 3 and 2 for nine modes off one `MODES` list |
| `tests/test_<mode>_accel.py` | the same three gates for a mode with its own fixture needs |
| `tests/test_numba_backend.py` | dispatch routing and CPU end-to-end |
| `tests/test_fast_batch_alignment.py` | batch-vs-single, padding masks, numba-vs-Triton |
| `tests/test_screen.py` | gate 5, store round-trip |
| `tests/test_screen_arrays.py` | array-vs-object bit-identity, driven off `_ARRAY_MODES` |
| `tests/test_screen_const_seeds.py` | WHICH seed generator ran, per mode class |
| `tests/test_screen_ndev_dispatch.py` | the one-dispatch-table rule for the multi-GPU worker |
| `tests/_es_fixture.py` | the per-pair early stop, on both CPU fine loops |

Most of these parametrize over the registry, so your mode joins them on the next run. **Drive
anything new off `CANONICAL_MODES` / `_ARRAY_MODES`, not a fresh hardcoded list** — three hardcoded
lists plus a five-mode store fixture once let six modes ship with zero coverage, and the negative
control had gone false.

## Gate 7 — run it at SCALE, not just on the fixture

Every gate above is cheap because its fixture is small. Two shipped regressions were invisible to
all of them, and both were found only by a benchmark-sized run:

- **Past 65,535 poses** the shape/ESP/avoid launches are sliced by `drivers/terms._chunked`, and
  code that is correct in one launch can be wrong across slices. It passed the per-molecule
  `N_real`/`M_real` whole, so from the second slice every pose was scored against another
  molecule's atom count — Tanimotos up to 1.1e5 on a 30,000-pair `vol` batch, and nothing below
  the limit could show it. CPU is immune, since the numba kernels take one call. If your mode
  adds a kernel argument, ask whether it is per-pose or per-molecule, and add it to `_MOL_KW` if
  it is the latter. (There is no 65,535 any more: every kernel launches a 1-D grid, so the bound
  is `grid.x` = 2^31-1, and the pharmacophore kernel runs 3,167,232 poses unchunked and correct.
  The slice that remains is `terms._launch_step`, the int32 pointer-offset ceiling derived from
  the pads because the kernels form `mol * N_pad * 3` in int32. If your kernel widens its offset
  differently — a fourth coordinate, a per-pose table — that formula is the one thing to adjust.
  Do not add a constant slice to a new kernel; measure, one process per variant, because the
  captured-graph cache is keyed by shape and an in-process A/B replays the first variant.)
- **A store built for your mode ALONE** may lack arrays a neighbouring mode was writing.
  `_flush` emits the with-H block under one `schema["charges"]`, so a mode that reads `cwh`
  without pulling that flag got a store with none of it.

So: one GPU run above 65,535 poses (K ≥ 8,192 at 8 seeds), and one store built for your mode by
itself. `tests/test_grid_chunked_launch.py` and `tests/test_store_minimal_schema.py` hold both,
and the second picks your mode up from the registry automatically.

## Do not weaken a tolerance to make a gate pass

If a gate fails, the spec or the kernel is wrong. The one legitimate reason a tolerance here is
loose is the seed-set difference described at the top, and that reason is already priced in.

Equally: do not report a gate as passing when you could not run it. Gate 2 and gate 3b need a CUDA
device and gate 5 needs a built store; "pending a CUDA box, and the Triton kernel is a mechanical
clone of the validated shape kernel" is an acceptable thing to write in a commit. Silence is not.
