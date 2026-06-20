# fast_shepherd_score — Merge-Back Audit Report

Audit target: `shepherd_score/` (the GPU-accelerated fork) vs the pristine `shepherd-score-original-repo/shepherd_score/`.
Goal: assess how cleanly this fork merges back into upstream **coleygroup/shepherd-score**, prioritizing (1) minimal API disruption, (2) respect for the original organization, (3) removal of experimental clutter.

---

## Executive summary & merge-readiness verdict

**Footprint is excellent for an upstream merge.** The fork is overwhelmingly *additive*: **14 new files**, only **8 truly-modified original files** (ignoring CRLF noise), and **zero deletions**. The acceleration lives almost entirely in new modules behind `try/except ImportError`, so the package imports and runs without Triton.

**But the fork is NOT byte-clean to merge as-is.** Four issues touch the *pre-existing public surface* and must be fixed first:

1. **`VAA_const` dead positional param in `objective_ROCS_overlay`** — silently shifts `precomputed_U`, a latent break for legacy positional callers. (HIGH)
2. **`apply_SE3_transform` rewritten in place** — dropped 3 validation checks, renamed a param, changed observable error behavior on a public function. (HIGH)
3. **Per-pair CUDA fast-path gated on `torch.cuda.is_available()`** — existing default `align_with_esp/esp_combo/pharm` calls silently switch kernels and drop `use_analytical` on any CUDA box, bypassing the additive `backend=` seam. (HIGH)
4. **`numba` (and `triton`) are undeclared dependencies** — the CPU fallback crashes at import on the very CPU-only boxes it targets. (HIGH)

Plus one repo-hygiene HIGH: **environment.yml was wholesale-replaced** with the developer's personal `GNNenv`.

Beyond those, there is a clean set of **confirmed-dead deletions** (verdicts held) and a clearly-marked set of **REFUTED delete claims** (live code — do NOT remove). The doc (`WHATS_NEW.md`) Organization diagram is materially inaccurate and has been corrected (see second deliverable).

**Verdict: MERGEABLE after a focused cleanup pass.** The additive structure is upstream-friendly; the blockers are a handful of edits to the 8 modified originals plus dependency declaration and dead-code removal. None require touching the original JAX/CPU default path's behavior.

---

## 1. API-disruption risk (fix before merge)

These touch pre-existing public/observable behavior. Highest priority.

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| **HIGH** | `alignment/_torch.py:28` | `VAA_const=None` inserted as 5th positional param **before** `precomputed_U`; never referenced in the body; no caller passes it; the one internal caller (`_torch.py:519`) uses keywords. A legacy positional call `objective_ROCS_overlay(se3, ref, fit, alpha, U)` now binds `U`→`VAA_const`, losing `precomputed_U`. **(verdict: held)** | **Delete the `VAA_const=None` parameter.** Restores the exact original signature; no GPU feature needs it. |
| **HIGH** | `alignment/utils/se3.py:181-227` (orig 136-175) | Public `apply_SE3_transform` rewritten in place: param renamed `SE3_transform`→`se3`, all **3** `ValueError` validation checks deleted, single instances auto-unsqueezed, and a **new (R,7) quaternion+translation path** added. Original tests call it positionally; dropped checks change observable error behavior (inputs upstream rejected are now silently accepted). | **Keep upstream `apply_SE3_transform(points, SE3_transform)` byte-for-byte** (matrix-only, with validation). Put the new quaternion-aware/broadcasting variant in a NEW function (e.g. `apply_SE3_transform_fast` or in `fast_se3.py`). |
| **HIGH** | `container/_core.py:724-766` (esp), `885-957` (esp_combo), `1082-1127` (pharm) | The CUDA fast path is injected into the EXISTING torch branch via `if torch.cuda.is_available():` (confirmed at `_core.py:725`), not behind the additive `backend=` seam. On any CUDA box the default path silently swaps to a Triton kernel and **`use_analytical` is never passed** to `fast_optimize_*`. `MoleculePairBatch.align_with_*` with default `backend='jax'` but `use_jax=False` reaches this via `_delegate_alignment`. | **Gate the per-pair fast path behind an explicit opt-in** (the same `backend=`/`use_fast=` seam), not `cuda.is_available()`. Honor or explicitly reject `use_analytical` when the fast kernel is selected; at minimum document it's ignored on CUDA. Do not silently swap kernels for existing default calls. |
| LOW | `generate_point_cloud.py:8-25` | `_LazyOpen3D` proxy replaces `import open3d as o3d` but preserves the public `o3d` name and attribute access; required for the fork's fork+CUDA multi-GPU pool (Open3D poisons fork+CUDA). Purely additive/behavior-preserving. | **Acceptable as-is.** Call it out in the PR. |

---

## 2. Unused / dead code (confirmed — safe to delete)

All entries below carry a **verdict that HELD** (`refuted=false`). Each is fork-added or fork-private, not in any `__all__`, no dynamic refs, and off the JAX/CPU default path.

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| HIGH | `container/_batch_align.py:72-77, 91-100, 148-273` | In-library **process-per-GPU** backend (`_MGPU_BACKEND`/`FSS_MGPU_BACKEND`, `_run_distributed_procs`, `_mgpu_proc_worker`, `_ProcStandIn`, `_MODE_SPEC`). Default is `'thread'`; process branch reachable only via env var; referenced only by `benchmarks/experiments/{mgpu_parity,mgpu_pool_probe}.py`. Memory marks it PAUSED/too-slow, superseded by `multi_gpu.py::MultiGPUAligner`. | **Remove the process backend** and collapse `_run_distributed` to the thread path. ⚠️ **Keep `_run_distributed` itself** — it is LIVE (called at `_batch_align.py:399,537,686,912,987,1196`). Keep the real driver `multi_gpu.py` (uses only `_DISPATCH_LOCAL`). |
| MED | `score/gaussian_overlap_triton.py:13-25` | `_select_block_size()` — sole definition, never called; BLOCK is chosen by `@triton.autotune`. Its docstring contradicts the current design. | Delete the function; also fix the stale wrapper docstring at line 217. |
| MED | `score/gaussian_overlap_triton.py:27-37` | `@triton.jit _load_xyz()` — never called by any kernel (each kernel inlines its own `tl.load`); a jit helper only callable from inside another jit kernel. | Delete lines 27-37. |
| MED | `score/gaussian_overlap_triton.py:801-958` (+ blanks 776-796) | `if __name__=="__main__" and "--trace-quat-grad"` scratch autograd-vs-kernel harness; nothing invokes it; local-only helpers; `import sys` exists only for it. | Delete/move to `tests/` or `benchmarks/`; drop blank padding; change `import math, sys`→`import math`. ⚠️ **Do NOT start the cut at line 759** — lines 762-774 are the LIVE `_batch_self_overlap` (widely imported). Safe cut is 776-958. |
| MED | `score/pharmacophore_overlap_triton.py:377-407, 410-453` | `batch_pharm_tanimoto` and `batch_pharm_overlap_with_transform` — defined only; no callers; not exported; net-new vs upstream. | Delete both (or keep only `batch_pharm_tanimoto` if a tanimoto entry point is wanted upstream). |
| MED | `alignment/utils/fast_common.py:142-181` | Public `legacy_seeds_torch` never called — only imported (dead) at `fast_surface_se3.py:30`, `fast_esp_se3.py:28`, `fast_esp_combo_se3.py:20`; drivers actually use `batched_seeds_torch`. | Delete the def and the 3 dead imports. |
| MED | `alignment/utils/fast_se3.py:89-111, 113-151` | `_fallback_quats` + `_legacy_seeds_torch` (fast_se3 copies) — `_legacy_seeds_torch` never called; `_fallback_quats` used only inside it. `fast_common` has its own live `_fallback_quats`. | Delete both; then the import `_initialize_se3_params as _legacy_init` (`fast_se3.py:17`) becomes unused — remove it too. Fix stale docstring ref at `fast_common.py:280`. |
| MED | `alignment/utils/se3.py:17-43` | Singular `quaternion_to_SE3` added but never called (name appears only in a docstring/comment); only `quaternions_to_SE3_batch` is used. Fork-added; absent upstream. | Delete the singular helper (or relocate to `fast_se3.py`); reword the `se3.py:46` docstring. |
| MED | `score/__init__.py:1-6,15-16` (Triton selection only) | The `from .gaussian_overlap_triton import gaussian_tanimoto` import **always fails** (symbol doesn't exist), so `_HAS_TRITON` is always False and the Triton single-pair path is permanently dead. **(NOT a delete-the-whole-file claim — see Refuted §6.)** | **Remove the dead Triton try-import** but KEEP the `gaussian_tanimoto` wrapper (it is imported by `objective.py:17`). Also fix the latent signature bug below. |
| LOW | `container/_core.py:159` | `self._shape_cache: dict[float, float] = {}` — set in `__init__`, never read or reassigned anywhere. Fork-only. | Delete the line. |
| LOW | `container/_core.py:1152-1234` | New public `score_with_vol` (80 lines) — zero callers; thin re-wrap of `get_overlap*`; fork-only; original deliberately omits a 'vol' scoring method. | Drop before upstreaming unless a concrete consumer exists. |
| LOW | `container/_core.py:341` | `_esp_bucketed_align = staticmethod(_ba._esp_bucketed_align)` — never accessed as a class attribute; callers use the module-level free function (`_batch_align.py:728,951`). | Remove this one binding (keep the other `_align_batch_*` bindings — those ARE used via the class). |
| LOW | `score/gaussian_overlap_triton.py:10` | `import numpy as np` unused (`.numpy()` calls are torch tensor methods). | Remove. |
| LOW | `alignment/utils/se3.py:14` | `import numpy as np` added by fork, never used in this file (numpy path lives in separate `se3_np`). | Remove. |
| LOW | `fast_surface_se3.py:9`, `fast_esp_se3.py:7`, `fast_esp_combo_se3.py:8` | `import torch.nn.functional as F` unused in all three (copy-paste leftover). | Remove the three lines. |
| LOW | `fast_surface_se3.py:15,24,31,33`; `fast_esp_se3.py:14,22,31`; `fast_esp_combo_se3.py:13,23` | Unused `fused_adam_qt`, `quat_mul`, and (surface only) `build_coarse_grid` imports. `build_coarse_grid` IS used in esp/combo (don't prune there). | Prune each import list. ⚠️ Surface/esp import `fused_adam_qt` in BOTH the try and except branches — remove from both. |
| LOW | `fast_se3.py:15,18,19` | `objective_ROCS_overlay`, `Path`, `suppress` imported, never used. | Remove the three imports. |
| LOW | `fast_se3.py:7,13` | `_HAS_TRITON` assigned in both import-guard branches but never read (branch selection uses `torch.cuda.is_available()`); inconsistent with sibling modules. | Remove the two assignments (or use the flag consistently). The same-named flag in `score/__init__.py` is a distinct, live symbol — leave it. |

---

## 3. Redundancy

Code duplication that is real but **not deletable** (live code). One safe in-place dedup is recommended; the rest are optional refactors flagged as REFUTED-as-removal (kept here for visibility, see §6).

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| LOW | `fast_surface_se3.py:332-335`, `fast_esp_se3.py:413-416`, `fast_esp_combo_se3.py:575-578`, `fast_pharm_se3.py:471-474` | **(verdict: held)** Byte-identical SE(3)-from-(q,t) epilogue duplicated in all 4 single-pair wrappers. The fork ALREADY has a bit-identical helper `quaternion_to_SE3(q,t)` at `se3.py:18-42`. | **In-place dedup**: call the existing helper (or add `se3_matrix_from_qt`). Preserve wrapper signatures and return shapes (public API & CPU path unaffected). ⚠️ Keep the wrappers — three are live public API (`_core.py:748,929,1106`). Note: `fast_optimize_ROCS_overlay` (surface non-batch, `fast_surface_se3.py:258`) is the one wrapper with no callers — a separate dead-code candidate. |
| MED | `score/gaussian_overlap_triton.py:581-644` (vs 99-184) | `_fused_surf_adam_step` duplicates the overlap+grad body of `_gauss_overlap_se3_tiled` verbatim (comment admits "copied VERBATIM … so Vab/dQ/dT are bit-identical"). | **Optional**: factor the inner body into a shared `@triton.jit` device function. Behavior-preserving refactor, not removal. |
| MED | `score/gaussian_overlap_esp_triton.py:134-185` (vs `gaussian_overlap_triton.py:136-184`) | ESP kernel grad block is a near-verbatim copy; only difference is the extra charge-weight term. | **Optional**: share the rotation+force+quaternion-grad math via a common jit helper parameterized by an optional charge factor. |
| LOW | `cpu_overlap.py:36-205` vs `gaussian_overlap_np.py:18-49` | **Not a true duplicate.** Original np path computes values only, single-instance, no gradient; `cpu_overlap` computes batched value+SE(3)-gradient to drive the optimizer. | **No removal on redundancy grounds.** |

---

## 4. Inefficiency

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| LOW (REFUTED as removal) | `gaussian_overlap_triton.py:206-208,242-244`; `gaussian_overlap_esp_triton.py:209-211,255-256` | `BLOCK/num_warps/num_stages` "accepted for back-compat but ignored" on the GPU side. **(verdict: refuted)** These are a deliberate **dual-backend signature contract**: the CPU drop-in (`cpu_overlap.py:88-92,208-212`) must accept the identical signature so the numba fallback is a true drop-in. | **Do NOT drop as dead.** They are inert on the GPU launch but load-bearing for the GPU↔CPU import symmetry. If cleaned, must be done in BOTH backends + all 3 call sites in lockstep. Low value; leave as-is for merge. |

---

## 5. Organization & merge-readiness

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| **HIGH** | `environment.yml:1-46` (orig 1-19) | Wholesale-replaced with personal `GNNenv` (python 3.10, torch 2.5.1+cu124, pyg/torch_scatter/sparse/cluster, lightning, jax, mdanalysis, prolif, plotly, tensorboard, …). None relate to the Triton feature. The only real dep the feature needs (`tqdm`) is already in pyproject.toml. CI uses `uv`/pyproject, not this file. **(verdict: held)** | **Revert environment.yml to upstream verbatim.** Keep GNNenv locally (gitignored) if needed. |
| MED | `codex_startup.sh:1-49` | Tracked in fork, absent upstream; bootstraps micromamba/condarc and an env `codex-env`; personal Codex agent hook, not library/CI code. **(verdict: held)** | **Drop from the merge** (`git rm` + .gitignore, or personal branch). |
| MED | `benchmarks/experiments/{cuda_init_probe,fork_test,fork_test2,fork_bisect,import_bisect,startup_probe}.py` | Six UNtracked one-off CUDA-fork/import-timing probes; not gitignored, so they'd leak into the PR; only callers are gitignored `slurm/*.sh`. **(verdict: held)** | **Delete the six untracked probes** (or gitignore `benchmarks/experiments/*probe*.py` + bisect scripts). ⚠️ Do NOT drop tracked `mgpu_parity.py`/`mgpu_pool_probe.py` while the process backend still ships — they are its parity gate. |
| MED | `container/multi_gpu.py:1-433`; `container/__init__.py:1-4` | `multi_gpu.py` (the real ~3.5–3.9× driver, `align_multi_gpu`/`MultiGPUAligner`) is LIVE but **not in the public surface** — `__init__.py` exports only `Molecule/MoleculePair/MoleculePairBatch`; reachable only via the deep `shepherd_score.container.multi_gpu` path. | **Decide its status before merge**: if supported, add to `container/__init__.py` `__all__` + document; if experimental, add a module docstring note. Don't leave it undocumented+unexported but live. |
| MED | `score/gaussian_overlap_triton.py:153-308,180-188,452-464` (REFUTED as removal) | Lab-notebook experiment scaffolding (`_GraphedFineSurf`, `_run_fused_fine`, `_PRUNE_*`, `_ES_*`). **(verdict: refuted)** — `_FINE_GRAPHS` is mutated in production (`_batch_align.py:120`), prune/ES knobs are read by `speedlab.py`, all three branches live inside the LIVE `coarse_fine_align_many`. | **Do NOT delete as a block** (breaks `_batch_align.py` + `speedlab.py`). Optional: trim the proven-regression `_FINE_FUSED` prototype (no external consumer) and condense the verbose verdict comments. |
| LOW | `fast_se3.py:193,430-431`; `fast_common.py:13`; `fast_esp_se3.py:287`; `fast_pharm_se3.py:307`; `fast_surface_se3.py:233`; `fast_esp_combo_se3.py:448` (REFUTED as removal) | Early-stop patience override has **3 mechanisms**; `FINE_ES_PATIENCE` affects vol/esp/pharm but not surf/combo; two parallel definitions. **(verdict: refuted)** — both definitions are live (read+written by `speedlab.py:110-111,130,142`); documented 1.33× lever. | **Optional consistency refactor** (unify on one override source applied in all 5 drivers). Not dead code — do not silently delete. |
| LOW | `cpu_overlap.py:149-153` | `fused_surf_step_batch` is a deliberate `NotImplementedError` stub for import-parity (its only caller is CUDA-gated). | Acceptable. Optional: a comment at the `fast_se3.py:12` import clarifying dead-on-CPU intent. |
| LOW | `slurm/` (23 scripts) | Entirely UNtracked (won't enter the merge); hardcoded `#SBATCH --partition=pi_melkin`, personal paths. | Keep untracked/local. If a launcher is wanted upstream, generalize one parameterized template into `benchmarks/`. |
| LOW | `POTENTIAL_FUTURE_WORK.md` | Untracked working-notes ("IDEA / not implemented"). (`MOLECULE_CONSTRUCTION_OVERHEAD.md` in the stale git-status snapshot no longer exists on disk.) | Keep local or fold into PR description / an issue; not a feature. |
| LOW | `WHATS_NEW.md`, `SPEED_EXPERIMENTS.md` (18KB), `SPEED_EXPERIMENTS_CPU.md` (51KB), `RELATED_WORK.md` (24KB) | Four tracked dev/process docs at repo root; README.md is byte-identical to upstream so the feature is undocumented in the canonical place. | Keep `WHATS_NEW.md` (or fold into README/CHANGELOG); move the experiment logs + `RELATED_WORK.md` under `docs/` or out of the merge. |
| LOW | `paper/` (fig1..fig4) | Entirely untracked manuscript-figure scaffolding (PDFs/PNGs/JSON, `__pycache__`). | Keep out of the merge; use a separate paper repo/branch. Don't commit generated binaries. |
| LOW | `_torch.py:572,660,885,890,1136` | Cosmetic-only diffs (trailing commas, a removed blank line, one comment). | **Revert** to minimize the merge diff (zero functional gain). |
| LOW | `se3.py:229-275` | The entire old `apply_SE3_transform` left commented-out beneath the rewrite (+ stray commented else-branch). | **Delete the commented block** (history is in VCS). |
| LOW | `protonation/protonate.py:10-11` | Only change is `from __future__ import annotations` + blank line; purely additive. | Acceptable. |
| LOW | `.gitattributes` / line endings | **Line-ending policy is SAFE.** Both repos' committed blobs are LF (fork 0 CRLF blobs, original 0 CRLF blobs); apparent CRLF noise is a Windows working-tree artifact only. | **No action.** Do NOT run a normalization pass (would create the churn you're avoiding). |
| LOW | `pyproject.toml:46` | Only substantive diff vs original is the added `tqdm`; version machinery, CI, docs, pytest config are byte-identical; generated artifacts correctly gitignored. | Keep `tqdm`. No packaging cleanup needed. Confirm `shepherd-score-original-repo/` is never committed into the PR. |

---

## 6. Correctness

| Sev | Location | Evidence | Recommendation |
|---|---|---|---|
| **HIGH** | `cpu_overlap.py:31` | `from numba import njit, prange` is a **hard top-level import**, but `numba` is in NO dependency group of `pyproject.toml` (confirmed: deps are rdkit/torch/open3d/py3Dmol/numpy/pandas/scipy/molscrub/tqdm; no `numba`, no `triton`). On a box lacking Triton (the fallback trigger) AND `numba`, `import fast_se3`/esp/surf/pharm raises ImportError — the fallback fails on the exact machines it targets. | **Add `numba` to dependencies** (or an optional `cpu` extra), OR wrap the numba import to degrade gracefully. **Also declare `triton`** as an optional extra. **This is the single most merge-blocking correctness issue in this area.** |
| MED | `tests/test_fast_batch_alignment.py:7-14` vs `benchmarks/experiments/cpu_*` | The only fast-aligner tests `pytest.skip` unless BOTH CUDA and Triton present, so `cpu_overlap` is never exercised in CI; its correctness is validated only in non-collected `benchmarks/experiments/` scripts (with `sys.modules.setdefault('open3d', MagicMock())` hacks). A shipped fallback with no CI test will rot. | **Promote a minimal numba value+grad-vs-torch-reference check (plus self-copy==1.0) into `tests/`** guarded by `importorskip('numba')`, gated to CPU. |
| MED | `fast_esp_combo_se3.py:11-16` | esp_combo does a **bare** `from ...score.gaussian_overlap_triton import (...)` with **no** `try/except cpu_overlap` fallback (confirmed), unlike its 4 siblings. On a CPU-only box, importing this module raises ImportError even though it has a `check_gpu_available()` CPU fallback at line 511. | **Wrap esp_combo's kernel import in the same try/except cpu_overlap fallback** as its siblings, or explicitly document esp_combo as GPU-only. As written it is inconsistent. |
| MED | `score/__init__.py:9` + `objective.py:187-191` | **Latent signature bug.** The wrapper is `gaussian_tanimoto(a, b, alpha=0.81)` but the only call site (`objective.py:187`) passes keyword args `centers_1=`/`centers_2=` → `TypeError` if reached. Masked today (only fires on `use_torch` + `num_points>=150` + CUDA). | **Fix when repairing the score/__init__ seam**: either rename wrapper params to `centers_1/centers_2`, or (cleaner) revert `objective.py:17` to import `get_overlap` from `gaussian_overlap` directly. |

---

## 7. Doc accuracy (`WHATS_NEW.md`) — all fixed in the corrected file

| Sev | Location | Evidence | Fix applied |
|---|---|---|---|
| **HIGH** | `WHATS_NEW.md:36-41` | Layer-3 diagram attributes new code to `_batch.py`/`_core.py`, which **pre-exist** and were only modified; the genuinely-ADDED `_batch_align.py` (1348 lines: bucketing, sub-batching, `_scatter_fill`, `_esp_bucketed_align`, sharding) is where that code physically lives. | Diagram now lists `_batch_align.py` (NEW) as the orchestration home and marks `_batch.py`/`_core.py` as modified-in-place. |
| **HIGH** | `WHATS_NEW.md:36-41` | `_batch_align.py` (largest new container module) is never mentioned. | Added with a description. |
| **HIGH** | `WHATS_NEW.md:36-41,73-76,166-174` | `multi_gpu.py` (432 lines, ADDED — `align_multi_gpu`/`MultiGPUAligner`, the explicit one-process-per-GPU driver that actually scales ~3.5–3.9×) is entirely omitted; the doc only describes "automatic" in-library sharding. | Added `multi_gpu.py` to the diagram and a new multi-GPU subsection distinguishing automatic in-library sharding from the explicit data-parallel driver. |
| MED | `WHATS_NEW.md:28-34,9-11,100-104` | `cpu_overlap.py` (354 lines, ADDED numba CPU fallback wired into all 4 fast_* drivers) omitted from the Layer-2 diagram and the fallback prose. | Added `cpu_overlap.py` to Layer 2 and corrected the fallback description (numba CPU path, not only JAX). |
| MED | `WHATS_NEW.md:15,36-41` | Heading "where the new code lives" points at the two files with the LEAST new code. | Reworded; added NEW vs modified-in-place markers. |
| LOW | `WHATS_NEW.md:38` | `_core.py → MoleculePair # PRIVATE` overstates it — `MoleculePair` is public (public `align_with_*`); only `_align_batch_*` helpers are private. | Reworded. |
| — | (kernel layer) | Layer 1 omitted `gaussian_overlap_esp_triton.py` from the listed kernels. | Now lists all four triton kernel files accurately. |

Additional WHATS_NEW corrections folded in for fidelity to code (not all from auditors, but required for accuracy): `triton`/`numba` are **optional, undeclared** deps (added a caveat to Requirements); the per-pair `use_analytical`-dropping CUDA behavior is noted as a known caveat; the `backend` matrix and seam description retained (verified accurate against `_batch.py:71` `_TRITON_BACKENDS = ("triton","cuda","gpu")` and the per-method `backend=` params).

---

## Quick action list (priority order)

1. **API**: delete `VAA_const` (`_torch.py:28`); restore pristine `apply_SE3_transform` + move new logic to a new fn; gate per-pair CUDA fast-path behind `backend=`/`use_fast=` and handle `use_analytical`.
2. **Deps**: declare `numba` + `triton` (optional `cpu`/`gpu` extras); add esp_combo's try/except cpu fallback; add a CPU-fallback CI test.
3. **Repo hygiene**: revert `environment.yml`; drop `codex_startup.sh`; delete the 6 untracked probes; revert cosmetic `_torch.py` diffs; delete the commented `se3.py:229-275` block.
4. **Dead code**: remove the items in §2 (process backend, dead triton helpers, `legacy_seeds_torch`/`_legacy_seeds_torch`, singular `quaternion_to_SE3`, `_shape_cache`, `score_with_vol`, unused imports, `_HAS_TRITON` in fast_se3, `_esp_bucketed_align` class binding).
5. **Optional**: in-place SE(3)-epilogue dedup via existing `quaternion_to_SE3`; kernel device-function sharing.
6. **Do NOT touch (REFUTED — live)**: `_overlap_in_chunks*`/`_self_overlap_*_chunks`, duplicated Adam loops + gather epilogues, BLOCK/num_warps/num_stages dual-backend kwargs, `_FINE_GRAPHS`/prune/ES-patience knobs, the `gaussian_tanimoto` wrapper, tracked `mgpu_parity.py`/`mgpu_pool_probe.py`.