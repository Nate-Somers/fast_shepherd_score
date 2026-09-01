"""Deterministic CPU fixture that guards the PER-PAIR early stop (WHATS_NEW B9).

This is a HELPER, not a collected test: the leading underscore keeps it out of pytest's
``python_files = test_*.py`` discovery. Run it directly::

    python tests/_es_fixture.py

WHAT IT GUARDS
--------------
Every fine loop in ``shepherd_score/accel`` USED to decide when to stop from a GLOBAL maximum::

    if step % 5 == 0:
        current_max = best_score.max().item()     # max over ALL pairs AND ALL seeds
        ...                                       # patience=2 -> break

``best_score`` has shape ``(PK,)`` with ``PK = K*P`` (K = pairs in the bucket, P = seeds),
laid out k-major, so ``.max()`` collapsed every pose of every pair to one scalar and a pair
that converged early halted optimization for every other pair sharing the bucket. With
``patience=2`` and a check every 5 steps the floor was 11 executed steps no matter how large
the configured budget was. The criterion is now taken PER PAIR, so that leak is closed.

The fixture holds it closed by putting a SELF-PAIR (a molecule against a copy of itself,
which scores 1.0 from the identity seed and can never improve) in the same bucket as four
genuinely different cross-pairs. Under the old rule the self-pair pinned the global max at
1.0 immediately, the whole bucket broke at step 10, and the cross-pairs got 11 steps instead
of their configured 30 / 40 / 50. A PASS today is every BASELINE row running its full budget
and printing ``deficit=+0.0000000000``; a row that reads 11 steps again is the defect back.

Four variants are printed per mode, plus a control:

  * REFERENCE -- the existing ``early_stop_patience`` keyword raised so the break can never
    fire and the configured budget runs in full. These are the scores the fix has to reach.
    ``MODE_SEEDS`` / ``MODE_STEPS`` are NOT touched; search effort is identical.
  * BASELINE  -- the shipped behaviour. Each cross-pair prints its ``deficit`` against the
    reference; with the per-pair criterion in place that deficit is +0.0000000000.
  * CONTROL   -- the same four cross-pairs with the self-pair dropped from the bucket and
    nothing else changed. They run their full budget and land back on the reference scores,
    which attributes the whole deficit to the other pair sharing the bucket.

THE TWO CPU CODE PATHS
----------------------
On CPU the modes do not all take the same route, and both routes carried the defect (both
were fixed, and both are covered here):

  * ``vol``       -> ``kernels/cpu_fused.py::fine_loop_cpu``  (fused numba loop, ``best.max()``)
  * ``vol_color`` -> ``kernels/cpu_fused.py::fine_loop_cpu``  (fused numba loop, ``best.max()``)
  * ``vol_lipo``  -> ``drivers/vol_lipo.py`` eager torch loop (no fused path exists for it)

So each mode is run twice: once on its default CPU route, and once with the ``cpu_fused``
entry points forced to fail so the eager driver loop runs instead. That covers the eager loop
for ``vol`` and ``vol_color`` too.

INSTRUMENTATION
---------------
No driver, kernel or container file is edited. Everything is a monkeypatch of a module
attribute, restored on exit:

  * ``drivers.shape.coarse_fine_align_many`` / ``drivers.vol_color.coarse_fine_vol_color_align_many``
    / ``drivers.vol_lipo.coarse_fine_vol_lipo_align_many`` -- wrapped only to OPEN a record,
    one per fine-loop invocation. (``aligners.py`` imports these inside the function body, and
    the two ``fast_optimize_*_batch`` wrappers call them as module globals, so patching the
    module attribute is enough.) A bucket split or a sub-batch split shows up as several
    records instead of hiding inside one total.
  * ``drivers.shape._overlap_in_chunks`` / ``drivers.vol_color._vc_overlaps`` /
    ``drivers.vol_lipo._vl_overlaps`` -- the value+grad symbol each EAGER loop calls exactly
    once per iteration. Each has exactly one eager call site; the other call site is inside
    the CUDA-graph class, which never runs on CPU.
  * ``kernels.cpu_fused.fine_loop_cpu`` -- wraps the ``overlap_fn`` closure it is handed, so
    the count is that FUSED loop's own iteration count.

Counting convention: the reported number is VALUE+GRAD EVALUATIONS, i.e. loop iterations
entered. The eager loops break *before* their Adam update, so an 11-iteration eager run
applies 10 Adam updates; the fused loop applies its Adam tail before the check, so an
11-iteration fused run applies 11. That asymmetry is a real difference between the two paths,
not an artifact of the counter.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings

# Run-as-a-script bootstrap: ``python tests/_es_fixture.py`` puts tests/ on sys.path, not the
# repo root, so the package would not import. Harmless when pytest imports this instead.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from shepherd_score.accel._modes import MODE_SEEDS, MODE_STEPS
from shepherd_score.container import Molecule, MoleculePair


MODES = ("vol", "vol_color", "vol_lipo")

# Fixed library. Heavy-atom counts are close enough (9-14) that the adaptive bucketer keeps
# all five pairs in ONE bucket, which is the point: the leak this guards against was a
# cross-pair one, inside a bucket. Entry 0 is the reference itself, so pair 0 is a
# self-overlap that scores 1.0.
SMILES = (
    "c1ccccc1CCO",                    # 2-phenylethanol   (ref, and the self-pair fit)
    "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
    "c1ccc(cc1)S(=O)(=O)N",           # benzenesulfonamide
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",   # caffeine
    "OCC1OC(O)C(O)C(O)C1O",           # glucopyranose
)
EMBED_SEED = 0xC0FFEE

# Per-mode result attribute and batched-aligner seam on MoleculePair.
_SCORE_ATTR = {"vol": "sim_aligned_vol_noH",
               "vol_color": "sim_aligned_vol_color",
               "vol_lipo": "sim_aligned_vol_lipo"}
_ALIGNER = {"vol": "_align_batch_vol",
            "vol_color": "_align_batch_vol_color",
            "vol_lipo": "_align_batch_vol_lipo"}


def _embed(smiles: str) -> Chem.Mol:
    """ETKDGv3 embed with a fixed seed -> identical coordinates on every run."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    assert AllChem.EmbedMolecule(mol, params) == 0, f"embed failed: {smiles}"
    return mol


def _molecule(smiles: str) -> Molecule:
    """A Molecule carrying everything the three modes read.

    ``pharm_multi_vector=False`` + ``feature_set='rdkit_base'`` gives the directionless
    pharmacophores ``vol_color`` needs. No ``num_surf_points``: none of these modes touch the
    surface, and building one would require open3d.
    """
    return Molecule(_embed(smiles), pharm_multi_vector=False, feature_set="rdkit_base")


def build_pairs() -> list[MoleculePair]:
    """Five pairs sharing one reference. Pair 0 is the self-pair that pins the global max."""
    cpu = torch.device("cpu")
    return [MoleculePair(_molecule(SMILES[0]), _molecule(s), do_center=True, device=cpu)
            for s in SMILES]


# Patience large enough that the early-stop branch can never fire, so the configured step
# budget runs in full. Used only for the reference variant.
_NO_EARLY_STOP = 1 << 30


@contextlib.contextmanager
def _instrumented(force_eager: bool, disable_early_stop: bool = False):
    """Count value+grad evaluations per fine-loop invocation; optionally force the eager path.

    Yields a list of ``[path, n_evals]`` records, one per fine loop that ran. ``force_eager``
    makes the two ``cpu_fused`` entry points raise, which the drivers catch (``except
    Exception: best_score = None``) and answer by running their eager loop.
    ``disable_early_stop`` raises ``early_stop_patience`` on the existing driver keyword so the
    whole configured budget runs -- the reference the fix has to reach. It changes NO search
    effort: ``MODE_SEEDS`` / ``MODE_STEPS`` are untouched, only the premature break is removed.
    """
    from shepherd_score.accel.drivers import shape, vol_color, vol_lipo
    from shepherd_score.accel.kernels import cpu_fused

    records: list[list] = []
    saved: list[tuple] = []

    def patch(module, name, factory):
        original = getattr(module, name)
        saved.append((module, name, original))
        setattr(module, name, factory(original))

    def open_record(original):
        """Wrapper that starts a new record for each fine-loop invocation."""
        def wrapper(*args, **kwargs):
            records.append(["-", 0])
            if disable_early_stop:
                kwargs["early_stop_patience"] = _NO_EARLY_STOP
            return original(*args, **kwargs)
        return wrapper

    def count_eager(original):
        """Wrapper on an eager loop's value+grad: one call == one iteration."""
        def wrapper(*args, **kwargs):
            records[-1][0] = "eager"
            records[-1][1] += 1
            return original(*args, **kwargs)
        return wrapper

    def count_fused(original):
        """Wrapper on fine_loop_cpu: count the overlap closure it drives."""
        def wrapper(overlap_fn, *args, **kwargs):
            slot = records[-1]

            def counting_overlap(q, t):
                slot[0] = "fused"
                slot[1] += 1
                return overlap_fn(q, t)

            return original(counting_overlap, *args, **kwargs)
        return wrapper

    patch(shape, "coarse_fine_align_many", open_record)
    patch(vol_color, "coarse_fine_vol_color_align_many", open_record)
    patch(vol_lipo, "coarse_fine_vol_lipo_align_many", open_record)
    patch(shape, "_overlap_in_chunks", count_eager)
    patch(vol_color, "_vc_overlaps", count_eager)
    patch(vol_lipo, "_vl_overlaps", count_eager)
    patch(cpu_fused, "fine_loop_cpu", count_fused)

    if force_eager:
        def refuse(_original):
            def wrapper(*args, **kwargs):
                raise RuntimeError(
                    "_es_fixture: fused CPU path disabled to force the eager loop")
            return wrapper
        patch(cpu_fused, "cpu_fused_shape", refuse)
        patch(cpu_fused, "cpu_fused_vol_color", refuse)

    try:
        yield records
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def run_mode(mode: str, force_eager: bool, disable_early_stop: bool = False,
             drop_self_pair: bool = False):
    """Align one bucket of pairs in ``mode``; return (scores, per-loop step records).

    ``drop_self_pair`` removes pair 0 from the bucket and changes NOTHING else -- same
    molecules, same seeds, same step budget, same untouched early-stop settings. It is the
    attribution control: if the four cross-pairs then run their full budget and land on the
    full-budget reference scores, the deficit they show in the five-pair bucket was caused by
    the OTHER pair sharing the bucket, not by anything about themselves.
    """
    pairs = build_pairs()
    if drop_self_pair:
        pairs = pairs[1:]
    aligner = getattr(MoleculePair, _ALIGNER[mode])
    with _instrumented(force_eager, disable_early_stop) as records:
        aligner(pairs, steps_fine=MODE_STEPS[mode])
    scores = [float(getattr(p, _SCORE_ATTR[mode])) for p in pairs]
    return scores, [tuple(r) for r in records]


# (label, force_eager, disable_early_stop). The reference runs first so the two BASELINE
# variants can be printed with their deficit against it.
VARIANTS = (
    ("REFERENCE full budget, default route", False, True),
    ("REFERENCE full budget, eager",         True,  True),
    ("BASELINE default CPU route",           False, False),
    ("BASELINE eager (fused forced off)",    True,  False),
)


def main() -> None:
    warnings.filterwarnings("ignore")           # the one-shot SVML notice is env, not signal
    import numba.core.config as nbcfg

    print("=" * 78)
    print("EARLY-STOP BASELINE FIXTURE  (tests/_es_fixture.py)")
    print("=" * 78)
    print(f"torch {torch.__version__}   rdkit {Chem.rdBase.rdkitVersion}   "
          f"numba SVML={bool(nbcfg.USING_SVML)}")
    print(f"device cpu   pairs 5 (pair 0 = self-overlap)   embed seed 0x{EMBED_SEED:X}")
    print("step counts are VALUE+GRAD EVALUATIONS per fine-loop invocation")
    print("REFERENCE variants raise early_stop_patience only; seeds/steps are untouched")
    print()

    reference: dict[str, list[float]] = {}
    for mode in MODES:
        seeds, steps = MODE_SEEDS[mode], MODE_STEPS[mode]
        for label, force_eager, no_es in VARIANTS:
            scores, records = run_mode(mode, force_eager, no_es)
            print(f"--- mode={mode}  seeds={seeds}  steps_configured={steps}  [{label}]")
            for i, (path, n) in enumerate(records):
                print(f"    fine-loop[{i}] path={path:<5s} steps_executed={n:3d} of {steps}")
            if no_es and not force_eager:
                reference[mode] = scores            # the target the fix must reach
            for i, score in enumerate(scores):
                tag = "self " if i == 0 else "cross"
                line = (f"    pair[{i}] {tag} {SMILES[0]:<24s} vs {SMILES[i]:<30s} "
                        f"score={score:.10f}")
                if not no_es:
                    line += f"   deficit={score - reference[mode][i]:+.10f}"
                print(line)
            print()

        # --- attribution control: same four cross-pairs, self-pair dropped, nothing else
        # changed. Full budget runs and the scores land back on the reference -> the deficit
        # above was leaked in from the other pair in the bucket.
        scores, records = run_mode(mode, force_eager=False, drop_self_pair=True)
        print(f"--- mode={mode}  seeds={seeds}  steps_configured={steps}  "
              f"[CONTROL self-pair dropped, early stop untouched]")
        for i, (path, n) in enumerate(records):
            print(f"    fine-loop[{i}] path={path:<5s} steps_executed={n:3d} of {steps}")
        for i, score in enumerate(scores):
            ref = reference[mode][i + 1]
            print(f"    pair[{i + 1}] cross {SMILES[0]:<24s} vs {SMILES[i + 1]:<30s} "
                  f"score={score:.10f}   vs_reference={score - ref:+.10f}")
        print()

    print("=" * 78)
    print("END")
    print("=" * 78)


if __name__ == "__main__":
    main()
