"""
Correctness / agreement smoke tests for the alignment backends.

This is the *accuracy* half of the benchmark suite expressed as assertions so it
can run in CI.  It does **not** assert on timing (machine dependent); the timing
story lives in ``benchmarks/`` and is exercised by ``test_runner_smoke``.

What is asserted
----------------
* Every available backend recovers a near-perfect overlap on a *noiseless*
  cohort, where the global optimum is a perfect (Tanimoto = 1) alignment.  This
  is the honest accuracy bar: "given a problem with a known recoverable optimum,
  did the backend find it?"
* The GPU fast/batch paths agree with the torch reference within a tolerance
  that reflects their being a different (approximate) optimiser.
* The size-bucketing metadata is correct (uniform -> 1 bucket, mixed -> many).

Markers: ``slow`` (all), ``cuda`` (GPU backends).  CPU backends always run.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.alignment_bench.workloads import make_cohort, MODES
from benchmarks.alignment_bench import backends as B

pytestmark = pytest.mark.slow

# A small, fast config so the suite is CI-friendly.
_CFG = B.BenchConfig(num_repeats=16, max_steps=60, steps_fine=60, topk=20)
_NOISELESS_FLOOR = 0.90   # every backend must reach at least this on noiseless data
_AGREE_TOL = 0.08         # max |fast - reference| per-pair score on noiseless data


def _cohort(mode, size_kind="uniform", size=20, n_pairs=6):
    return make_cohort(mode, n_pairs=n_pairs, size_kind=size_kind,
                       size=size, size_range=(12, 36), noise=0.0, seed=1)


def _run(backend, cohort):
    state = backend.prepare(cohort, _CFG)
    return backend.run(state)


def _cpu_reference(cohort):
    ref = [b for b in B.all_backends() if b.name == B.REFERENCE_BACKEND][0]
    return _run(ref, cohort).scores


# ---------------------------------------------------------------------------
# CPU backends (always available)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_cpu_reference_recovers_optimum(mode):
    """The torch reference optimiser must recover a near-perfect overlap."""
    cohort = _cohort(mode)
    scores = _cpu_reference(cohort)
    assert np.all(np.isfinite(scores))
    assert scores.min() >= _NOISELESS_FLOOR, f"{mode}: min score {scores.min():.3f}"
    assert scores.max() <= 1.0 + 1e-4


@pytest.mark.parametrize("mode", MODES)
def test_cpu_analytical_agrees_with_autograd(mode):
    cohort = _cohort(mode)
    ref = _cpu_reference(cohort)
    ana = [b for b in B.all_backends() if b.name == "cpu_single_analytical"][0]
    sc = _run(ana, cohort).scores
    assert np.all(np.isfinite(sc))
    assert sc.min() >= _NOISELESS_FLOOR
    # Analytical and autograd optimise the same objective; allow modest slack.
    assert np.max(np.abs(sc - ref)) <= _AGREE_TOL


# ---------------------------------------------------------------------------
# GPU backends
# ---------------------------------------------------------------------------
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("mode", MODES)
def test_gpu_fast_single_recovers_and_agrees(mode):
    cohort = _cohort(mode)
    ref = _cpu_reference(cohort)
    fast = [b for b in B.all_backends() if b.name == "gpu_single_fast"][0]
    ok, reason = fast.available()
    if not ok:
        pytest.skip(reason)
    sc = _run(fast, cohort).scores
    assert np.all(np.isfinite(sc)), f"{mode}: fast produced non-finite scores"
    assert sc.min() >= _NOISELESS_FLOOR, f"{mode}: fast min {sc.min():.3f}"
    assert np.max(np.abs(sc - ref)) <= _AGREE_TOL


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("mode", MODES)
def test_gpu_batch_recovers_and_agrees(mode):
    cohort = _cohort(mode)
    ref = _cpu_reference(cohort)
    batch = [b for b in B.all_backends() if b.name == "gpu_multi_batch"][0]
    ok, reason = batch.available()
    if not ok:
        pytest.skip(reason)
    out = _run(batch, cohort)
    assert np.all(np.isfinite(out.scores)), f"{mode}: batch produced non-finite scores"
    assert out.scores.min() >= _NOISELESS_FLOOR, f"{mode}: batch min {out.scores.min():.3f}"
    assert np.max(np.abs(out.scores - ref)) <= _AGREE_TOL


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_bucketing_counts():
    """Uniform cohort -> single bucket; mixed cohort -> several buckets."""
    batch = [b for b in B.all_backends() if b.name == "gpu_multi_batch"][0]
    ok, reason = batch.available()
    if not ok:
        pytest.skip(reason)
    uni = make_cohort("surf", n_pairs=12, size_kind="uniform", size=20, noise=0.0, seed=2)
    mix = make_cohort("surf", n_pairs=12, size_kind="mixed", size_range=(10, 60),
                      noise=0.0, seed=2)
    assert batch._n_buckets(uni) == 1
    assert batch._n_buckets(mix) >= 2


# ---------------------------------------------------------------------------
# Runner smoke (no timing assertions)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_multidevice_shmap_simulated():
    """Exercise the multi-GPU shard_map path on N *simulated* devices.

    Runs in a subprocess so the ``XLA_FLAGS`` device-count override is applied
    before JAX is imported (and so it does not perturb the device count seen by
    the rest of the test session).  This validates the exact code that would run
    on a real multi-GPU host -- only the device kind differs.
    """
    import os
    import subprocess
    import sys

    try:
        import jax  # noqa: F401
    except Exception:
        pytest.skip("JAX not installed")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["SIM_DEVICES"] = "4"
    # ensure the subprocess can import the top-level `benchmarks` package
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.validate_multigpu"],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=900,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "platform=cpu devices=4" in out, out
    assert "MULTI-DEVICE VALIDATION OK" in out, out


def test_runner_smoke():
    from benchmarks.alignment_bench.runner import run_matrix, to_markdown
    results = run_matrix(modes=("vol",), n_pairs=4, uniform_size=14,
                         mixed_range=(10, 20), noise=0.0, seed=0,
                         cfg=B.BenchConfig(num_repeats=6, max_steps=30, steps_fine=30),
                         warmup=1, repeats=1)
    assert results["records"], "runner produced no records"
    # The CPU reference must always be present and finite.
    refs = [r for r in results["records"]
            if r["backend"] == B.REFERENCE_BACKEND and r["available"]]
    assert refs and np.isfinite(refs[0]["score"]["mean"])
    md = to_markdown(results)
    assert "Alignment speed + accuracy benchmark" in md
    assert "Fairness ledger" in md
