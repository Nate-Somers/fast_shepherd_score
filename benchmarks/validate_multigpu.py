"""
Validate the multi-device (multi-GPU) shard_map alignment path.

Because ``shard_map`` shards over ``jax.devices()``, we can exercise the exact
multi-GPU code path on a single-GPU (or CPU-only) box by asking JAX to fork N
*virtual* devices.  The env vars MUST be set before JAX is imported, which is
why this lives in its own entry point / subprocess.

Usage
-----
    SIM_DEVICES=4 python -m benchmarks.validate_multigpu          # simulate 4 CPU devices
    python -m benchmarks.validate_multigpu --real                 # use real devices as-is

On a real multi-GPU host, ``--real`` shards across the physical GPUs (no env
flags needed; optionally restrict with CUDA_VISIBLE_DEVICES).
"""
import os
import sys


def _configure_simulated_devices() -> int:
    """Set XLA flags for N virtual CPU devices. Must run before importing jax."""
    n = int(os.environ.get("SIM_DEVICES", "4"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + f" --xla_force_host_platform_device_count={n}").strip()
    return n


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    real = "--real" in argv

    if not real:
        want = _configure_simulated_devices()
    else:
        want = None

    # Imports happen *after* the env is configured.
    import numpy as np
    import jax

    n_dev = len(jax.devices())
    plat = jax.devices()[0].platform
    print(f"[multigpu] platform={plat} devices={n_dev}")
    if want is not None and n_dev != want:
        print(f"[multigpu] WARNING expected {want} devices, got {n_dev} "
              f"(XLA_FLAGS may have been set too late)")
    if n_dev <= 1:
        print("[multigpu] only 1 device visible -- nothing to shard. "
              "Set SIM_DEVICES>1 or run on a multi-GPU host.")
        return 2

    from benchmarks.alignment_bench.workloads import make_cohort
    from benchmarks.alignment_bench import backends as B
    from benchmarks.alignment_bench import jax_shmap

    backend = [b for b in B.all_backends() if b.name == "gpu_multi_jax_shmap"][0]
    ok, reason = backend.available()
    if not ok:
        print(f"[multigpu] backend unavailable: {reason}")
        return 2

    cfg = B.BenchConfig(num_repeats=8, max_steps=60, lr_cpu=0.1)
    n_pairs = max(10, n_dev * 3 + 1)   # not a multiple of n_dev -> exercises padding
    all_ok = True
    for mode in jax_shmap.SUPPORTED:
        cohort = make_cohort(mode, n_pairs=n_pairs, size_kind="mixed",
                             size_range=(10, 30), noise=0.0, seed=3)
        state = backend.prepare(cohort, cfg)
        out = backend.run(state)
        n_real = out.scores.shape[0]
        total = out.extra["total_padded"]
        finite = bool(np.all(np.isfinite(out.scores)))
        recovered = float(out.scores.min())
        sharded_ok = (total % n_dev == 0) and (total >= n_pairs)
        status = "OK" if (finite and recovered >= 0.85 and n_real == n_pairs and sharded_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[multigpu] mode={mode:5s} devices={out.extra['n_devices']} "
              f"pairs={n_real} padded_total={total} "
              f"min={recovered:.3f} mean={out.scores.mean():.3f} -> {status}")

    if all_ok:
        print("MULTI-DEVICE VALIDATION OK")
        return 0
    print("MULTI-DEVICE VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
