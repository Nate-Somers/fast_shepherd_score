"""
Multi-device (multi-GPU) alignment via JAX ``shard_map``.

This is the engine behind the ``gpu_multi_jax_shmap`` backend.  It distributes a
flat batch of pairs across ``len(jax.devices())`` devices: on a multi-GPU box
those are the GPUs; with no flag JAX picks all visible GPUs automatically.

How to make it actually run on >1 device
-----------------------------------------
* **Real multi-GPU:** nothing special -- JAX sees every visible GPU and
  ``shard_map`` shards the leading (pair) axis across them.  Set
  ``CUDA_VISIBLE_DEVICES`` to choose which GPUs participate.
* **Validate on a 1-GPU / CPU box:** JAX can fork N *virtual* CPU devices, which
  exercises the identical ``shard_map`` + ``vmap`` + ``scan`` code path.  Set,
  **before the first JAX import**::

      JAX_PLATFORMS=cpu
      XLA_FLAGS=--xla_force_host_platform_device_count=4

  Then ``len(jax.devices()) == 4`` and this module shards across 4 devices.
  ``benchmarks/validate_multigpu.py`` does exactly this in a subprocess.

The upstream ``shard_map`` kernels require the flat leading axis (number of
pairs) to be a multiple of the device count, and every pair padded to a common
point count with binary masks.  This module owns that bookkeeping so the rest of
the benchmark does not need to know about it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

# Keep JAX from grabbing the whole GPU at import time so it can coexist with the
# torch / Triton backends in the same process (set before the first jax import).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# Modes this engine supports (esp uses the masked volumetric-ESP kernel; surf
# reuses the masked volumetric (Gaussian-overlap) kernel on surface points).
SUPPORTED = ("vol", "surf", "esp", "pharm")


def jax_device_count() -> int:
    """Number of JAX devices, or 0 if JAX is unavailable."""
    try:
        import jax
        return len(jax.devices())
    except Exception:
        return 0


def jax_platform() -> str:
    try:
        import jax
        return jax.devices()[0].platform
    except Exception:
        return "none"


# ---------------------------------------------------------------------------
# Input preparation (numpy -> padded/masked batches + self-overlaps + se3 init)
# ---------------------------------------------------------------------------
def _pad_clouds(clouds: List[np.ndarray], max_n: int):
    """Stack variable-length (n_i, 3) clouds into (K, max_n, 3) + masks (K, max_n)."""
    K = len(clouds)
    pos = np.zeros((K, max_n, 3), dtype=np.float32)
    mask = np.zeros((K, max_n), dtype=np.float32)
    for i, c in enumerate(clouds):
        n = c.shape[0]
        pos[i, :n] = c
        mask[i, :n] = 1.0
    return pos, mask


def _pad_scalars(vals: List[np.ndarray], max_n: int):
    """Stack variable-length (n_i,) scalar-per-point arrays into (K, max_n)."""
    K = len(vals)
    out = np.zeros((K, max_n), dtype=np.float32)
    for i, v in enumerate(vals):
        out[i, : v.shape[0]] = v
    return out


def _pad_types(types: List[np.ndarray], max_n: int):
    K = len(types)
    out = np.zeros((K, max_n), dtype=np.int32)
    for i, t in enumerate(types):
        out[i, : t.shape[0]] = t
    return out


def _se3_init_batch(refs, fits, num_repeats: int) -> np.ndarray:
    """(K, R, 7) SE(3) initialisations built from the *unpadded* clouds."""
    import torch
    from shepherd_score.alignment import _initialize_se3_params
    out = []
    for ref, fit in zip(refs, fits):
        s = _initialize_se3_params(
            ref_points=torch.tensor(np.asarray(ref), dtype=torch.float32),
            fit_points=torch.tensor(np.asarray(fit), dtype=torch.float32),
            num_repeats=num_repeats,
        ).detach()
        if s.dim() == 1:
            s = s.unsqueeze(0)
        out.append(s.cpu().numpy().astype(np.float32))
    return np.stack(out)  # (K, R, 7)


def _tile_to_multiple(n_pairs: int, n_dev: int) -> Tuple[int, np.ndarray]:
    """Return (total, index_map) padding the pair axis up to a multiple of n_dev
    by repeating real pairs (never zero-mask pairs, which would divide by zero)."""
    if n_dev <= 1:
        return n_pairs, np.arange(n_pairs)
    total = ((n_pairs + n_dev - 1) // n_dev) * n_dev
    idx = np.arange(total) % n_pairs
    return total, idx


def prepare_inputs(mode: str, cohort, cfg) -> Dict[str, object]:
    """Build all device arrays for one cohort + mode. JITs happen on first run()."""
    import jax.numpy as jnp
    from shepherd_score.alignment._jax import (
        VAB_2nd_order_jax_mask,
        VAB_2nd_order_esp_jax_mask,
        get_overlap_pharm_jax_vectorized_mask,
    )

    n_dev = jax_device_count()
    n_pairs = len(cohort)
    total, idx = _tile_to_multiple(n_pairs, n_dev)
    pairs = [cohort.pairs[i] for i in idx]

    state: Dict[str, object] = {"mode": mode, "n_pairs": n_pairs, "total": total,
                                "n_dev": n_dev}

    if mode in ("vol", "surf"):
        ref_c = [(p.ref.atom_pos if mode == "vol" else p.ref.surf_pos) for p in pairs]
        fit_c = [(p.fit.atom_pos if mode == "vol" else p.fit.surf_pos) for p in pairs]
        N = max(c.shape[0] for c in ref_c)
        M = max(c.shape[0] for c in fit_c)
        ref_pos, mask_r = _pad_clouds(ref_c, N)
        fit_pos, mask_f = _pad_clouds(fit_c, M)
        VAA = np.array([float(VAB_2nd_order_jax_mask(jnp.array(ref_pos[i]), jnp.array(ref_pos[i]),
                        jnp.array(mask_r[i]), jnp.array(mask_r[i]), cfg.alpha)) for i in range(total)],
                       dtype=np.float32)
        VBB = np.array([float(VAB_2nd_order_jax_mask(jnp.array(fit_pos[i]), jnp.array(fit_pos[i]),
                        jnp.array(mask_f[i]), jnp.array(mask_f[i]), cfg.alpha)) for i in range(total)],
                       dtype=np.float32)
        se3 = _se3_init_batch(ref_c, fit_c, cfg.num_repeats)
        state.update(ref=jnp.array(ref_pos), fit=jnp.array(fit_pos),
                     mask_r=jnp.array(mask_r), mask_f=jnp.array(mask_f),
                     VAA=jnp.array(VAA), VBB=jnp.array(VBB), se3=jnp.array(se3))

    elif mode == "esp":
        ref_c = [p.ref.surf_pos for p in pairs]
        fit_c = [p.fit.surf_pos for p in pairs]
        ref_e = [p.ref.surf_esp for p in pairs]
        fit_e = [p.fit.surf_esp for p in pairs]
        N = max(c.shape[0] for c in ref_c)
        M = max(c.shape[0] for c in fit_c)
        ref_pos, mask_r = _pad_clouds(ref_c, N)
        fit_pos, mask_f = _pad_clouds(fit_c, M)
        ref_ch = _pad_scalars(ref_e, N)[..., None]   # (K, N, 1)
        fit_ch = _pad_scalars(fit_e, M)[..., None]
        VAA = np.array([float(VAB_2nd_order_esp_jax_mask(
                        jnp.array(ref_pos[i]), jnp.array(ref_pos[i]),
                        jnp.array(ref_ch[i]), jnp.array(ref_ch[i]),
                        jnp.array(mask_r[i]), jnp.array(mask_r[i]), cfg.alpha, cfg.lam))
                        for i in range(total)], dtype=np.float32)
        VBB = np.array([float(VAB_2nd_order_esp_jax_mask(
                        jnp.array(fit_pos[i]), jnp.array(fit_pos[i]),
                        jnp.array(fit_ch[i]), jnp.array(fit_ch[i]),
                        jnp.array(mask_f[i]), jnp.array(mask_f[i]), cfg.alpha, cfg.lam))
                        for i in range(total)], dtype=np.float32)
        se3 = _se3_init_batch(ref_c, fit_c, cfg.num_repeats)
        state.update(ref=jnp.array(ref_pos), fit=jnp.array(fit_pos),
                     ref_ch=jnp.array(ref_ch), fit_ch=jnp.array(fit_ch),
                     mask_r=jnp.array(mask_r), mask_f=jnp.array(mask_f),
                     VAA=jnp.array(VAA), VBB=jnp.array(VBB), se3=jnp.array(se3))

    elif mode == "pharm":
        ref_a = [p.ref.pharm_ancs for p in pairs]
        fit_a = [p.fit.pharm_ancs for p in pairs]
        N = max(a.shape[0] for a in ref_a)
        M = max(a.shape[0] for a in fit_a)
        ref_anc, mask_r = _pad_clouds(ref_a, N)
        fit_anc, mask_f = _pad_clouds(fit_a, M)
        ref_vec, _ = _pad_clouds([p.ref.pharm_vecs for p in pairs], N)
        fit_vec, _ = _pad_clouds([p.fit.pharm_vecs for p in pairs], M)
        ref_ph = _pad_types([p.ref.pharm_types for p in pairs], N)
        fit_ph = _pad_types([p.fit.pharm_types for p in pairs], M)
        ref_self = np.array([float(get_overlap_pharm_jax_vectorized_mask(
                        jnp.array(ref_ph[i]), jnp.array(ref_ph[i]),
                        jnp.array(ref_anc[i]), jnp.array(ref_anc[i]),
                        jnp.array(ref_vec[i]), jnp.array(ref_vec[i]),
                        jnp.array(mask_r[i]), jnp.array(mask_r[i]),
                        extended_points=False, only_extended=False)) for i in range(total)],
                       dtype=np.float32)
        fit_self = np.array([float(get_overlap_pharm_jax_vectorized_mask(
                        jnp.array(fit_ph[i]), jnp.array(fit_ph[i]),
                        jnp.array(fit_anc[i]), jnp.array(fit_anc[i]),
                        jnp.array(fit_vec[i]), jnp.array(fit_vec[i]),
                        jnp.array(mask_f[i]), jnp.array(mask_f[i]),
                        extended_points=False, only_extended=False)) for i in range(total)],
                       dtype=np.float32)
        se3 = _se3_init_batch(ref_a, fit_a, cfg.num_repeats)
        state.update(ref_ph=jnp.array(ref_ph), fit_ph=jnp.array(fit_ph),
                     ref_anc=jnp.array(ref_anc), fit_anc=jnp.array(fit_anc),
                     ref_vec=jnp.array(ref_vec), fit_vec=jnp.array(fit_vec),
                     mask_r=jnp.array(mask_r), mask_f=jnp.array(mask_f),
                     ref_self=jnp.array(ref_self), fit_self=jnp.array(fit_self),
                     se3=jnp.array(se3))
    else:
        raise ValueError(f"jax_shmap does not support mode={mode!r}")

    # block until all device arrays are materialised so prepare timing is honest
    import jax
    for v in state.values():
        if hasattr(v, "block_until_ready"):
            v.block_until_ready()
    return state


def run(state: Dict[str, object], cfg) -> np.ndarray:
    """Run the appropriate shard_map kernel; return per-pair best scores (n_pairs,)."""
    import numpy as np
    import jax
    from shepherd_score.alignment import _jax_parallel as JP

    mode = state["mode"]
    n_pairs = state["n_pairs"]

    if mode in ("vol", "surf"):
        _, _, scores = JP.optimize_ROCS_overlay_jax_vol_shmap(
            state["ref"], state["fit"], state["mask_r"], state["mask_f"],
            state["VAA"], state["VBB"], state["se3"],
            cfg.alpha, cfg.lr_cpu, cfg.max_steps)
    elif mode == "esp":
        _, _, scores = JP.optimize_ROCS_esp_overlay_jax_vol_esp_shmap(
            state["ref"], state["fit"], state["ref_ch"], state["fit_ch"],
            state["mask_r"], state["mask_f"], state["VAA"], state["VBB"], state["se3"],
            cfg.alpha, cfg.lam, cfg.lr_cpu, cfg.max_steps)
    elif mode == "pharm":
        _, _, _, scores = JP.optimize_pharm_overlay_jax_pharm_shmap(
            state["ref_ph"], state["fit_ph"], state["ref_anc"], state["fit_anc"],
            state["ref_vec"], state["fit_vec"], state["mask_r"], state["mask_f"],
            state["ref_self"], state["fit_self"], state["se3"],
            cfg.similarity, False, False, cfg.lr_cpu, cfg.max_steps)
    else:
        raise ValueError(mode)

    scores = np.asarray(jax.device_get(scores), dtype=float)
    return scores[:n_pairs]
