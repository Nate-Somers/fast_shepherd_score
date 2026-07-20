"""MoleculePairBatch: batch of MoleculePair objects for fast sequential JAX alignment."""
from importlib.metadata import version as _pkg_version
from typing import List, Optional, Tuple

import numpy as np

from shepherd_score.container._core import MoleculePair
from shepherd_score.container._batch_utils import (
    _pad_arrays,
    _dispatch_parallel,
    _align_vol_shmap,
    _align_vol_esp_shmap,
    _align_surf_shmap,
    _align_esp_shmap,
    _align_pharm_shmap,
    _align_vol_worker,
    _align_vol_esp_worker,
    _align_surf_worker,
    _align_esp_worker,
    _align_pharm_worker,
)


def _resolve_backend(backend):
    """Resolve ``backend=None`` to the device-aware default: ``triton`` when CUDA is
    available, else ``numba``. Explicit values (including ``"jax"``) pass through. Shares
    the single source of truth with the screening front-end (``screen._default_backend``)
    via a lazy import (no circular import at module load)."""
    if backend is not None:
        return backend
    from shepherd_score.screen import _default_backend
    return _default_backend()


def _default_steps(mode: str) -> int:
    """Per-mode default fine-step count from the single-source table in
    ``accel.batch.aligners`` (``_steps_for``). Lazy-imported so this module stays importable
    without the accel/torch stack. Used to resolve ``max_num_steps=None`` in ``align_with_*``."""
    from shepherd_score.accel.batch.aligners import _steps_for
    return _steps_for(mode)


def _default_seeds(mode: str) -> int:
    """Per-mode default SE(3) seed count (``MODE_SEEDS``) from the single-source table in
    ``accel.batch.aligners`` (``_seeds_for``). Lazy-imported, same as ``_default_steps``. Used to
    resolve ``num_repeats=None`` in ``align_with_*`` so the shipped per-mode default is what
    actually runs. Resolved to an int here (rather than passing ``None`` down) because the JAX,
    cpu_pool and per-pair fallback branches all require a concrete count."""
    from shepherd_score.accel.batch.aligners import _seeds_for
    return _seeds_for(mode)


def _compute_bucket_splits(sizes_a, sizes_b, num_buckets):
    """Sort pairs by (max(a,b), min(a,b)) and split into buckets.

    Parameters
    ----------
    sizes_a, sizes_b : array-like of int
        Per-pair sizes (e.g. atom counts) for the two molecules.
    num_buckets : int
        Number of buckets.  ``<= 1`` returns a single bucket with all
        pairs in their original order (no sorting).

    Returns
    -------
    list of list of int
        Each inner list is a bucket of global pair indices.
    """
    n = len(sizes_a)
    if num_buckets <= 1:
        return [list(range(n))]
    sizes_a = np.asarray(sizes_a)
    sizes_b = np.asarray(sizes_b)
    sort_keys = np.array([np.minimum(sizes_a, sizes_b),
                           np.maximum(sizes_a, sizes_b)])
    sorted_order = np.lexsort(sort_keys)
    num_buckets_actual = min(num_buckets, n)
    return [
        arr.tolist()
        for arr in np.array_split(sorted_order, num_buckets_actual)
        if len(arr) > 0
    ]


class MoleculePairBatch:
    """Batch of MoleculePair objects for fast sequential JAX alignment.

    Pads all atom coordinate arrays to common max shapes so JAX's XLA compiler
    reuses the same compiled function for every pair, avoiding recompilation.
    This modifies each MoleculePair in-place (stores results on the pair).

    This is currently optimized for CPU. A GPU-optimized version would
    benefit from optimizing batches of pairs and using a GPU-optimized alignment.
    """

    def __init__(self, pairs: List[MoleculePair]):
        self.pairs = pairs

    # Backend names that route to the Triton/CUDA MoleculePair._align_batch_* path.
    _TRITON_BACKENDS = ("triton", "cuda", "gpu")
    # Backend names that force the batched **CPU numba** path: the *same*
    # coarse-to-fine drivers the Triton path uses, with the numba kernels in place
    # of Triton. This is the explicit, portable counterpart to relying on the
    # Triton path's import-time CPU fallback.
    _NUMBA_BACKENDS = ("numba", "cpu")

    def _prepare_numba(self):
        """Force the batched CPU numba path for a ``backend="numba"`` call.

        Moves every pair onto CPU. Kernel selection is per-call and device-driven
        (see :mod:`shepherd_score.accel.kernels.dispatch`), so the CPU
        tensors run the numba kernels **even in a process that also has the Triton
        GPU kernels loaded** -- e.g. to reserve the GPU for another task or to run a
        deterministic CPU pass on a GPU box.
        """
        import torch
        try:
            import numba  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "backend='numba' needs the 'numba' package (pip install "
                "'shepherd-score[cpu]').") from exc
        cpu = torch.device("cpu")
        for p in self.pairs:
            p.device = cpu

    def _run_fast_or_fallthrough(self, backend, align_fn, align_kwargs, score_attr,
                                 transform_attr, fit_attr, return_aligned, *,
                                 num_workers: int = 1, precheck=None):
        """Shared ``backend=`` dispatch for the modes whose fast path is a single
        ``_triton_align`` call (vol/vol_esp/surf/esp/esp_combo).

        Returns ``(handled, result)``: on a triton/numba backend it runs the batched
        path and returns ``(True, result)``; on ``"jax"`` it returns ``(False, None)``
        so the caller runs the original JAX path; any other value raises ``ValueError``.
        ``precheck`` (if given) runs AFTER ``_prepare_numba`` for the numba backend,
        preserving the original ordering of mode-specific guards (e.g. the ``no_H``
        check).
        """
        backend = _resolve_backend(backend)
        if backend in self._TRITON_BACKENDS or backend in self._NUMBA_BACKENDS:
            if backend in self._NUMBA_BACKENDS:
                self._prepare_numba()
            if precheck is not None:
                precheck()
            return True, self._triton_align(align_fn, align_kwargs, score_attr,
                                            transform_attr, fit_attr, return_aligned,
                                            num_workers=num_workers)
        if backend != "jax":
            raise ValueError(f"unknown backend {backend!r}; use 'jax', 'triton', or 'numba'")
        return False, None

    def _triton_align(self, align_fn, align_kwargs, score_attr, transform_attr,
                      fit_attr, return_aligned, num_workers: int = 1):
        """Route a batch alignment through the Triton ``MoleculePair._align_batch_*``
        path with ZERO extra alignment work.

        ``num_workers > 1`` on the CPU (numba) path shards the pairs across a persistent
        single-threaded process pool (:mod:`shepherd_score.accel.cpu_pool`) for
        near-linear multi-core scaling. Pairs are independent, so sharding does not
        change the optimization problem, but results agree to convergence tolerance
        rather than bitwise: the fine loop's early-stop tests a batch-GLOBAL max, so a
        pair's step count depends on which pairs share its shard. It is ignored on CUDA
        tensors and for modes the pool does not cover -- only ``vol``, ``surf``,
        ``surf_esp`` and ``pharm`` have a pool path; the rest run the single call.

        ``align_fn(self.pairs, **align_kwargs)`` is byte-identical to calling the
        Triton static method directly, so alignment throughput is unchanged (and it
        inherits the same multi-GPU sharding via ``_should_distribute``). Scores +
        SE(3) transforms are read from the in-place results the Triton path writes
        (also populated in-place by the multi-GPU path).

        ``aligned_list`` (transformed fit points) is OFF by default — the Triton
        path's primary outputs are the score and the stored transform, and a user
        can apply the transform themselves. When ``return_aligned=True`` it is built
        GPU-batched from the already-cached fit tensor (``fit_attr``), grouped by
        device so multi-GPU shards (whose tensors live on different devices) are
        handled correctly.
        """
        pairs = self.pairs
        mode = align_fn.__name__.replace("_align_batch_", "")
        if (num_workers and num_workers > 1 and pairs
                and pairs[0].device.type == "cpu"):
            from shepherd_score.accel import cpu_pool as _cpu_pool
            if mode in _cpu_pool.POOL_MODES:
                _cpu_pool.align_pairs(mode, pairs, num_workers, align_kwargs)
            else:
                align_fn(pairs, **align_kwargs)          # mode not pooled -> single call
        else:
            align_fn(pairs, **align_kwargs)              # <- identical to standalone Triton call
        scores = np.array([float(getattr(p, score_attr)) for p in pairs])
        if not return_aligned:
            return scores, [None] * len(pairs)

        import torch
        aligned = [None] * len(pairs)
        by_dev: dict = {}
        for i, p in enumerate(pairs):
            by_dev.setdefault(getattr(p, fit_attr).device, []).append(i)
        for dev, idxs in by_dev.items():
            fits = [getattr(pairs[i], fit_attr) for i in idxs]
            Ss = torch.stack([torch.as_tensor(getattr(pairs[i], transform_attr),
                                              dtype=torch.float32) for i in idxs]).to(dev)
            fit_pad = torch.nn.utils.rnn.pad_sequence(fits, batch_first=True)  # (k, Mpad, 3)
            # aligned = fit @ R.T + t  (row-vector SE(3) convention)
            aligned_pad = torch.baddbmm(Ss[:, :3, 3][:, None, :], fit_pad,
                                        Ss[:, :3, :3].transpose(1, 2)).cpu().numpy()
            for j, i in enumerate(idxs):
                aligned[i] = aligned_pad[j, :fits[j].shape[0]]
        return scores, aligned

    def _pad_and_mask_vol(self, no_H: bool = True, include_charges: bool = False):
        """Extract, pad, and create masks for volumetric (and optionally ESP) alignment.

        Does NOT modify the pair objects. Returns padded arrays and masks.

        Parameters
        ----------
        no_H : bool
            If True, use heavy-atom positions (atom_pos). If False, use all atoms.
        include_charges : bool
            If True, also extract and pad partial charge arrays. The returned tuple
            per entry gains two extra elements: ``(ref_pos_pad, fit_pos_pad,
            ref_ch_pad, fit_ch_pad, mask_ref, mask_fit, orig_ref, orig_fit)``.
            If False, each entry is ``(ref_padded, fit_padded, mask_ref, mask_fit,
            orig_ref, orig_fit)``.

        Returns
        -------
        entries : list of tuples
        max_ref_len : int
        max_fit_len : int
        """
        if no_H:
            ref_pos_arrays = [p.ref_molec.atom_pos for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.atom_pos for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges[p.ref_molec._nonH_atoms_idx]
                                 for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges[p.fit_molec._nonH_atoms_idx]
                                 for p in self.pairs]
        else:
            ref_pos_arrays = [p.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            fit_pos_arrays = [p.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
                              for p in self.pairs]
            if include_charges:
                ref_ch_arrays = [p.ref_molec.partial_charges for p in self.pairs]
                fit_ch_arrays = [p.fit_molec.partial_charges for p in self.pairs]

        ref_padded, masks_ref, orig_refs, max_ref_len = _pad_arrays(ref_pos_arrays)
        fit_padded, masks_fit, orig_fits, max_fit_len = _pad_arrays(fit_pos_arrays)

        if include_charges:
            ref_ch_padded, _, _, _ = _pad_arrays(ref_ch_arrays)
            fit_ch_padded, _, _, _ = _pad_arrays(fit_ch_arrays)
            entries = [
                (rp, fp, rc, fc, mr, mf, ori_r, ori_f)
                for rp, fp, rc, fc, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, ref_ch_padded, fit_ch_padded,
                    masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]
        else:
            entries = [
                (rp, fp, mr, mf, ori_r, ori_f)
                for rp, fp, mr, mf, ori_r, ori_f in zip(
                    ref_padded, fit_padded, masks_ref, masks_fit, orig_refs, orig_fits
                )
            ]

        return entries, max_ref_len, max_fit_len

    def align_with_vol(self,
                       no_H: bool = True,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = None,
                       num_workers: int = 1,
                       use_shmap: bool = True,
                       num_buckets: int = 1,
                       verbose: bool = False,
                       backend: Optional[str] = None,
                       alpha: float = 0.81,
                       return_aligned: bool = False,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks
        and processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_noH`` / ``pair.sim_aligned_vol_noH`` (when ``no_H=True``)
        - ``pair.transform_vol``     / ``pair.sim_aligned_vol``      (when ``no_H=False``)

        Parameters
        ----------
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        num_workers : int
            Number of parallel workers.  ``1`` (default) runs sequentially
            in-process. When ``use_shmap=True`` (the default), this value is informational;
            actual parallelism equals ``len(jax.devices())``, which is set by
            ``XLA_FLAGS`` **before** JAX is first imported. When ``use_shmap=False``
            use ``multiprocessing`` with a ``'spawn'`` start method.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global atom-count maximum —
            lowest overhead for typical use.  Values > 1 sort pairs by
            ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        backend : str
            ``"jax"`` (default) uses the JAX/XLA path below. ``"triton"`` (aliases
            ``"cuda"``/``"gpu"``) routes to the Triton ``MoleculePair._align_batch_vol``
            kernel path (heavy-atom only), which also handles multi-GPU internally via
            ``_should_distribute``. ``"numba"`` (alias ``"cpu"``) runs that *same*
            batched driver on CPU with the numba kernels -- it forces every pair onto
            CPU and requires a Triton-free process (otherwise use ``"triton"``).
            ``max_num_steps`` maps to the ``steps_fine`` count.
        return_aligned : bool
            For the Triton backend, skip building ``aligned_list`` when ``False``
            (pure delegation, zero overhead over a direct ``_align_batch_vol`` call).
        """
        def _no_H_guard():
            if not no_H:
                raise NotImplementedError(
                    "the Triton/numba vol backend aligns heavy atoms only (no_H=True)")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol")
        if num_repeats is None:
            num_repeats = _default_seeds("vol")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol,
            dict(alpha=alpha, steps_fine=max_num_steps),
            "sim_aligned_vol_noH", "transform_vol_noH", "_fit_xyz_t", return_aligned,
            num_workers=num_workers, precheck=_no_H_guard)
        if handled:
            return _result

        # build raw (unpadded) position arrays for every pair
        raw_refs, raw_fits, trans_centers_list = [], [], []
        for pair in self.pairs:
            if no_H:
                ref_pos = pair.ref_molec.atom_pos
                fit_pos = pair.fit_molec.atom_pos
            else:
                ref_pos = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                fit_pos = pair.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
            raw_refs.append(ref_pos)
            raw_fits.append(fit_pos)

            tc = None
            if trans_init:
                tc = ref_pos  # already numpy; worker copies implicitly
            trans_centers_list.append(tc)

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_list = [None] * n_pairs

        if use_shmap and num_workers > 1:  # shard_map path (single process, multi-device)
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            pair_data = list(zip(raw_refs, raw_fits, trans_centers_list))
            results = _align_vol_shmap(
                pair_data, num_workers, num_repeats, lr, max_num_steps, verbose,
                num_buckets=num_buckets,
            )
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                if no_H:
                    pair.transform_vol_noH = se3_transform
                    pair.sim_aligned_vol_noH = score
                else:
                    pair.transform_vol = se3_transform
                    pair.sim_aligned_vol = score

        elif num_workers > 1:  # multiprocessing path
            pair_data = list(zip(raw_refs, raw_fits, trans_centers_list))
            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            # Primary key: max(ref, fit) — dominates padding; secondary: min.
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_vol_worker, num_workers,
                (num_repeats, lr, max_num_steps, verbose),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    if no_H:
                        pair.transform_vol_noH = se3_transform
                        pair.sim_aligned_vol_noH = score
                    else:
                        pair.transform_vol = se3_transform
                        pair.sim_aligned_vol = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_vol. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_ROCS_overlay_jax_mask

            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_refs = [raw_refs[i] for i in bucket_idx_list]
                bucket_fits = [raw_fits[i] for i in bucket_idx_list]
                ref_padded_b, masks_ref_b, _orig_refs_b, _ = _pad_arrays(bucket_refs)
                fit_padded_b, masks_fit_b, orig_fits_b, _ = _pad_arrays(bucket_fits)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    aligned_pts, se3_transform, score = optimize_ROCS_overlay_jax_mask(
                        ref_points=jnp.array(ref_padded_b[local_j]),
                        fit_points=jnp.array(fit_padded_b[local_j]),
                        mask_ref=jnp.array(masks_ref_b[local_j]),
                        mask_fit=jnp.array(masks_fit_b[local_j]),
                        alpha=0.81,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers_list[global_i],
                        lr=lr,
                        max_num_steps=max_num_steps,
                        verbose=verbose,
                    )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_pts = np.array(aligned_pts)[:orig_fits_b[local_j]]
                    scores[global_i] = score

                    if no_H:
                        pair.transform_vol_noH = se3_transform
                        pair.sim_aligned_vol_noH = score
                    else:
                        pair.transform_vol = se3_transform
                        pair.sim_aligned_vol = score

                    aligned_list[global_i] = aligned_pts

        return scores, aligned_list

    def align_with_vol_esp(self,
                           lam: float = 0.1,
                           no_H: bool = True,
                           num_repeats: int = None,
                           trans_init: bool = False,
                           lr: float = 0.1,
                           max_num_steps: int = None,
                           num_workers: int = 1,
                           use_shmap: bool = True,
                           num_buckets: int = 1,
                           verbose: bool = False,
                           backend: Optional[str] = None,
                           alpha: float = 0.81,
                           return_aligned: bool = False,
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using padded masked volumetric ESP similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks
        and processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_esp_noH`` / ``pair.sim_aligned_vol_esp_noH`` (when ``no_H=True``)
        - ``pair.transform_vol_esp``     / ``pair.sim_aligned_vol_esp``      (when ``no_H=False``)

        Parameters
        ----------
        lam : float
            Partial charge weighting parameter. Typically 0.1 for volumetric.
        no_H : bool
            Whether to exclude hydrogens. Default is True.
        num_repeats : int
            Number of SE(3) initializations per pair.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        trans_init : bool
            If True, initialize translations to each ref atom position. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global atom-count maximum —
            lowest overhead for typical use.  Values > 1 sort pairs by
            ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates (unpadded) for each pair.
        """
        # Build raw (unpadded) per-pair data tuples (plain numpy — picklable).
        def _no_H_guard():
            if not no_H:
                raise NotImplementedError(
                    "the Triton/numba vol_esp backend aligns heavy atoms only (no_H=True)")
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_esp")
        if num_repeats is None:
            num_repeats = _default_seeds("vol_esp")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol_esp,
            dict(alpha=alpha, lam=lam, num_repeats=num_repeats,
                 trans_init=trans_init, lr=lr, steps_fine=max_num_steps),
            "sim_aligned_vol_esp_noH", "transform_vol_esp_noH", "_fit_xyz_t",
            return_aligned, precheck=_no_H_guard)
        if handled:
            return _result

        raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list = [], [], [], [], []
        for pair in self.pairs:
            if no_H:
                ref_pos = pair.ref_molec.atom_pos
                fit_pos = pair.fit_molec.atom_pos
                ref_ch = pair.ref_molec.partial_charges[pair.ref_molec._nonH_atoms_idx]
                fit_ch = pair.fit_molec.partial_charges[pair.fit_molec._nonH_atoms_idx]
            else:
                ref_pos = pair.ref_molec.mol.GetConformer().GetPositions().astype(np.float32)
                fit_pos = pair.fit_molec.mol.GetConformer().GetPositions().astype(np.float32)
                ref_ch = pair.ref_molec.partial_charges
                fit_ch = pair.fit_molec.partial_charges
            raw_refs.append(ref_pos)
            raw_fits.append(fit_pos)
            raw_ref_ch.append(ref_ch)
            raw_fit_ch.append(fit_ch)
            tc = ref_pos if trans_init else None
            trans_centers_list.append(tc)

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_list = [None] * n_pairs

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            pair_data = list(zip(raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list))
            results = _align_vol_esp_shmap(
                pair_data, num_workers, lam, num_repeats, lr, max_num_steps, verbose,
                num_buckets=num_buckets,
            )
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                if no_H:
                    pair.transform_vol_esp_noH = se3_transform
                    pair.sim_aligned_vol_esp_noH = score
                else:
                    pair.transform_vol_esp = se3_transform
                    pair.sim_aligned_vol_esp = score

        elif num_workers > 1: # parallel
            pair_data = list(zip(raw_refs, raw_fits, raw_ref_ch, raw_fit_ch, trans_centers_list))
            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_vol_esp_worker, num_workers,
                (lam, num_repeats, lr, max_num_steps, verbose),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    if no_H:
                        pair.transform_vol_esp_noH = se3_transform
                        pair.sim_aligned_vol_esp_noH = score
                    else:
                        pair.transform_vol_esp = se3_transform
                        pair.sim_aligned_vol_esp = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_vol_esp. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_ROCS_esp_overlay_jax_mask

            ref_sizes = np.array([len(r) for r in raw_refs])
            fit_sizes = np.array([len(f) for f in raw_fits])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_refs = [raw_refs[i] for i in bucket_idx_list]
                bucket_fits = [raw_fits[i] for i in bucket_idx_list]
                bucket_ref_ch = [raw_ref_ch[i] for i in bucket_idx_list]
                bucket_fit_ch = [raw_fit_ch[i] for i in bucket_idx_list]
                ref_padded_b, masks_ref_b, _orig_refs_b, _ = _pad_arrays(bucket_refs)
                fit_padded_b, masks_fit_b, orig_fits_b, _ = _pad_arrays(bucket_fits)
                ref_ch_padded_b, _, _, _ = _pad_arrays(bucket_ref_ch)
                fit_ch_padded_b, _, _, _ = _pad_arrays(bucket_fit_ch)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    aligned_pts, se3_transform, score = optimize_ROCS_esp_overlay_jax_mask(
                        ref_points=jnp.array(ref_padded_b[local_j]),
                        fit_points=jnp.array(fit_padded_b[local_j]),
                        ref_charges=jnp.array(ref_ch_padded_b[local_j]),
                        fit_charges=jnp.array(fit_ch_padded_b[local_j]),
                        mask_ref=jnp.array(masks_ref_b[local_j]),
                        mask_fit=jnp.array(masks_fit_b[local_j]),
                        alpha=0.81,
                        lam=lam,
                        num_repeats=num_repeats,
                        trans_centers=trans_centers_list[global_i],
                        lr=lr,
                        max_num_steps=max_num_steps,
                        verbose=verbose,
                    )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_pts = np.array(aligned_pts)[:orig_fits_b[local_j]]
                    scores[global_i] = score

                    if no_H:
                        pair.transform_vol_esp_noH = se3_transform
                        pair.sim_aligned_vol_esp_noH = score
                    else:
                        pair.transform_vol_esp = se3_transform
                        pair.sim_aligned_vol_esp = score

                    aligned_list[global_i] = aligned_pts

        return scores, aligned_list

    def _delegate_alignment(self, method_name: str, score_attr: str, **kwargs):
        """Delegate alignment to each MoleculePair's method and collect results.

        Parameters
        ----------
        method_name : str
            Name of the MoleculePair method to call (e.g. 'align_with_surf').
        score_attr : str
            Name of the attribute on MoleculePair where the score is stored after alignment.
        **kwargs
            Forwarded to each pair's method.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit coordinates for each pair.
        """
        aligned_list = []
        scores = np.zeros(len(self.pairs))
        for i, pair in enumerate(self.pairs):
            aligned_pts = getattr(pair, method_name)(**kwargs)
            scores[i] = float(getattr(pair, score_attr))
            aligned_list.append(aligned_pts)
        return scores, aligned_list

    def align_with_surf(self,
                        alpha: float,
                        num_repeats: int = None,
                        trans_init: bool = False,
                        lr: float = 0.1,
                        max_num_steps: int = None,
                        use_jax: bool = True,
                        use_analytical: bool = True,
                        num_workers: int = 1,
                        use_shmap: bool = False,
                        verbose: bool = False,
                        backend: Optional[str] = None,
                        return_aligned: bool = False,
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using surface similarity.

        Surface arrays are the same size across all pairs so no padding or
        size-sorting is needed.  It is not recommended to use multiprocessing
        due to this reason.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf`` and ``pair.sim_aligned_surf``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        num_repeats : int
            Number of SE(3) initializations per pair.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            Whether to use JAX shard_map for parallel alignment. Default is False.
            Performance is better when use_shmap is False on cpu.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        backend : str
            ``"jax"`` (default) or ``"triton"`` (aliases ``"cuda"``/``"gpu"``) which
            routes to ``MoleculePair._align_batch_surf`` (multi-GPU-aware). ``return_aligned``
            controls building the aligned-surface list (off by default = pure delegation).
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("surf")
        if num_repeats is None:
            num_repeats = _default_seeds("surf")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_surf,
            dict(alpha=alpha, steps_fine=max_num_steps),
            "sim_aligned_surf", "transform_surf", "_fit_surf_t", return_aligned,
            num_workers=num_workers)
        if handled:
            return _result

        n_pairs = len(self.pairs)
        pair_data = [
            (pair.ref_molec.surf_pos,
             pair.fit_molec.surf_pos,
             pair.ref_molec.atom_pos if trans_init else None)
            for pair in self.pairs
        ]

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_surf_shmap(
                pair_data, num_workers, alpha, num_repeats, lr, max_num_steps, verbose,
            )
            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                pair.transform_surf = se3_transform
                pair.sim_aligned_surf = score
            return scores, aligned_list

        elif num_workers > 1: # parallel
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, None, _align_surf_worker, num_workers,
                (alpha, num_repeats, lr, max_num_steps, use_jax, use_analytical, verbose),
            )

            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    pair.transform_surf = se3_transform
                    pair.sim_aligned_surf = score
            return scores, aligned_list

        # sequential
        return self._delegate_alignment(
            'align_with_surf', 'sim_aligned_surf',
            alpha=alpha,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def align_with_surf_esp(self,
                       alpha: float,
                       lam: float = 0.3,
                       num_repeats: int = None,
                       trans_init: bool = False,
                       lr: float = 0.1,
                       max_num_steps: int = None,
                       use_jax: bool = True,
                       use_analytical: bool = True,
                       num_workers: int = 1,
                       use_shmap: bool = False,
                       verbose: bool = False,
                       backend: Optional[str] = None,
                       return_aligned: bool = False,
                       ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using ESP+surface similarity.

        Surface arrays are the same size across all pairs so no padding or
        size-sorting is needed.  It is not recommended to use multiprocessing
        due to this reason.

        ``surf_esp`` is the canonical name for the legacy ``esp`` mode (alias kept).
        Results are stored in-place on each MoleculePair:
        - ``pair.transform_surf_esp`` and ``pair.sim_aligned_surf_esp``

        Parameters
        ----------
        alpha : float
            Gaussian width parameter for overlap.
        lam : float
            Weighting factor for ESP scoring. Scaled internally. Default is 0.3.
        num_repeats : int
            Number of SE(3) initializations per pair.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        trans_init : bool
            Apply translation initialization for alignment. Default is False.
        lr : float
            Optimizer learning rate. Default is 0.1.
        max_num_steps : int
            Maximum optimization steps.
            Default (``None``) is the per-mode value in ``shepherd_score/accel/_modes.py``
            (MODE_SEEDS / MODE_STEPS).
        use_jax : bool
            Whether to use JAX backend. Default is True.
        use_analytical : bool
            Whether to use analytical gradients (PyTorch only). Default is True.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            Whether to use JAX shard_map for parallel alignment. Default is False.
            Performance is better when use_shmap is False on cpu.
        verbose : bool
            Print scores per pair. Default is False.

        Returns
        -------
        scores : np.ndarray
            Scores for each pair. Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates for each pair.
        backend : str
            ``"jax"`` (default) or ``"triton"`` (aliases ``"cuda"``/``"gpu"``) which
            routes to ``MoleculePair._align_batch_surf_esp`` (multi-GPU-aware; it applies the
            same internal LAM_SCALING as this path, so ``lam`` is consistent across
            backends). ``return_aligned`` controls the aligned-surface list.
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("surf_esp")
        if num_repeats is None:
            num_repeats = _default_seeds("surf_esp")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_surf_esp,
            dict(alpha=alpha, lam=lam, num_repeats=num_repeats,
                 trans_init=trans_init, lr=lr, steps_fine=max_num_steps),
            "sim_aligned_surf_esp", "transform_surf_esp", "_fit_surf_t", return_aligned,
            num_workers=num_workers)
        if handled:
            return _result

        from shepherd_score.score.constants import LAM_SCALING
        lam_scaled = float(LAM_SCALING * lam)

        n_pairs = len(self.pairs)
        pair_data = [
            (pair.ref_molec.surf_pos,
             pair.fit_molec.surf_pos,
             pair.ref_molec.surf_esp,
             pair.fit_molec.surf_esp,
             pair.ref_molec.atom_pos if trans_init else None)
            for pair in self.pairs
        ]

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_esp_shmap(
                pair_data, num_workers, alpha, lam_scaled, num_repeats, lr, max_num_steps, verbose,
            )
            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for i, (score, se3_transform, aligned_pts) in enumerate(results):
                scores[i] = score
                aligned_list[i] = aligned_pts
                pair = self.pairs[i]
                pair.transform_surf_esp = se3_transform
                pair.sim_aligned_surf_esp = score
            return scores, aligned_list

        elif num_workers > 1: # parallel
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, None, _align_esp_worker, num_workers,
                (alpha, lam_scaled, num_repeats, lr, max_num_steps,
                 use_jax, use_analytical, verbose),
            )

            scores = np.zeros(n_pairs)
            aligned_list = [None] * n_pairs
            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_pts) in zip(idx_list, chunk_result):
                    scores[global_i] = score
                    aligned_list[global_i] = aligned_pts
                    pair = self.pairs[global_i]
                    pair.transform_surf_esp = se3_transform
                    pair.sim_aligned_surf_esp = score
            return scores, aligned_list

        # sequential
        return self._delegate_alignment(
            'align_with_surf_esp', 'sim_aligned_surf_esp',
            alpha=alpha,
            lam=lam,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            use_jax=use_jax,
            use_analytical=use_analytical,
            verbose=verbose,
        )

    def align_with_vol_and_surf_esp(self,
                             alpha: float,
                             lam: float = 0.001,
                             probe_radius: float = 1.0,
                             esp_weight: float = 0.5,
                             num_repeats: int = None,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = None,
                             verbose: bool = False,
                             backend: Optional[str] = None,
                             return_aligned: bool = False,
                             ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using ShaEP-style ESP-combo similarity. ``vol_and_surf_esp``
        is the canonical name for the legacy ``esp_combo`` mode (alias kept).

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_and_surf_esp`` and ``pair.sim_aligned_vol_and_surf_esp``

        Parameters
        ----------
        alpha, lam, probe_radius, esp_weight : float
            ShaEP combo parameters (see ``MoleculePair.align_with_vol_and_surf_esp``).
        num_repeats, trans_init, lr, max_num_steps, verbose
            Standard optimization controls.
        backend : str
            ``"jax"`` (default) runs the per-pair CPU/torch path sequentially via
            ``MoleculePair.align_with_vol_and_surf_esp``. ``"triton"`` (aliases
            ``"cuda"``/``"gpu"``) routes to the batched
            ``MoleculePair._align_batch_vol_and_surf_esp`` GPU kernel (multi-GPU-aware).
            ``"numba"`` (alias ``"cpu"``) runs the same batched path on CPU via the
            numba kernels (the ESP channel is the fused ``esp_comparison_batch``).
        return_aligned : bool
            For the Triton backend, build the aligned-fit-surface list when ``True``.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit surface coordinates per pair (``None`` entries unless
            ``return_aligned=True`` on the Triton backend).
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_and_surf_esp")
        if num_repeats is None:
            num_repeats = _default_seeds("vol_and_surf_esp")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol_and_surf_esp,
            dict(alpha=alpha, lam=lam, probe_radius=probe_radius,
                 esp_weight=esp_weight, trans_init=trans_init,
                 num_repeats=num_repeats, lr=lr, steps_fine=max_num_steps),
            "sim_aligned_vol_and_surf_esp", "transform_vol_and_surf_esp", "_fit_surf_t",
            return_aligned)
        if handled:
            return _result

        return self._delegate_alignment(
            'align_with_vol_and_surf_esp', 'sim_aligned_vol_and_surf_esp',
            alpha=alpha,
            lam=lam,
            probe_radius=probe_radius,
            esp_weight=esp_weight,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

    # legacy method aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp)
    align_with_esp = align_with_surf_esp
    align_with_esp_combo = align_with_vol_and_surf_esp

    def align_with_vol_color(self,
                             color_weight: float = 0.5,
                             alpha: float = 0.81,
                             num_repeats: int = None,
                             trans_init: bool = False,
                             lr: float = 0.1,
                             max_num_steps: int = None,
                             verbose: bool = False,
                             backend: Optional[str] = None,
                             return_aligned: bool = False,
                             ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using the ROCS/ROSHAMBO-style ``vol_color`` combo
        (atom-Gaussian shape + directionless pharmacophore color).

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_color`` and ``pair.sim_aligned_vol_color``

        Parameters
        ----------
        color_weight : float
            Weight of the color channel in [0, 1] (default 0.5).
        alpha : float
            Gaussian width for the shape overlap (default 0.81, volumetric).
        num_repeats, trans_init, lr, max_num_steps, verbose
            Standard optimization controls.
        backend : str
            ``"jax"`` (default) runs the per-pair torch path sequentially via
            ``MoleculePair.align_with_vol_color``. ``"triton"`` (aliases ``"cuda"``/
            ``"gpu"``) and ``"numba"`` (alias ``"cpu"``) route to the batched
            ``MoleculePair._align_batch_vol_color`` driver — BOTH the shape channel and
            the directionless color channel run on the device-dispatched kernels (Triton
            on CUDA, numba on CPU; where every cloud is small they fuse into one launch),
            so the batched path runs on either device. The batched
            path descends on the JOINT weighted gradient -- both the shape and the color
            channel steer the pose. NOTE ``backend="jax"`` is a misnomer for this mode:
            there is no JAX kernel, so it runs the per-pair PyTorch path sequentially.
        return_aligned : bool
            For the batched backend, build the aligned-fit-atom list when ``True``.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates per pair (``None`` entries unless
            ``return_aligned=True`` on the batched backend).
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_color")
        if num_repeats is None:
            num_repeats = _default_seeds("vol_color")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol_color,
            dict(alpha=alpha, color_weight=color_weight, trans_init=trans_init,
                 num_repeats=num_repeats, lr=lr, steps_fine=max_num_steps),
            "sim_aligned_vol_color", "transform_vol_color", "_fit_xyz_t",
            return_aligned)
        if handled:
            return _result

        return self._delegate_alignment(
            'align_with_vol_color', 'sim_aligned_vol_color',
            color_weight=color_weight,
            alpha=alpha,
            num_repeats=num_repeats,
            trans_init=trans_init,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

    def align_with_vol_tversky(self,
                               tversky_alpha: float = 0.95,
                               tversky_beta: float = 0.05,
                               alpha: float = 0.81,
                               num_repeats: int = None,
                               lr: float = 0.1,
                               max_num_steps: int = None,
                               verbose: bool = False,
                               backend: Optional[str] = None,
                               return_aligned: bool = False,
                               ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using the asymmetric "fits-inside" ``vol_tversky`` shape overlay.

        Same atom-centred Gaussian *shape* channel as ``vol`` (heavy atoms), scored with a
        **Tversky** reduction ``AB / (AB + tversky_alpha*(AA-AB) + tversky_beta*(BB-AB))`` instead
        of Tanimoto. With the defaults (``tversky_alpha=0.95``, ``tversky_beta=0.05``) missing
        reference volume is penalized heavily and extra fit volume barely, so the score rewards
        the *reference* (query) being contained in the fit. The score is NOT bounded to [0, 1] --
        a small dense query inside a larger molecule can legitimately exceed 1.0 -- and is never
        clamped.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_tversky`` and ``pair.sim_aligned_vol_tversky``

        Parameters
        ----------
        tversky_alpha : float
            Weight on missing reference volume ``AA - AB`` (default 0.95). Named to avoid the
            Gaussian width ``alpha``.
        tversky_beta : float
            Weight on extra fit volume ``BB - AB`` (default 0.05).
        alpha : float
            Gaussian width for the shape overlap (default 0.81, volumetric heavy atoms).
        num_repeats, lr, max_num_steps, verbose
            Standard optimization controls. ``num_repeats`` / ``max_num_steps`` default
            (``None``) to the per-mode ``MODE_SEEDS`` / ``MODE_STEPS`` in ``accel/_modes.py``.
        backend : str
            ``None`` (default) resolves device-aware (Triton on CUDA else numba). ``"triton"``
            (aliases ``"cuda"``/``"gpu"``) and ``"numba"`` (alias ``"cpu"``) route to the batched
            ``MoleculePair._align_batch_vol_tversky`` driver, which reuses the shape kernel and
            applies the Tversky reduction on the host. ``"jax"`` falls back to the per-pair
            PyTorch path (there is no JAX kernel for this mode).
        return_aligned : bool
            For the batched backend, build the aligned-fit-atom list when ``True``.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,). NOT clipped to [0, 1].
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates per pair (``None`` entries unless
            ``return_aligned=True`` on the batched backend).
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_tversky")
        if num_repeats is None:
            num_repeats = _default_seeds("vol_tversky")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol_tversky,
            dict(alpha=alpha, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta,
                 steps_fine=max_num_steps),
            "sim_aligned_vol_tversky", "transform_vol_tversky", "_fit_xyz_t",
            return_aligned)
        if handled:
            return _result

        return self._delegate_alignment(
            'align_with_vol_tversky', 'sim_aligned_vol_tversky',
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            alpha=alpha,
            num_repeats=num_repeats,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

    def align_with_vol_lipo(self,
                            lipo_weight: float = 0.5,
                            alpha: float = 0.81,
                            lam: float = 0.1,
                            num_repeats: int = None,
                            lr: float = 0.1,
                            max_num_steps: int = None,
                            verbose: bool = False,
                            backend: Optional[str] = None,
                            return_aligned: bool = False,
                            ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Align all pairs using the ``vol_lipo`` overlay: atom-centred Gaussian *shape*
        (volume) + per-atom *lipophilicity* overlap, blended
        ``(1-lipo_weight)*shape_Tanimoto + lipo_weight*lipo_Tanimoto``.

        The shape channel is the heavy-atom Gaussian volume overlap (identical to ``vol``); the
        lipophilicity channel overlays the per-atom Crippen atomic logP contributions -- placed
        at the TRUE-heavy atom centres -- like an ESP/partial-charge field (matched by value so
        hydrophobic overlaps hydrophobic), with the atom-centred ``lam=0.1`` (raw). Both the fit
        shape centres AND the fit lipophilicity centres move rigidly under the same SE(3) pose,
        and BOTH channels steer the pose (joint weighted gradient). Each channel self-normalises
        to a Tanimoto, so a self-copy scores 1.000.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_vol_lipo`` and ``pair.sim_aligned_vol_lipo``

        Parameters
        ----------
        lipo_weight : float
            Weight of the lipophilicity channel in [0, 1]; shape gets ``1 - lipo_weight``
            (default 0.5).
        alpha : float
            Gaussian width for the shape AND lipophilicity overlaps (default 0.81, volumetric
            heavy atoms).
        lam : float
            Value ("charge") weighting for the lipophilicity ESP overlap (default 0.1, raw /
            atom-centred, NOT LAM_SCALING-scaled).
        num_repeats, lr, max_num_steps, verbose
            Standard optimization controls. ``num_repeats`` / ``max_num_steps`` default
            (``None``) to the per-mode ``MODE_SEEDS`` / ``MODE_STEPS`` in ``accel/_modes.py``.
        backend : str
            ``None`` (default) resolves device-aware (Triton on CUDA else numba). ``"triton"``
            (aliases ``"cuda"``/``"gpu"``) and ``"numba"`` (alias ``"cpu"``) route to the batched
            ``MoleculePair._align_batch_vol_lipo`` driver, which reuses the shape kernel (shape
            channel) and the fused ESP kernel (lipophilicity channel, logP as charges).
            ``"jax"`` falls back to the per-pair PyTorch path (there is no JAX kernel for this
            mode).
        return_aligned : bool
            For the batched backend, build the aligned-fit-atom list when ``True``.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_list : list of np.ndarray
            Aligned fit atom coordinates per pair (``None`` entries unless
            ``return_aligned=True`` on the batched backend).
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("vol_lipo")
        if num_repeats is None:
            num_repeats = _default_seeds("vol_lipo")
        handled, _result = self._run_fast_or_fallthrough(
            backend, MoleculePair._align_batch_vol_lipo,
            dict(lipo_weight=lipo_weight, alpha=alpha, lam=lam,
                 num_repeats=num_repeats, lr=lr, steps_fine=max_num_steps),
            "sim_aligned_vol_lipo", "transform_vol_lipo", "_fit_xyz_t",
            return_aligned)
        if handled:
            return _result

        return self._delegate_alignment(
            'align_with_vol_lipo', 'sim_aligned_vol_lipo',
            lipo_weight=lipo_weight,
            alpha=alpha,
            lam=lam,
            num_repeats=num_repeats,
            lr=lr,
            max_num_steps=max_num_steps,
            verbose=verbose,
        )

    def _pad_and_mask_pharm(self):
        """Extract, pad, and create masks for pharmacophore alignment.

        Validates that all pairs have pharmacophore data. Does NOT modify the
        pair objects. Returns padded arrays and masks.

        Returns
        -------
        entries : list of tuples
            Each tuple is (ref_ptypes, fit_ptypes,
                           ref_ancs_pad, fit_ancs_pad,
                           ref_vecs_pad, fit_vecs_pad,
                           mask_ref, mask_fit,
                           orig_ref_len, orig_fit_len).
        max_ref_len : int
        max_fit_len : int
        """
        for i, pair in enumerate(self.pairs):
            if (pair.ref_molec.pharm_types is None or
                    pair.fit_molec.pharm_types is None):
                raise ValueError(
                    f'Pair {i} is missing pharmacophore data. '
                    'Create Molecule objects with pharm_multi_vector set to True or False.'
                )

        DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES

        ref_types_list = [p.ref_molec.pharm_types for p in self.pairs]
        fit_types_list = [p.fit_molec.pharm_types for p in self.pairs]

        max_ref_len = max(t.shape[0] for t in ref_types_list)
        max_fit_len = max(t.shape[0] for t in fit_types_list)

        ref_ancs_padded, masks_ref, orig_refs, _ = _pad_arrays([p.ref_molec.pharm_ancs for p in self.pairs])
        fit_ancs_padded, masks_fit, orig_fits, _ = _pad_arrays([p.fit_molec.pharm_ancs for p in self.pairs])
        ref_vecs_padded, _, _, _ = _pad_arrays([p.ref_molec.pharm_vecs for p in self.pairs])
        fit_vecs_padded, _, _, _ = _pad_arrays([p.fit_molec.pharm_vecs for p in self.pairs])

        entries = []
        for (ref_types, fit_types,
             ref_ancs_pad, fit_ancs_pad,
             ref_vecs_pad, fit_vecs_pad,
             mask_ref, mask_fit,
             orig_ref, orig_fit) in zip(
                ref_types_list, fit_types_list,
                ref_ancs_padded, fit_ancs_padded,
                ref_vecs_padded, fit_vecs_padded,
                masks_ref, masks_fit,
                orig_refs, orig_fits
        ):
            ref_types_pad = np.full(max_ref_len, DUMMY_TYPE, dtype=np.int32)
            ref_types_pad[:orig_ref] = ref_types
            fit_types_pad = np.full(max_fit_len, DUMMY_TYPE, dtype=np.int32)
            fit_types_pad[:orig_fit] = fit_types

            entries.append((ref_types_pad, fit_types_pad,
                            ref_ancs_pad, fit_ancs_pad,
                            ref_vecs_pad, fit_vecs_pad,
                            mask_ref, mask_fit,
                            orig_ref, orig_fit))

        return entries, max_ref_len, max_fit_len

    def align_with_pharm(self,
                         similarity: str = 'tanimoto',
                         extended_points: bool = False,
                         only_extended: bool = False,
                         num_repeats: int = None,
                         trans_init: bool = False,
                         lr: float = 0.1,
                         max_num_steps: int = None,
                         num_workers: int = 1,
                         use_shmap: bool = True,
                         num_buckets: int = 1,
                         verbose: bool = False,
                         backend: Optional[str] = None,
                         return_aligned: bool = False,
                         ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Align all pairs using padded masked pharmacophore similarity via JAX.

        Because all padded arrays have the same shape, JAX's XLA compiler
        reuses one compiled kernel for every pair — no recompilation overhead.

        When ``num_workers > 1`` the pairs are split into size-sorted chunks and
        processed in parallel. It is recommended to use ``use_shmap=True``
        instead of ``multiprocessing`` for this setting.

        Results are stored in-place on each MoleculePair:
        - ``pair.transform_pharm`` and ``pair.sim_aligned_pharm``

        Parameters
        ----------
        similarity : str
            One of ``'tanimoto'``, ``'tversky'``, ``'tversky_ref'``, ``'tversky_fit'``.
        extended_points : bool
            Score HBA/HBD with extended-point Gaussians.
        only_extended : bool
            When ``extended_points`` is True, ignore anchor overlaps.
        num_repeats : int
            Number of SE(3) initializations per pair.
        trans_init : bool
            If True, initialize translations to each ref pharmacophore anchor.
        lr : float
            Optimizer learning rate.
        max_num_steps : int
            Maximum optimization steps.
        num_workers : int
            Number of parallel worker processes. ``1`` (default) runs
            sequentially in-process. Values greater than ``len(self.pairs)``
            are clamped to ``len(self.pairs)``.
        use_shmap : bool
            If ``True`` and ``num_workers > 1``, use ``jax.shard_map`` + ``vmap``
            to parallelise across virtual CPU devices in a single process.
            Requires ``XLA_FLAGS=--xla_force_host_platform_device_count=N``
            to be set before any JAX import.  Uses ``lax.scan`` (fixed steps,
            no early stopping) instead of the ``while_loop``-based sequential
            path.  Required on Linux HPC if num_workers > 1 where ``multiprocessing``
            spawn can be unreliable with JAX.  Default is ``True``.
        num_buckets : int
            ``1`` (default) pads all pairs to the global pharmacophore-count
            maximum — lowest overhead for typical use.  Values > 1 sort pairs
            by ``(max(ref,fit), min(ref,fit))`` and process each bucket
            separately with reduced per-bucket padding, which can be
            beneficial for large heterogeneous molecule sets.
        verbose : bool
            Print scores per pair.

        Returns
        -------
        scores : np.ndarray
            Shape: (N,).
        aligned_anchors_list : list of np.ndarray
            Aligned fit pharmacophore anchors (unpadded) for each pair.
        aligned_vectors_list : list of np.ndarray
            Aligned fit pharmacophore vectors (unpadded) for each pair.
        backend : str
            ``"jax"`` (default) or ``"triton"`` (aliases ``"cuda"``/``"gpu"``) which
            routes to ``MoleculePair._align_batch_pharm`` (multi-GPU-aware). With
            ``return_aligned=True`` the aligned anchors (rotate+translate) and vectors
            (rotate only) are rebuilt GPU-batched from the cached fit tensors.
        """
        if max_num_steps is None:
            max_num_steps = _default_steps("pharm")
        if num_repeats is None:
            num_repeats = _default_seeds("pharm")
        backend = _resolve_backend(backend)
        if backend in self._TRITON_BACKENDS or backend in self._NUMBA_BACKENDS:
            if backend in self._NUMBA_BACKENDS:
                self._prepare_numba()
            _pharm_kw = dict(similarity=similarity, extended_points=extended_points,
                             only_extended=only_extended, trans_init=trans_init,
                             num_repeats=num_repeats, steps_fine=max_num_steps, lr=lr)
            if (num_workers and num_workers > 1 and self.pairs
                    and self.pairs[0].device.type == "cpu"):
                # CPU multi-core: shard pairs across the persistent single-threaded pool
                # (bit-identical; align_pairs also caches _*_pharm_*_t for return_aligned).
                from shepherd_score.accel import cpu_pool as _cpu_pool
                _cpu_pool.align_pairs("pharm", self.pairs, num_workers, _pharm_kw)
            else:
                MoleculePair._align_batch_pharm(self.pairs, **_pharm_kw)
            pairs = self.pairs
            n = len(pairs)
            scores = np.array([float(p.sim_aligned_pharm) for p in pairs])
            if not return_aligned:
                return scores, [None] * n, [None] * n
            import torch
            anchors = [None] * n
            vectors = [None] * n
            by_dev: dict = {}
            for i, p in enumerate(pairs):
                by_dev.setdefault(p._fit_pharm_ancs_t.device, []).append(i)
            for dev, idxs in by_dev.items():
                Ss = torch.stack([torch.as_tensor(pairs[i].transform_pharm,
                                                  dtype=torch.float32) for i in idxs]).to(dev)
                Rt = Ss[:, :3, :3].transpose(1, 2)
                anc = [pairs[i]._fit_pharm_ancs_t for i in idxs]
                vec = [pairs[i]._fit_pharm_vecs_t for i in idxs]
                _ps = torch.nn.utils.rnn.pad_sequence
                anc_al = torch.baddbmm(Ss[:, :3, 3][:, None, :], _ps(anc, batch_first=True), Rt).cpu().numpy()
                vec_al = torch.bmm(_ps(vec, batch_first=True), Rt).cpu().numpy()       # vectors: rotate only
                for j, i in enumerate(idxs):
                    anchors[i] = anc_al[j, :anc[j].shape[0]]
                    vectors[i] = vec_al[j, :vec[j].shape[0]]
            return scores, anchors, vectors
        if backend != "jax":
            raise ValueError(f"unknown backend {backend!r}; use 'jax', 'triton', or 'numba'")

        # Validate pharmacophore data and collect raw arrays for all pairs.
        for idx, pair in enumerate(self.pairs):
            if (pair.ref_molec.pharm_types is None or
                    pair.fit_molec.pharm_types is None):
                raise ValueError(
                    f'Pair {idx} is missing pharmacophore data. '
                    'Create Molecule objects with pharm_multi_vector set to True or False.'
                )

        n_pairs = len(self.pairs)
        scores = np.zeros(n_pairs)
        aligned_anchors_list = [None] * n_pairs
        aligned_vectors_list = [None] * n_pairs

        # Build raw (unpadded) per-pair data tuples (plain numpy — picklable).
        pair_data = []
        for pair in self.pairs:
            tc = pair.ref_molec.pharm_ancs if trans_init else None
            pair_data.append((
                pair.ref_molec.pharm_types,
                pair.fit_molec.pharm_types,
                pair.ref_molec.pharm_ancs,
                pair.fit_molec.pharm_ancs,
                pair.ref_molec.pharm_vecs,
                pair.fit_molec.pharm_vecs,
                tc,
                pair.ref_molec.pharm_ancs,
                pair.fit_molec.pharm_ancs,
            ))

        if use_shmap and num_workers > 1:  # shard_map path
            _jax_ver = _pkg_version("jax")
            _jax_ver_tuple = tuple(int(x) for x in _jax_ver.split(".")[:2])
            if _jax_ver_tuple < (0, 9):
                raise RuntimeError(
                    f"use_shmap=True requires JAX >= 0.9.0, but found JAX {_jax_ver}. "
                    "Either upgrade JAX (which requires Python >= 3.11) or set use_shmap=False."
                )

            results = _align_pharm_shmap(
                pair_data, num_workers, similarity, extended_points, only_extended,
                num_repeats, lr, max_num_steps, verbose, num_buckets=num_buckets,
            )
            for i, (score, se3_transform, aligned_ancs, aligned_vecs) in enumerate(results):
                scores[i] = score
                aligned_anchors_list[i] = aligned_ancs
                aligned_vectors_list[i] = aligned_vecs
                pair = self.pairs[i]
                pair.transform_pharm = se3_transform
                pair.sim_aligned_pharm = score

        elif num_workers > 1: # parallel
            ref_sizes = np.array([len(d[2]) for d in pair_data])  # ref_ancs
            fit_sizes = np.array([len(d[3]) for d in pair_data])  # fit_ancs
            # Primary key: max(ref, fit) — dominates padding; secondary: min.
            sort_keys = np.array([np.minimum(ref_sizes, fit_sizes),
                                   np.maximum(ref_sizes, fit_sizes)])
            index_splits, chunk_results = _dispatch_parallel(
                pair_data, sort_keys, _align_pharm_worker, num_workers,
                (similarity, extended_points, only_extended,
                 num_repeats, lr, max_num_steps, verbose),
            )

            for idx_list, chunk_result in zip(index_splits, chunk_results):
                for global_i, (score, se3_transform, aligned_ancs, aligned_vecs) in zip(
                    idx_list, chunk_result
                ):
                    scores[global_i] = score
                    aligned_anchors_list[global_i] = aligned_ancs
                    aligned_vectors_list[global_i] = aligned_vecs
                    pair = self.pairs[global_i]
                    pair.transform_pharm = se3_transform
                    pair.sim_aligned_pharm = score

        else: # sequential
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    'JAX is required for MoleculePairBatch.align_with_pharm. '
                    'Install it with: pip install "shepherd-score[jax]"'
                ) from exc

            from shepherd_score.alignment_jax import optimize_pharm_overlay_jax_vectorized_mask

            DUMMY_TYPE = 8  # index of 'Dummy' in P_TYPES
            ref_types_list = [p.ref_molec.pharm_types for p in self.pairs]
            fit_types_list = [p.fit_molec.pharm_types for p in self.pairs]
            ref_ancs_list = [p.ref_molec.pharm_ancs for p in self.pairs]
            fit_ancs_list = [p.fit_molec.pharm_ancs for p in self.pairs]
            ref_vecs_list = [p.ref_molec.pharm_vecs for p in self.pairs]
            fit_vecs_list = [p.fit_molec.pharm_vecs for p in self.pairs]

            ref_sizes = np.array([len(a) for a in ref_ancs_list])
            fit_sizes = np.array([len(a) for a in fit_ancs_list])
            bucket_splits = _compute_bucket_splits(ref_sizes, fit_sizes, num_buckets)

            for bucket_idx_list in bucket_splits:
                bucket_ref_ancs = [ref_ancs_list[i] for i in bucket_idx_list]
                bucket_fit_ancs = [fit_ancs_list[i] for i in bucket_idx_list]
                bucket_ref_vecs = [ref_vecs_list[i] for i in bucket_idx_list]
                bucket_fit_vecs = [fit_vecs_list[i] for i in bucket_idx_list]

                ref_ancs_padded, masks_ref, orig_refs_b, max_ref_b = _pad_arrays(bucket_ref_ancs)
                fit_ancs_padded, masks_fit, orig_fits_b, max_fit_b = _pad_arrays(bucket_fit_ancs)
                ref_vecs_padded, _, _, _ = _pad_arrays(bucket_ref_vecs)
                fit_vecs_padded, _, _, _ = _pad_arrays(bucket_fit_vecs)

                for local_j, global_i in enumerate(bucket_idx_list):
                    pair = self.pairs[global_i]
                    orig_ref = orig_refs_b[local_j]
                    orig_fit = orig_fits_b[local_j]

                    ref_types_pad = np.full(max_ref_b, DUMMY_TYPE, dtype=np.int32)
                    ref_types_pad[:orig_ref] = ref_types_list[global_i]
                    fit_types_pad = np.full(max_fit_b, DUMMY_TYPE, dtype=np.int32)
                    fit_types_pad[:orig_fit] = fit_types_list[global_i]

                    trans_centers = pair.ref_molec.pharm_ancs if trans_init else None

                    aligned_ancs, aligned_vecs, se3_transform, score = \
                        optimize_pharm_overlay_jax_vectorized_mask(
                            ref_pharms=jnp.array(ref_types_pad),
                            fit_pharms=jnp.array(fit_types_pad),
                            ref_anchors=jnp.array(ref_ancs_padded[local_j]),
                            fit_anchors=jnp.array(fit_ancs_padded[local_j]),
                            ref_vectors=jnp.array(ref_vecs_padded[local_j]),
                            fit_vectors=jnp.array(fit_vecs_padded[local_j]),
                            mask_ref=jnp.array(masks_ref[local_j]),
                            mask_fit=jnp.array(masks_fit[local_j]),
                            similarity=similarity,
                            extended_points=extended_points,
                            only_extended=only_extended,
                            num_repeats=num_repeats,
                            trans_centers=trans_centers,
                            init_ref_anchors=pair.ref_molec.pharm_ancs,
                            init_fit_anchors=pair.fit_molec.pharm_ancs,
                            lr=lr,
                            max_num_steps=max_num_steps,
                            verbose=verbose,
                        )

                    se3_transform = np.array(se3_transform)
                    score = float(np.array(score))
                    aligned_ancs = np.array(aligned_ancs)[:orig_fit]
                    aligned_vecs = np.array(aligned_vecs)[:orig_fit]

                    scores[global_i] = score
                    pair.transform_pharm = se3_transform
                    pair.sim_aligned_pharm = score
                    aligned_anchors_list[global_i] = aligned_ancs
                    aligned_vectors_list[global_i] = aligned_vecs

        return scores, aligned_anchors_list, aligned_vectors_list
