# shepherd_score/accel/batch/__init__.py
"""Batched multi-GPU GPU/Triton aligners for MoleculePair, split into
``_pad`` (bucketing / sub-batching / scatter), ``_dispatch`` (multi-GPU sharding
+ the CPU-pool tensor spec), and ``aligners`` (the six ``_align_batch_*`` free
functions). The full prior ``accel.batch`` attribute surface is re-exported here
so external imports (``accel.batch._align_batch_*`` / ``._MODE_SPEC`` / etc.) are
unchanged."""
from ._pad import (
    _band_key, _subbatched_align, _scatter_fill, _PAIR_FOOTPRINT_BYTES, _BAND,
)
from ._dispatch import (
    _DISPATCH_LOCAL, _dev_idx, _MIN_SHARD_PER_DEVICE, _should_distribute,
    _run_distributed, _MODE_SPEC, _ProcStandIn,
)
from .aligners import (
    _align_batch_vol, _align_batch_surf, _align_batch_surf_esp, _align_batch_vol_esp,
    _align_batch_vol_and_surf_esp, _align_batch_pharm, _align_batch_vol_color,
    _align_batch_vol_tversky, _align_batch_vol_lipo, _align_batch_vol_esp_tversky,
    _esp_bucketed_align,
    # legacy mode aliases (esp -> surf_esp, esp_combo -> vol_and_surf_esp)
    _align_batch_esp, _align_batch_esp_combo,
    _ALIGN_WORKSPACES, _INT_BUFFER_CACHE,
)
