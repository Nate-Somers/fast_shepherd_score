# shepherd_score/accel/batch/__init__.py
"""Batched GPU/Triton aligners for MoleculePair: ``_pad`` (bucketing, sub-batching, scatter),
``_dispatch`` (multi-GPU sharding and the process-pool tensor spec), ``aligners`` (the generic
``_align_batch_<mode>`` functions, one per registry mode) and ``_arrays`` (the array-native
screen aligner). The ``accel.batch`` attribute surface (``_align_batch_<mode>`` for every
canonical mode and both legacy aliases, ``_MODE_SPEC``, ``_batch_upload``, ...) is re-exported
here."""
from ._pad import (
    _band_key, _subbatched_align, _scatter_fill, _PAIR_FOOTPRINT_BYTES, _BAND,
)
from ._dispatch import (
    _DISPATCH_LOCAL, _dev_idx, _MIN_SHARD_PER_DEVICE, _should_distribute,
    _run_distributed, _MODE_SPEC, _ProcStandIn,
)
from . import aligners as _aligners
from .aligners import _batch_upload, _ALIGN_WORKSPACES, _INT_BUFFER_CACHE, _seeds_for, _steps_for
from .._modes import SPECS as _SPECS, LEGACY_MODE_ALIASES as _LEGACY

for _m in _SPECS:
    globals()[f"_align_batch_{_m}"] = getattr(_aligners, f"_align_batch_{_m}")
for _legacy, _canon in _LEGACY.items():
    globals()[f"_align_batch_{_legacy}"] = getattr(_aligners, f"_align_batch_{_canon}")
del _m, _legacy, _canon
