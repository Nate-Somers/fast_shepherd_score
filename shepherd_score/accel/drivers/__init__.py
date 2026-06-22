"""Batched coarse-to-fine SE(3) alignment drivers -- one module per mode
(:mod:`shape` / vol, :mod:`surface`, :mod:`esp`, :mod:`esp_combo`, :mod:`pharm`).

Each driver runs seed generation, the Adam fine-tuning loop, and per-pair best-pose
selection over the whole batch on the dispatched kernels
(:mod:`shepherd_score.accel.kernels`). Shared helpers live in
:mod:`~shepherd_score.accel.drivers._common`.
"""
