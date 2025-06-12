# --- Fast Triton kernel -------------------------------------------------
try:
    from .gaussian_overlap_triton import gaussian_tanimoto as _gauss_tanimoto
    _HAS_TRITON = True
except (ImportError, RuntimeError):      # triton not installed or no CUDA
    _HAS_TRITON = False

# optional convenience wrapper that picks the best available backend
def gaussian_tanimoto(a, b, alpha=0.81):
    """
    Select fastest available backend:
        • Triton on CUDA               (nanosecond-fast)
        • falls back to old Torch code (µs-fast) when GPU/Triton absent
    """
    if _HAS_TRITON and a.is_cuda and b.is_cuda:
        return _gauss_tanimoto(a, b, alpha)
    # lazy-import so we don’t pay the cost unless needed
    from .gaussian_overlap import get_overlap as _slow
    return _slow(a, b, alpha)