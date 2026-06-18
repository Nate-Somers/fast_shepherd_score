"""
Why does per-pair throughput drop from 4096 -> 10000 and does it STABILIZE?

Warm, single-process sweep (footprint cache calibrated from smaller sizes first,
workspaces cleared between sizes to avoid accumulation). For each size prints
time, mol/s, and the sub-batch chunk size the sub-batcher chose (K_sub0) so we
can see exactly when/why chunking kicks in and whether mol/s asymptotes.
"""
import os
os.environ["SUBBATCH_DEBUG"] = "1"
import io
import time
import contextlib
import re
import torch

from benchmarks.real_workloads import make_real_cohort
from shepherd_score.container import MoleculePair
import shepherd_score.container._core as _cc


def align(mode, pairs):
    if mode == "surf":
        MoleculePair.align_batch_surf(pairs, alpha=0.81, steps_fine=100)
    else:
        MoleculePair.align_batch_esp(pairs, alpha=0.81, lam=0.3, num_repeats=50,
                                     topk=30, steps_fine=100, lr=0.075)


def chunks_of(buf):
    """Parse 'K=.. K_sub0=..' from captured SUBBATCH_DEBUG; return list of (K,sub)."""
    out = []
    for m in re.finditer(r"K=(\d+) .*?K_sub0=(\d+)", buf):
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def main():
    sizes = [2048, 4096, 6144, 8192, 10000, 14000, 20000]
    for mode in ["surf", "esp"]:
        print(f"\n{'='*60}\n{mode} same-bucket warm scaling\n{'='*60}")
        print(f'{"batch":>6} {"time s":>8} {"mol/s":>9} {"chunks":>7} {"K_sub0":>8}')
        for nb in sizes:
            co = make_real_cohort(mode, n_pairs=nb, bucket_kind="same", seed=3)
            pairs = [MoleculePair(p.ref, p.fit, do_center=False, device=torch.device("cuda"))
                     for p in co.pairs]
            align(mode, pairs); torch.cuda.synchronize()           # warm 1 (calibrate footprint)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                align(mode, pairs); torch.cuda.synchronize()       # warm 2 (capture chunking)
            cks = chunks_of(buf.getvalue())
            K, sub = (cks[-1] if cks else (nb, nb))
            nchunks = (K + sub - 1) // sub
            ts = []
            for _ in range(3):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                align(mode, pairs); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = min(ts)
            print(f'{nb:6d} {t:8.3f} {nb/t:9.1f} {nchunks:7d} {sub:8d}', flush=True)
            del pairs, co
            _cc._ALIGN_WORKSPACES.clear(); _cc._INT_BUFFER_CACHE.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
