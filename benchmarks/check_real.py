"""Quick check: real-molecule table + bucketing structure per mode."""
from benchmarks.real_workloads import molecule_table, make_real_cohort
from shepherd_score.container._core import _band_key

for mode in ("surf", "pharm"):
    print(f"\n=== mode={mode}: per-molecule bucketing count (band=16) ===")
    tab = molecule_table(mode)
    for name, heavy, cnt in tab:
        print(f"  {name:16s} heavy={heavy:2d}  {mode}_count={cnt:3d}  band={_band_key(cnt)}")
    for bk in ("same", "cross"):
        co = make_real_cohort(mode, n_pairs=16, bucket_kind=bk)
        bands = sorted(set(_band_key(p.n_ref) for p in co.pairs))
        print(f"  cohort {bk:5s}: n_buckets={len(bands)} bands={bands} "
              f"pool={co.meta['pool'][:4]}{'...' if len(co.meta['pool'])>4 else ''}")
print("\nOK")
