"""Re-render speed_plot.png / speed_table.md from a saved fork plot_data.json,
merging in a cached snapshot of the upstream-original panel (so the slow original
baselines are NOT re-run). Used to refresh the plot after fork-only re-measures.

Usage:
    python -m benchmarks.replot <fork_plot_data.json> [out_dir]
"""
import json
import os
import sys

from benchmarks.headline import render_plot, render_table, MODES, BUCKETS, SIZES, DEFAULT_CAP

# Cached upstream-original throughput (pairs/s) from the last full `headline`
# run (fork + original). Original repo is untouched, so these are stable; we
# snapshot them here to avoid re-running the slow capped subprocess baselines.
ORIG = {
    "vol|same":   [[1, 0.5], [10, 15.4], [100, 161.1], [1000, 168.1], [10000, 167.8]],
    "vol|cross":  [[1, 1.6], [10, 11.7], [100, 61.6], [1000, 59.4]],
    "surf|same":  [[1, 1.8], [10, 3.9], [100, 55.6], [1000, 54.4]],
    "surf|cross": [[1, 2.0], [10, 3.2], [100, 25.5], [1000, 38.7]],
    "esp|same":   [[1, 2.3], [10, 4.9], [100, 31.8], [1000, 31.6]],
    "esp|cross":  [[1, 2.3], [10, 3.7], [100, 17.3], [1000, 21.5]],
    "pharm|same": [[1, 1.1], [10, 12.0]],
    "pharm|cross":[[1, 1.6], [10, 11.9]],
}


def main():
    fork_json = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "plot_data.json")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(__file__)

    with open(fork_json) as fh:
        data = json.load(fh)

    print("fork cells found:")
    for k in sorted(data):
        pts = data[k].get("fork", [])
        print(f"  {k:12s} {len(pts)} sizes: {[p[0] for p in pts]}")

    for k, rows in ORIG.items():
        data.setdefault(k, {})["orig"] = rows

    out_json = os.path.join(out_dir, "plot_data.json")
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {out_json}")

    render_table(data, MODES, BUCKETS, SIZES, os.path.join(out_dir, "speed_table.md"))
    render_plot(data, MODES, BUCKETS, SIZES, DEFAULT_CAP, os.path.join(out_dir, "speed_plot.png"))


if __name__ == "__main__":
    main()
