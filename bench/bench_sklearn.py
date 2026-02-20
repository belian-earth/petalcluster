"""Benchmark: sklearn clustering for comparison with petalcluster.

Run from project root:
  Rscript bench/gen_data.R            # generate shared datasets (once)
  uv run bench/bench_sklearn.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.5",
#     "numpy>=1.26",
# ]
# ///

import csv
import gc
import re
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

warnings.filterwarnings("ignore", category=FutureWarning)


def bench(fn, x, min_iters: int = 3, min_seconds: float = 0.5) -> float:
    """Return median wall-clock seconds over at least min_iters runs."""
    times = []
    while len(times) < min_iters or sum(times) < min_seconds:
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2]


def bench_sklearn(data_dir: Path, d: int) -> list[tuple]:
    """Run benchmarks for datasets with dimensionality d."""
    pattern = f"blobs_*_d{d}.csv"
    csv_files = sorted(data_dir.glob(pattern))
    if not csv_files:
        raise SystemExit(f"No files matching {pattern}. Run bench/gen_data.R first.")

    datasets = {}
    for f in csv_files:
        x = np.loadtxt(f, delimiter=",", skiprows=1)
        datasets[x.shape[0]] = x

    sizes = sorted(datasets.keys())
    print(f"\n===== d = {d} =====")
    print(f"Loaded {len(sizes)} datasets: {', '.join(f'n={n:,}' for n in sizes)}\n")

    results = []

    # -- DBSCAN ----------------------------------------------------------------
    print("=== DBSCAN ===")
    for n in sizes:
        x = datasets[n]
        print(f"  n={n:,} ... ", end="", flush=True)
        model = DBSCAN(eps=3.0, min_samples=5)
        t = bench(model.fit_predict, x)
        results.append(("DBSCAN", f"n={n:,}", n, d, "sklearn", t))
        print(f"{t:.4f}s")

    # -- HDBSCAN ---------------------------------------------------------------
    print("\n=== HDBSCAN ===")
    for n in sizes:
        x = datasets[n]
        print(f"  n={n:,} ... ", end="", flush=True)
        model = HDBSCAN(min_samples=5, min_cluster_size=15)
        t = bench(model.fit_predict, x)
        results.append(("HDBSCAN", f"n={n:,}", n, d, "sklearn", t))
        print(f"{t:.4f}s")

    # -- OPTICS ----------------------------------------------------------------
    print("\n=== OPTICS ===")
    for n in sizes:
        x = datasets[n]
        print(f"  n={n:,} ... ", end="", flush=True)
        model = OPTICS(max_eps=3.0, min_samples=5)
        t = bench(model.fit_predict, x)
        results.append(("OPTICS", f"n={n:,}", n, d, "sklearn", t))
        print(f"{t:.4f}s")

    return results


def main():
    data_dir = Path(__file__).parent / "data"

    all_results = []
    for d in [2, 10]:
        all_results.extend(bench_sklearn(data_dir, d))

    # -- Write CSV -------------------------------------------------------------
    csv_path = Path(__file__).parent / "results_sklearn.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "dataset", "n", "dims", "package", "median_s"])
        writer.writerows(all_results)

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
