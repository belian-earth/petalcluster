"""Benchmark: the Python counterpart of each shoal algorithm.

Run from project root:
  Rscript bench/gen_data.R            # generate shared datasets (once)
  uv run bench/bench_sklearn.py
  BENCH_ONLY=Ward uv run bench/bench_sklearn.py   # one algorithm, merged into results

Settings match bench_r.R: the same k, restarts, iteration cap and linkage.
scikit-learn covers DBSCAN, HDBSCAN, k-means and the Gaussian mixture; SciPy
covers Ward linkage; EVoC is compared with its reference implementation. The
reference implementation is JIT-compiled, so it is warmed on a small input
before timing.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.5",
#     "scipy>=1.13",
#     "numpy>=1.26",
#     "evoc==0.3.1",
#     "matplotlib>=3.8",
# ]
# ///

import csv
import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"


def bench(fn, x, reps: int = 3) -> float:
    times = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def load_family(prefix: str, d: int) -> list[np.ndarray]:
    files = sorted(DATA.glob(f"{prefix}_*_d{d}.csv"))
    if not files:
        raise SystemExit(f"No {prefix} data for d={d}. Run bench/gen_data.R first.")
    data = [np.loadtxt(f, delimiter=",", skiprows=1) for f in files]
    return sorted(data, key=lambda a: a.shape[0])


def evoc_fit(x):
    import evoc
    evoc.EVoC(base_min_cluster_size=15).fit_predict(x.astype(np.float32))


# (algorithm, family, package, function, max_n)
BENCHMARKS = [
    ("DBSCAN", "blobs", "sklearn",
     lambda x: DBSCAN(eps=3.0, min_samples=5).fit_predict(x), np.inf),
    ("HDBSCAN", "blobs", "sklearn",
     lambda x: HDBSCAN(min_samples=5, min_cluster_size=15).fit_predict(x), np.inf),
    ("k-means", "blobs", "sklearn",
     lambda x: KMeans(n_clusters=5, n_init=10, max_iter=300).fit_predict(x), np.inf),
    ("GMM", "blobs", "sklearn",
     lambda x: GaussianMixture(n_components=5, covariance_type="full",
                               max_iter=100).fit_predict(x), np.inf),
    ("Ward", "blobs", "scipy",
     lambda x: linkage(x, method="ward"), np.inf),
    ("Distances", "blobs", "scipy",
     lambda x: pdist(x), np.inf),
    ("EVoC", "emb", "evoc", evoc_fit, np.inf),
]

DIMS = {"blobs": [2, 10], "emb": [48]}


def main():
    rows = []
    only = os.environ.get("BENCH_ONLY", "")
    benchmarks = [b for b in BENCHMARKS if not only or b[0] == only]
    if not benchmarks:
        raise SystemExit(f"No benchmark named {only}")
    if any(b[0] == "EVoC" for b in benchmarks):
        # Warm the reference EVoC's JIT so timings measure the algorithm.
        evoc_fit(load_family("emb", 48)[0])

    for algorithm, family, package, fn, max_n in benchmarks:
        for d in DIMS[family]:
            print(f"\n=== {algorithm}, {family} d={d} ===")
            for x in load_family(family, d):
                n = x.shape[0]
                print(f"  n={n:6d} ... ", end="", flush=True)
                t = bench(fn, x) if n <= max_n else float("nan")
                print("skipped" if np.isnan(t) else f"{package} {t:.3f}s")
                rows.append((algorithm, family, n, d, package, "" if np.isnan(t) else t))

    out = HERE / "results_sklearn.csv"
    if only and out.exists():
        with out.open(newline="") as f:
            kept = [r for r in csv.reader(f) if r and r[0] not in (only, "algorithm")]
        rows = [tuple(r) for r in kept] + rows
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "family", "n", "dims", "package", "median_s"])
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
