# Density-Based Clustering via Rust

R bindings to the
[petal-clustering](https://github.com/petabi/petal-clustering) Rust
crate by [Petabi, Inc.](https://github.com/petabi), providing fast
implementations of three density-based clustering algorithms:

- [`petal_dbscan()`](https://belian-earth.github.io/petalcluster/reference/petal_dbscan.md)
  — DBSCAN: finds clusters as dense regions separated by areas of lower
  density.

- [`petal_hdbscan()`](https://belian-earth.github.io/petalcluster/reference/petal_hdbscan.md)
  — HDBSCAN: hierarchical extension of DBSCAN that adapts to clusters of
  varying density.

- [`petal_optics()`](https://belian-earth.github.io/petalcluster/reference/petal_optics.md)
  — OPTICS: produces a reachability ordering that captures cluster
  structure at multiple density thresholds.

All three accept numeric matrices or data frames, support Euclidean and
cosine distance metrics, and return S3 objects with
[`print()`](https://rdrr.io/r/base/print.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods.

## Author

**Maintainer**: First Last <first.last@example.com>
