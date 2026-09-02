# HDBSCAN Clustering

Hierarchical density-based spatial clustering of applications with
noise.

## Usage

``` r
shoal_hdbscan(
  x,
  alpha = 1,
  min_samples = 15L,
  min_cluster_size = 15L,
  metric = c("euclidean", "cosine"),
  boruvka = TRUE,
  partial_labels = NULL
)
```

## Arguments

- x:

  A numeric matrix or data frame. Data frames are coerced to a matrix
  using their numeric columns (non-numeric columns are dropped).

- alpha:

  Smoothing parameter for mutual reachability distance. Default `1.0`.

- min_samples:

  Minimum neighbourhood size. Default `15L`.

- min_cluster_size:

  Minimum cluster size. Default `15L`.

- metric:

  Distance metric, one of `"euclidean"` or `"cosine"`.

- boruvka:

  Whether to build the minimum spanning tree with a tree-accelerated
  Boruvka search rather than Prim's algorithm. Default `TRUE`, which is
  the faster choice in low dimensions. The acceleration depends on a
  spatial index, and above a few dozen columns it becomes slower than
  the plain search: on 20,000 rows in 64 dimensions Boruvka takes around
  2.5 times as long as `boruvka = FALSE`. Both give the same clustering.

- partial_labels:

  Optional named list for semi-supervised clustering. Names are cluster
  IDs (as strings), values are integer vectors of 1-indexed point
  indices. `NULL` (default) for fully unsupervised clustering.

## Value

An object of class `c("shoal_hdbscan", "shoal_clustering")`: a list with
components `cluster` (integer vector of cluster IDs, `NA` for noise),
`n_clusters`, `n_noise`, `data` (the input matrix), `algorithm`,
`params`, `metric`, and `outlier_scores` (GLOSH scores).

## Examples

``` r
res <- shoal_hdbscan(as.matrix(iris[, 1:4]))
res
#> 
#> ── HDBSCAN Clustering 
#> Metric: "euclidean"
#> Parameters: alpha = 1, min_samples = 15, min_cluster_size = 15, boruvka = TRUE
#> Clusters: 2, Noise points: 0
#> GLOSH outlier scores: median 0.159, max 0.657
```
