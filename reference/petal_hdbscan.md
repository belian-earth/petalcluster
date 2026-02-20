# HDBSCAN Clustering

Hierarchical density-based spatial clustering of applications with
noise.

## Usage

``` r
petal_hdbscan(
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

  Whether to use Boruvka's algorithm for MST construction. Default
  `TRUE`.

- partial_labels:

  Optional named list for semi-supervised clustering. Names are cluster
  IDs (as strings), values are integer vectors of 1-indexed point
  indices. `NULL` (default) for fully unsupervised clustering.

## Value

An object of class `"petal_hdbscan"`: a list with components `cluster`
(integer vector of cluster IDs, `NA` for noise), `n_clusters`,
`n_noise`, `data` (the input matrix), `params`, `metric`, and
`outlier_scores` (GLOSH scores).
