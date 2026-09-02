# DBSCAN Clustering

Density-based spatial clustering of applications with noise.

## Usage

``` r
shoal_dbscan(x, eps = 0.5, min_samples = 5L, metric = c("euclidean", "cosine"))
```

## Arguments

- x:

  A numeric matrix or data frame. Data frames are coerced to a matrix
  using their numeric columns (non-numeric columns are dropped).

- eps:

  Neighbourhood radius. Default `0.5`.

- min_samples:

  Minimum number of points to form a dense region. Default `5L`.

- metric:

  Distance metric, one of `"euclidean"` or `"cosine"`.

## Value

An object of class `c("shoal_dbscan", "shoal_clustering")`: a list with
components `cluster` (integer vector of cluster IDs, `NA` for noise),
`n_clusters`, `n_noise`, `data` (the input matrix), `algorithm`,
`params`, and `metric`.

## Examples

``` r
res <- shoal_dbscan(as.matrix(iris[, 1:4]), eps = 0.5, min_samples = 5L)
res
#> 
#> ── DBSCAN Clustering 
#> Metric: "euclidean"
#> Parameters: eps = 0.5, min_samples = 5
#> Clusters: 2, Noise points: 17
```
