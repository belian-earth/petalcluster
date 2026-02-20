# OPTICS Clustering

Ordering points to identify the clustering structure.

## Usage

``` r
petal_optics(x, eps = 0.5, min_samples = 5L, metric = c("euclidean", "cosine"))
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

An object of class `"petal_optics"`: a list with components `cluster`
(integer vector of cluster IDs, `NA` for noise), `n_clusters`,
`n_noise`, `data` (the input matrix), `params`, and `metric`.
