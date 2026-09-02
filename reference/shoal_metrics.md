# Internal Cluster Validity Indices

Computes the Calinski-Harabasz and Davies-Bouldin indices, both of which
work from the data rather than a distance matrix, and both of which are
used the same way: fit at several values of `k` and compare.

## Usage

``` r
shoal_metrics(x, cluster = NULL)
```

## Arguments

- x:

  A numeric matrix or data frame, or a clustering result from this
  package, in which case both the data and the assignment are taken from
  it.

- cluster:

  A clustering result or an integer vector of cluster IDs. Required
  unless `x` is itself a clustering result. `NA` entries are excluded.

## Value

A one-row data frame with columns `n`, `k`, `calinski_harabasz` and
`davies_bouldin`.

## Details

- **Calinski-Harabasz** is between-cluster dispersion over
  within-cluster dispersion, each per degree of freedom. **Higher is
  better.**

- **Davies-Bouldin** averages, over clusters, the worst-case ratio of
  combined within-cluster scatter to the distance between centroids.
  **Lower is better.**

Both assume roughly convex, centroid-shaped clusters, so they suit
[`shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/shoal_kmeans.md)
and
[`shoal_gmm()`](https://belian-earth.github.io/shoal/reference/shoal_gmm.md)
better than the density-based algorithms. For arbitrary cluster shapes
prefer
[`shoal_silhouette()`](https://belian-earth.github.io/shoal/reference/shoal_silhouette.md),
which needs only a distance matrix.

## See also

[`shoal_silhouette()`](https://belian-earth.github.io/shoal/reference/shoal_silhouette.md).

## Examples

``` r
fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)
shoal_metrics(fit)
#>     n k calinski_harabasz davies_bouldin
#> 1 150 3          561.6278      0.6619715

# Comparing candidate values of k
do.call(rbind, lapply(2:5, function(k) {
  shoal_metrics(shoal_kmeans(as.matrix(iris[, 1:4]), k = k))
}))
#>     n k calinski_harabasz davies_bouldin
#> 1 150 2          513.9245      0.4042928
#> 2 150 3          561.6278      0.6619715
#> 3 150 4          530.4871      0.7757009
#> 4 150 5          495.5415      0.8059652
```
