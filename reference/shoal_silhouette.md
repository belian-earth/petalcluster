# Silhouette Widths

For each observation, compares the mean distance to the rest of its own
cluster (`a`) with the smallest mean distance to any other cluster
(`b`), giving a width of `(b - a) / max(a, b)`. Widths near 1 indicate a
well-placed observation, near 0 one on a boundary, and negative ones an
observation that sits closer to a different cluster.

## Usage

``` r
shoal_silhouette(d, cluster)
```

## Arguments

- d:

  A [stats::dist](https://rdrr.io/r/stats/dist.html) object, or a
  numeric matrix or data frame, in which case Euclidean distances are
  computed with
  [`shoal_dist()`](https://belian-earth.github.io/petalcluster/reference/shoal_dist.md)
  first.

- cluster:

  A clustering result from this package, or an integer vector of cluster
  IDs. `NA` entries, the noise points of the density-based algorithms,
  are excluded along with their distances.

## Value

A data frame with one row per observation and columns `cluster`,
`neighbour` (the nearest other cluster) and `width`. The mean width is
attached as the `avg_width` attribute.

## Details

An observation alone in its cluster is given a width of 0 by convention,
since it has no within-cluster distances to average.

## See also

[`shoal_metrics()`](https://belian-earth.github.io/petalcluster/reference/shoal_metrics.md)
for indices computed from the data instead.

## Examples

``` r
x <- as.matrix(iris[, 1:4])
fit <- shoal_kmeans(x, k = 3L)
sil <- shoal_silhouette(shoal_dist(x), fit)
attr(sil, "avg_width")
#> [1] 0.552819
```
