# Hierarchical Agglomerative Clustering

Agglomerative clustering via the Rust
[kodama](https://github.com/diffeo/kodama) crate, a port of
*fastcluster*. Returns a standard
[stats::hclust](https://rdrr.io/r/stats/hclust.html) object, so
[`stats::cutree()`](https://rdrr.io/r/stats/cutree.html),
[`stats::as.dendrogram()`](https://rdrr.io/r/stats/dendrogram.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) and the rest of
R's hierarchical clustering ecosystem work on the result unchanged.

## Usage

``` r
shoal_hclust(
  d,
  method = c("complete", "single", "average", "weighted", "ward", "centroid", "median")
)
```

## Arguments

- d:

  A [stats::dist](https://rdrr.io/r/stats/dist.html) object, or a
  numeric matrix or data frame, in which case Euclidean distances are
  computed with
  [`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md)
  first.

- method:

  Linkage method. One of `"complete"`, `"single"`, `"average"`,
  `"weighted"`, `"ward"`, `"centroid"` or `"median"`.

## Value

An object of class `"hclust"` with components `merge`, `height`,
`order`, `labels`, `method`, `call` and `dist.method`.

## Linkage methods

Dissimilarities are squared internally for `"ward"`, `"centroid"` and
`"median"`, with the square root taken afterwards, the fastcluster and
SciPy convention. Two consequences differ from
[`stats::hclust()`](https://rdrr.io/r/stats/hclust.html):

- `"ward"` corresponds to R's `"ward.D2"`, not `"ward.D"`.

- `"centroid"` and `"median"` take plain distances here, whereas
  [`stats::hclust()`](https://rdrr.io/r/stats/hclust.html) expects them
  to be squared beforehand.

`"weighted"` is R's `"mcquitty"` (WPGMA); `"average"` is UPGMA.

`"centroid"` and `"median"` can produce inversions, a merge at a lower
height than the one before it. This is a property of the methods, not a
bug, but [`stats::cutree()`](https://rdrr.io/r/stats/cutree.html)
rejects such trees, so a warning is issued when it happens.

## See also

[`shoal_dist()`](https://belian-earth.github.io/shoal/reference/shoal_dist.md)
for building the distance matrix.

## Examples

``` r
d <- shoal_dist(as.matrix(iris[, 1:4]))
fit <- shoal_hclust(d, method = "ward")
cutree(fit, k = 3)
#>   [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#>  [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#>  [75] 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 3 3 3 3 2 3 3 3 3
#> [112] 3 3 2 2 3 3 3 3 2 3 2 3 2 3 3 2 2 3 3 3 3 3 2 2 3 3 3 2 3 3 3 2 3 3 3 2 3
#> [149] 3 2
```
