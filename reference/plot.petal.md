# Plot clustering results

Produces a scatter plot matrix (pairs plot) of clustered data, colored
by cluster assignment. For 2-column data a single scatter plot is
produced instead. Noise points (`NA` cluster) are shown as grey crosses.

## Usage

``` r
# S3 method for class 'petal_dbscan'
plot(
  x,
  xcol = NULL,
  ycol = NULL,
  pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
  ...
)

# S3 method for class 'petal_hdbscan'
plot(
  x,
  xcol = NULL,
  ycol = NULL,
  pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
  ...
)

# S3 method for class 'petal_optics'
plot(
  x,
  xcol = NULL,
  ycol = NULL,
  pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
  ...
)
```

## Arguments

- x:

  A clustering result object.

- xcol, ycol:

  Optional column name or index to plot on the x/y axis. When both are
  supplied, a single scatter plot is produced instead of a pairs matrix.

- pal:

  Character vector of colours for clusters. Defaults to
  `grDevices::hcl.colors(n, "Roma")` where `n` is the number of
  clusters.

- ...:

  Additional arguments passed to
  [`pairs()`](https://rdrr.io/r/graphics/pairs.html) or
  [`plot.default()`](https://rdrr.io/r/graphics/plot.default.html).

## Value

`x`, invisibly.

## Details

When `xcol` and `ycol` are supplied, a single scatter plot of those two
variables is produced instead of the full pairs matrix. Columns can be
specified by name or integer index.
