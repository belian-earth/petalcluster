# Threads used by the Rust backends

The parallel parts of the package run on a thread pool owned by it.
`shoal_threads(n)` rebuilds that pool with `n` threads;
`shoal_threads()` reports the current size. The setting takes effect
immediately and lasts for the session.

## Usage

``` r
shoal_threads(n)
```

## Arguments

- n:

  Number of threads, a positive whole number. Omit to query.

## Value

The number of threads in the pool, invisibly when setting.

## Details

Parallel work covers
[`shoal_evoc()`](https://belian-earth.github.io/petalcluster/reference/shoal_evoc.md)'s
neighbour search, spanning tree and node embedding;
[`shoal_dbscan()`](https://belian-earth.github.io/petalcluster/reference/shoal_dbscan.md)'s
neighbour queries and
[`shoal_hdbscan()`](https://belian-earth.github.io/petalcluster/reference/shoal_hdbscan.md)'s
core distances and spanning tree;
[`shoal_kmeans()`](https://belian-earth.github.io/petalcluster/reference/shoal_kmeans.md)'s
assignment step and initialisation, which the Gaussian mixture shares
through its k-means start;
[`shoal_dist()`](https://belian-earth.github.io/petalcluster/reference/shoal_dist.md);
and
[`shoal_silhouette()`](https://belian-earth.github.io/petalcluster/reference/shoal_silhouette.md).
Hierarchical clustering and the mixture's EM iterations are
single-threaded. Results never depend on the thread count.

## Default

On load the pool takes, in order of precedence, the `shoal.threads`
option, the `RAYON_NUM_THREADS` environment variable, or one thread per
logical core. When `_R_CHECK_LIMIT_CORES_` is set, as it is by
`R CMD check --as-cran`, the automatic default is capped at 2 to respect
the CRAN policy for checks; an explicit option or variable is still
honoured.

## Examples

``` r
old <- shoal_threads()
shoal_threads(2)
shoal_threads()
#> [1] 2
shoal_threads(old)
```
