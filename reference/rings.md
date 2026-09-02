# Concentric rings with noise

Three concentric rings of 300, 400 and 500 points with radii 0.5, 1.2
and 2.0, plus 60 uniformly scattered noise points. A standard case where
density-based methods succeed and centroid-based ones cannot.

## Usage

``` r
rings
```

## Format

A numeric matrix with 1260 rows and columns `x` and `y`. No labels are
included; the generating code is in `data-raw/rings.R`.

## Source

Simulated; see `data-raw/rings.R`.

## Examples

``` r
res <- shoal_hdbscan(rings, min_cluster_size = 15L, min_samples = 5L)
res
#> 
#> ── HDBSCAN Clustering 
#> Metric: "euclidean"
#> Parameters: alpha = 1, min_samples = 5, min_cluster_size = 15, boruvka = TRUE
#> Clusters: 3, Noise points: 26
#> GLOSH outlier scores: median 0.079, max 0.965
```
