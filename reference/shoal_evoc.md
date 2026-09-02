# EVoC: Embedding Vector Oriented Clustering

Direct multi-granularity clustering of embedding vectors, via an in-tree
Rust port of McInnes's [EVoC](https://github.com/TutteInstitute/evoc)
(Tutte Institute). Instead of clustering in the original space, EVoC
builds a nearest-neighbour graph under cosine distance, learns a compact
node embedding from it, and density-clusters that embedding at several
granularities at once. On collections of pre-built embeddings it is
orders of magnitude faster than running HDBSCAN on the raw vectors, at
comparable quality.

## Usage

``` r
shoal_evoc(
  x,
  n_neighbors = 15L,
  noise_level = 0.5,
  min_cluster_size = 5L,
  min_samples = 5L,
  n_epochs = 50L,
  dim = NULL,
  max_layers = 10L,
  layer = "auto",
  seed = 1L
)
```

## Arguments

- x:

  A numeric matrix or data frame of embedding vectors, one per row. Data
  frames are coerced to a matrix using their numeric columns
  (non-numeric columns are dropped). Must have more rows than
  `n_neighbors`.

- n_neighbors:

  Neighbourhood size for the nearest-neighbour graph. Default `15L`.

- noise_level:

  Tolerance for spreading points out in the learned embedding; higher
  values let clusters absorb more of their surroundings. Default `0.5`.

- min_cluster_size:

  Minimum cluster size at the finest layer; at least 2. Default `5L`,
  matching upstream, which is calibrated for large inputs; on
  collections of a few thousand rows or fewer it tends to over-fragment,
  and something like `15L` recovers the structure far more reliably.

- min_samples:

  Minimum neighbourhood size for the density estimation. Default `5L`.

- n_epochs:

  Training epochs for the node embedding. Default `50L`.

- dim:

  Dimension of the learned node embedding. `NULL` (default) uses the
  upstream rule `min(max(n_neighbors / 4, 4), 15)`.

- max_layers:

  Maximum number of cluster layers to return. Default `10L`.

- layer:

  Which layer populates `cluster`: `"auto"` (default) selects the layer
  with the highest persistence score, matching upstream behaviour; an
  integer selects that layer directly (1 is the finest).

- seed:

  Non-negative whole-number seed. Stored and passed as a double, so
  values beyond the integer range are safe. Default `1L`.

## Value

An object of class `c("shoal_evoc", "shoal_clustering")`: a list with
components `cluster` (integer vector for the selected layer, `NA` for
noise), `n_clusters`, `n_noise`, `data`, `algorithm`, `params`, `metric`
(always `"cosine"`), and the multi-layer results: `layers` (list of
integer cluster vectors, finest first), `strengths` (list of numeric
membership-strength vectors, aligned with `layers`), `persistence`
(numeric persistence score per layer), `layer` (index of the selected
layer) and `embedding` (the learned node-embedding matrix).

## Domain

EVoC assumes cosine geometry: rows are L2-normalised internally and
treated as directions, which is the right model for text/image embedding
vectors and the wrong one for general tabular data. It also wants scale:
behaviour is calibrated for thousands to millions of rows. On small or
low-dimensional data it over-fragments and marks much of the input as
noise (the reference implementation behaves the same way); reach for
[`shoal_hdbscan()`](https://belian-earth.github.io/shoal/reference/shoal_hdbscan.md)
there.

## Layers

A single flat clustering hides a genuine modelling choice, and the
upstream heuristic for making it (most persistent layer) is not reliable
on every shape of data. Every layer is therefore returned, finest first,
along with its persistence score and per-point membership strengths;
`layer` only chooses which one populates `cluster` for printing,
plotting and the single-partition helpers. Pick a different layer
afterwards by indexing `layers` directly; the fit does not need to be
rerun.

## Reproducibility

EVoC is stochastic (neighbour search, graph partitioning and the
embedding all draw random numbers), so `seed` is a parameter rather than
being taken from R's RNG, as in
[`shoal_kmeans()`](https://belian-earth.github.io/shoal/reference/shoal_kmeans.md).
The same seed, parameters and data give bitwise-identical results
regardless of thread count, a stronger guarantee than the reference
implementation's, whose parallel stages can race. Expect different seeds
to give somewhat different clusterings; on data EVoC suits, the
structure they agree on is real.

## References

McInnes, L. (2023). *EVoC: Embedding Vector Oriented Clustering*.
<https://github.com/TutteInstitute/evoc>

## Examples

``` r
# Embedding-like data: directions with angular spread, not raw tabular data.
set.seed(1)
centres <- matrix(runif(6 * 48, -1, 1) * 0.6, nrow = 6)
x <- centres[rep(1:6, each = 140L), ] + matrix(rnorm(840 * 48, sd = 0.1), ncol = 48)

fit <- shoal_evoc(x, min_cluster_size = 15L)
fit
#> 
#> ── EVoC Clustering 
#> Metric: "cosine"
#> Parameters: n_neighbors = 15, noise_level = 0.5, min_cluster_size = 15,
#> min_samples = 5, n_epochs = 50, seed = 1
#> Clusters: 6, Noise points: 0
#> Layers (finest first, ✔ = selected):
#>   1: 7 clusters, 79 noise, persistence 0
#> ✔ 2: 6 clusters, 0 noise, persistence 23.81
# every granularity remains available:
vapply(fit$layers, function(l) length(unique(l[!is.na(l)])), integer(1))
#> [1] 7 6
```
