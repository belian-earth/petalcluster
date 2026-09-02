# Gaussian Mixture Model

Fits a mixture of multivariate Gaussians by expectation-maximisation,
via the Rust [linfa](https://github.com/rust-ml/linfa) toolkit.

## Usage

``` r
shoal_gmm(
  x,
  k,
  init = c("kmeans", "random"),
  n_runs = 1L,
  max_iter = 100L,
  tolerance = 0.001,
  reg_covariance = 1e-06,
  seed = 1L
)
```

## Arguments

- x:

  A numeric matrix or data frame. Data frames are coerced to a matrix
  using their numeric columns (non-numeric columns are dropped).

- k:

  Number of mixture components. Required, since there is no sensible
  default for the central modelling decision.

- init:

  Initialisation method: `"kmeans"` (default) or `"random"`.

- n_runs:

  Number of restarts, keeping the best fit. Default `1L`: unlike
  [`shoal_kmeans()`](https://belian-earth.github.io/petalcluster/reference/shoal_kmeans.md)'s
  10, a single run is the norm for EM (sklearn does the same) because
  the k-means initialisation already starts close and each run is
  expensive. Raise it for small, multimodal problems.

- max_iter:

  Maximum EM iterations per run. Default `100L`.

- tolerance:

  Convergence threshold on the log-likelihood. Default `1e-3`.

- reg_covariance:

  Value added to the diagonal of each covariance matrix to keep it
  positive definite. Default `1e-6`.

- seed:

  Non-negative whole-number seed for initialisation. Stored and passed
  as a double, so values beyond the integer range are safe. Default
  `1L`.

## Value

An object of class `c("shoal_gmm", "shoal_clustering")`: a list with
components `cluster`, `n_clusters`, `n_noise` (always `0`), `data`,
`algorithm`, `params`, `posterior` (an `n x k` matrix of
responsibilities), `weights`, `means`, `covariances` (a `k x p x p`
array) and `loglik`.

## Details

Unlike the other algorithms here, a GMM is generative: it gives each
observation a probability of belonging to each component rather than a
single label. `posterior` holds those responsibilities; `cluster` is
their row-wise maximum, provided for consistency with the rest of the
package.

Clusters are elliptical rather than spherical, so a GMM handles
correlated and differently-scaled features that k-means would split
badly.

## Covariance structure

Only full covariance matrices are supported: each component gets its own
unconstrained matrix. The constrained families `mclust` offers
(spherical, diagonal, tied) are not available upstream.

## Choosing k

A [`logLik()`](https://rdrr.io/r/stats/logLik.html) method is provided,
so [`stats::AIC()`](https://rdrr.io/r/stats/AIC.html) and
[`stats::BIC()`](https://rdrr.io/r/stats/AIC.html) work directly on a
fitted model and can be compared across `k`. Unlike the within-cluster
sum of squares reported by
[`shoal_kmeans()`](https://belian-earth.github.io/petalcluster/reference/shoal_kmeans.md),
these penalise parameter count and so have an interior optimum.

## See also

[`predict.shoal_gmm()`](https://belian-earth.github.io/petalcluster/reference/predict.shoal_gmm.md),
[`logLik.shoal_gmm()`](https://belian-earth.github.io/petalcluster/reference/logLik.shoal_gmm.md).

## Examples

``` r
fit <- shoal_gmm(as.matrix(iris[, 1:4]), k = 3L)
fit
#> 
#> ── Gaussian Mixture Clustering 
#> Parameters: k = 3, init = kmeans, n_runs = 1, seed = 1
#> Clusters: 3, Noise points: 0
#> Log-likelihood: -180.196, BIC: 580.859
#> Mixing proportions: 0.333, 0.365, 0.301
BIC(fit)
#> [1] 580.8594
head(fit$posterior)
#>      [,1]         [,2]         [,3]
#> [1,]    1 6.398292e-35 9.065168e-44
#> [2,]    1 2.599542e-28 8.538368e-31
#> [3,]    1 4.211987e-30 9.372959e-36
#> [4,]    1 2.711941e-26 1.467012e-31
#> [5,]    1 2.683188e-35 3.367511e-46
#> [6,]    1 3.360128e-35 8.486090e-45
```
