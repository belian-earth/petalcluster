# Log-Likelihood of a Fitted Mixture

Implementing this gives
[`stats::AIC()`](https://rdrr.io/r/stats/AIC.html) and
[`stats::BIC()`](https://rdrr.io/r/stats/AIC.html) for free, which is
how the number of components is normally chosen.

## Usage

``` r
# S3 method for class 'shoal_gmm'
logLik(object, ...)
```

## Arguments

- object:

  A fitted
  [`shoal_gmm()`](https://belian-earth.github.io/shoal/reference/shoal_gmm.md)
  model.

- ...:

  Ignored.

## Value

An object of class `"logLik"`, with `df` and `nobs` attributes.

## Details

The degrees of freedom count the free parameters of a full-covariance
mixture: `k * p` means, `k * p * (p + 1) / 2` distinct covariance
entries and `k - 1` independent mixing proportions.

## Examples

``` r
fit <- shoal_gmm(as.matrix(iris[, 1:4]), k = 3L)
logLik(fit)
#> 'log Lik.' -180.1957 (df=44)
AIC(fit)
#> [1] 448.3915
BIC(fit)
#> [1] 580.8594
```
