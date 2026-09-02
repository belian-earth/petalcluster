# Predict Mixture Membership

Predict Mixture Membership

## Usage

``` r
# S3 method for class 'shoal_gmm'
predict(object, newdata = NULL, type = c("class", "posterior"), ...)
```

## Arguments

- object:

  A fitted
  [`shoal_gmm()`](https://belian-earth.github.io/petalcluster/reference/shoal_gmm.md)
  model.

- newdata:

  A numeric matrix or data frame with the same columns as the data the
  model was fitted to. Omit it to return the training assignment.

- type:

  `"class"` (default) for the most likely component, or `"posterior"`
  for the full `n x k` matrix of responsibilities.

- ...:

  Ignored.

## Value

An integer vector of component IDs, or an `n x k` numeric matrix of
posterior probabilities.

## Examples

``` r
fit <- shoal_gmm(as.matrix(iris[1:100, 1:4]), k = 2L)
predict(fit, as.matrix(iris[101:150, 1:4]))
#>  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#> [39] 2 2 2 2 2 2 2 2 2 2 2 2
head(predict(fit, type = "posterior"))
#>      [,1]         [,2]
#> [1,]    1 1.533810e-26
#> [2,]    1 3.345318e-19
#> [3,]    1 5.977894e-22
#> [4,]    1 3.633872e-19
#> [5,]    1 9.938109e-28
#> [6,]    1 4.480187e-27
```
