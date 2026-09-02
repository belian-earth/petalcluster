test_that("shoal_gmm returns a soft partition of every observation", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 3L)

  expect_s3_class(fit, "shoal_gmm")
  expect_s3_class(fit, "shoal_clustering")
  expect_length(fit$cluster, nrow(x))
  expect_false(anyNA(fit$cluster))
  expect_equal(fit$n_noise, 0L)
})

test_that("the posterior is a proper set of responsibilities", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 3L)

  expect_equal(dim(fit$posterior), c(nrow(x), 3L))
  expect_true(all(fit$posterior >= 0 & fit$posterior <= 1))
  expect_equal(rowSums(fit$posterior), rep(1, nrow(x)), tolerance = 1e-10)

  # cluster is the argmax of the posterior, by construction.
  expect_equal(fit$cluster, max.col(fit$posterior, ties.method = "first"))
})

test_that("mixture parameters have the right shape and constraints", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 3L)

  expect_length(fit$weights, 3L)
  expect_equal(sum(fit$weights), 1, tolerance = 1e-10)
  expect_true(all(fit$weights > 0))

  expect_equal(dim(fit$means), c(3L, 4L))
  expect_equal(colnames(fit$means), colnames(x))
  expect_equal(dim(fit$covariances), c(3L, 4L, 4L))

  # Each covariance slice must be symmetric and positive definite. Getting the
  # column-major reshape wrong would break symmetry, so this pins the layout.
  for (j in 1:3) {
    sigma <- fit$covariances[j, , ]
    expect_equal(sigma, t(sigma), tolerance = 1e-10)
    expect_true(all(eigen(sigma, symmetric = TRUE, only.values = TRUE)$values > 0))
  }
})

test_that("the log-likelihood agrees with an independent computation", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 2L)

  # Densities computed the long way round, without the log-sum-exp shortcut.
  dens <- vapply(seq_len(2L), function(j) {
    sigma <- fit$covariances[j, , ]
    md <- stats::mahalanobis(x, center = fit$means[j, ], cov = sigma)
    fit$weights[j] * exp(-0.5 * md) /
      sqrt((2 * pi)^ncol(x) * det(sigma))
  }, numeric(nrow(x)))

  expect_equal(fit$loglik, sum(log(rowSums(dens))), tolerance = 1e-8)
})

test_that("logLik gives working AIC and BIC", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 3L)

  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
  expect_equal(as.numeric(ll), fit$loglik)
  expect_equal(attr(ll, "nobs"), nrow(x))

  # Full covariance: k means, k covariance matrices, k - 1 free weights.
  k <- 3L
  p <- 4L
  expect_equal(attr(ll, "df"), k * p + k * p * (p + 1) / 2 + (k - 1))

  expect_equal(AIC(fit), -2 * fit$loglik + 2 * attr(ll, "df"), tolerance = 1e-10)
  expect_equal(
    BIC(fit),
    -2 * fit$loglik + log(nrow(x)) * attr(ll, "df"),
    tolerance = 1e-10
  )
})

test_that("BIC has an interior optimum across k", {
  # Unlike k-means inertia, BIC penalises parameter count, so it should not
  # simply improve forever as components are added.
  x <- as.matrix(iris[, 1:4])
  bics <- vapply(1:5, function(k) BIC(shoal_gmm(x, k = k, seed = 1L)), numeric(1L))
  expect_false(which.min(bics) == length(bics))
})

test_that("a mixture recovers well-separated components", {
  set.seed(8)
  blob <- function(cx, cy, n) cbind(stats::rnorm(n, cx, 0.3), stats::rnorm(n, cy, 0.3))
  x <- rbind(blob(0, 0, 60), blob(8, 0, 60), blob(4, 8, 60))
  truth <- rep(1:3, each = 60)

  fit <- shoal_gmm(x, k = 3L, seed = 1L)
  expect_true(all(table(truth, fit$cluster) %in% c(0L, 60L)))
})

test_that("the seed makes runs reproducible", {
  x <- as.matrix(iris[, 1:4])
  a <- shoal_gmm(x, k = 3L, seed = 5L)
  b <- shoal_gmm(x, k = 3L, seed = 5L)

  expect_equal(a$cluster, b$cluster)
  expect_equal(a$loglik, b$loglik)
  expect_equal(a$means, b$means)
})

test_that("predict reproduces the fit and handles both types", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 3L, seed = 1L)

  # Fitting and prediction share one density implementation, so these are equal
  # rather than merely close.
  expect_equal(predict(fit, x), fit$cluster)
  expect_equal(predict(fit, x, type = "posterior"), fit$posterior)
  expect_equal(predict(fit), fit$cluster)
  expect_equal(predict(fit, type = "posterior"), fit$posterior)

  new <- as.matrix(iris[c(1, 60, 120), 1:4])
  expect_length(predict(fit, new), 3L)
  expect_equal(dim(predict(fit, new, type = "posterior")), c(3L, 3L))
})

test_that("shoal_gmm validates its inputs", {
  x <- as.matrix(iris[1:20, 1:4])

  expect_error(shoal_gmm(x, k = 0L), "positive integer")
  expect_error(shoal_gmm(x, k = 21L), "cannot exceed")
  expect_error(shoal_gmm(x, k = 2L, init = "spectral"))
  expect_error(shoal_gmm(x, k = 2L, tolerance = 0), "positive")
  expect_error(shoal_gmm(x, k = 2L, seed = -1L), "non-negative")

  fit <- shoal_gmm(x, k = 2L)
  expect_error(predict(fit, as.matrix(iris[, 1:2])), "fitted on")
  expect_error(predict(fit, x, type = "logits"))
})
