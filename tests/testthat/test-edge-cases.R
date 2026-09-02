# Error paths and degenerate inputs that the main test files do not reach.

test_that("inputs with no rows or a bad seed are refused", {
  empty <- matrix(numeric(0), ncol = 2L)
  expect_error(shoal_kmeans(empty, k = 1L), "at least 1 row")
  expect_error(shoal_kmeans(as.matrix(iris[, 1:4]), k = 2L, seed = "a"), "single integer")
})

test_that("a dist describing fewer than 2 observations is refused", {
  one <- structure(double(0), Size = 1L, class = "dist")
  expect_error(shoal_hclust(one), "at least 2 observations")
})

test_that("silhouette and metrics reject unusable inputs", {
  x <- as.matrix(iris[1:20, 1:4])
  square <- as.matrix(dist(x))
  expect_error(shoal_silhouette(square, rep(1:2, 10)), "square distance matrix")
  expect_error(shoal_metrics(x, rep(NA_integer_, 20)), "no clustered observations")

  # A single clustered observation leaves too few to compare; the subsetting
  # of the distance matrix to one point happens on the way to that error.
  d <- shoal_dist(x[1:4, ])
  expect_error(shoal_silhouette(d, c(1L, NA, NA, NA)), "at least 2 clusters")
  sub <- subset_dist(d, c(TRUE, FALSE, FALSE, FALSE))
  expect_s3_class(sub, "dist")
  expect_identical(attr(sub, "Size"), 1L)
  expect_length(sub, 0L)
})

test_that("a singular component covariance is reported, not silently used", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_gmm(x, k = 2L)
  fit$covariances[1, , ] <- 0
  expect_error(predict(fit, x), "singular covariance")
})

test_that("plot column references must be names or indices", {
  res <- shoal_dbscan(as.matrix(iris[, 1:4]), eps = 0.5, min_samples = 5L)
  pdf(NULL)
  on.exit(dev.off())
  expect_error(plot(res, xcol = TRUE, ycol = 2), "column name or index")
  expect_null(plot_legend(character(0), 0L, 0L))
})

test_that("an all-noise result plots with a noise-only legend", {
  # eps too small for anything to be dense: every point is noise.
  res <- shoal_dbscan(as.matrix(iris[, 1:2]), eps = 1e-6, min_samples = 5L)
  expect_identical(res$n_clusters, 0L)
  expect_identical(res$n_noise, nrow(res$data))
  pdf(NULL)
  on.exit(dev.off())
  expect_no_error(plot(res))
})

test_that("silhouette accepts raw data and computes the distances itself", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L)
  expect_equal(shoal_silhouette(x, fit), shoal_silhouette(shoal_dist(x), fit))
})
