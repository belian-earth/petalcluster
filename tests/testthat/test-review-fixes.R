# Regression tests for the findings of the branch review.

test_that("non-finite rows are screened like missing ones", {
  x <- as.matrix(iris[1:20, 1:4])
  x[3L, 2L] <- Inf
  x[7L, 4L] <- -Inf

  expect_warning(d <- shoal_dist(x), "non-finite")
  expect_equal(attr(d, "Size"), 18L)
  # And nothing non-finite survives into the distances themselves.
  expect_true(all(is.finite(d)))

  expect_warning(shoal_dbscan(x, eps = 0.5, min_samples = 3L), "non-finite")
})

test_that("validator errors name the caller's argument, not 'x'", {
  fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)

  bad <- as.matrix(iris[1:5, 1:4])
  bad[2L, 1L] <- NA_real_
  expect_error(predict(fit, bad), "newdata")

  gfit <- shoal_gmm(as.matrix(iris[, 1:4]), k = 2L)
  expect_error(predict(gfit, bad), "newdata")
})

test_that("k is required, with a clear absence error", {
  x <- as.matrix(iris[, 1:4])
  expect_error(shoal_kmeans(x), "absent")
  expect_error(shoal_gmm(x), "absent")
})

test_that("a large whole-number seed survives into params intact", {
  x <- as.matrix(iris[1:30, 1:4])
  big <- 2^40
  fit <- shoal_kmeans(x, k = 2L, seed = big)
  expect_identical(fit$params$seed, big)
  expect_false(is.na(fit$params$seed))
})

test_that("predict warns when newdata column names disagree", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L, seed = 1L)

  renamed <- x
  colnames(renamed) <- paste0("V", 1:4)
  expect_warning(p <- predict(fit, renamed), "Column names")
  # Matching stays positional, so the assignment itself is unchanged.
  expect_equal(p, fit$cluster)

  # Same names, same order: silent.
  expect_silent(predict(fit, x))
})

test_that("a square distance matrix is refused as raw data", {
  d <- shoal_dist(as.matrix(iris[1:10, 1:4]))
  m <- as.matrix(d)

  expect_error(shoal_hclust(m), "as.dist")
  expect_error(shoal_silhouette(m, rep(1:2, each = 5L)), "as.dist")

  # And the suggested escape hatch works.
  expect_s3_class(shoal_hclust(stats::as.dist(m)), "hclust")
})

test_that("dist sizes past the integer square root do not overflow", {
  # 50,000 observations give 1,249,975,000 pairs, beyond R's integer range
  # in the intermediate products.
  expect_identical(dist_index(50000L, 50000L - 1L, 50000L), 1249975000)
  expect_false(anyNA(dist_index(50000L, c(1L, 49999L), c(2L, 50000L))))

  fake <- structure(double(3), Size = 50000L, class = "dist")
  expect_error(shoal_hclust(fake), "1249975000")
})

test_that("subset_dist agrees with subsetting the full matrix", {
  x <- as.matrix(iris[1:12, 1:4])
  rownames(x) <- letters[1:12]
  d <- shoal_dist(x)
  keep <- rep(c(TRUE, TRUE, FALSE), times = 4L)

  direct <- stats::as.dist(as.matrix(d)[keep, keep, drop = FALSE])
  sub <- subset_dist(d, keep)

  expect_equal(as.double(sub), as.double(direct), tolerance = 1e-15)
  expect_equal(attr(sub, "Size"), sum(keep))
  expect_equal(attr(sub, "Labels"), letters[1:12][keep])
  expect_equal(attr(sub, "method"), "euclidean")
})

test_that("non-finite scalar parameters are rejected", {
  x <- as.matrix(iris[1:20, 1:4])
  expect_error(shoal_kmeans(x, k = 2L, tolerance = Inf), "finite")
  expect_error(shoal_dbscan(x, eps = Inf), "finite")
})

test_that("na_action is matched strictly, not partially", {
  x <- as.matrix(iris[1:5, 1:4])
  expect_error(check_numeric_matrix(x, na_action = "d"))
})
