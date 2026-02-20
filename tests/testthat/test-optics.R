test_that("petal_optics works with matrix input", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_optics(x, eps = 1.0, min_samples = 5L)

  expect_s3_class(res, "petal_optics")
  expect_type(res$cluster, "integer")
  expect_length(res$cluster, nrow(x))
  expect_true(res$n_clusters >= 1L)
  expect_equal(sum(is.na(res$cluster)), res$n_noise)
  expect_equal(res$params, list(eps = 1.0, min_samples = 5L))
  expect_equal(res$metric, "euclidean")
  expect_true(is.matrix(res$data))
  expect_equal(dim(res$data), c(150L, 4L))
})

test_that("petal_optics works with data frame input", {
  res <- petal_optics(iris, eps = 1.0, min_samples = 5L)

  expect_s3_class(res, "petal_optics")
  expect_length(res$cluster, nrow(iris))
})

test_that("petal_optics supports cosine metric", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_optics(x, eps = 0.01, min_samples = 5L, metric = "cosine")

  expect_s3_class(res, "petal_optics")
  expect_equal(res$metric, "cosine")
})

test_that("petal_optics cluster IDs are sequential from 1", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_optics(x, eps = 1.0, min_samples = 5L)

  ids <- sort(unique(na.omit(res$cluster)))
  if (length(ids) > 0) {
    expect_equal(ids, seq_len(res$n_clusters))
  }
})

test_that("petal_optics validates inputs", {
  x <- as.matrix(iris[, 1:4])

  expect_error(petal_optics(1:10), "numeric matrix or data frame")
  expect_error(petal_optics(x, eps = -1), "positive")
  expect_error(petal_optics(x, min_samples = 0L), "positive integer")
  expect_error(petal_optics(x, metric = "manhattan"))
})

test_that("print.petal_optics returns invisibly", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_optics(x, eps = 1.0, min_samples = 5L)
  expect_snapshot(print(res))
})

test_that("plot.petal_optics runs without error", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:4])
  res <- petal_optics(x, eps = 1.0, min_samples = 5L)
  vdiffr::expect_doppelganger("optics-pairs", plot(res))
})
