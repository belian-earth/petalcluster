test_that("petal_dbscan works with matrix input", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)

  expect_s3_class(res, "petal_dbscan")
  expect_type(res$cluster, "integer")
  expect_length(res$cluster, nrow(x))
  expect_true(res$n_clusters >= 1L)
  expect_equal(
    res$n_clusters + (res$n_noise > 0L),
    length(unique(na.omit(res$cluster))) + (res$n_noise > 0L)
  )
  expect_equal(sum(is.na(res$cluster)), res$n_noise)
  expect_equal(res$params, list(eps = 0.5, min_samples = 5L))
  expect_equal(res$metric, "euclidean")
  expect_true(is.matrix(res$data))
  expect_equal(dim(res$data), c(150L, 4L))
})

test_that("petal_dbscan works with data frame input", {
  res <- petal_dbscan(iris, eps = 0.5, min_samples = 5L)

  expect_s3_class(res, "petal_dbscan")
  expect_length(res$cluster, nrow(iris))
  # data frame coerced to matrix with numeric columns only
  expect_true(is.matrix(res$data))
  expect_equal(ncol(res$data), 4L)
})

test_that("petal_dbscan supports cosine metric", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.01, min_samples = 5L, metric = "cosine")

  expect_s3_class(res, "petal_dbscan")
  expect_equal(res$metric, "cosine")
})

test_that("petal_dbscan cluster IDs are sequential from 1", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)

  ids <- sort(unique(na.omit(res$cluster)))
  expect_equal(ids, seq_len(res$n_clusters))
})

test_that("petal_dbscan validates inputs", {
  x <- as.matrix(iris[, 1:4])

  expect_error(petal_dbscan(1:10), "numeric matrix or data frame")
  expect_error(petal_dbscan(x, eps = -1), "positive")
  expect_error(petal_dbscan(x, eps = 0), "positive")
  expect_error(petal_dbscan(x, min_samples = 0L), "positive integer")
  expect_error(petal_dbscan(x, min_samples = 1.5), "single integer")
  expect_error(petal_dbscan(x, metric = "manhattan"))
})

test_that("petal_dbscan rejects data frame with < 2 numeric columns", {
  df <- data.frame(a = letters[1:10], b = 1:10)
  expect_error(petal_dbscan(df), "at least 2 numeric columns")
})

test_that("print.petal_dbscan returns invisibly", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  expect_snapshot(print(res))
})

test_that("plot.petal_dbscan runs without error", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  vdiffr::expect_doppelganger("dbscan-pairs", plot(res))
})

test_that("plot.petal_dbscan works with 2-column data", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:2])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  vdiffr::expect_doppelganger("dbscan-2col", plot(res))
})

test_that("plot xcol/ycol by name produces scatter plot", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  vdiffr::expect_doppelganger(
    "dbscan-xcol-name",
    plot(res, xcol = "Petal.Length", ycol = "Petal.Width")
  )
})

test_that("plot xcol/ycol by index produces scatter plot", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  vdiffr::expect_doppelganger("dbscan-xcol-idx", plot(res, xcol = 3, ycol = 4))
})

test_that("plot errors when only xcol or ycol supplied", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  expect_error(plot(res, xcol = 1), "xcol.*ycol.*together")
  expect_error(plot(res, ycol = 1), "xcol.*ycol.*together")
})

test_that("plot errors for invalid column name", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  expect_error(plot(res, xcol = "nope", ycol = "Sepal.Length"), "not found")
})

test_that("plot errors for out-of-range column index", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_dbscan(x, eps = 0.5, min_samples = 5L)
  expect_error(plot(res, xcol = 0, ycol = 1), "between 1 and")
  expect_error(plot(res, xcol = 1, ycol = 99), "between 1 and")
})
