test_that("petal_hdbscan works with matrix input", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x)

  expect_s3_class(res, "petal_hdbscan")
  expect_type(res$cluster, "integer")
  expect_length(res$cluster, nrow(x))
  expect_true(res$n_clusters >= 1L)
  expect_equal(sum(is.na(res$cluster)), res$n_noise)
  expect_equal(res$metric, "euclidean")
  expect_equal(
    res$params,
    list(alpha = 1.0, min_samples = 15L, min_cluster_size = 15L, boruvka = TRUE)
  )
  expect_true(is.matrix(res$data))
  expect_equal(dim(res$data), c(150L, 4L))
})

test_that("petal_hdbscan returns outlier scores", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x)

  expect_type(res$outlier_scores, "double")
  expect_length(res$outlier_scores, nrow(x))
  expect_true(all(res$outlier_scores >= 0))
})

test_that("petal_hdbscan works with data frame input", {
  res <- petal_hdbscan(iris)

  expect_s3_class(res, "petal_hdbscan")
  expect_length(res$cluster, nrow(iris))
})

test_that("petal_hdbscan supports cosine metric", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x, metric = "cosine")

  expect_s3_class(res, "petal_hdbscan")
  expect_equal(res$metric, "cosine")
})

test_that("petal_hdbscan supports custom parameters", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x, alpha = 0.5, min_samples = 5L, min_cluster_size = 5L,
                       boruvka = FALSE)

  expect_s3_class(res, "petal_hdbscan")
  expect_equal(res$params$alpha, 0.5)
  expect_equal(res$params$min_samples, 5L)
  expect_equal(res$params$min_cluster_size, 5L)
  expect_false(res$params$boruvka)
})

test_that("petal_hdbscan cluster IDs are sequential from 1", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x)

  ids <- sort(unique(na.omit(res$cluster)))
  if (length(ids) > 0) {
    expect_equal(ids, seq_len(res$n_clusters))
  }
})

test_that("petal_hdbscan validates inputs", {
  x <- as.matrix(iris[, 1:4])

  expect_error(petal_hdbscan("not a matrix"), "numeric matrix or data frame")
  expect_error(petal_hdbscan(x, alpha = -1), "positive")
  expect_error(petal_hdbscan(x, min_samples = 0L), "positive integer")
  expect_error(petal_hdbscan(x, min_cluster_size = 0L), "positive integer")
  expect_error(petal_hdbscan(x, boruvka = "yes"), "TRUE.*FALSE")
  expect_error(petal_hdbscan(x, partial_labels = list(1:3)), "named list")
})

test_that("print.petal_hdbscan returns invisibly", {
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x)
  expect_snapshot(print(res))
})

test_that("plot.petal_hdbscan runs without error", {
  skip_if_not_installed("vdiffr")
  x <- as.matrix(iris[, 1:4])
  res <- petal_hdbscan(x)
  vdiffr::expect_doppelganger("hdbscan-pairs", plot(res))
})
