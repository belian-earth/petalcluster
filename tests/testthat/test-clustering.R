test_that("results carry both the subclass and the shared parent", {
  x <- as.matrix(iris[, 1:4])
  db <- shoal_dbscan(x, eps = 0.5, min_samples = 5L)
  hdb <- shoal_hdbscan(x)

  expect_s3_class(db, "shoal_dbscan")
  expect_s3_class(db, "shoal_clustering")
  expect_equal(class(db), c("shoal_dbscan", "shoal_clustering"))

  expect_s3_class(hdb, "shoal_hdbscan")
  expect_s3_class(hdb, "shoal_clustering")
  expect_equal(class(hdb), c("shoal_hdbscan", "shoal_clustering"))
})

test_that("every result exposes the components print() and plot() rely on", {
  x <- as.matrix(iris[, 1:4])

  for (res in list(
    shoal_dbscan(x, eps = 0.5, min_samples = 5L),
    shoal_hdbscan(x)
  )) {
    expect_type(res$algorithm, "character")
    expect_type(res$params, "list")
    expect_named(res$params)
    expect_type(res$cluster, "integer")
    expect_type(res$n_clusters, "integer")
    expect_type(res$n_noise, "integer")
    expect_true(is.matrix(res$data))
  }
})

test_that("n_clusters and n_noise are derived from cluster", {
  x <- as.matrix(iris[, 1:4])
  res <- shoal_dbscan(x, eps = 0.5, min_samples = 5L)

  expect_equal(res$n_clusters, length(unique(res$cluster[!is.na(res$cluster)])))
  expect_equal(res$n_noise, sum(is.na(res$cluster)))
})

test_that("format_params renders name = value pairs", {
  expect_equal(
    format_params(list(eps = 0.5, min_samples = 5L)),
    "eps = 0.5, min_samples = 5"
  )
  expect_equal(format_params(list(boruvka = TRUE)), "boruvka = TRUE")
  expect_equal(format_params(list()), "none")
})

test_that("format_params tolerates a multi-value parameter", {
  expect_equal(format_params(list(k = c(2L, 3L))), "k = 2, 3")
})

test_that("a single print method serves every algorithm", {
  # print.shoal_clustering is dispatched to via the parent class, and
  # print.shoal_hdbscan extends it rather than replacing it.
  expect_false(is.null(getS3method("print", "shoal_clustering")))
  expect_false(is.null(getS3method("print", "shoal_hdbscan")))
  expect_null(getS3method("print", "shoal_dbscan", optional = TRUE))
})

test_that("dropped algorithms are no longer exported", {
  expect_false(exists("shoal_optics", where = asNamespace("shoal"), inherits = FALSE))
  expect_false("petal_optics" %in% getNamespaceExports("shoal"))
})
