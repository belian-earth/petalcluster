test_that("shoal_threads reports and sets the pool size", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))
  expect_gte(old, 1L)

  expect_invisible(shoal_threads(2))
  expect_identical(shoal_threads(), 2L)
  shoal_threads(1)
  expect_identical(shoal_threads(), 1L)

  expect_error(shoal_threads(0), "positive integer")
  expect_error(shoal_threads("many"), "single integer")
})

test_that("results do not depend on the thread count", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))
  x <- as.matrix(iris[, 1:4])
  set.seed(1)
  emb <- matrix(runif(4 * 32, -1, 1) * 0.6, nrow = 4)[rep(1:4, each = 60L), ] +
    matrix(rnorm(240 * 32, sd = 0.1), ncol = 32)

  run <- function() list(
    dist = shoal_dist(x),
    dbscan = shoal_dbscan(x, eps = 0.5, min_samples = 5L)$cluster,
    hdbscan = shoal_hdbscan(x)$cluster,
    evoc = shoal_evoc(emb, min_cluster_size = 15L)$layers,
    kmeans = shoal_kmeans(x, k = 3L)[c("cluster", "centroids", "inertia")],
    gmm = shoal_gmm(x, k = 3L)[c("cluster", "means", "loglik")],
    sil = shoal_silhouette(shoal_dist(x), shoal_kmeans(x, k = 3L))$width
  )
  shoal_threads(1)
  one <- run()
  shoal_threads(4)
  four <- run()
  expect_identical(one, four)
})
