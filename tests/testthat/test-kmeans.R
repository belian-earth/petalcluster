test_that("shoal_kmeans partitions every observation", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L)

  expect_s3_class(fit, "shoal_kmeans")
  expect_s3_class(fit, "shoal_clustering")
  expect_length(fit$cluster, nrow(x))
  # k-means has no noise class: every point is assigned.
  expect_false(anyNA(fit$cluster))
  expect_equal(fit$n_noise, 0L)
  expect_equal(fit$n_clusters, 3L)
  expect_setequal(unique(fit$cluster), 1:3)
})

test_that("centroids and sizes describe the partition they came from", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L)

  expect_equal(dim(fit$centroids), c(3L, 4L))
  expect_equal(colnames(fit$centroids), colnames(x))
  expect_equal(sum(fit$sizes), nrow(x))
  expect_equal(sort(fit$sizes), sort(as.integer(table(fit$cluster))))

  # Each centroid is the mean of the points assigned to it.
  for (j in 1:3) {
    expect_equal(
      unname(fit$centroids[j, ]),
      unname(colMeans(x[fit$cluster == j, , drop = FALSE])),
      tolerance = 1e-6
    )
  }
})

test_that("inertia is the within-cluster sum of squares", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L)

  wss <- sum(vapply(seq_len(nrow(x)), function(i) {
    sum((x[i, ] - fit$centroids[fit$cluster[i], ])^2)
  }, numeric(1L)))

  expect_equal(fit$inertia, wss, tolerance = 1e-6)
})

test_that("inertia decreases as k grows", {
  x <- as.matrix(iris[, 1:4])
  inertias <- vapply(2:6, function(k) shoal_kmeans(x, k = k)$inertia, numeric(1L))
  expect_false(is.unsorted(rev(inertias)))
})

test_that("the seed makes runs reproducible", {
  x <- as.matrix(iris[, 1:4])

  a <- shoal_kmeans(x, k = 3L, seed = 42L)
  b <- shoal_kmeans(x, k = 3L, seed = 42L)
  expect_equal(a$cluster, b$cluster)
  expect_equal(a$centroids, b$centroids)

  # And R's own RNG state is irrelevant to it.
  set.seed(1)
  c1 <- shoal_kmeans(x, k = 3L, seed = 7L)$cluster
  set.seed(999)
  c2 <- shoal_kmeans(x, k = 3L, seed = 7L)$cluster
  expect_equal(c1, c2)
})

test_that("k-means recovers well-separated blobs", {
  set.seed(4)
  blob <- function(cx, cy, n) cbind(stats::rnorm(n, cx, 0.2), stats::rnorm(n, cy, 0.2))
  x <- rbind(blob(0, 0, 40), blob(10, 0, 40), blob(5, 10, 40))
  truth <- rep(1:3, each = 40)

  fit <- shoal_kmeans(x, k = 3L, seed = 1L)

  # A perfect bijection between true blobs and found clusters: every cell of
  # the cross-tabulation is either empty or a whole blob, whatever the labelling.
  expect_true(all(table(truth, fit$cluster) %in% c(0L, 40L)))
})

test_that("every init method produces a valid partition", {
  x <- as.matrix(iris[, 1:4])

  for (init in c("kmeans++", "kmeans_parallel", "random")) {
    fit <- shoal_kmeans(x, k = 3L, init = init, seed = 2L)
    expect_setequal(unique(fit$cluster), 1:3)
    expect_equal(fit$params$init, init)
  }
})

test_that("k = 1 and k = nrow(x) are both handled", {
  x <- as.matrix(iris[1:10, 1:4])

  one <- shoal_kmeans(x, k = 1L)
  expect_equal(unique(one$cluster), 1L)
  expect_equal(dim(one$centroids), c(1L, 4L))

  all_singletons <- shoal_kmeans(x, k = 10L)
  expect_equal(nrow(all_singletons$centroids), 10L)
  expect_lte(all_singletons$n_clusters, 10L)
})

test_that("shoal_kmeans validates its inputs", {
  x <- as.matrix(iris[1:10, 1:4])

  expect_error(shoal_kmeans(x, k = 0L), "positive integer")
  expect_error(shoal_kmeans(x, k = 11L), "cannot exceed")
  expect_error(shoal_kmeans(x, k = 2L, init = "forgy"))
  expect_error(shoal_kmeans(x, k = 2L, n_runs = 0L), "positive integer")
  expect_error(shoal_kmeans(x, k = 2L, tolerance = -1), "positive")
  expect_error(shoal_kmeans(x, k = 2L, seed = -1L), "non-negative")
  expect_error(shoal_kmeans(1:10), "numeric matrix or data frame")
})

test_that("predict assigns new observations to the nearest centroid", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L, seed = 1L)

  # Predicting the training data reproduces the fitted assignment.
  expect_equal(predict(fit, x), fit$cluster)
  # Omitting newdata returns the training assignment unchanged.
  expect_equal(predict(fit), fit$cluster)

  # A point placed on a centroid is assigned to that centroid.
  expect_equal(predict(fit, fit$centroids), 1:3)
})

test_that("predict checks newdata against the fitted model", {
  fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)

  expect_error(predict(fit, as.matrix(iris[, 1:2])), "fitted on")

  with_na <- as.matrix(iris[1:5, 1:4])
  with_na[2L, 1L] <- NA_real_
  # Dropping rows would leave the result misaligned with newdata, so this errors.
  expect_error(predict(fit, with_na), "missing values")
})

test_that("density-based results have no predict method", {
  # The capability genuinely does not exist for them, so R's own dispatch
  # error is the right failure rather than a hand-written one.
  db <- shoal_dbscan(as.matrix(iris[, 1:4]))
  expect_error(predict(db), "no applicable method")
})
