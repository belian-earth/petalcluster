# Silhouette widths computed straight from the definition, for comparison.
naive_silhouette <- function(m, cluster) {
  ids <- sort(unique(cluster))
  vapply(seq_along(cluster), function(i) {
    own <- cluster[i]
    if (sum(cluster == own) <= 1L) {
      return(0)
    }
    a <- mean(m[i, cluster == own & seq_along(cluster) != i])
    b <- min(vapply(setdiff(ids, own), function(c) mean(m[i, cluster == c]), numeric(1L)))
    (b - a) / max(a, b)
  }, numeric(1L))
}

test_that("silhouette widths match the definition", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L, seed = 1L)
  d <- shoal_dist(x)

  sil <- shoal_silhouette(d, fit)
  expect_equal(sil$width, naive_silhouette(as.matrix(d), fit$cluster), tolerance = 1e-10)
  expect_equal(attr(sil, "avg_width"), mean(sil$width))
})

test_that("silhouette returns a tidy frame describing every observation", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L, seed = 1L)

  sil <- shoal_silhouette(shoal_dist(x), fit)

  expect_s3_class(sil, "data.frame")
  expect_equal(nrow(sil), nrow(x))
  expect_named(sil, c("cluster", "neighbour", "width"))
  expect_equal(sil$cluster, fit$cluster)
  # A point is never its own nearest neighbouring cluster.
  expect_true(all(sil$neighbour != sil$cluster))
  expect_true(all(sil$width >= -1 & sil$width <= 1))
})

test_that("well-separated clusters score near 1", {
  set.seed(2)
  x <- rbind(
    cbind(stats::rnorm(30, 0, 0.1), stats::rnorm(30, 0, 0.1)),
    cbind(stats::rnorm(30, 20, 0.1), stats::rnorm(30, 0, 0.1))
  )
  sil <- shoal_silhouette(shoal_dist(x), rep(1:2, each = 30))
  expect_gt(attr(sil, "avg_width"), 0.9)
})

test_that("a deliberately wrong assignment scores negative", {
  set.seed(2)
  x <- rbind(
    cbind(stats::rnorm(30, 0, 0.1), stats::rnorm(30, 0, 0.1)),
    cbind(stats::rnorm(30, 20, 0.1), stats::rnorm(30, 0, 0.1))
  )
  # Interleaved labels cut across both true groups.
  sil <- shoal_silhouette(shoal_dist(x), rep(1:2, times = 30))
  expect_lt(attr(sil, "avg_width"), 0)
})

test_that("noise points are excluded rather than distorting the result", {
  x <- as.matrix(iris[, 1:4])
  cluster <- shoal_kmeans(x, k = 3L, seed = 1L)$cluster
  cluster[1:10] <- NA_integer_

  sil <- shoal_silhouette(shoal_dist(x), cluster)

  expect_equal(nrow(sil), nrow(x) - 10L)
  # Identical to computing on the reduced data from the outset.
  direct <- shoal_silhouette(shoal_dist(x[-(1:10), ]), cluster[-(1:10)])
  expect_equal(sil$width, direct$width, tolerance = 1e-10)
})

test_that("shoal_metrics reports both indices", {
  fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L, seed = 1L)
  m <- shoal_metrics(fit)

  expect_s3_class(m, "data.frame")
  expect_equal(nrow(m), 1L)
  expect_named(m, c("n", "k", "calinski_harabasz", "davies_bouldin"))
  expect_equal(m$n, 150L)
  expect_equal(m$k, 3L)
  expect_gt(m$calinski_harabasz, 0)
  expect_gt(m$davies_bouldin, 0)
})

test_that("Calinski-Harabasz matches its definition", {
  x <- as.matrix(iris[, 1:4])
  cluster <- shoal_kmeans(x, k = 3L, seed = 1L)$cluster

  grand <- colMeans(x)
  ids <- sort(unique(cluster))
  between <- sum(vapply(ids, function(c) {
    members <- x[cluster == c, , drop = FALSE]
    nrow(members) * sum((colMeans(members) - grand)^2)
  }, numeric(1L)))
  within <- sum(vapply(ids, function(c) {
    members <- x[cluster == c, , drop = FALSE]
    sum(sweep(members, 2L, colMeans(members))^2)
  }, numeric(1L)))

  expected <- (between / (length(ids) - 1L)) / (within / (nrow(x) - length(ids)))
  expect_equal(shoal_metrics(x, cluster)$calinski_harabasz, expected, tolerance = 1e-8)
})

test_that("Davies-Bouldin matches its definition", {
  x <- as.matrix(iris[, 1:4])
  cluster <- shoal_kmeans(x, k = 3L, seed = 1L)$cluster
  ids <- sort(unique(cluster))

  cent <- t(vapply(ids, function(c) colMeans(x[cluster == c, , drop = FALSE]), numeric(ncol(x))))
  scatter <- vapply(seq_along(ids), function(j) {
    members <- x[cluster == ids[j], , drop = FALSE]
    mean(sqrt(rowSums(sweep(members, 2L, cent[j, ])^2)))
  }, numeric(1L))

  expected <- mean(vapply(seq_along(ids), function(a) {
    max(vapply(setdiff(seq_along(ids), a), function(b) {
      (scatter[a] + scatter[b]) / sqrt(sum((cent[a, ] - cent[b, ])^2))
    }, numeric(1L)))
  }, numeric(1L)))

  expect_equal(shoal_metrics(x, cluster)$davies_bouldin, expected, tolerance = 1e-8)
})

test_that("the indices agree on the right k for clean blobs", {
  set.seed(5)
  blob <- function(cx, cy, n) cbind(stats::rnorm(n, cx, 0.3), stats::rnorm(n, cy, 0.3))
  x <- rbind(blob(0, 0, 50), blob(9, 0, 50), blob(4.5, 9, 50))

  scores <- do.call(rbind, lapply(2:6, function(k) {
    shoal_metrics(shoal_kmeans(x, k = k, seed = 1L))
  }))

  expect_equal(scores$k[which.max(scores$calinski_harabasz)], 3L)
  expect_equal(scores$k[which.min(scores$davies_bouldin)], 3L)
})

test_that("both accept a bare integer vector as well as a result object", {
  x <- as.matrix(iris[, 1:4])
  fit <- shoal_kmeans(x, k = 3L, seed = 1L)

  expect_equal(shoal_metrics(fit), shoal_metrics(x, fit$cluster))
  expect_equal(
    shoal_silhouette(shoal_dist(x), fit)$width,
    shoal_silhouette(shoal_dist(x), fit$cluster)$width
  )
})

test_that("validation functions check their inputs", {
  x <- as.matrix(iris[1:20, 1:4])
  d <- shoal_dist(x)
  cluster <- rep(1:2, each = 10L)

  expect_error(shoal_silhouette(d, rep(1L, 20L)), "at least 2 clusters")
  expect_error(shoal_silhouette(d, cluster[1:5]), "describes")
  expect_error(shoal_silhouette(d, letters[1:20]), "integer vector")
  expect_error(shoal_silhouette(d, rep(NA_integer_, 20L)), "no clustered")

  expect_error(shoal_metrics(x), "required unless")
  expect_error(shoal_metrics(x, rep(1L, 20L)), "at least 2 clusters")
  expect_error(shoal_metrics(x, cluster[1:5]), "but")
})
