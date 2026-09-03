# Reference search in R: the full distance matrix, each row ordered by
# distance then index, first k kept. Slow, but independent of the Rust path
# apart from the metric kernel itself, which test-dist.R checks.
reference_knn <- function(x, k, query = NULL, metric = "euclidean", p = 2, cov = NULL) {
  if (is.null(query)) {
    full <- as.matrix(shoal_dist(x, metric = metric, p = p, cov = cov))
    diag(full) <- Inf
  } else {
    m <- nrow(query)
    if (identical(metric, "mahalanobis")) {
      # Whiten with the reference set's covariance, then plain Euclidean.
      ch <- chol(cov %||% stats::cov(x))
      w <- function(z) t(backsolve(ch, t(z), transpose = TRUE))
      full <- as.matrix(stats::dist(rbind(w(query), w(x))))[seq_len(m), m + seq_len(nrow(x))]
    } else {
      full <- as.matrix(shoal_dist(rbind(query, x), metric = metric, p = p))
      full <- full[seq_len(m), m + seq_len(nrow(x))]
    }
  }
  n <- ncol(full)
  ord <- t(apply(full, 1L, function(d) order(d, seq_len(n))[seq_len(k)]))
  id <- matrix(as.integer(ord), ncol = k)
  dist <- matrix(full[cbind(rep(seq_len(nrow(full)), k), as.vector(id))], ncol = k)
  list(id = id, dist = dist)
}

test_that("shoal_knn returns the documented structure", {
  x <- as.matrix(iris[, 1:4])
  rownames(x) <- paste0("r", seq_len(nrow(x)))
  nn <- shoal_knn(x, k = 5L)

  expect_s3_class(nn, "shoal_knn")
  expect_named(nn, c("id", "dist", "k", "metric", "search"))
  expect_identical(dim(nn$id), c(150L, 5L))
  expect_identical(dim(nn$dist), c(150L, 5L))
  expect_true(is.integer(nn$id))
  expect_true(is.double(nn$dist))
  expect_identical(dimnames(nn$id), list(rownames(x), as.character(1:5)))
  expect_identical(dimnames(nn$dist), dimnames(nn$id))
  expect_identical(nn$k, 5L)
  expect_identical(nn$metric, "euclidean")
  expect_identical(nn$search, "kdtree")

  # Nearest first, and never the row itself.
  expect_true(all(nn$dist[, -1L] >= nn$dist[, -5L]))
  expect_false(any(nn$id == seq_len(nrow(x))))
})

test_that("every metric agrees with the ordered distance matrix", {
  set.seed(11)
  x <- matrix(stats::runif(40 * 6, min = 0.5, max = 5), nrow = 40L)
  metrics <- c(
    "euclidean", "maximum", "manhattan", "canberra", "minkowski", "cosine",
    "correlation", "mahalanobis"
  )
  for (m in metrics) {
    nn <- shoal_knn(x, k = 4L, metric = m, p = 3)
    ref <- reference_knn(x, k = 4L, metric = m, p = 3)
    expect_identical(unname(nn$id), ref$id, label = m)
    expect_equal(unname(nn$dist), ref$dist, tolerance = 1e-12, label = m)
  }

  # Binary needs 0/1 data to be meaningful.
  b <- matrix(stats::rbinom(30 * 8, 1, 0.4), nrow = 30L)
  nn <- shoal_knn(b, k = 3L, metric = "binary")
  ref <- reference_knn(b, k = 3L, metric = "binary")
  expect_identical(unname(nn$id), ref$id)
  expect_equal(unname(nn$dist), ref$dist, tolerance = 1e-12)
})

test_that("shoal_knn agrees with dbscan::kNN on Euclidean data", {
  skip_if_not_installed("dbscan")
  x <- as.matrix(iris[, 1:4])

  # iris has many exactly tied distances. dbscan's kd-tree sums squares in a
  # different order, so its ties can differ in the last bit and sort the
  # other way round. Distances must agree everywhere; ids are compared only at
  # positions whose distance is unique within the row, with one neighbour
  # beyond k included so a tie across the boundary is seen too.
  expect_same_neighbours <- function(x, k, query = NULL) {
    ours <- shoal_knn(x, k = k, query = query)
    theirs <- dbscan::kNN(x, k = k, query = query)
    expect_equal(unname(ours$dist), unname(theirs$dist), tolerance = 1e-12)

    wider <- shoal_knn(x, k = k + 1L, query = query)$dist
    unique_pos <- t(apply(wider, 1L, function(d) {
      !(duplicated(d) | duplicated(d, fromLast = TRUE))
    }))[, seq_len(k), drop = FALSE]
    expect_identical(unname(ours$id)[unique_pos], unname(theirs$id)[unique_pos])
    # Most positions are untied, so the comparison has teeth.
    expect_gt(mean(unique_pos), 0.5)
  }

  expect_same_neighbours(x, k = 6L)
  expect_same_neighbours(x, k = 4L, query = x[c(3, 77, 140), ])
})

test_that("the kd-tree and the scan give identical results", {
  set.seed(9)
  # Rounded values give many exact ties, which is where the two could
  # legitimately differ if the tie rule were not shared.
  x <- round(matrix(stats::rnorm(600 * 3), ncol = 3), 1)
  q <- round(matrix(stats::rnorm(25 * 3), ncol = 3), 1)
  metrics <- c(
    "euclidean", "manhattan", "maximum", "minkowski", "mahalanobis", "cosine",
    "correlation"
  )
  for (m in metrics) {
    for (k in c(1L, 7L, 30L)) {
      brute <- shoal_knn(x, k, metric = m, p = 3, search = "brute")
      tree <- shoal_knn(x, k, metric = m, p = 3, search = "kdtree")
      expect_identical(brute$id, tree$id, label = paste(m, k))
      expect_identical(brute$dist, tree$dist, label = paste(m, k))
      brute <- shoal_knn(x, k, query = q, metric = m, p = 3, search = "brute")
      tree <- shoal_knn(x, k, query = q, metric = m, p = 3, search = "kdtree")
      expect_identical(brute$id, tree$id, label = paste("query", m, k))
      expect_identical(brute$dist, tree$dist, label = paste("query", m, k))
    }
  }
  expect_identical(shoal_knn(x, 599L, search = "brute")$id, shoal_knn(x, 599L, search = "kdtree")$id)
})

test_that("search = 'auto' picks the tree only where it applies", {
  narrow <- matrix(stats::runif(100 * 3), ncol = 3)
  wide <- matrix(stats::runif(100 * 20), ncol = 20)
  expect_identical(shoal_knn(narrow, 3L)$search, "kdtree")
  expect_identical(shoal_knn(wide, 3L)$search, "brute")
  expect_identical(shoal_knn(narrow, 3L, metric = "cosine")$search, "kdtree")
  expect_identical(shoal_knn(narrow, 3L, metric = "canberra")$search, "brute")
  expect_identical(shoal_knn(narrow, 3L, metric = "mahalanobis")$search, "kdtree")
  expect_identical(shoal_knn(wide, 3L, search = "kdtree")$search, "kdtree")
  expect_identical(shoal_knn(narrow, 3L, search = "brute")$search, "brute")
  expect_error(shoal_knn(narrow, 3L, metric = "canberra", search = "kdtree"), "does not support")
  expect_error(shoal_knn(narrow, 3L, metric = "binary", search = "kdtree"), "does not support")
  expect_error(shoal_knn(narrow, 3L, search = "balltree"), "must be one of")
})

test_that("ties are broken by row index and duplicates are kept", {
  x <- rbind(c(0, 0), c(0, 0), c(0, 0), c(1, 0), c(-1, 0), c(5, 5))
  for (s in c("brute", "kdtree")) {
    nn <- shoal_knn(x, k = 3L, search = s)
    expect_identical(unname(nn$id[1L, ]), c(2L, 3L, 4L))
    expect_identical(unname(nn$dist[1L, ]), c(0, 0, 1))
    expect_identical(unname(nn$id[4L, ]), c(1L, 2L, 3L))
    expect_identical(unname(nn$id[6L, ]), c(4L, 1L, 2L))
  }
})

test_that("query rows keep an identical data row as their nearest neighbour", {
  x <- as.matrix(iris[, 1:4])
  nn <- shoal_knn(x, k = 2L, query = x[1:3, ])
  expect_identical(unname(nn$id[, 1L]), 1:3)
  expect_identical(unname(nn$dist[, 1L]), c(0, 0, 0))
  expect_identical(rownames(nn$id), NULL)

  q <- x[c(10, 20), ]
  rownames(q) <- c("a", "b")
  nn <- shoal_knn(x, k = 2L, query = q)
  expect_identical(rownames(nn$id), c("a", "b"))
})

test_that("query search agrees with the reference for every metric", {
  set.seed(5)
  x <- matrix(stats::runif(30 * 5, min = 0.5, max = 5), nrow = 30L)
  q <- matrix(stats::runif(7 * 5, min = 0.5, max = 5), nrow = 7L)
  for (m in c("euclidean", "manhattan", "cosine", "correlation", "mahalanobis")) {
    nn <- shoal_knn(x, k = 5L, query = q, metric = m)
    ref <- reference_knn(x, k = 5L, query = q, metric = m)
    expect_identical(unname(nn$id), ref$id, label = m)
    expect_equal(unname(nn$dist), ref$dist, tolerance = 1e-10, label = m)
  }

  # A supplied covariance is applied to both sets.
  cv <- diag(c(1, 2, 3, 4, 5))
  nn <- shoal_knn(x, k = 3L, query = q, metric = "mahalanobis", cov = cv)
  ref <- reference_knn(x, k = 3L, query = q, metric = "mahalanobis", cov = cv)
  expect_identical(unname(nn$id), ref$id)
  expect_equal(unname(nn$dist), ref$dist, tolerance = 1e-10)
})

test_that("k = 1 and k = n - 1 both work", {
  x <- as.matrix(iris[1:10, 1:4])
  one <- shoal_knn(x, k = 1L)
  expect_identical(dim(one$id), c(10L, 1L))
  all <- shoal_knn(x, k = 9L)
  expect_identical(dim(all$id), c(10L, 9L))
  # With every other row a neighbour, each row of id is a permutation.
  for (i in 1:10) {
    expect_setequal(unname(all$id[i, ]), setdiff(1:10, i))
  }
  expect_identical(dim(shoal_knn(x, k = 10L, query = x[1:2, ])$id), c(2L, 10L))
})

test_that("input validation follows dbscan::kNN", {
  x <- as.matrix(iris[, 1:4])
  expect_error(shoal_knn(x, k = 0L), "positive integer")
  expect_error(shoal_knn(x, k = 150L), "at most 149")
  expect_error(shoal_knn(x, k = 151L, query = x[1:2, ]), "at most 150")
  expect_error(shoal_knn(x, k = 3L, query = x[1:2, 1:3]), "same columns")

  # Rows with NA are refused, not dropped: dropping would renumber the ids.
  xn <- x
  xn[5L, 2L] <- NA
  expect_error(shoal_knn(xn, k = 3L), "missing or non-finite")
  expect_error(shoal_knn(x, k = 3L, query = xn[1:6, ]), "missing or non-finite")
  expect_no_warning(try(shoal_knn(xn, k = 3L), silent = TRUE))

  # Non-finite distances are an error even when the offending row would not
  # have been among the neighbours.
  z <- rbind(x[1:20, ], 0)
  expect_error(shoal_knn(z, k = 2L, metric = "cosine", search = "brute"), "non-finite")
  expect_error(shoal_knn(z, k = 2L, metric = "cosine", search = "kdtree"), "non-finite")

  expect_error(shoal_knn(x, k = 3L, metric = "hamming"), "must be one of")
})

test_that("data frames are accepted and non-numeric columns dropped", {
  nn_df <- shoal_knn(iris, k = 3L)
  nn_mat <- shoal_knn(as.matrix(iris[, 1:4]), k = 3L)
  expect_identical(nn_df$id, nn_mat$id)
  expect_identical(nn_df$dist, nn_mat$dist)
  expect_identical(
    shoal_knn(iris, k = 2L, query = iris[1:3, ])$id,
    shoal_knn(as.matrix(iris[, 1:4]), k = 2L, query = as.matrix(iris[1:3, 1:4]))$id
  )
})

test_that("results do not depend on the thread count", {
  old <- shoal_threads()
  on.exit(shoal_threads(old))
  set.seed(3)
  x <- matrix(stats::rnorm(500 * 8), nrow = 500L)
  shoal_threads(1)
  one <- shoal_knn(x, k = 7L, metric = "cosine")
  shoal_threads(4)
  four <- shoal_knn(x, k = 7L, metric = "cosine")
  expect_identical(one, four)
})

test_that("print and plot methods work", {
  nn <- shoal_knn(as.matrix(iris[, 1:4]), k = 4L, metric = "manhattan")
  printed <- paste(cli::cli_fmt(print(nn)), collapse = "\n")
  expect_match(printed, "k-Nearest Neighbours")
  expect_match(printed, "manhattan")
  expect_match(printed, "Search: \"kdtree\"")
  expect_match(printed, "Points: 150, Neighbours: 4")
  expect_match(printed, "Distance to neighbour 4")
  expect_invisible(print(nn))

  pdf(NULL)
  on.exit(dev.off())
  expect_invisible(plot(nn))
  expect_no_error(plot(nn, which = 2L, main = "eps", col = "red"))
  expect_error(plot(nn, which = 5L), "at most 4")
  expect_error(plot(nn, which = 0L), "positive integer")
})
