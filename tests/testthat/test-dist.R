test_that("shoal_dist returns a plain dist object", {
  x <- as.matrix(iris[, 1:4])
  d <- shoal_dist(x)

  # Deliberately not a subclass: code that tests class(x) != "dist" would error.
  expect_identical(class(d), "dist")
  expect_equal(attr(d, "Size"), nrow(x))
  expect_false(attr(d, "Diag"))
  expect_false(attr(d, "Upper"))
  expect_equal(attr(d, "method"), "euclidean")
  expect_length(d, nrow(x) * (nrow(x) - 1L) / 2L)
})

test_that("the condensed vector uses R's dist ordering", {
  # The layout is the thing most likely to be silently wrong, so check every
  # pair against the position R's own indexing formula puts it in.
  set.seed(42)
  x <- matrix(stats::rnorm(24), nrow = 6L)
  n <- nrow(x)

  v <- as.double(shoal_dist(x))
  full <- as.matrix(stats::dist(x))
  r_index <- function(i, j, n) n * (i - 1L) - i * (i - 1L) / 2L + j - i

  for (i in seq_len(n - 1L)) {
    for (j in seq(i + 1L, n)) {
      expect_equal(v[r_index(i, j, n)], full[i, j], tolerance = 1e-12)
    }
  }
})

test_that("metrics shared with stats::dist agree numerically", {
  set.seed(7)
  x <- matrix(stats::runif(60, min = 0.5, max = 5), nrow = 12L)
  metrics <- c("euclidean", "maximum", "manhattan", "canberra")

  # Compared as named lists so a failure names the metric that diverged.
  ours <- stats::setNames(
    lapply(metrics, function(m) as.double(shoal_dist(x, metric = m))), metrics
  )
  theirs <- stats::setNames(
    lapply(metrics, function(m) as.double(stats::dist(x, method = m))), metrics
  )
  expect_equal(ours, theirs, tolerance = 1e-10)

  expect_equal(
    as.double(shoal_dist(x, metric = "minkowski", p = 3)),
    as.double(stats::dist(x, method = "minkowski", p = 3)),
    tolerance = 1e-10
  )
})

test_that("canberra matches stats::dist when zero terms are present", {
  # Terms with a zero numerator and denominator are dropped and the total
  # rescaled; this is the branch that is easy to get wrong.
  x <- rbind(
    c(0, 1, 2, 0),
    c(0, 3, 0, 4),
    c(1, 0, 2, 5)
  )
  expect_equal(
    as.double(shoal_dist(x, metric = "canberra")),
    as.double(stats::dist(x, method = "canberra")),
    tolerance = 1e-10
  )
})

test_that("binary matches stats::dist on presence/absence data", {
  set.seed(3)
  x <- matrix(stats::rbinom(80, size = 1L, prob = 0.4), nrow = 10L)
  expect_equal(
    as.double(shoal_dist(x, metric = "binary")),
    as.double(stats::dist(x, method = "binary")),
    tolerance = 1e-12
  )
})

test_that("cosine agrees with the definition used by the density algorithms", {
  set.seed(11)
  x <- matrix(stats::runif(40, min = 0.1, max = 3), nrow = 8L)

  cos_dist <- function(a, b) 1 - sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
  expected <- unlist(lapply(seq_len(nrow(x) - 1L), function(i) {
    vapply(seq(i + 1L, nrow(x)), function(j) cos_dist(x[i, ], x[j, ]), numeric(1L))
  }))

  expect_equal(as.double(shoal_dist(x, metric = "cosine")), expected, tolerance = 1e-12)
})

test_that("correlation is 1 minus the Pearson correlation", {
  set.seed(13)
  x <- matrix(stats::rnorm(50), nrow = 10L)

  expected <- unlist(lapply(seq_len(nrow(x) - 1L), function(i) {
    vapply(seq(i + 1L, nrow(x)), function(j) 1 - stats::cor(x[i, ], x[j, ]), numeric(1L))
  }))

  expect_equal(as.double(shoal_dist(x, metric = "correlation")), expected, tolerance = 1e-12)
})

test_that("the result works with the rest of R's dist ecosystem", {
  x <- as.matrix(iris[1:20, 1:4])
  d <- shoal_dist(x)

  expect_equal(dim(as.matrix(d)), c(20L, 20L))
  expect_equal(diag(as.matrix(d)), rep(0, 20L), ignore_attr = TRUE)
  expect_s3_class(stats::hclust(d), "hclust")
  expect_equal(nrow(stats::cmdscale(d, k = 2L)), 20L)
})

test_that("row names are carried through as labels", {
  x <- as.matrix(iris[1:5, 1:4])
  rownames(x) <- letters[1:5]
  expect_equal(attr(shoal_dist(x), "Labels"), letters[1:5])
})

test_that("shoal_dist validates its inputs", {
  x <- as.matrix(iris[, 1:4])

  expect_error(shoal_dist(x, metric = "jaccard"))
  expect_error(shoal_dist(x, p = -1), "positive")
  expect_error(shoal_dist(1:10), "numeric matrix or data frame")
  expect_error(shoal_dist(matrix(1:4, nrow = 1L)), "at least 2")
})

test_that("degenerate rows are reported rather than returned as NaN", {
  # A zero row has no direction, so cosine distance is undefined.
  x <- rbind(c(0, 0, 0), c(1, 2, 3), c(2, 1, 0))
  expect_error(shoal_dist(x, metric = "cosine"), "non-finite")
})

test_that("mahalanobis matches stats::mahalanobis pair by pair", {
  x <- as.matrix(iris[, 1:4])
  d <- shoal_dist(x, metric = "mahalanobis")
  expect_s3_class(d, "dist")
  expect_identical(attr(d, "method"), "mahalanobis")

  S <- stats::cov(x)
  full <- as.matrix(d)
  for (pair in list(c(1L, 2L), c(1L, 51L), c(51L, 101L), c(10L, 150L))) {
    i <- pair[1]; j <- pair[2]
    expect_equal(full[i, j], sqrt(stats::mahalanobis(x[i, ], x[j, ], S)), tolerance = 1e-10)
  }

  # With the identity as covariance it is plain Euclidean distance.
  expect_equal(
    as.numeric(shoal_dist(x, metric = "mahalanobis", cov = diag(4))),
    as.numeric(shoal_dist(x)),
    tolerance = 1e-12
  )
  # Rescaling a column changes nothing: the metric is scale-invariant.
  y <- x
  y[, 1] <- y[, 1] * 1000
  expect_equal(as.numeric(shoal_dist(y, metric = "mahalanobis")), as.numeric(d), tolerance = 1e-8)
})

test_that("mahalanobis validates its covariance", {
  x <- as.matrix(iris[, 1:4])
  expect_error(shoal_dist(x[1:3, ], metric = "mahalanobis"), "more rows than columns")
  expect_error(shoal_dist(x, metric = "mahalanobis", cov = diag(3)), "4 x 4")
  asym <- diag(4); asym[1, 2] <- 1
  expect_error(shoal_dist(x, metric = "mahalanobis", cov = asym), "symmetric")
  singular <- diag(c(1, 1, 1, 0))
  expect_error(shoal_dist(x, metric = "mahalanobis", cov = singular), "not positive definite")
  constant <- cbind(x, 1)
  expect_error(shoal_dist(constant, metric = "mahalanobis"), "not positive definite")
})

test_that("canberra uses |x| + |y| in the denominator, as stats::dist() does", {
  set.seed(1)
  x <- matrix(rnorm(60), ncol = 3)  # signed values, where the two forms differ
  expect_equal(
    as.double(shoal_dist(x, metric = "canberra")),
    as.double(stats::dist(x, method = "canberra"))
  )
  # Opposite signs give a term of exactly 1, so a pair differing in sign on
  # every coordinate is at the maximum.
  expect_equal(as.double(shoal_dist(rbind(c(1, 2, 3), c(-1, -2, -3)), metric = "canberra")), 3)
})
