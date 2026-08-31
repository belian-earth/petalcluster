# EVoC is stochastic and calibrated for embedding-like data, so quality
# assertions here use the regime it is built for (normalised directions with
# moderate angular spread, n in the hundreds) and generous bounds. The port
# itself is held to the Python reference by the fixture suite in evoc-port/;
# these tests cover the R surface.

evoc_test_data <- function(k = 6L, n_per = 140L, d = 48L, seed = 1L) {
  set.seed(seed)
  centres <- matrix(stats::runif(k * d, -1, 1) * 0.6, nrow = k)
  x <- centres[rep(seq_len(k), each = n_per), ] +
    matrix(stats::rnorm(k * n_per * d, sd = 0.1), ncol = d)
  list(x = x, truth = rep(seq_len(k), each = n_per))
}

# Adjusted Rand index, with NA (noise) treated as its own class.
ari <- function(a, b) {
  a <- ifelse(is.na(a), -1L, a)
  b <- ifelse(is.na(b), -1L, b)
  tab <- table(a, b)
  comb2 <- function(x) x * (x - 1) / 2
  sum_ij <- sum(comb2(tab))
  sum_a <- sum(comb2(rowSums(tab)))
  sum_b <- sum(comb2(colSums(tab)))
  expected <- sum_a * sum_b / comb2(sum(tab))
  max_index <- (sum_a + sum_b) / 2
  (sum_ij - expected) / (max_index - expected)
}

test_that("shoal_evoc returns aligned multi-layer results", {
  x <- evoc_test_data()$x
  fit <- shoal_evoc(x)

  expect_s3_class(fit, "shoal_evoc")
  expect_s3_class(fit, "shoal_clustering")
  expect_length(fit$cluster, nrow(x))

  n_layers <- length(fit$layers)
  expect_gte(n_layers, 1L)
  expect_length(fit$strengths, n_layers)
  expect_length(fit$persistence, n_layers)
  for (i in seq_len(n_layers)) {
    expect_length(fit$layers[[i]], nrow(x))
    expect_length(fit$strengths[[i]], nrow(x))
    expect_type(fit$layers[[i]], "integer")
  }

  # Layers come finest first.
  ks <- vapply(fit$layers, function(l) length(unique(l[!is.na(l)])), integer(1L))
  expect_false(is.unsorted(rev(ks)))

  # The selected layer is what populates cluster.
  expect_gte(fit$layer, 1L)
  expect_lte(fit$layer, n_layers)
  expect_identical(fit$cluster, fit$layers[[fit$layer]])

  expect_equal(fit$metric, "cosine")
  expect_equal(nrow(fit$embedding), nrow(x))
})

test_that("shoal_evoc recovers embedding-like cluster structure", {
  data <- evoc_test_data()
  fit <- shoal_evoc(data$x)

  # Some layer captures the true partition well. The bound is generous:
  # calibration runs on this data shape score 0.86-0.92.
  best <- max(vapply(fit$layers, function(l) ari(data$truth, l), numeric(1L)))
  expect_gte(best, 0.75)

  # The auto-selected layer lands near the true granularity, without
  # relabelling most of the data as noise.
  expect_gte(fit$n_clusters, 4L)
  expect_lte(fit$n_clusters, 12L)
  expect_lte(fit$n_noise, nrow(data$x) / 2L)
})

test_that("the seed makes runs reproducible and R's RNG is irrelevant", {
  x <- evoc_test_data(k = 3L, n_per = 60L, d = 16L)$x

  a <- shoal_evoc(x, seed = 42L)
  b <- shoal_evoc(x, seed = 42L)
  expect_equal(a$layers, b$layers)
  expect_equal(a$strengths, b$strengths)
  expect_equal(a$persistence, b$persistence)
  expect_equal(a$embedding, b$embedding)

  set.seed(1)
  c1 <- shoal_evoc(x, seed = 7L)$cluster
  set.seed(999)
  c2 <- shoal_evoc(x, seed = 7L)$cluster
  expect_equal(c1, c2)
})

test_that("layer selects the partition that populates cluster", {
  x <- evoc_test_data()$x
  auto <- shoal_evoc(x)

  # The automatic choice is the most persistent layer.
  expect_equal(auto$layer, which.max(auto$persistence))

  # An explicit index selects that layer; the other layers are unchanged.
  finest <- shoal_evoc(x, layer = 1L)
  expect_equal(finest$layer, 1L)
  expect_identical(finest$cluster, finest$layers[[1L]])
  expect_equal(finest$layers, auto$layers)

  n_layers <- length(auto$layers)
  expect_error(shoal_evoc(x, layer = n_layers + 1L), "layer")
  expect_error(shoal_evoc(x, layer = 0L), "layer")
  expect_error(shoal_evoc(x, layer = "finest"), "layer")
})

test_that("shoal_evoc validates its inputs", {
  x <- evoc_test_data(k = 2L, n_per = 30L, d = 8L)$x

  expect_error(shoal_evoc(1:10), "numeric matrix or data frame")
  expect_error(shoal_evoc(x[1:10, ]), "more rows")
  expect_error(shoal_evoc(x, n_neighbors = 0L), "positive integer")
  expect_error(shoal_evoc(x, noise_level = -1), "positive")
  expect_error(shoal_evoc(x, min_cluster_size = 0L), "positive integer")
  expect_error(shoal_evoc(x, min_samples = 0L), "positive integer")
  expect_error(shoal_evoc(x, n_epochs = 0L), "positive integer")
  expect_error(shoal_evoc(x, dim = 0L), "positive integer")
  expect_error(shoal_evoc(x, max_layers = 0L), "positive integer")
  expect_error(shoal_evoc(x, seed = -1L), "non-negative")
})

test_that("evoc results have no predict method", {
  x <- evoc_test_data(k = 2L, n_per = 30L, d = 8L)$x
  fit <- shoal_evoc(x)
  expect_error(predict(fit), "no applicable method")
})
