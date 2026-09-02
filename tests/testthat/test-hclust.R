# Two partitions are equivalent if they induce the same grouping, regardless of
# how the groups happen to be numbered.
same_partition <- function(a, b) {
  identical(outer(a, a, "=="), outer(b, b, "=="))
}

test_that("shoal_hclust returns a usable hclust object", {
  d <- shoal_dist(as.matrix(iris[, 1:4]))
  fit <- shoal_hclust(d, method = "complete")

  expect_s3_class(fit, "hclust")
  expect_equal(dim(fit$merge), c(149L, 2L))
  expect_length(fit$height, 149L)
  expect_setequal(fit$order, seq_len(150L))
  expect_equal(fit$method, "complete")
  expect_equal(fit$dist.method, "euclidean")
})

test_that("merge heights match stats::hclust", {
  # If the condensed ordering were wrong, these would diverge immediately.
  d <- shoal_dist(as.matrix(iris[, 1:4]))
  rd <- stats::dist(as.matrix(iris[, 1:4]))

  equivalent <- list(
    single = "single",
    complete = "complete",
    average = "average",
    weighted = "mcquitty",
    ward = "ward.D2"
  )

  ours <- lapply(names(equivalent), function(m) shoal_hclust(d, method = m)$height)
  theirs <- lapply(equivalent, function(m) stats::hclust(rd, method = m)$height)
  names(ours) <- names(equivalent)

  expect_equal(ours, theirs, tolerance = 1e-9)
})

test_that("cutree gives the same partitions as stats::hclust", {
  d <- shoal_dist(as.matrix(iris[, 1:4]))
  rd <- stats::dist(as.matrix(iris[, 1:4]))

  grid <- expand.grid(
    method = c("single", "complete", "average"),
    k = c(2L, 3L, 5L),
    stringsAsFactors = FALSE
  )
  agree <- vapply(seq_len(nrow(grid)), function(i) {
    m <- grid$method[i]
    k <- grid$k[i]
    same_partition(
      stats::cutree(shoal_hclust(d, method = m), k = k),
      stats::cutree(stats::hclust(rd, method = m), k = k)
    )
  }, logical(1L))
  names(agree) <- paste(grid$method, grid$k, sep = "/")

  expect_equal(agree[!agree], stats::setNames(logical(0L), character(0L)))
})

test_that("the merge matrix is a valid hclust encoding", {
  fit <- shoal_hclust(shoal_dist(as.matrix(iris[1:30, 1:4])), method = "average")
  n <- 30L

  expect_true(all(fit$merge != 0L))
  # Every observation is merged in exactly once.
  expect_setequal(-fit$merge[fit$merge < 0L], seq_len(n))
  # Every intermediate cluster is consumed exactly once, and only after it exists.
  expect_setequal(fit$merge[fit$merge > 0L], seq_len(n - 2L))
  # A step may only reference clusters formed at strictly earlier steps.
  forward_refs <- vapply(seq_len(nrow(fit$merge)), function(k) {
    positives <- fit$merge[k, fit$merge[k, ] > 0L]
    any(positives >= k)
  }, logical(1L))
  expect_false(any(forward_refs))
})

test_that("the leaf order draws a dendrogram without crossings", {
  fit <- shoal_hclust(shoal_dist(as.matrix(iris[1:40, 1:4])), method = "complete")

  expect_setequal(fit$order, seq_len(40L))
  expect_length(unique(fit$order), 40L)
  # as.dendrogram validates the whole structure, order included.
  expect_s3_class(stats::as.dendrogram(fit), "dendrogram")
})

test_that("a matrix is accepted directly and routed through shoal_dist", {
  x <- as.matrix(iris[1:25, 1:4])
  expect_equal(
    shoal_hclust(x, method = "ward")$height,
    shoal_hclust(shoal_dist(x), method = "ward")$height,
    tolerance = 1e-12
  )
})

test_that("labels are carried from the dist object", {
  x <- as.matrix(iris[1:6, 1:4])
  rownames(x) <- letters[1:6]
  expect_equal(shoal_hclust(shoal_dist(x))$labels, letters[1:6])
})

test_that("inversions in centroid and median linkage are warned about", {
  # Centroid and median make no monotonicity guarantee; cutree rejects such
  # trees, so the user needs to be told rather than finding out later.
  set.seed(19)
  x <- matrix(stats::rnorm(120), nrow = 30L)
  d <- shoal_dist(x)

  for (m in c("centroid", "median")) {
    fit <- suppressWarnings(shoal_hclust(d, method = m))
    if (is.unsorted(fit$height)) {
      expect_warning(shoal_hclust(d, method = m), "inversions")
    }
  }
  succeed()
})

test_that("shoal_hclust validates its inputs", {
  d <- shoal_dist(as.matrix(iris[1:10, 1:4]))

  expect_error(shoal_hclust(d, method = "nope"))

  bad <- d
  bad[1L] <- NA_real_
  expect_error(shoal_hclust(bad), "missing or non-finite")

  truncated <- structure(
    as.double(d)[1:5], Size = 10L, Diag = FALSE, Upper = FALSE,
    method = "euclidean", class = "dist"
  )
  expect_error(shoal_hclust(truncated), "expected")
})

test_that("raw data takes the fused path and matches the two-step result", {
  x <- as.matrix(iris[, 1:4])
  rownames(x) <- paste0("obs", seq_len(nrow(x)))
  fused <- shoal_hclust(x, method = "average")
  stepwise <- shoal_hclust(shoal_dist(x), method = "average")
  for (component in c("merge", "height", "order", "labels", "method", "dist.method")) {
    expect_equal(fused[[component]], stepwise[[component]])
  }
  expect_identical(fused$labels, rownames(x))

  # A data frame is accepted, and inversion warnings still surface.
  expect_s3_class(shoal_hclust(iris[, 1:4]), "hclust")
  expect_warning(shoal_hclust(x[1:60, ], method = "centroid"), "inversions")

  # One row has no pairs to cluster; refused before any distance is computed.
  expect_error(shoal_hclust(x[1, , drop = FALSE]), "at least 2 rows")
})
