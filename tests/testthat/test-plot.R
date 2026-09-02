test_that("shoal_palette switches from Dark 2 to turbo past eight clusters", {
  expect_identical(shoal_palette(0), character(0))
  expect_length(shoal_palette(1), 1L)
  expect_identical(shoal_palette(8), unname(grDevices::palette.colors(8, "Dark 2")))
  expect_identical(shoal_palette(3), shoal_palette(8)[1:3])
  expect_identical(shoal_palette(12), viridisLite::turbo(12))
  expect_error(shoal_palette(-1), "non-negative")
})

test_that("col and pch override the cluster aesthetics", {
  res <- shoal_dbscan(as.matrix(iris[, 1:2]), eps = 0.5, min_samples = 5L)
  n <- nrow(res$data)

  aes <- cluster_aesthetics(res$cluster, shoal_palette(res$n_clusters))
  expect_true(aes$legend)
  expect_identical(aes$col[is.na(res$cluster)], rep("grey60", res$n_noise))
  expect_identical(aes$pch[is.na(res$cluster)], rep(4L, res$n_noise))

  aes <- cluster_aesthetics(res$cluster, shoal_palette(res$n_clusters), col = "black")
  expect_false(aes$legend)
  expect_identical(aes$col, rep("black", n))
  expect_identical(aes$pch[is.na(res$cluster)], rep(4L, res$n_noise))

  aes <- cluster_aesthetics(res$cluster, shoal_palette(res$n_clusters), pch = 1:2)
  expect_false(aes$legend)
  expect_identical(aes$pch, rep_len(1:2, n))
})

test_that("plot() accepts col and pch without a duplicate-argument error", {
  res <- shoal_dbscan(as.matrix(iris[, 1:4]), eps = 0.5, min_samples = 5L)
  pdf(NULL)
  on.exit(dev.off())
  expect_no_error(plot(res, col = "black"))
  expect_no_error(plot(res, pch = 3))
  expect_no_error(plot(res, xcol = 1, ycol = 2, col = grey(seq(0, 1, length.out = nrow(res$data)))))
  expect_no_error(plot(res, xcol = "Sepal.Length", ycol = "Petal.Length", pch = 1))
})

test_that("a palette too short for the clusters is refused", {
  res <- shoal_kmeans(as.matrix(iris[, 1:2]), k = 3L)
  pdf(NULL)
  on.exit(dev.off())
  expect_error(plot(res, pal = c("red", "blue")), "2 colours")
})

test_that("many clusters plot with the turbo palette", {
  res <- shoal_kmeans(as.matrix(iris[, 1:2]), k = 10L)
  vdiffr::expect_doppelganger("kmeans-ten-clusters", function() plot(res))
})
