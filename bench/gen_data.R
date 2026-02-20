# Generate shared benchmark datasets for multiple dimensionalities.
# Run from project root: Rscript bench/gen_data.R

dir.create("bench/data", showWarnings = FALSE)

set.seed(42)
sizes <- c(500L, 1000L, 2000L, 5000L, 10000L, 20000L, 50000L)
dims <- c(2L, 10L)

make_blobs <- function(n, k = 5, d = 10) {
  centres <- matrix(rnorm(k * d, sd = 10), nrow = k)
  labels <- sample.int(k, n, replace = TRUE)
  centres[labels, , drop = FALSE] + matrix(rnorm(n * d), nrow = n)
}

for (d in dims) {
  cat(sprintf("--- d = %d ---\n", d))
  for (n in sizes) {
    x <- make_blobs(n, d = d)
    path <- sprintf("bench/data/blobs_%d_d%d.csv", n, d)
    write.csv(x, path, row.names = FALSE)
    cat(sprintf("  %s  (%d x %d)\n", path, nrow(x), ncol(x)))
  }
}

cat("Done. Generated", length(sizes) * length(dims), "datasets.\n")
