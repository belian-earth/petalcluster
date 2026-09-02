# Generate shared benchmark datasets.
# Run from project root: Rscript bench/gen_data.R
#
# Two families:
#   blobs_{n}_d{d}.csv  five Gaussian blobs in d = 2 and 10 dimensions, for
#                       the general-purpose algorithms.
#   emb_{n}_d48.csv     embedding-like data for EVoC: eight topics of unequal
#                       size as directions in 48 dimensions plus 5% scattered
#                       points, rows normalised to unit length.

dir.create("bench/data", showWarnings = FALSE)

sizes <- c(500L, 1000L, 2000L, 5000L, 10000L, 20000L, 50000L)

make_blobs <- function(n, k = 5, d = 10) {
  centres <- matrix(rnorm(k * d, sd = 10), nrow = k)
  labels <- sample.int(k, n, replace = TRUE)
  centres[labels, , drop = FALSE] + matrix(rnorm(n * d), nrow = n)
}

make_emb <- function(n, k = 8, d = 48) {
  weights <- c(20, 16, 14, 12, 10, 8, 4, 2)
  n_noise <- round(0.05 * n)
  sizes <- round((n - n_noise) * weights / sum(weights))
  sizes[1] <- sizes[1] + (n - n_noise - sum(sizes))
  centres <- matrix(runif(k * d, -1, 1) * 0.6, nrow = k)
  x <- rbind(
    centres[rep(seq_len(k), times = sizes), ] + matrix(rnorm(sum(sizes) * d, sd = 0.1), ncol = d),
    matrix(runif(n_noise * d, -1, 1) * 0.6, ncol = d)
  )
  x / sqrt(rowSums(x^2))
}

set.seed(42)
for (d in c(2L, 10L)) {
  cat(sprintf("--- blobs, d = %d ---\n", d))
  for (n in sizes) {
    path <- sprintf("bench/data/blobs_%d_d%d.csv", n, d)
    write.csv(make_blobs(n, d = d), path, row.names = FALSE)
    cat(sprintf("  %s\n", path))
  }
}

cat("--- embeddings, d = 48 ---\n")
for (n in sizes) {
  path <- sprintf("bench/data/emb_%d_d48.csv", n)
  write.csv(make_emb(n), path, row.names = FALSE)
  cat(sprintf("  %s\n", path))
}

cat("Done.\n")
