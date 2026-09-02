# Benchmark: shoal against the best R alternative for each algorithm.
# Run from project root:
#   Rscript bench/gen_data.R          # generate shared datasets (once)
#   NOT_CRAN=true R CMD INSTALL .     # release build (once)
#   Rscript bench/bench_r.R
#   BENCH_ONLY=Ward Rscript bench/bench_r.R   # one algorithm, merged into results
#
# Every comparison uses matched settings: the same k, the same number of
# restarts, the same iteration cap, the same linkage. Results are medians of
# three runs. Alternatives that are quadratic in memory or known to crash at
# scale are capped by `max_n` and reported as NA above it. R alternatives run
# in a subprocess so a crash costs one row rather than the run.

library(shoal)
`%||%` <- function(a, b) if (is.null(a)) b else a

# -- Helpers -------------------------------------------------------------------

time_it <- function(f, reps = 3L) {
  times <- numeric(reps)
  for (i in seq_len(reps)) {
    gc(FALSE)
    t0 <- proc.time()[["elapsed"]]
    f()
    times[i] <- proc.time()[["elapsed"]] - t0
  }
  stats::median(times)
}

# Time `f(x)` in a fresh R process; NA if it fails or exceeds the timeout.
safe_time <- function(f, x, reps = 3L, timeout = 600) {
  tryCatch(
    callr::r(
      function(f, x, reps) {
        times <- numeric(reps)
        for (i in seq_len(reps)) {
          gc(FALSE)
          t0 <- proc.time()[["elapsed"]]
          f(x)
          times[i] <- proc.time()[["elapsed"]] - t0
        }
        stats::median(times)
      },
      args = list(f = f, x = x, reps = reps),
      timeout = timeout
    ),
    error = function(e) {
      message("  [alternative failed: ", conditionMessage(e), "]")
      NA_real_
    }
  )
}

load_family <- function(prefix, d) {
  files <- sort(Sys.glob(sprintf("bench/data/%s_*_d%d.csv", prefix, d)))
  if (length(files) == 0) stop("Run bench/gen_data.R first.")
  datasets <- lapply(files, function(f) as.matrix(read.csv(f)))
  datasets[order(vapply(datasets, nrow, integer(1)))]
}

# -- Benchmark definitions -----------------------------------------------------
#
# Each entry: the shoal call, the alternative, the alternative's package name,
# the largest n the alternative is run at and, optionally, the largest n
# shoal itself is run at.

benchmarks <- list(
  list(
    algorithm = "DBSCAN", family = "blobs",
    shoal = function(x) shoal_dbscan(x, eps = 3.0, min_samples = 5L),
    alt = function(x) dbscan::dbscan(x, eps = 3.0, minPts = 5L),
    alt_name = "dbscan", max_n = Inf
  ),
  list(
    algorithm = "HDBSCAN", family = "blobs",
    shoal = function(x) shoal_hdbscan(x, min_samples = 5L, min_cluster_size = 15L),
    alt = function(x) dbscan::hdbscan(x, minPts = 5L),
    alt_name = "dbscan", max_n = 20000L  # full distance matrix
  ),
  list(
    algorithm = "k-means", family = "blobs",
    shoal = function(x) shoal_kmeans(x, k = 5L, n_runs = 10L, max_iter = 300L),
    alt = function(x) stats::kmeans(x, 5L, nstart = 10, iter.max = 300),
    alt_name = "stats", max_n = Inf
  ),
  list(
    algorithm = "GMM", family = "blobs",
    shoal = function(x) shoal_gmm(x, k = 5L, max_iter = 100L),
    # Mclust() evaluates unqualified calls, so its namespace must be attached.
    alt = function(x) { library(mclust); Mclust(x, G = 5, modelNames = "VVV", verbose = FALSE) },
    alt_name = "mclust", max_n = 20000L
  ),
  list(
    algorithm = "Ward", family = "blobs",
    shoal = function(x) shoal_hclust(shoal_dist(x), method = "ward"),
    alt = function(x) stats::hclust(stats::dist(x), method = "ward.D2"),
    alt_name = "stats", max_n = 20000L,  # n^2 / 2 distances: 1.6 GB at 20k
    max_n_shoal = 20000L
  ),
  list(
    algorithm = "Distances", family = "blobs",
    shoal = function(x) shoal_dist(x),
    alt = function(x) stats::dist(x),
    alt_name = "stats", max_n = 20000L,   # n^2 / 2 doubles: 1.6 GB at 20k
    max_n_shoal = 20000L
  ),
  list(
    algorithm = "EVoC", family = "emb",
    shoal = function(x) shoal_evoc(x, min_cluster_size = 15L),
    alt = NULL, alt_name = NULL, max_n = 0L  # no R alternative
  )
)

# -- Run -----------------------------------------------------------------------

dims_for <- list(blobs = c(2L, 10L), emb = 48L)
rows <- list()

only <- Sys.getenv("BENCH_ONLY")
if (nzchar(only)) {
  benchmarks <- Filter(function(b) b$algorithm == only, benchmarks)
  if (length(benchmarks) == 0) stop("No benchmark named ", only)
}

for (b in benchmarks) {
  for (d in dims_for[[b$family]]) {
    datasets <- load_family(b$family, d)
    cat(sprintf("\n=== %s, %s d=%d ===\n", b$algorithm, b$family, d))
    for (x in datasets) {
      n <- nrow(x)
      if (n > (b$max_n_shoal %||% Inf)) next
      cat(sprintf("  n=%6d ... ", n))
      t_shoal <- time_it(function() b$shoal(x))
      cat(sprintf("shoal %.3fs", t_shoal))
      rows[[length(rows) + 1]] <- data.frame(
        algorithm = b$algorithm, family = b$family, n = n, dims = d,
        package = "shoal", median_s = t_shoal
      )
      if (!is.null(b$alt)) {
        t_alt <- if (n <= b$max_n) safe_time(b$alt, x) else NA_real_
        cat(sprintf(" | %s %s", b$alt_name, if (is.na(t_alt)) "skipped" else sprintf("%.3fs", t_alt)))
        rows[[length(rows) + 1]] <- data.frame(
          algorithm = b$algorithm, family = b$family, n = n, dims = d,
          package = b$alt_name, median_s = t_alt
        )
      }
      cat("\n")
    }
  }
}

results <- do.call(rbind, rows)
csv_path <- file.path("bench", "results_r.csv")
if (nzchar(only) && file.exists(csv_path)) {
  previous <- read.csv(csv_path)
  results <- rbind(previous[previous$algorithm != only, ], results)
}
write.csv(results, csv_path, row.names = FALSE)
cat("\nSaved:", csv_path, "\n")
