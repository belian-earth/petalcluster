# Benchmark: petalcluster vs dbscan R package
# Run from project root:
#   Rscript bench/gen_data.R          # generate shared datasets (once)
#   NOT_CRAN=true R CMD INSTALL .     # release build (once)
#   Rscript bench/bench_r.R

library(petalcluster)

# -- Helpers -------------------------------------------------------------------

# Time an expression using proc.time for sub-second resolution.
time_it <- function(expr, min_iters = 3L) {
  expr <- substitute(expr)
  env <- parent.frame()
  times <- numeric(min_iters)
  for (i in seq_len(min_iters)) {
    gc(FALSE)
    t0 <- proc.time()["elapsed"]
    eval(expr, env)
    t1 <- proc.time()["elapsed"]
    times[i] <- t1 - t0
  }
  stats::median(times)
}

# Run dbscan pkg benchmark in a subprocess; returns median seconds or NA on crash.
safe_time_dbscan <- function(x, fn_name, args, min_iters = 3L) {
  tryCatch(
    callr::r(
      function(x, fn_name, args, min_iters) {
        fn <- getExportedValue("dbscan", fn_name)
        times <- numeric(0)
        for (i in seq_len(min_iters)) {
          gc(FALSE)
          t <- system.time(do.call(fn, c(list(x), args)))["elapsed"]
          times <- c(times, t)
        }
        stats::median(times)
      },
      args = list(x = x, fn_name = fn_name, args = args, min_iters = min_iters),
      timeout = 300
    ),
    error = function(e) {
      message("  [dbscan crashed: ", conditionMessage(e), "]")
      NA_real_
    }
  )
}

# -- Benchmark function --------------------------------------------------------

bench_r <- function(d) {
  pattern <- sprintf("bench/data/blobs_*_d%d.csv", d)
  data_files <- sort(Sys.glob(pattern))
  if (length(data_files) == 0) stop("Run bench/gen_data.R first.")

  datasets <- lapply(data_files, function(f) as.matrix(read.csv(f)))
  sizes <- vapply(datasets, nrow, integer(1))
  names(datasets) <- paste0("n=", format(sizes, big.mark = ",", trim = TRUE))

  cat(sprintf("\n===== d = %d =====\n", d))
  cat("Loaded", length(datasets), "datasets:",
      paste(names(datasets), collapse = ", "), "\n\n")

  # -- DBSCAN --
  cat("=== DBSCAN ===\n")
  dbscan_rows <- lapply(seq_along(sizes), function(i) {
    x <- datasets[[i]]
    nm <- names(datasets)[i]
    n <- sizes[i]
    cat(nm, "... ")

    t_petal <- time_it(petal_dbscan(x, eps = 3.0, min_samples = 5L))
    t_dbscan <- safe_time_dbscan(x, "dbscan", list(eps = 3.0, minPts = 5L))

    cat("done\n")
    data.frame(algorithm = "DBSCAN", dataset = nm, n = n, dims = d,
               package = c("petalcluster", "dbscan"),
               median_s = c(t_petal, t_dbscan))
  })

  # -- HDBSCAN --
  cat("\n=== HDBSCAN ===\n")
  hdbscan_rows <- lapply(seq_along(sizes), function(i) {
    x <- datasets[[i]]
    nm <- names(datasets)[i]
    n <- sizes[i]
    cat(nm, "... ")

    t_petal <- time_it(petal_hdbscan(x, min_samples = 5L, min_cluster_size = 15L))
    if (n > 20000L) {
      cat("[dbscan skipped, n > 20k] ")
      t_dbscan <- NA_real_
    } else {
      t_dbscan <- safe_time_dbscan(x, "hdbscan", list(minPts = 5L))
    }

    cat("done\n")
    data.frame(algorithm = "HDBSCAN", dataset = nm, n = n, dims = d,
               package = c("petalcluster", "dbscan"),
               median_s = c(t_petal, t_dbscan))
  })

  # -- OPTICS --
  cat("\n=== OPTICS ===\n")
  optics_rows <- lapply(seq_along(sizes), function(i) {
    x <- datasets[[i]]
    nm <- names(datasets)[i]
    n <- sizes[i]
    cat(nm, "... ")

    t_petal <- time_it(petal_optics(x, eps = 3.0, min_samples = 5L))
    t_dbscan <- safe_time_dbscan(x, "optics", list(eps = 3.0, minPts = 5L))

    cat("done\n")
    data.frame(algorithm = "OPTICS", dataset = nm, n = n, dims = d,
               package = c("petalcluster", "dbscan"),
               median_s = c(t_petal, t_dbscan))
  })

  do.call(rbind, c(dbscan_rows, hdbscan_rows, optics_rows))
}

# -- Run for each dimensionality -----------------------------------------------

all_results <- do.call(rbind, lapply(c(2L, 10L), bench_r))

csv_path <- file.path("bench", "results_r.csv")
write.csv(all_results, csv_path, row.names = FALSE)
cat("\nSaved:", csv_path, "\n")
