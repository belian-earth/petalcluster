# Collate R and sklearn benchmark results, print table, and produce scaling plot.
# Run after bench_r.R and bench_sklearn.py:
#   Rscript bench/compare.R

r_csv <- "bench/results_r.csv"
py_csv <- "bench/results_sklearn.csv"

if (!file.exists(r_csv)) stop("Run bench/bench_r.R first")
if (!file.exists(py_csv)) stop("Run bench/bench_sklearn.py first (uv run bench/bench_sklearn.py)")

r_data  <- read.csv(r_csv)
py_data <- read.csv(py_csv)

# Drop NA rows (from crashes / skips)
r_data <- r_data[!is.na(r_data$median_s), ]
py_data <- py_data[!is.na(py_data$median_s), ]

common_cols <- c("algorithm", "n", "dims", "package", "median_s")
all_data <- rbind(r_data[, common_cols], py_data[, common_cols])

dims_values <- sort(unique(all_data$dims))

# -- Tables --------------------------------------------------------------------

cat(strrep("=", 75), "\n")
cat("Cross-language Benchmark: petalcluster vs dbscan (R) vs sklearn (Python)\n")
cat(strrep("=", 75), "\n\n")

for (d in dims_values) {
  cat(sprintf("***** d = %d dimensions *****\n\n", d))
  for (algo in c("DBSCAN", "HDBSCAN", "OPTICS")) {
    cat("---", algo, paste0(strrep("-", 60)), "\n")
    sub <- all_data[all_data$algorithm == algo & all_data$dims == d, ]

    if (nrow(sub) == 0) next

    wide <- reshape(
      sub,
      direction  = "wide",
      idvar      = "n",
      timevar    = "package",
      v.names    = "median_s"
    )
    names(wide) <- gsub("median_s\\.", "", names(wide))
    wide$dims <- NULL

    wide <- wide[order(wide$n), ]

    if ("dbscan" %in% names(wide)) {
      wide$"vs_dbscan(R)" <- round(wide$dbscan / wide$petalcluster, 2)
    }
    if ("sklearn" %in% names(wide)) {
      wide$"vs_sklearn" <- round(wide$sklearn / wide$petalcluster, 2)
    }

    print(wide, row.names = FALSE, digits = 3)
    cat("\n")
  }
}

cat("Speedup > 1 means petalcluster is faster.\n\n")

# -- Scaling plot (stacked: one row per dimensionality) ------------------------

pkg_styles <- list(
  petalcluster = list(col = "#E41A1C", pch = 19, lty = 1),
  dbscan       = list(col = "#377EB8", pch = 17, lty = 2),
  sklearn      = list(col = "#4DAF4A", pch = 15, lty = 3)
)

algos <- c("DBSCAN", "HDBSCAN", "OPTICS")
packages <- unique(all_data$package)
n_dims <- length(dims_values)

plot_path <- "bench/scaling.png"
png(plot_path, width = 1200, height = 400 * n_dims, res = 120)
par(mfrow = c(n_dims, 3), mar = c(4.5, 4.5, 2, 1), cex = 0.9)

for (d in dims_values) {
  for (algo in algos) {
    sub <- all_data[all_data$algorithm == algo & all_data$dims == d &
                    !is.na(all_data$median_s), ]
    if (nrow(sub) == 0) {
      plot.new()
      next
    }

    xlim <- range(sub$n)
    ylim <- range(sub$median_s)

    plot(
      NULL, xlim = xlim, ylim = ylim,
      xlab = "n (points)", ylab = "Time (seconds)",
      main = sprintf("%s (d=%d)", algo, d),
      xaxt = "n"
    )

    # nice x-axis labels
    at <- sort(unique(sub$n))
    labels <- ifelse(at >= 1000, paste0(at / 1000, "k"), as.character(at))
    axis(1, at = at, labels = labels)

    for (pkg in packages) {
      dd <- sub[sub$package == pkg & !is.na(sub$median_s), ]
      if (nrow(dd) == 0) next
      dd <- dd[order(dd$n), ]
      s <- pkg_styles[[pkg]]
      lines(dd$n, dd$median_s, col = s$col, lty = s$lty, lwd = 2)
      points(dd$n, dd$median_s, col = s$col, pch = s$pch, cex = 1.2)
    }

    if (algo == algos[1] && d == dims_values[1]) {
      legend(
        "topleft",
        legend = c("petalcluster (Rust)", "dbscan (R/C++)", "sklearn (Python)"),
        col = vapply(pkg_styles, `[[`, character(1), "col"),
        pch = vapply(pkg_styles, `[[`, numeric(1), "pch"),
        lty = vapply(pkg_styles, `[[`, numeric(1), "lty"),
        lwd = 2, bty = "n", cex = 0.85
      )
    }
  }
}

invisible(dev.off())
cat("Saved:", plot_path, "\n")
