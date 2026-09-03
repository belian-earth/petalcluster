# Collate the R and Python benchmark results, print speedup tables, and draw
# the scaling figure. Run after bench_r.R and bench_sklearn.py:
#   Rscript bench/compare.R

r_csv <- "bench/results_r.csv"
py_csv <- "bench/results_sklearn.csv"
if (!file.exists(r_csv)) stop("Run bench/bench_r.R first")
if (!file.exists(py_csv)) stop("Run bench/bench_sklearn.py first (uv run bench/bench_sklearn.py)")

cols <- c("algorithm", "family", "n", "dims", "package", "median_s")
all_data <- rbind(read.csv(r_csv)[, cols], read.csv(py_csv)[, cols])
all_data <- all_data[!is.na(all_data$median_s), ]

algorithms <- c("DBSCAN", "HDBSCAN", "k-means", "GMM", "Ward", "Distances", "kNN", "EVoC")
panels <- unique(all_data[, c("algorithm", "dims")])
panels <- panels[order(match(panels$algorithm, algorithms), panels$dims), ]

# -- Tables --------------------------------------------------------------------

cat(strrep("=", 78), "\n")
cat("shoal against the best R and Python alternatives (median seconds, 3 runs)\n")
cat(strrep("=", 78), "\n")

for (i in seq_len(nrow(panels))) {
  algo <- panels$algorithm[i]
  d <- panels$dims[i]
  sub <- all_data[all_data$algorithm == algo & all_data$dims == d, ]
  wide <- reshape(sub[, c("n", "package", "median_s")], direction = "wide",
                  idvar = "n", timevar = "package", v.names = "median_s")
  names(wide) <- gsub("median_s\\.", "", names(wide))
  wide <- wide[order(wide$n), ]
  for (pkg in setdiff(names(wide), c("n", "shoal"))) {
    wide[[paste0("speedup_vs_", pkg)]] <- round(wide[[pkg]] / wide$shoal, 1)
  }
  cat(sprintf("\n--- %s, d = %d ---\n", algo, d))
  print(wide, row.names = FALSE, digits = 3)
}
cat("\nSpeedup > 1 means shoal is faster.\n\n")

# -- Scaling figure: one panel per algorithm and dimensionality, log-x --------

styles <- list(
  shoal  = list(col = "#D95F02", pch = 19, lty = 1, label = "shoal (Rust)"),
  dbscan = list(col = "#1B9E77", pch = 17, lty = 2, label = "dbscan (R)"),
  stats  = list(col = "#1B9E77", pch = 17, lty = 2, label = "stats (R)"),
  mclust = list(col = "#1B9E77", pch = 17, lty = 2, label = "mclust (R)"),
  sklearn = list(col = "#7570B3", pch = 15, lty = 3, label = "scikit-learn"),
  scipy  = list(col = "#7570B3", pch = 15, lty = 3, label = "SciPy"),
  evoc   = list(col = "#7570B3", pch = 15, lty = 3, label = "evoc (Python)")
)

n_panels <- nrow(panels)
n_col <- 2L
n_row <- ceiling(n_panels / n_col)

plot_path <- "bench/scaling.png"
png(plot_path, width = 1000, height = 360 * n_row, res = 120)
par(mfrow = c(n_row, n_col), mar = c(4, 4.2, 2.5, 1), cex = 0.85)

for (i in seq_len(nrow(panels))) {
  algo <- panels$algorithm[i]
  d <- panels$dims[i]
  sub <- all_data[all_data$algorithm == algo & all_data$dims == d, ]

  plot(NULL, xlim = range(sub$n), ylim = range(sub$median_s), log = "x",
       xlab = "n (points)", ylab = "seconds", xaxt = "n",
       main = sprintf("%s, d = %d", algo, d))
  at <- sort(unique(sub$n))
  axis(1, at = at, labels = ifelse(at >= 1000, paste0(at / 1000, "k"), at))
  grid(col = "grey90", lty = 1)

  present <- unique(sub$package)
  for (pkg in present) {
    dd <- sub[sub$package == pkg, ]
    dd <- dd[order(dd$n), ]
    s <- styles[[pkg]]
    lines(dd$n, dd$median_s, col = s$col, lty = s$lty, lwd = 2)
    points(dd$n, dd$median_s, col = s$col, pch = s$pch, cex = 1.1)
  }
  legend("topleft",
         legend = vapply(present, function(p) styles[[p]]$label, character(1)),
         col = vapply(present, function(p) styles[[p]]$col, character(1)),
         pch = vapply(present, function(p) styles[[p]]$pch, numeric(1)),
         lty = vapply(present, function(p) styles[[p]]$lty, numeric(1)),
         lwd = 2, bty = "n", cex = 0.85)
}

invisible(dev.off())
cat("Saved:", plot_path, "\n")
