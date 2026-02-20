#' Plot clustering results
#'
#' Produces a scatter plot matrix (pairs plot) of clustered data, colored by
#' cluster assignment. For 2-column data a single scatter plot is produced
#' instead. Noise points (`NA` cluster) are shown as grey crosses.
#'
#' When `xcol` and `ycol` are supplied, a single scatter plot of those two
#' variables is produced instead of the full pairs matrix. Columns can be
#' specified by name or integer index.
#'
#' @param x A clustering result object.
#' @param xcol,ycol Optional column name or index to plot on the x/y axis.
#'   When both are supplied, a single scatter plot is produced instead of a
#'   pairs matrix.
#' @param pal Character vector of colours for clusters. Defaults to
#'   `grDevices::hcl.colors(n, "Roma")` where `n` is the number of clusters.
#' @param ... Additional arguments passed to [pairs()] or [plot.default()].
#'
#' @returns `x`, invisibly.
#'
#' @name plot.petal
NULL

#' @rdname plot.petal
#' @export
plot.petal_dbscan <- function(x, xcol = NULL, ycol = NULL,
                              pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
                              ...) {
  plot_clusters(x, title = "DBSCAN", xcol = xcol, ycol = ycol, pal = pal, ...)
}

#' @rdname plot.petal
#' @export
plot.petal_hdbscan <- function(x, xcol = NULL, ycol = NULL,
                               pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
                               ...) {
  plot_clusters(x, title = "HDBSCAN", xcol = xcol, ycol = ycol, pal = pal, ...)
}

#' @rdname plot.petal
#' @export
plot.petal_optics <- function(x, xcol = NULL, ycol = NULL,
                              pal = grDevices::hcl.colors(max(x$n_clusters, 2L), "Roma"),
                              ...) {
  plot_clusters(x, title = "OPTICS", xcol = xcol, ycol = ycol, pal = pal, ...)
}

#' Build cluster colour/pch vectors
#' @noRd
cluster_aesthetics <- function(cluster, n_clusters, pal) {
  is_noise <- is.na(cluster)

  pt_col <- rep(NA_character_, length(cluster))
  pt_pch <- rep(19L, length(cluster))

  clustered <- !is_noise
  if (any(clustered)) {
    pt_col[clustered] <- pal[cluster[clustered]]
  }
  if (any(is_noise)) {
    pt_col[is_noise] <- "grey60"
    pt_pch[is_noise] <- 4L
  }

  list(col = pt_col, pch = pt_pch, pal = pal)
}

#' Resolve a column reference (name or index) to an integer index
#' @noRd
resolve_col <- function(col, data, arg) {
  if (is.character(col)) {
    idx <- match(col, colnames(data))
    if (is.na(idx)) {
      cli::cli_abort(
        "Column {.val {col}} not found in data.",
        call = rlang::caller_env(2)
      )
    }
    idx
  } else if (is.numeric(col)) {
    col <- as.integer(col)
    if (col < 1L || col > ncol(data)) {
      cli::cli_abort(
        "{.arg {arg}} must be between 1 and {ncol(data)}.",
        call = rlang::caller_env(2)
      )
    }
    col
  } else {
    cli::cli_abort(
      "{.arg {arg}} must be a column name or index.",
      call = rlang::caller_env(2)
    )
  }
}

#' Shared plotting logic
#' @noRd
plot_clusters <- function(obj, title, xcol = NULL, ycol = NULL, pal, ...) {
  data <- obj$data
  cluster <- obj$cluster
  aes <- cluster_aesthetics(cluster, obj$n_clusters, pal)

  scatter <- !is.null(xcol) || ncol(data) == 2L

  # Add bottom margin for legend on scatter plots, scaled to legend rows
  if (scatter) {
    n_items <- obj$n_clusters + (obj$n_noise > 0L)
    max_per_row <- max(floor(graphics::par("pin")[1] / 1.0), 1L)
    n_rows <- ceiling(n_items / max_per_row)
    extra <- 1.0 + n_rows * 0.8
    opar <- graphics::par(mar = c(5.1, 4.1, 4.1, 2.1) + c(extra, 0, 0, 0))
    on.exit(graphics::par(opar), add = TRUE)
  }

  # User-specified x/y columns -> single scatter plot
  if (!is.null(xcol) || !is.null(ycol)) {
    if (is.null(xcol) || is.null(ycol)) {
      cli::cli_abort(
        "Both {.arg xcol} and {.arg ycol} must be supplied together."
      )
    }
    xi <- resolve_col(xcol, data, "xcol")
    yi <- resolve_col(ycol, data, "ycol")
    plot(
      data[, xi],
      data[, yi],
      col = aes$col,
      pch = aes$pch,
      xlab = colnames(data)[xi] %||% paste0("V", xi),
      ylab = colnames(data)[yi] %||% paste0("V", yi),
      main = title,
      ...
    )
    plot_legend(aes$pal, obj$n_clusters, obj$n_noise)
  } else if (ncol(data) == 2L) {
    plot(
      data[, 1L],
      data[, 2L],
      col = aes$col,
      pch = aes$pch,
      xlab = colnames(data)[1L] %||% "X1",
      ylab = colnames(data)[2L] %||% "X2",
      main = title,
      ...
    )
    plot_legend(aes$pal, obj$n_clusters, obj$n_noise)
  } else {
    graphics::pairs(
      data,
      col = aes$col,
      pch = aes$pch,
      main = title,
      ...
    )
  }

  invisible(obj)
}

#' Add a legend below the plot, wrapping into multiple rows if needed
#' @noRd
plot_legend <- function(pal, n_clusters, n_noise) {
  legend_labels <- paste("Cluster", seq_len(n_clusters))
  legend_col <- pal
  legend_pch <- rep(19L, n_clusters)

  if (n_noise > 0L) {
    legend_labels <- c(legend_labels, "Noise")
    legend_col <- c(legend_col, "grey60")
    legend_pch <- c(legend_pch, 4L)
  }

  n_items <- length(legend_labels)
  # Estimate how many items fit per row (~1 inch each)
  max_per_row <- max(floor(graphics::par("pin")[1] / 1.0), 1L)
  legend_ncol <- min(n_items, max_per_row)

  # Draw in the bottom margin using a physical offset (inches) so the gap

  # is consistent regardless of data scale.
  graphics::par(xpd = NA)
  usr <- graphics::par("usr")
  pin <- graphics::par("pin")  # plot region height in inches
  csi <- graphics::par("csi")  # line height in inches
  mgp1 <- graphics::par("mgp")[1]  # axis label line (default 3)

  # Place legend below: axis label line + 1.5 lines of padding
  offset_inches <- (mgp1 + 1.5) * csi
  y_offset <- offset_inches * diff(usr[3:4]) / pin[2]

  graphics::legend(
    x = mean(usr[1:2]),
    y = usr[3] - y_offset,
    legend = legend_labels,
    col = legend_col,
    pch = legend_pch,
    bty = "n",
    cex = 0.8,
    ncol = legend_ncol,
    xjust = 0.5
  )
}
