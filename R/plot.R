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
#' Colours come from `pal`, one per cluster, with noise in grey. To colour
#' points by something other than their cluster, pass `col` directly: it is
#' recycled to one entry per observation and used as is. The same goes for
#' `pch`. The cluster legend is drawn only when neither is overridden, since
#' it would no longer describe the points.
#'
#' @param x A clustering result object.
#' @param xcol,ycol Optional column name or index to plot on the x/y axis.
#'   When both are supplied, a single scatter plot is produced instead of a
#'   pairs matrix.
#' @param pal Character vector of colours for clusters, indexed by cluster ID.
#'   Defaults to [shoal_palette()] for the number of clusters found.
#' @param col,pch Optional per-observation colours and plotting characters,
#'   recycled to the number of rows. When given they replace the cluster
#'   colouring and the noise crosses respectively, and no legend is drawn.
#' @param ... Additional arguments passed to [pairs()] or [plot.default()].
#'
#' @returns `x`, invisibly.
#'
#' @examples
#' res <- shoal_hdbscan(rings, min_cluster_size = 15L, min_samples = 5L)
#' plot(res)
#'
#' # Colour by something else entirely, e.g. an outlier score.
#' plot(res, col = grey(1 - res$outlier_scores), pch = 19)
#'
#' @name plot.shoal
NULL

#' @rdname plot.shoal
#' @export
plot.shoal_clustering <- function(x, xcol = NULL, ycol = NULL,
                                  pal = shoal_palette(x$n_clusters),
                                  col = NULL, pch = NULL, ...) {
  plot_clusters(
    x, title = x$algorithm, xcol = xcol, ycol = ycol, pal = pal,
    col = col, pch = pch, ...
  )
}

#' Default cluster palette
#'
#' Qualitative colours for up to eight clusters, a continuous rainbow beyond
#' that. Up to eight, the ColorBrewer "Dark 2" palette: distinct, saturated
#' hues that hold up on white and print well. Past eight no qualitative set
#' stays distinguishable, so the turbo colour map is sampled evenly instead;
#' adjacent cluster IDs then get adjacent hues, which is at least legible.
#'
#' @param n Number of colours.
#'
#' @returns A character vector of `n` hex colours.
#'
#' @examples
#' shoal_palette(3)
#' shoal_palette(12)
#'
#' @export
shoal_palette <- function(n) {
  check_count(n)
  n <- as.integer(n)
  if (n == 0L) {
    return(character(0L))
  }
  if (n <= 8L) {
    unname(grDevices::palette.colors(n, palette = "Dark 2"))
  } else {
    viridisLite::turbo(n)
  }
}

#' Build cluster colour/pch vectors
#'
#' `col` and `pch` given by the user take precedence over the cluster mapping.
#' @noRd
cluster_aesthetics <- function(cluster, pal, col = NULL, pch = NULL) {
  n <- length(cluster)
  is_noise <- is.na(cluster)

  if (is.null(col)) {
    pt_col <- rep(NA_character_, n)
    clustered <- !is_noise
    if (any(clustered)) {
      pt_col[clustered] <- pal[cluster[clustered]]
    }
    pt_col[is_noise] <- "grey60"
  } else {
    pt_col <- rep_len(col, n)
  }

  if (is.null(pch)) {
    pt_pch <- rep(19L, n)
    pt_pch[is_noise] <- 4L
  } else {
    pt_pch <- rep_len(pch, n)
  }

  list(col = pt_col, pch = pt_pch, pal = pal, legend = is.null(col) && is.null(pch))
}

#' Resolve a column reference (name or index) to an integer index
#' @noRd
resolve_col <- function(col, data, arg, call) {
  if (is.character(col)) {
    idx <- match(col, colnames(data))
    if (is.na(idx)) {
      cli::cli_abort("Column {.val {col}} not found in data.", call = call)
    }
    idx
  } else if (is.numeric(col)) {
    col <- as.integer(col)
    if (col < 1L || col > ncol(data)) {
      cli::cli_abort("{.arg {arg}} must be between 1 and {ncol(data)}.", call = call)
    }
    col
  } else {
    cli::cli_abort("{.arg {arg}} must be a column name or index.", call = call)
  }
}

#' Shared plotting logic
#' @noRd
plot_clusters <- function(obj, title, xcol = NULL, ycol = NULL, pal,
                          col = NULL, pch = NULL, ...,
                          call = rlang::caller_env()) {
  data <- obj$data
  cluster <- obj$cluster

  if (length(pal) < obj$n_clusters) {
    cli::cli_abort(
      "{.arg pal} has {length(pal)} colour{?s}, but there are {obj$n_clusters} clusters.",
      call = call
    )
  }
  aes <- cluster_aesthetics(cluster, pal, col = col, pch = pch)

  scatter <- !is.null(xcol) || ncol(data) == 2L

  # Add bottom margin for the legend on scatter plots, scaled to legend rows
  if (scatter && aes$legend) {
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
        "Both {.arg xcol} and {.arg ycol} must be supplied together.",
        call = call
      )
    }
    xi <- resolve_col(xcol, data, "xcol", call = call)
    yi <- resolve_col(ycol, data, "ycol", call = call)
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
    if (aes$legend) {
      plot_legend(aes$pal, obj$n_clusters, obj$n_noise)
    }
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
    if (aes$legend) {
      plot_legend(aes$pal, obj$n_clusters, obj$n_noise)
    }
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
  legend_col <- pal[seq_len(n_clusters)]
  legend_pch <- rep(19L, n_clusters)

  if (n_noise > 0L) {
    legend_labels <- c(legend_labels, "Noise")
    legend_col <- c(legend_col, "grey60")
    legend_pch <- c(legend_pch, 4L)
  }

  n_items <- length(legend_labels)
  if (n_items == 0L) {
    return(invisible(NULL))
  }
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
