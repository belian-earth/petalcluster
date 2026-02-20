#' Plot clustering results
#'
#' Produces a scatter plot matrix (pairs plot) of clustered data, colored by
#' cluster assignment. For 2-column data a single scatter plot is produced
#' instead. Noise points (`NA` cluster) are shown as grey crosses.
#'
#' @param x A clustering result object.
#' @param ... Additional arguments passed to [pairs()] or [plot.default()].
#'
#' @returns `x`, invisibly.
#'
#' @name plot.petal
NULL

#' @rdname plot.petal
#' @export
plot.petal_dbscan <- function(x, ...) {
  plot_clusters(x, title = "DBSCAN", ...)
}

#' @rdname plot.petal
#' @export
plot.petal_hdbscan <- function(x, ...) {
  plot_clusters(x, title = "HDBSCAN", ...)
}

#' @rdname plot.petal
#' @export
plot.petal_optics <- function(x, ...) {
  plot_clusters(x, title = "OPTICS", ...)
}

#' Build cluster colour/pch vectors
#' @noRd
cluster_aesthetics <- function(cluster, n_clusters) {
  is_noise <- is.na(cluster)

  if (n_clusters <= 8L) {
    pal <- grDevices::palette.colors(max(n_clusters, 2L), "R4")[seq_len(
      n_clusters
    )]
  } else {
    pal <- grDevices::hcl.colors(n_clusters, "Dynamic")
  }

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

#' Shared plotting logic
#' @noRd
plot_clusters <- function(obj, title, ...) {
  data <- obj$data
  cluster <- obj$cluster
  aes <- cluster_aesthetics(cluster, obj$n_clusters)

  if (ncol(data) == 2L) {
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

#' Add a legend (used for 2-column plots only)
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

  graphics::legend(
    "topright",
    legend = legend_labels,
    col = legend_col,
    pch = legend_pch,
    bty = "n",
    cex = 0.8
  )
}
