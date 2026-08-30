#' Print a clustering result
#'
#' A single method serves every algorithm: the heading, parameters and cluster
#' counts all come from components the shared constructor guarantees. Algorithms
#' with extra output specialise and call [NextMethod()].
#'
#' @param x A clustering result.
#' @param ... Ignored.
#'
#' @returns `x`, invisibly.
#'
#' @name print.shoal
NULL

#' @rdname print.shoal
#' @export
print.shoal_clustering <- function(x, ...) {
  params <- format_params(x$params)

  cli::cli_h3("{x$algorithm} Clustering")
  if (!is.null(x$metric)) {
    cli::cli_text("Metric: {.val {x$metric}}")
  }
  cli::cli_text("Parameters: {params}")
  cli::cli_text("Clusters: {x$n_clusters}, Noise points: {x$n_noise}")
  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_hdbscan <- function(x, ...) {
  NextMethod()

  med <- round(stats::median(x$outlier_scores), 3L)
  mx <- round(max(x$outlier_scores), 3L)
  cli::cli_text("GLOSH outlier scores: median {med}, max {mx}")

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_kmeans <- function(x, ...) {
  NextMethod()

  inertia <- signif(x$inertia, 5L)
  cli::cli_text("Within-cluster sum of squares: {inertia}")
  cli::cli_text("Cluster sizes: {x$sizes}")

  invisible(x)
}
