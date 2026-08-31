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
print.shoal_evoc <- function(x, ...) {
  NextMethod()

  cli::cli_text("Layers (finest first, {cli::symbol$star} = selected):")
  for (i in seq_along(x$layers)) {
    l <- x$layers[[i]]
    k <- length(unique(l[!is.na(l)]))
    noise <- sum(is.na(l))
    persistence <- signif(x$persistence[[i]], 4L)
    marker <- if (i == x$layer) cli::symbol$star else " "
    cli::cli_text(
      "{marker} {i}: {k} cluster{?s}, {noise} noise, persistence {persistence}"
    )
  }

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_kmeans <- function(x, ...) {
  NextMethod()

  inertia <- signif(x$inertia, 5L)
  sizes <- paste(x$sizes, collapse = ", ")
  cli::cli_text("Within-cluster sum of squares: {inertia}")
  cli::cli_text("Cluster sizes: {sizes}")

  invisible(x)
}

#' @rdname print.shoal
#' @export
print.shoal_gmm <- function(x, ...) {
  NextMethod()

  loglik <- signif(x$loglik, 6L)
  bic <- signif(stats::BIC(x), 6L)
  weights <- paste(signif(x$weights, 3L), collapse = ", ")
  cli::cli_text("Log-likelihood: {loglik}, BIC: {bic}")
  cli::cli_text("Mixing proportions: {weights}")

  invisible(x)
}
