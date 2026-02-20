#' @export
print.petal_dbscan <- function(x, ...) {
  cli::cli_h3("DBSCAN Clustering")
  cli::cli_text("Metric: {.val {x$metric}}")
  cli::cli_text("Parameters: eps = {x$params$eps}, min_samples = {x$params$min_samples}")
  cli::cli_text("Clusters: {x$n_clusters}, Noise points: {x$n_noise}")
  invisible(x)
}

#' @export
print.petal_hdbscan <- function(x, ...) {
  cli::cli_h3("HDBSCAN Clustering")
  cli::cli_text("Metric: {.val {x$metric}}")
  cli::cli_text(
    "Parameters: alpha = {x$params$alpha}, min_samples = {x$params$min_samples}, min_cluster_size = {x$params$min_cluster_size}"
  )
  cli::cli_text("Clusters: {x$n_clusters}, Noise points: {x$n_noise}")
  invisible(x)
}

#' @export
print.petal_optics <- function(x, ...) {
  cli::cli_h3("OPTICS Clustering")
  cli::cli_text("Metric: {.val {x$metric}}")
  cli::cli_text("Parameters: eps = {x$params$eps}, min_samples = {x$params$min_samples}")
  cli::cli_text("Clusters: {x$n_clusters}, Noise points: {x$n_noise}")
  invisible(x)
}
