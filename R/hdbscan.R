#' HDBSCAN Clustering
#'
#' Hierarchical density-based spatial clustering of applications with noise.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped).
#' @param alpha Smoothing parameter for mutual reachability distance. Default `1.0`.
#' @param min_samples Minimum neighbourhood size. Default `15L`.
#' @param min_cluster_size Minimum cluster size. Default `15L`.
#' @param metric Distance metric, one of `"euclidean"` or `"cosine"`.
#' @param boruvka Whether to use Boruvka's algorithm for MST construction. Default `TRUE`.
#' @param partial_labels Optional named list for semi-supervised clustering.
#'   Names are cluster IDs (as strings), values are integer vectors of 1-indexed
#'   point indices. `NULL` (default) for fully unsupervised clustering.
#'
#' @returns An object of class `"petal_hdbscan"`: a list with components
#'   `cluster` (integer vector of cluster IDs, `NA` for noise),
#'   `n_clusters`, `n_noise`, `data` (the input matrix), `params`, `metric`,
#'   and `outlier_scores` (GLOSH scores).
#'
#' @export
petal_hdbscan <- function(x, alpha = 1.0, min_samples = 15L, min_cluster_size = 15L,
                          metric = c("euclidean", "cosine"), boruvka = TRUE,
                          partial_labels = NULL) {
  x <- check_numeric_matrix(x)
  check_positive_number(alpha)
  check_positive_integer(min_samples)
  check_positive_integer(min_cluster_size)
  metric <- rlang::arg_match(metric)

  if (!rlang::is_bool(boruvka)) {
    cli::cli_abort("{.arg boruvka} must be {.code TRUE} or {.code FALSE}.")
  }

  if (!is.null(partial_labels)) {
    if (!is.list(partial_labels) || is.null(names(partial_labels))) {
      cli::cli_abort("{.arg partial_labels} must be a named list or {.code NULL}.")
    }
  }

  result <- rust_hdbscan(
    x, alpha, as.integer(min_samples), as.integer(min_cluster_size),
    metric, boruvka, partial_labels
  )

  structure(
    list(
      cluster = result$cluster,
      n_clusters = result$n_clusters,
      n_noise = result$n_noise,
      data = x,
      params = list(
        alpha = alpha,
        min_samples = as.integer(min_samples),
        min_cluster_size = as.integer(min_cluster_size),
        boruvka = boruvka
      ),
      metric = metric,
      outlier_scores = result$outlier_scores
    ),
    class = "petal_hdbscan"
  )
}
