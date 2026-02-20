#' OPTICS Clustering
#'
#' Ordering points to identify the clustering structure.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped).
#' @param eps Neighbourhood radius. Default `0.5`.
#' @param min_samples Minimum number of points to form a dense region. Default `5L`.
#' @param metric Distance metric, one of `"euclidean"` or `"cosine"`.
#'
#' @returns An object of class `"petal_optics"`: a list with components
#'   `cluster` (integer vector of cluster IDs, `NA` for noise),
#'   `n_clusters`, `n_noise`, `data` (the input matrix), `params`, and `metric`.
#'
#' @export
petal_optics <- function(x, eps = 0.5, min_samples = 5L, metric = c("euclidean", "cosine")) {
  x <- check_numeric_matrix(x)
  check_positive_number(eps)
  check_positive_integer(min_samples)
  metric <- rlang::arg_match(metric)

  result <- rust_optics(x, eps, as.integer(min_samples), metric)

  structure(
    list(
      cluster = result$cluster,
      n_clusters = result$n_clusters,
      n_noise = result$n_noise,
      data = x,
      params = list(eps = eps, min_samples = as.integer(min_samples)),
      metric = metric
    ),
    class = "petal_optics"
  )
}
