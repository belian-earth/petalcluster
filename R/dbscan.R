#' DBSCAN Clustering
#'
#' Density-based spatial clustering of applications with noise.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped).
#' @param eps Neighbourhood radius. Default `0.5`.
#' @param min_samples Minimum number of points to form a dense region. Default `5L`.
#' @param metric Distance metric, one of `"euclidean"` or `"cosine"`.
#'
#' @returns An object of class `c("shoal_dbscan", "shoal_clustering")`: a list
#'   with components `cluster` (integer vector of cluster IDs, `NA` for noise),
#'   `n_clusters`, `n_noise`, `data` (the input matrix), `algorithm`, `params`,
#'   and `metric`.
#'
#' @examples
#' res <- shoal_dbscan(as.matrix(iris[, 1:4]), eps = 0.5, min_samples = 5L)
#' res
#'
#' @export
shoal_dbscan <- function(x, eps = 0.5, min_samples = 5L, metric = c("euclidean", "cosine")) {
  x <- check_numeric_matrix(x)
  check_positive_number(eps)
  check_positive_integer(min_samples)
  metric <- rlang::arg_match(metric)

  cluster <- rust_dbscan(x, eps, as.integer(min_samples), metric)

  new_clustering(
    cluster = cluster,
    data = x,
    algorithm = "DBSCAN",
    subclass = "shoal_dbscan",
    params = list(eps = eps, min_samples = as.integer(min_samples)),
    metric = metric
  )
}
