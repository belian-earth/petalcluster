#' Construct a clustering result
#'
#' Shared constructor for every clustering algorithm in the package. `n_clusters`
#' and `n_noise` are derived from `cluster` here rather than returned from Rust,
#' so the Rust side only ever has to produce the assignment vector.
#'
#' Every result carries two classes: an algorithm-specific one (`subclass`) and
#' the shared parent `"shoal_clustering"`. Generic behaviour lives on the parent;
#' algorithms specialise only where they genuinely differ.
#'
#' @param cluster Integer vector of cluster IDs, `NA` for noise.
#' @param data The matrix the model was fitted to.
#' @param algorithm Human-readable algorithm name, used by `print()` and `plot()`.
#' @param subclass Algorithm-specific class, e.g. `"shoal_dbscan"`.
#' @param params Named list of the parameters the algorithm was called with.
#' @param ... Further components stored on the result, e.g. `metric` or
#'   `outlier_scores`.
#'
#' @returns An object of class `c(subclass, "shoal_clustering")`.
#' @noRd
new_clustering <- function(cluster, data, algorithm, subclass, params, ...) {
  cluster <- as.integer(cluster)

  structure(
    c(
      list(
        cluster = cluster,
        n_clusters = length(unique(cluster[!is.na(cluster)])),
        n_noise = sum(is.na(cluster)),
        data = data,
        algorithm = algorithm,
        params = params
      ),
      list(...)
    ),
    class = c(subclass, "shoal_clustering")
  )
}

#' Format a parameter list as `name = value` pairs
#'
#' Values are formatted individually and collapsed, so a parameter holding a
#' vector prints without erroring.
#'
#' @noRd
format_params <- function(p) {
  if (length(p) == 0L) {
    return("none")
  }
  vals <- vapply(p, function(v) paste(format(v), collapse = ", "), character(1L))
  paste(names(p), vals, sep = " = ", collapse = ", ")
}
