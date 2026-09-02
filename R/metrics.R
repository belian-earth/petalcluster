#' Silhouette Widths
#'
#' For each observation, compares the mean distance to the rest of its own
#' cluster (`a`) with the smallest mean distance to any other cluster (`b`),
#' giving a width of `(b - a) / max(a, b)`. Widths near 1 indicate a
#' well-placed observation, near 0 one on a boundary, and negative ones an
#' observation that sits closer to a different cluster.
#'
#' An observation alone in its cluster is given a width of 0 by convention,
#' since it has no within-cluster distances to average.
#'
#' @param d A [stats::dist] object, or a numeric matrix or data frame, in which
#'   case Euclidean distances are computed with [shoal_dist()] first.
#' @param cluster A clustering result from this package, or an integer vector of
#'   cluster IDs. `NA` entries, the noise points of the density-based
#'   algorithms, are excluded along with their distances.
#'
#' @returns A data frame with one row per observation and columns `cluster`,
#'   `neighbour` (the nearest other cluster) and `width`. The mean width is
#'   attached as the `avg_width` attribute.
#'
#' @seealso [shoal_metrics()] for indices computed from the data instead.
#'
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' fit <- shoal_kmeans(x, k = 3L)
#' sil <- shoal_silhouette(shoal_dist(x), fit)
#' attr(sil, "avg_width")
#'
#' @export
shoal_silhouette <- function(d, cluster) {
  cluster <- as_cluster_vector(cluster)

  if (!inherits(d, "dist")) {
    if (looks_like_dist_matrix(d)) {
      cli::cli_abort(c(
        "{.arg d} looks like a square distance matrix, not raw data.",
        "i" = "Pass {.code as.dist(d)} to use it as distances."
      ))
    }
    d <- shoal_dist(d)
  }
  n <- attr(d, "Size")

  if (length(cluster) != n) {
    cli::cli_abort(
      "{.arg cluster} has {length(cluster)} element{?s}, but {.arg d} describes {n} observation{?s}."
    )
  }

  # Noise points belong to no cluster, so drop them and their distances rather
  # than letting them distort every other point's mean.
  keep <- !is.na(cluster)
  if (!any(keep)) {
    cli::cli_abort("{.arg cluster} contains no clustered observations.")
  }
  if (!all(keep)) {
    d <- subset_dist(d, keep)
    cluster <- cluster[keep]
    n <- sum(keep)
  }

  ids <- sort(unique(cluster))
  if (length(ids) < 2L) {
    cli::cli_abort("{.arg cluster} must contain at least 2 clusters.")
  }
  compact <- match(cluster, ids)

  if (!is.double(d)) {
    d <- as.double(d)
  }
  # Read in place on the Rust side: no copy of the distance vector.
  res <- rust_silhouette(d, as.integer(n), as.integer(compact), length(ids))

  out <- data.frame(
    cluster = ids[compact],
    neighbour = ids[res$neighbour],
    width = res$width
  )
  attr(out, "avg_width") <- mean(res$width)
  out
}

#' Internal Cluster Validity Indices
#'
#' Computes the Calinski-Harabasz and Davies-Bouldin indices, both of which
#' work from the data rather than a distance matrix, and both of which are used
#' the same way: fit at several values of `k` and compare.
#'
#' - **Calinski-Harabasz** is between-cluster dispersion over within-cluster
#'   dispersion, each per degree of freedom. **Higher is better.**
#' - **Davies-Bouldin** averages, over clusters, the worst-case ratio of
#'   combined within-cluster scatter to the distance between centroids.
#'   **Lower is better.**
#'
#' Both assume roughly convex, centroid-shaped clusters, so they suit
#' [shoal_kmeans()] and [shoal_gmm()] better than the density-based algorithms.
#' For arbitrary cluster shapes prefer [shoal_silhouette()], which needs only a
#' distance matrix.
#'
#' @param x A numeric matrix or data frame, or a clustering result from this
#'   package, in which case both the data and the assignment are taken from it.
#' @param cluster A clustering result or an integer vector of cluster IDs.
#'   Required unless `x` is itself a clustering result. `NA` entries are
#'   excluded.
#'
#' @returns A one-row data frame with columns `n`, `k`, `calinski_harabasz` and
#'   `davies_bouldin`.
#'
#' @seealso [shoal_silhouette()].
#'
#' @examples
#' fit <- shoal_kmeans(as.matrix(iris[, 1:4]), k = 3L)
#' shoal_metrics(fit)
#'
#' # Comparing candidate values of k
#' do.call(rbind, lapply(2:5, function(k) {
#'   shoal_metrics(shoal_kmeans(as.matrix(iris[, 1:4]), k = k))
#' }))
#'
#' @export
shoal_metrics <- function(x, cluster = NULL) {
  if (inherits(x, "shoal_clustering")) {
    cluster <- if (is.null(cluster)) x$cluster else as_cluster_vector(cluster)
    x <- x$data
  } else {
    if (is.null(cluster)) {
      cli::cli_abort("{.arg cluster} is required unless {.arg x} is a clustering result.")
    }
    cluster <- as_cluster_vector(cluster)
  }

  x <- check_numeric_matrix(x, na_action = "error")

  if (length(cluster) != nrow(x)) {
    cli::cli_abort(
      "{.arg cluster} has {length(cluster)} element{?s}, but {.arg x} has {nrow(x)} row{?s}."
    )
  }

  keep <- !is.na(cluster)
  if (!any(keep)) {
    cli::cli_abort("{.arg cluster} contains no clustered observations.")
  }
  x <- x[keep, , drop = FALSE]
  cluster <- cluster[keep]

  ids <- sort(unique(cluster))
  if (length(ids) < 2L) {
    cli::cli_abort("{.arg cluster} must contain at least 2 clusters.")
  }
  compact <- match(cluster, ids)

  res <- rust_cluster_indices(x, as.integer(compact), length(ids))

  data.frame(
    n = nrow(x),
    k = length(ids),
    calinski_harabasz = res$calinski_harabasz,
    davies_bouldin = res$davies_bouldin
  )
}

#' Accept either a clustering result or a bare vector of cluster IDs
#' @noRd
as_cluster_vector <- function(cluster, call = rlang::caller_env()) {
  if (inherits(cluster, "shoal_clustering")) {
    return(cluster$cluster)
  }
  if (!rlang::is_integerish(cluster)) {
    cli::cli_abort(
      "{.arg cluster} must be a clustering result or an integer vector.",
      call = call
    )
  }
  as.integer(cluster)
}
