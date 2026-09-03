#' k-Nearest Neighbours
#'
#' Finds the `k` nearest neighbours of every row of `x` among the other rows,
#' or of every row of `query` among the rows of `x`, by an exact search in
#' Rust. Every metric of [shoal_dist()] is available and the distances agree
#' with it exactly, but only the `k` nearest are kept, so the memory is
#' proportional to `k` times the number of rows rather than to the square of
#' it. Where a distance matrix stops being possible somewhere in the tens of
#' thousands of rows, a neighbour search does not.
#'
#' Two exact searches are available and give identical results, tie order
#' included. `"brute"` compares every query row with every data row, in
#' parallel over queries; it serves every metric. `"kdtree"` builds an
#' axis-aligned kd-tree and skips regions that cannot hold a nearer point;
#' it serves every metric but `"canberra"` and `"binary"`, whose distances
#' no rectangle can bound. `"cosine"` and `"correlation"` are searched
#' through a unit-normalised (and, for correlation, centred) copy of the
#' rows, on which Euclidean distance orders points exactly as the metric
#' does; the distances reported are still the metric itself. A tree prunes
#' well in a few dimensions and hardly at all beyond about ten, where the
#' scan is several times faster than the kd-tree searches in the dbscan and
#' FNN packages even on one thread. The default `"auto"` takes the tree when
#' the metric allows it and `ncol(x)` is at most 8, and the scan otherwise.
#' The tree is built in parallel, so on a large low-dimensional set the
#' build is a small part of the total. The tree path holds one extra copy
#' of `x`, its rows reordered so that each leaf is contiguous in memory.
#'
#' The conventions follow `dbscan::kNN()`, so code written for its results
#' works on these. Without `query`, each row is excluded from its own
#' neighbours. With `query`, every row of `x` is a candidate, so a query
#' identical to a data row finds that row at distance zero. Neighbours are
#' sorted by distance, and equal distances by row index, so the result is
#' fully determined by the input. Rows with missing or non-finite values are
#' an error rather than being dropped as [shoal_dist()] does: dropping rows
#' would renumber the indices so they no longer point into the caller's
#' matrix.
#'
#' The `plot()` method draws each point's distance to its `k`-th neighbour in
#' increasing order. The knee of that curve is the usual choice of `eps` for
#' [shoal_dbscan()] with `min_samples = k + 1`; the point itself counts
#' towards `min_samples`, hence the one.
#'
#' @inheritParams shoal_dist
#' @param x A numeric matrix or data frame of the points to search among.
#'   Data frames are coerced to a matrix using their numeric columns
#'   (non-numeric columns are dropped). Rows containing missing or non-finite
#'   values are an error.
#' @param k Number of neighbours to find. Must be less than `nrow(x)` when
#'   `query` is `NULL`, and at most `nrow(x)` otherwise.
#' @param query Optional numeric matrix or data frame of points to find
#'   neighbours for, with the same columns as `x`. `NULL` (default) searches
#'   the rows of `x` against one another.
#' @param cov Covariance matrix for `metric = "mahalanobis"`, `ncol(x)` square
#'   and positive definite. `NULL` (default) uses [stats::cov()] of `x`, the
#'   reference set, whichever of `x` and `query` is being measured. Ignored
#'   for other metrics.
#' @param search Search algorithm: `"auto"` (default), `"kdtree"` or
#'   `"brute"`. See Details. `"kdtree"` is an error for a metric the tree
#'   cannot bound.
#'
#' @returns An object of class `"shoal_knn"`: a list with components `id`, an
#'   integer matrix of row indices into `x`, and `dist`, a numeric matrix of
#'   the corresponding distances. Both have one row per point searched for
#'   and `k` columns, nearest first, with the row names of `x` or `query` and
#'   column names `1` to `k`. `k`, `metric` and `search` record the call,
#'   `search` being the algorithm actually used.
#'
#' @seealso [shoal_dist()] for the full distance matrix, and
#'   [shoal_dbscan()], whose `eps` the `plot()` method helps to choose.
#'
#' @examples
#' x <- as.matrix(iris[, 1:4])
#' nn <- shoal_knn(x, k = 5L)
#' nn
#' head(nn$id)
#'
#' # The distance to the fifth neighbour, sorted: read eps off the knee.
#' plot(nn)
#'
#' # Neighbours of new points among the rows of x, by cosine distance.
#' shoal_knn(x, k = 3L, query = x[c(1, 51, 101), ], metric = "cosine")
#'
#' @export
shoal_knn <- function(x, k, query = NULL,
                      metric = c(
                        "euclidean", "maximum", "manhattan", "canberra",
                        "binary", "minkowski", "cosine", "correlation",
                        "mahalanobis"
                      ),
                      p = 2, cov = NULL,
                      search = c("auto", "kdtree", "brute")) {
  x <- check_numeric_matrix(x, na_action = "error")
  check_positive_integer(k)
  k <- as.integer(k)
  metric <- rlang::arg_match(metric)
  check_positive_number(p)
  search <- rlang::arg_match(search)

  tree_ok <- !metric %in% c("canberra", "binary")
  if (identical(search, "kdtree") && !tree_ok) {
    cli::cli_abort(c(
      "{.code search = \"kdtree\"} does not support {.val {metric}}.",
      "i" = "No rectangle bounds that distance; use {.code search = \"brute\"}."
    ))
  }
  if (identical(search, "auto")) {
    search <- if (tree_ok && ncol(x) <= kd_tree_max_dims) "kdtree" else "brute"
  }

  if (!is.null(query)) {
    query <- check_numeric_matrix(query, na_action = "error")
    if (ncol(query) != ncol(x)) {
      cli::cli_abort(c(
        "{.arg query} must have the same columns as {.arg x}.",
        "i" = "{.arg x} has {ncol(x)} column{?s} and {.arg query} has {ncol(query)}."
      ))
    }
  }

  # Without a query each row is its own excluded candidate, so one fewer
  # neighbour is available.
  available <- nrow(x) - if (is.null(query)) 1L else 0L
  if (k > available) {
    cli::cli_abort(c(
      "{.arg k} must be at most {available} for {nrow(x)} row{?s} of {.arg x}.",
      "i" = if (is.null(query)) "A row is not its own neighbour." else NULL
    ))
  }

  if (identical(metric, "mahalanobis")) {
    ch <- whitening_factor(x, cov)
    x <- apply_whitening(x, ch)
    if (!is.null(query)) {
      query <- apply_whitening(query, ch)
    }
    res <- rust_knn(x, query, k, "euclidean", 2, search)
  } else {
    res <- rust_knn(x, query, k, metric, as.double(p), search)
  }

  # Checked as the distances were computed: no second pass over the result.
  if (!res$finite) {
    cli::cli_abort(c(
      "Distance computation produced non-finite values.",
      "i" = "{.val {metric}} is undefined for zero-variance or all-zero rows."
    ))
  }

  dn <- list(rownames(query %||% x), seq_len(k))
  dimnames(res$id) <- dn
  dimnames(res$dist) <- dn

  structure(
    list(id = res$id, dist = res$dist, k = k, metric = metric, search = search),
    class = "shoal_knn"
  )
}

#' Widest data the kd-tree is chosen for by `search = "auto"`
#'
#' Measured on 20,000 Gaussian points, k = 10. Seconds, tree against scan:
#' 2 dims 0.03 vs 1.43 on one thread; 8 dims 1.41 vs 1.86 (0.13 vs 0.22 on
#' 20 threads); 10 dims 3.20 vs 2.16 (0.30 vs 0.30); 16 dims 11.6 vs 2.7.
#' Cosine crosses at the same point: 8 dims 1.20 vs 2.70, 10 dims 3.11 vs
#' 3.05. Pruning weakens as dimension grows while the scan's per-pair cost
#' only grows linearly, so a fixed crossover in dimension is a fair rule,
#' and the odd wrong call near it costs little either way.
#'
#' @noRd
kd_tree_max_dims <- 8L

#' @rdname shoal_knn
#' @param ... For `plot()`, further arguments to [plot.default()]; `main`,
#'   `xlab` and `ylab` given here replace the defaults. Ignored by `print()`.
#' @export
print.shoal_knn <- function(x, ...) {
  kth <- x$dist[, x$k]
  cli::cli_h3("k-Nearest Neighbours")
  cli::cli_text("Metric: {.val {x$metric}}, Search: {.val {x$search}}")
  cli::cli_text("Points: {nrow(x$id)}, Neighbours: {x$k}")
  cli::cli_text(
    "Distance to neighbour {x$k}: min {signif(min(kth), 4)}, ",
    "median {signif(stats::median(kth), 4)}, max {signif(max(kth), 4)}"
  )
  invisible(x)
}

#' @rdname shoal_knn
#' @param which Which neighbour's distance to plot, between `1` and `x$k`.
#'   Default `x$k`.
#' @export
plot.shoal_knn <- function(x, which = x$k, ...) {
  check_positive_integer(which)
  if (which > x$k) {
    cli::cli_abort("{.arg which} must be at most {x$k}, the number of neighbours found.")
  }
  d <- sort(x$dist[, which])
  args <- list(...)
  defaults <- list(
    main = "k-Nearest Neighbour Distance",
    xlab = "Points sorted by distance",
    ylab = sprintf("Distance to neighbour %d (%s)", which, x$metric),
    type = "l"
  )
  args <- c(args, defaults[setdiff(names(defaults), names(args))])
  do.call(graphics::plot, c(list(x = seq_along(d), y = d), args))
  invisible(x)
}
