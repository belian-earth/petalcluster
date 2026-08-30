#' Pairwise Distance Matrix
#'
#' Computes a pairwise distance matrix in Rust and returns a standard
#' [stats::dist] object, so the result works anywhere a `dist` does —
#' [shoal_hclust()], [stats::cmdscale()], [stats::as.dendrogram()],
#' `cluster::pam()` and so on.
#'
#' Metrics shared with [stats::dist()] follow its definitions exactly, including
#' the way `"canberra"` drops and rescales degenerate terms and the way
#' `"binary"` treats non-zero entries as "on". `"cosine"` matches the
#' `metric = "cosine"` option on [shoal_dbscan()] and [shoal_hdbscan()].
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped). Rows
#'   containing missing values are removed.
#' @param metric Distance metric. One of `"euclidean"`, `"maximum"`,
#'   `"manhattan"`, `"canberra"`, `"binary"`, `"minkowski"`, `"cosine"` or
#'   `"correlation"`.
#' @param p Power for `metric = "minkowski"`. Ignored otherwise. Default `2`.
#'
#' @returns An object of class `"dist"`: the lower triangle of the distance
#'   matrix stored column-major, with `Size`, `Labels`, `Diag`, `Upper`,
#'   `method` and `call` attributes.
#'
#' @seealso [shoal_hclust()], which consumes the result.
#'
#' @examples
#' d <- shoal_dist(as.matrix(iris[, 1:4]))
#' d
#'
#' @export
shoal_dist <- function(x,
                       metric = c(
                         "euclidean", "maximum", "manhattan", "canberra",
                         "binary", "minkowski", "cosine", "correlation"
                       ),
                       p = 2) {
  x <- check_numeric_matrix(x)
  metric <- rlang::arg_match(metric)
  check_positive_number(p)

  if (nrow(x) < 2L) {
    cli::cli_abort("{.arg x} must have at least 2 rows to compute distances.")
  }

  values <- rust_dist(x, metric, as.double(p))

  if (anyNA(values) || any(!is.finite(values))) {
    cli::cli_abort(c(
      "Distance computation produced non-finite values.",
      "i" = "{.val {metric}} is undefined for zero-variance or all-zero rows."
    ))
  }

  new_dist(
    values,
    n = nrow(x),
    labels = rownames(x),
    method = metric,
    call = match.call()
  )
}

#' Construct a `dist` object
#'
#' Deliberately plain `class = "dist"` rather than a subclass: a good deal of
#' the code that consumes distance matrices tests `class(x) != "dist"`, which
#' errors outright under R >= 4.2 when the class vector has length two.
#'
#' @param v Lower triangle of the distance matrix, column-major, length
#'   `n * (n - 1) / 2`.
#' @param n Number of observations.
#' @param labels Optional observation labels.
#' @param method Name of the metric used.
#' @param call The call that produced the object.
#'
#' @noRd
new_dist <- function(v, n, labels = NULL, method = "euclidean", call = NULL) {
  structure(
    as.double(v),
    Size = as.integer(n),
    Labels = labels,
    Diag = FALSE,
    Upper = FALSE,
    method = method,
    call = call,
    class = "dist"
  )
}
