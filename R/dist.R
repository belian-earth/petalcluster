#' Pairwise Distance Matrix
#'
#' Computes a pairwise distance matrix in Rust and returns a standard
#' [stats::dist] object, so the result works anywhere a `dist` does:
#' [shoal_hclust()], [stats::cmdscale()], [stats::as.dendrogram()],
#' `cluster::pam()` and so on.
#'
#' Metrics shared with [stats::dist()] follow its definitions exactly, including
#' the way `"canberra"` drops and rescales degenerate terms and the way
#' `"binary"` treats non-zero entries as "on". Note that `"canberra"` divides
#' by `|x| + |y|`, which is what [stats::dist()] computes; its documentation
#' writes `|x + y|`, and the two differ on signed data. `"cosine"` matches the
#' `metric = "cosine"` option on [shoal_dbscan()] and [shoal_hdbscan()].
#'
#' `"mahalanobis"` is the Euclidean distance after the columns have been
#' decorrelated and scaled by a covariance matrix, by default the sample
#' covariance of `x`: features on different scales or correlated with one
#' another then count once rather than several times. It is computed by
#' whitening `x` with the Cholesky factor of the covariance and taking
#' Euclidean distances, so it costs the same as `"euclidean"` plus one
#' `p x p` factorisation. Pass `cov` to measure against a reference
#' covariance rather than the sample's own.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped). Rows
#'   containing missing or non-finite values are removed.
#' @param metric Distance metric. One of `"euclidean"`, `"maximum"`,
#'   `"manhattan"`, `"canberra"`, `"binary"`, `"minkowski"`, `"cosine"`,
#'   `"correlation"` or `"mahalanobis"`.
#' @param p Power for `metric = "minkowski"`. Ignored otherwise. Default `2`.
#' @param cov Covariance matrix for `metric = "mahalanobis"`, `ncol(x)` square
#'   and positive definite. `NULL` (default) uses [stats::cov()] of `x`, which
#'   needs more rows than columns. Ignored for other metrics.
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
#' # Mahalanobis: petal length and width are strongly correlated in iris, so
#' # their shared variation counts once.
#' m <- shoal_dist(as.matrix(iris[, 1:4]), metric = "mahalanobis")
#'
#' @export
shoal_dist <- function(x,
                       metric = c(
                         "euclidean", "maximum", "manhattan", "canberra",
                         "binary", "minkowski", "cosine", "correlation",
                         "mahalanobis"
                       ),
                       p = 2, cov = NULL) {
  x <- check_numeric_matrix(x)
  metric <- rlang::arg_match(metric)
  check_positive_number(p)

  if (nrow(x) < 2L) {
    cli::cli_abort("{.arg x} must have at least 2 rows to compute distances.")
  }

  if (identical(metric, "mahalanobis")) {
    values <- rust_dist(whiten(x, cov), "euclidean", 2)
  } else {
    values <- rust_dist(x, metric, as.double(p))
  }

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

#' Whiten `x` so that Euclidean distances become Mahalanobis distances
#'
#' With `S = R'R` from the Cholesky factor `R`, the Mahalanobis distance
#' `sqrt((u - v)' S^-1 (u - v))` equals the Euclidean distance between
#' `R^-T u` and `R^-T v`. Transforming every row once is `O(n p^2)`, against
#' the `O(n^2 p)` of the distances themselves.
#'
#' @noRd
whiten <- function(x, cov, call = rlang::caller_env()) {
  p <- ncol(x)
  if (is.null(cov)) {
    if (nrow(x) <= p) {
      cli::cli_abort(c(
        "{.arg x} needs more rows than columns for its sample covariance to be invertible.",
        "i" = "It has {nrow(x)} row{?s} and {p} column{?s}. Supply {.arg cov} instead."
      ), call = call)
    }
    cov <- stats::cov(x)
  } else {
    if (!is.matrix(cov) || !is.numeric(cov) || !identical(dim(cov), c(p, p))) {
      cli::cli_abort(
        "{.arg cov} must be a {p} x {p} numeric matrix to match the columns of {.arg x}.",
        call = call
      )
    }
    if (!isSymmetric(unname(cov))) {
      cli::cli_abort("{.arg cov} must be symmetric.", call = call)
    }
  }

  ch <- tryCatch(chol(cov), error = function(e) {
    cli::cli_abort(c(
      "The covariance matrix is not positive definite.",
      "i" = "A constant or collinear column makes it singular; drop it, or supply a regularised {.arg cov}."
    ), call = call)
  })

  # Row i becomes R^-T x_i; solving against the transpose avoids an inverse.
  z <- t(backsolve(ch, t(x), transpose = TRUE))
  dimnames(z) <- dimnames(x)
  z
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

#' Subset a `dist` object to a set of kept observations
#'
#' Works directly on the condensed vector via R's own indexing formula, so it
#' never materialises the full `n x n` matrix, which matters at exactly the
#' sizes where distance matrices are already straining memory.
#'
#' @param d A `dist` object.
#' @param keep Logical vector, one entry per observation.
#'
#' @noRd
subset_dist <- function(d, keep) {
  n <- attr(d, "Size")
  pos <- which(keep)
  m <- length(pos)

  if (m < 2L) {
    return(new_dist(double(0L), n = m,
                    labels = attr(d, "Labels")[keep],
                    method = attr(d, "method")))
  }

  # Every kept pair (a < b), enumerated in dist's column-major order.
  a <- rep(seq_len(m - 1L), times = (m - 1L):1L)
  b <- sequence((m - 1L):1L) + a

  old_index <- dist_index(n, pos[a], pos[b])

  new_dist(
    as.double(d)[old_index],
    n = m,
    labels = attr(d, "Labels")[keep],
    method = attr(d, "method")
  )
}

#' R's condensed `dist` index of the pair `i < j` among `n` observations
#'
#' Computed in double: the products overflow R's integers past 46,341
#' observations, well within the sizes a distance matrix can reach.
#' @noRd
dist_index <- function(n, i, j) {
  n <- as.double(n)
  i <- as.double(i)
  j <- as.double(j)
  n * (i - 1) - i * (i - 1) / 2 + j - i
}

#' Does a matrix look like a square distance matrix rather than data?
#'
#' Symmetric, zero diagonal, non-negative: essentially nothing but a distance
#' matrix satisfies all three, and treating one as raw data would silently
#' compute distances between rows of distances.
#'
#' @noRd
looks_like_dist_matrix <- function(x) {
  is.matrix(x) &&
    is.numeric(x) &&
    nrow(x) == ncol(x) &&
    nrow(x) > 2L &&
    !anyNA(x) &&
    all(x >= 0) &&
    all(diag(x) == 0) &&
    isSymmetric(unname(x))
}
