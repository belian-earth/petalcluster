#' Hierarchical Agglomerative Clustering
#'
#' Agglomerative clustering via the Rust
#' \href{https://github.com/diffeo/kodama}{kodama} crate, a port of
#' \emph{fastcluster}. Returns a standard [stats::hclust] object, so
#' [stats::cutree()], [stats::as.dendrogram()], `plot()` and the rest of R's
#' hierarchical clustering ecosystem work on the result unchanged.
#'
#' # Linkage methods
#'
#' Dissimilarities are squared internally for `"ward"`, `"centroid"` and
#' `"median"`, with the square root taken afterwards, the fastcluster and SciPy
#' convention. Two consequences differ from [stats::hclust()]:
#'
#' - `"ward"` corresponds to R's `"ward.D2"`, not `"ward.D"`.
#' - `"centroid"` and `"median"` take plain distances here, whereas
#'   [stats::hclust()] expects them to be squared beforehand.
#'
#' `"weighted"` is R's `"mcquitty"` (WPGMA); `"average"` is UPGMA.
#'
#' `"centroid"` and `"median"` can produce inversions, a merge at a lower
#' height than the one before it. This is a property of the methods, not a bug,
#' but [stats::cutree()] rejects such trees, so a warning is issued when it
#' happens.
#'
#' @param d A [stats::dist] object, or a numeric matrix or data frame, in which
#'   case Euclidean distances are computed with [shoal_dist()] first.
#' @param method Linkage method. One of `"complete"`, `"single"`, `"average"`,
#'   `"weighted"`, `"ward"`, `"centroid"` or `"median"`.
#'
#' @returns An object of class `"hclust"` with components `merge`, `height`,
#'   `order`, `labels`, `method`, `call` and `dist.method`.
#'
#' @seealso [shoal_dist()] for building the distance matrix.
#'
#' @examples
#' d <- shoal_dist(as.matrix(iris[, 1:4]))
#' fit <- shoal_hclust(d, method = "ward")
#' cutree(fit, k = 3)
#'
#' @export
shoal_hclust <- function(d,
                         method = c(
                           "complete", "single", "average", "weighted",
                           "ward", "centroid", "median"
                         )) {
  method <- rlang::arg_match(method)

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
  if (is.null(n) || n < 2L) {
    cli::cli_abort("{.arg d} must describe at least 2 observations.")
  }

  # In double: the integer product overflows past 46,341 observations.
  expected <- as.double(n) * (n - 1) / 2
  if (length(d) != expected) {
    cli::cli_abort(
      "{.arg d} has {length(d)} dissimilarit{?y/ies}, expected {expected} for {n} observations."
    )
  }
  # NA is screened here, cheaply; Inf is caught on the Rust side in the same
  # pass that copies `d`, with the same message. A separate is.finite() pass
  # would allocate a logical vector as long as `d`.
  if (anyNA(d)) {
    cli::cli_abort("{.arg d} must not contain missing or non-finite values.")
  }
  if (!is.double(d)) {
    d <- as.double(d)
  }

  # The dist object goes across as is; the copy kodama needs is made once
  # on the Rust side rather than here as well.
  res <- rust_hclust(d, as.integer(n), method)

  if (is.unsorted(res$height)) {
    cli::cli_warn(c(
      "The dendrogram contains inversions.",
      "i" = "{.val {method}} linkage does not guarantee monotone merge heights.",
      "i" = "{.fn cutree} will reject this tree."
    ))
  }

  structure(
    list(
      merge = res$merge,
      height = res$height,
      order = res$order,
      labels = attr(d, "Labels"),
      method = method,
      call = match.call(),
      dist.method = attr(d, "method")
    ),
    class = "hclust"
  )
}
