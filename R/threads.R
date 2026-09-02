#' Threads used by the Rust backends
#'
#' The parallel parts of the package run on a thread pool owned by it.
#' `shoal_threads(n)` rebuilds that pool with `n` threads; `shoal_threads()`
#' reports the current size. The setting takes effect immediately and lasts
#' for the session.
#'
#' Parallel work covers [shoal_evoc()]'s neighbour search, spanning tree and
#' node embedding; [shoal_dbscan()]'s neighbour queries and
#' [shoal_hdbscan()]'s core distances and spanning tree; [shoal_dist()]; and
#' [shoal_silhouette()]. k-means, Gaussian mixtures and hierarchical
#' clustering are single-threaded. Results never depend on the thread count.
#'
#' # Default
#'
#' On load the pool takes, in order of precedence, the `shoal.threads` option,
#' the `RAYON_NUM_THREADS` environment variable, or one thread per logical
#' core. When `_R_CHECK_LIMIT_CORES_` is set, as it is by
#' `R CMD check --as-cran`, the automatic default is capped at 2 to respect
#' the CRAN policy for checks; an explicit option or variable is still
#' honoured.
#'
#' @param n Number of threads, a positive whole number. Omit to query.
#'
#' @returns The number of threads in the pool, invisibly when setting.
#'
#' @examples
#' old <- shoal_threads()
#' shoal_threads(2)
#' shoal_threads()
#' shoal_threads(old)
#'
#' @export
shoal_threads <- function(n) {
  if (missing(n)) {
    return(rust_get_threads())
  }
  check_positive_integer(n)
  rust_set_threads(as.integer(n))
  invisible(as.integer(n))
}

#' The thread count to start the pool with
#' @noRd
default_threads <- function() {
  opt <- getOption("shoal.threads")
  if (!is.null(opt) && rlang::is_scalar_integerish(opt) && opt >= 1) {
    return(as.integer(opt))
  }
  env <- Sys.getenv("RAYON_NUM_THREADS")
  if (nzchar(env) && !is.na(suppressWarnings(as.integer(env))) && as.integer(env) > 0L) {
    return(as.integer(env))
  }
  cores <- max(parallel::detectCores(logical = TRUE), 1L, na.rm = TRUE)
  limit <- Sys.getenv("_R_CHECK_LIMIT_CORES_")
  if (nzchar(limit) && !identical(tolower(limit), "false")) {
    return(min(cores, 2L))
  }
  cores
}

.onLoad <- function(libname, pkgname) {
  # rextendr loads the package to discover Rust functions before it has
  # written their wrappers, so the setter may not exist yet at that moment.
  ns <- topenv(environment())
  if (is.function(get0("rust_set_threads", envir = ns, inherits = FALSE))) {
    rust_set_threads(default_threads())
  }
}
