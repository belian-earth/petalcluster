#' @title Density-Based Clustering via Rust
#'
#' @description
#' R bindings to the
#' \href{https://github.com/petabi/petal-clustering}{petal-clustering} Rust
#' crate by \href{https://github.com/petabi}{Petabi, Inc.}, providing fast
#' implementations of three density-based clustering algorithms:
#'
#' - [petal_dbscan()] — DBSCAN: finds clusters as dense regions separated by
#'   areas of lower density.
#' - [petal_hdbscan()] — HDBSCAN: hierarchical extension of DBSCAN that adapts
#'   to clusters of varying density.
#' - [petal_optics()] — OPTICS: produces a reachability ordering that captures
#'   cluster structure at multiple density thresholds.
#'
#' All three accept numeric matrices or data frames, support Euclidean and
#' cosine distance metrics, and return S3 objects with `print()` and `plot()`
#' methods.
#'
#' @keywords internal
#' @aliases petalcluster-package
"_PACKAGE"

#' @useDynLib petalcluster, .registration = TRUE
NULL
