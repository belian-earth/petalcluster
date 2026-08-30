#' @title High-Performance Clustering via Rust
#'
#' @description
#' A small, deliberately foundational set of clustering algorithms with Rust
#' backends, presented behind one consistent interface. Every result carries the
#' shared `"shoal_clustering"` class, so `print()` and `plot()` behave the same
#' way whichever algorithm produced it.
#'
#' Currently available:
#'
#' - [shoal_dbscan()] — DBSCAN: finds clusters as dense regions separated by
#'   areas of lower density.
#' - [shoal_hdbscan()] — HDBSCAN: hierarchical extension of DBSCAN that adapts
#'   to clusters of varying density.
#'
#' Both accept numeric matrices or data frames, support Euclidean and cosine
#' distance metrics, and return S3 objects with `print()` and `plot()` methods.
#'
#' The density-based algorithms are bindings to the
#' \href{https://github.com/petabi/petal-clustering}{petal-clustering} Rust
#' crate by \href{https://github.com/petabi}{Petabi, Inc.}
#'
#' @keywords internal
#' @aliases shoal-package
"_PACKAGE"

#' @useDynLib shoal, .registration = TRUE
#' @importFrom rlang %||%
NULL
