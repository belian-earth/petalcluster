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
#' - [shoal_kmeans()] — k-means: partitions into a fixed number of clusters, and
#'   can assign new observations to them.
#' - [shoal_gmm()] — Gaussian mixtures: soft, probabilistic assignment with
#'   elliptical clusters, and `AIC()`/`BIC()` for choosing the component count.
#' - [shoal_dbscan()] — DBSCAN: finds clusters as dense regions separated by
#'   areas of lower density.
#' - [shoal_hdbscan()] — HDBSCAN: hierarchical extension of DBSCAN that adapts
#'   to clusters of varying density.
#' - [shoal_evoc()] — EVoC: direct multi-granularity clustering of embedding
#'   vectors, returning every cluster layer rather than one flat partition.
#' - [shoal_hclust()] — agglomerative hierarchical clustering with seven linkage
#'   methods, returning a standard [stats::hclust] object.
#' - [shoal_dist()] — pairwise distance matrices, returning a standard
#'   [stats::dist] object.
#' - [shoal_silhouette()] and [shoal_metrics()] — validity measures for choosing
#'   the number of clusters.
#'
#' k-means and the density-based algorithms take numeric matrices or data frames
#' directly.
#' Hierarchical clustering works from a distance matrix, so it accepts anything
#' [stats::dist()] would produce as well as raw data.
#'
#' Returning R's own `dist` and `hclust` classes rather than bespoke ones is
#' deliberate: `cutree()`, `as.dendrogram()`, `cmdscale()` and the rest of the
#' ecosystem then work on the results without any glue.
#'
#' The density-based algorithms are bindings to the
#' \href{https://github.com/petabi/petal-clustering}{petal-clustering} Rust
#' crate by \href{https://github.com/petabi}{Petabi, Inc.}; hierarchical
#' clustering is bound to \href{https://github.com/diffeo/kodama}{kodama}, a
#' Rust port of \emph{fastcluster}; k-means and Gaussian mixtures to
#' \href{https://github.com/rust-ml/linfa}{linfa}; and EVoC to an in-tree
#' Rust port of the
#' \href{https://github.com/TutteInstitute/evoc}{reference implementation},
#' validated against fixtures generated from it.
#'
#' @keywords internal
#' @aliases shoal-package
"_PACKAGE"

#' @useDynLib shoal, .registration = TRUE
#' @importFrom rlang %||%
NULL
