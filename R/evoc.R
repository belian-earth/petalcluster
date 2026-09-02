#' EVoC: Embedding Vector Oriented Clustering
#'
#' Direct multi-granularity clustering of embedding vectors, via an in-tree
#' Rust port of McInnes's
#' \href{https://github.com/TutteInstitute/evoc}{EVoC} (Tutte Institute).
#' Instead of clustering in the original space, EVoC builds a nearest-neighbour
#' graph under cosine distance, learns a compact node embedding from it, and
#' density-clusters that embedding at several granularities at once. On
#' collections of pre-built embeddings it is orders of magnitude faster than
#' running HDBSCAN on the raw vectors, at comparable quality.
#'
#' # Domain
#'
#' EVoC assumes cosine geometry: rows are L2-normalised internally and treated
#' as directions, which is the right model for text/image embedding vectors and
#' the wrong one for general tabular data. It also wants scale: behaviour is
#' calibrated for thousands to millions of rows. On small or low-dimensional
#' data it over-fragments and marks much of the input as noise (the reference
#' implementation behaves the same way); reach for [shoal_hdbscan()] there.
#'
#' # Layers
#'
#' A single flat clustering hides a genuine modelling choice, and the upstream
#' heuristic for making it (most persistent layer) is not reliable on every
#' shape of data. Every layer is therefore returned, finest first, along with
#' its persistence score and per-point membership strengths; `layer` only
#' chooses which one populates `cluster` for printing, plotting and the
#' single-partition helpers. Pick a different layer afterwards by indexing
#' `layers` directly; the fit does not need to be rerun.
#'
#' # Reproducibility
#'
#' EVoC is stochastic (neighbour search, graph partitioning and the embedding
#' all draw random numbers), so `seed` is a parameter rather than being taken
#' from R's RNG, as in [shoal_kmeans()]. The same seed, parameters and data
#' give bitwise-identical results regardless of thread count, a stronger
#' guarantee than the reference implementation's, whose parallel stages can
#' race. Expect different seeds to give somewhat different clusterings; on
#' data EVoC suits, the structure they agree on is real.
#'
#' @param x A numeric matrix or data frame of embedding vectors, one per row.
#'   Data frames are coerced to a matrix using their numeric columns
#'   (non-numeric columns are dropped). Must have more rows than `n_neighbors`.
#' @param n_neighbors Neighbourhood size for the nearest-neighbour graph.
#'   Default `15L`.
#' @param noise_level Tolerance for spreading points out in the learned
#'   embedding; higher values let clusters absorb more of their surroundings.
#'   Default `0.5`.
#' @param min_cluster_size Minimum cluster size at the finest layer; at
#'   least 2. Default `5L`, matching upstream, which is calibrated for large inputs;
#'   on collections of a few thousand rows or fewer it tends to
#'   over-fragment, and something like `15L` recovers the structure far more
#'   reliably.
#' @param min_samples Minimum neighbourhood size for the density estimation.
#'   Default `5L`.
#' @param n_epochs Training epochs for the node embedding. Default `50L`.
#' @param dim Dimension of the learned node embedding. `NULL` (default) uses
#'   the upstream rule `min(max(n_neighbors / 4, 4), 15)`.
#' @param max_layers Maximum number of cluster layers to return. Default `10L`.
#' @param layer Which layer populates `cluster`: `"auto"` (default) selects the
#'   layer with the highest persistence score, matching upstream behaviour; an
#'   integer selects that layer directly (1 is the finest).
#' @param seed Non-negative whole-number seed. Stored and passed as a double,
#'   so values beyond the integer range are safe. Default `1L`.
#'
#' @returns An object of class `c("shoal_evoc", "shoal_clustering")`: a list
#'   with components `cluster` (integer vector for the selected layer, `NA`
#'   for noise), `n_clusters`, `n_noise`, `data`, `algorithm`, `params`,
#'   `metric` (always `"cosine"`), and the multi-layer results: `layers`
#'   (list of integer cluster vectors, finest first), `strengths` (list of
#'   numeric membership-strength vectors, aligned with `layers`),
#'   `persistence` (numeric persistence score per layer), `layer` (index of
#'   the selected layer) and `embedding` (the learned node-embedding matrix).
#'
#' @references
#' McInnes, L. (2023). *EVoC: Embedding Vector Oriented Clustering*.
#' <https://github.com/TutteInstitute/evoc>
#'
#' @examples
#' # Embedding-like data: directions with angular spread, not raw tabular data.
#' set.seed(1)
#' centres <- matrix(runif(6 * 48, -1, 1) * 0.6, nrow = 6)
#' x <- centres[rep(1:6, each = 140L), ] + matrix(rnorm(840 * 48, sd = 0.1), ncol = 48)
#'
#' fit <- shoal_evoc(x, min_cluster_size = 15L)
#' fit
#' # every granularity remains available:
#' vapply(fit$layers, function(l) length(unique(l[!is.na(l)])), integer(1))
#'
#' @export
shoal_evoc <- function(x,
                       n_neighbors = 15L,
                       noise_level = 0.5,
                       min_cluster_size = 5L,
                       min_samples = 5L,
                       n_epochs = 50L,
                       dim = NULL,
                       max_layers = 10L,
                       layer = "auto",
                       seed = 1L) {
  x <- check_numeric_matrix(x)
  check_positive_integer(n_neighbors)
  check_positive_number(noise_level)
  check_positive_integer(min_cluster_size)
  if (min_cluster_size < 2L) {
    # The condensed tree treats a size-1 cluster as a point, and the reference
    # implementation mislabels its sibling in that case; 1 is never meaningful.
    cli::cli_abort("{.arg min_cluster_size} must be at least 2.")
  }
  check_positive_integer(min_samples)
  check_positive_integer(n_epochs)
  if (!is.null(dim)) {
    check_positive_integer(dim)
  }
  check_positive_integer(max_layers)
  check_count(seed)

  # The port works in single precision; anything `as.single()` cannot hold
  # would become Inf, then NaN once rows are recentred, deep in the pipeline.
  if (max(abs(x)) > 3.4028235e38) {
    cli::cli_abort(c(
      "{.arg x} contains values too large for single precision.",
      "i" = "EVoC uses cosine geometry, so rescaling {.arg x} does not change the result."
    ))
  }

  if (nrow(x) <= n_neighbors) {
    cli::cli_abort(
      "{.arg x} must have more rows ({nrow(x)}) than {.arg n_neighbors} ({n_neighbors})."
    )
  }

  auto_layer <- rlang::is_string(layer) && identical(layer, "auto")
  if (!auto_layer && !rlang::is_scalar_integerish(layer)) {
    cli::cli_abort(
      "{.arg layer} must be {.val auto} or a single layer index."
    )
  }

  result <- rust_evoc(
    x,
    as.integer(n_neighbors),
    as.double(noise_level),
    as.integer(min_cluster_size),
    as.integer(min_samples),
    as.integer(n_epochs),
    as.integer(dim %||% 0L),
    0.2, # min_similarity_threshold, the upstream default for layer diversity
    as.integer(max_layers),
    20L, # n_label_prop_iter, the upstream default
    as.double(seed)
  )

  n_layers <- length(result$layers)
  if (auto_layer) {
    # Upstream behaviour: the most persistent layer wins (the base layer
    # carries persistence 0, so a lone base layer selects itself).
    selected <- which.max(result$persistence)
  } else {
    selected <- as.integer(layer)
    if (selected < 1L || selected > n_layers) {
      cli::cli_abort(
        "{.arg layer} is {selected}, but this fit has {n_layers} layer{?s}."
      )
    }
  }

  new_clustering(
    cluster = result$layers[[selected]],
    data = x,
    algorithm = "EVoC",
    subclass = "shoal_evoc",
    params = list(
      n_neighbors = as.integer(n_neighbors),
      noise_level = noise_level,
      min_cluster_size = as.integer(min_cluster_size),
      min_samples = as.integer(min_samples),
      n_epochs = as.integer(n_epochs),
      # Kept as double: a whole-number seed can exceed the integer range.
      seed = as.numeric(seed)
    ),
    metric = "cosine",
    layers = lapply(result$layers, as.integer),
    strengths = result$strengths,
    persistence = as.numeric(result$persistence),
    layer = selected,
    embedding = result$embedding
  )
}
