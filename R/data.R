#' Concentric rings with noise
#'
#' Three concentric rings of 300, 400 and 500 points with radii 0.5, 1.2 and
#' 2.0, plus 60 uniformly scattered noise points. A standard case where
#' density-based methods succeed and centroid-based ones cannot.
#'
#' @format A numeric matrix with 1260 rows and columns `x` and `y`. No labels
#'   are included; the generating code is in `data-raw/rings.R`.
#' @source Simulated; see `data-raw/rings.R`.
#' @examples
#' res <- shoal_hdbscan(rings, min_cluster_size = 15L, min_samples = 5L)
#' res
"rings"

#' Sentence embeddings of newsgroup posts
#'
#' 2,400 posts from eight groups of the 20 Newsgroups corpus, 300 per group,
#' embedded with the `all-MiniLM-L6-v2` sentence-transformer. Real embedding
#' vectors, for trying [shoal_evoc()] and the other algorithms on the kind of
#' input EVoC is built for.
#'
#' The model's 384-dimensional output was reduced to its first 64 uncentred
#' singular directions, with rows re-normalised to unit length, to keep the
#' package small. Uncentred so that direction, and hence cosine similarity, is
#' preserved up to the truncation; 62 percent of the squared norm is retained,
#' and clusterings of the reduced and full vectors agree closely.
#'
#' @format A list with three components, aligned by position:
#' \describe{
#'   \item{embedding}{A 2400 by 64 numeric matrix of unit-length rows.}
#'   \item{group}{A factor with eight levels: the newsgroup each post came
#'     from, e.g. `"sci.space"` or `"rec.autos"`.}
#'   \item{snippet}{The first 120 characters of each post, whitespace
#'     collapsed, for seeing what a cluster contains.}
#' }
#' @source The 20 Newsgroups corpus as distributed by scikit-learn, with
#'   headers, footers and quoted text removed. Generating code is in
#'   `data-raw/newsgroups.py` and `data-raw/newsgroups.R`.
#' @examples
#' fit <- shoal_evoc(newsgroups$embedding, min_cluster_size = 15L)
#' table(fit$cluster, newsgroups$group, useNA = "ifany")
"newsgroups"
