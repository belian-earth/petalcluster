## Package the newsgroups embeddings produced by data-raw/newsgroups.py.
##
## The 384-dimensional embeddings would be 2.6 MB compressed, so the first 64
## uncentred singular directions are kept and rows re-normalised to unit
## length. Uncentred, so that direction (and hence cosine similarity) is
## preserved up to the truncation; this keeps 62% of the squared norm and
## clusters indistinguishably from the full vectors in practice.

raw <- as.matrix(read.csv("data-raw/newsgroups_embedding.csv", header = FALSE))
group <- readLines("data-raw/newsgroups_group.txt")
snippet <- read.csv("data-raw/newsgroups_snippet.csv", header = FALSE,
                    stringsAsFactors = FALSE)[[1]]
stopifnot(nrow(raw) == length(group), nrow(raw) == length(snippet))

sv <- svd(raw)
k <- 64L
embedding <- sv$u[, seq_len(k)] %*% diag(sv$d[seq_len(k)])
embedding <- embedding / sqrt(rowSums(embedding^2))
embedding <- signif(embedding, 5L)
dimnames(embedding) <- NULL

newsgroups <- list(
  embedding = embedding,
  group = factor(group),
  snippet = snippet
)
usethis::use_data(newsgroups, overwrite = TRUE, compress = "xz")
