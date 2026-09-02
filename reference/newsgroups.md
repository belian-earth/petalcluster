# Sentence embeddings of newsgroup posts

2,400 posts from eight groups of the 20 Newsgroups corpus, 300 per
group, embedded with the `all-MiniLM-L6-v2` sentence-transformer. Real
embedding vectors, for trying
[`shoal_evoc()`](https://belian-earth.github.io/shoal/reference/shoal_evoc.md)
and the other algorithms on the kind of input EVoC is built for.

## Usage

``` r
newsgroups
```

## Format

A list with three components, aligned by position:

- embedding:

  A 2400 by 64 numeric matrix of unit-length rows.

- group:

  A factor with eight levels: the newsgroup each post came from, e.g.
  `"sci.space"` or `"rec.autos"`.

- snippet:

  The first 120 characters of each post, whitespace collapsed, for
  seeing what a cluster contains.

## Source

The 20 Newsgroups corpus as distributed by scikit-learn, with headers,
footers and quoted text removed. Generating code is in
`data-raw/newsgroups.py` and `data-raw/newsgroups.R`.

## Details

The model's 384-dimensional output was reduced to its first 64 uncentred
singular directions, with rows re-normalised to unit length, to keep the
package small. Uncentred so that direction, and hence cosine similarity,
is preserved up to the truncation; 62 percent of the squared norm is
retained, and clusterings of the reduced and full vectors agree closely.

## Examples

``` r
fit <- shoal_evoc(newsgroups$embedding, min_cluster_size = 15L)
table(fit$cluster, newsgroups$group, useNA = "ifany")
#>       
#>        comp.graphics misc.forsale rec.autos rec.sport.hockey sci.med sci.space
#>   1                0            0         0                0       0         0
#>   2                0            0         0               81       0         0
#>   3                0            0         0               66       0         0
#>   4               10          235         3                1       0         1
#>   5              261            6         0                0       1         4
#>   6                1            0         0                1     265        12
#>   7                0            0         0                0       1         0
#>   8                0            0         1                0       0         0
#>   9                1           24       267                0       1         3
#>   10               0            0         0                0       0        78
#>   11               0            0         0                0       0        68
#>   <NA>            27           35        29              151      32       134
#>       
#>        soc.religion.christian talk.politics.mideast
#>   1                         1                    87
#>   2                         0                     0
#>   3                         0                     0
#>   4                         0                     1
#>   5                         0                     0
#>   6                         7                     0
#>   7                       266                     1
#>   8                         6                   182
#>   9                         1                     0
#>   10                        0                     1
#>   11                        0                     0
#>   <NA>                     19                    28
```
