# Package index

## Clustering

Clustering algorithms.

- [`shoal_kmeans()`](https://belian-earth.github.io/petalcluster/reference/shoal_kmeans.md)
  : K-Means Clustering
- [`shoal_gmm()`](https://belian-earth.github.io/petalcluster/reference/shoal_gmm.md)
  : Gaussian Mixture Model
- [`shoal_dbscan()`](https://belian-earth.github.io/petalcluster/reference/shoal_dbscan.md)
  : DBSCAN Clustering
- [`shoal_hdbscan()`](https://belian-earth.github.io/petalcluster/reference/shoal_hdbscan.md)
  : HDBSCAN Clustering
- [`shoal_evoc()`](https://belian-earth.github.io/petalcluster/reference/shoal_evoc.md)
  : EVoC: Embedding Vector Oriented Clustering
- [`shoal_hclust()`](https://belian-earth.github.io/petalcluster/reference/shoal_hclust.md)
  : Hierarchical Agglomerative Clustering

## Distances

Pairwise distance matrices.

- [`shoal_dist()`](https://belian-earth.github.io/petalcluster/reference/shoal_dist.md)
  : Pairwise Distance Matrix

## Validation

Assessing a clustering, and choosing the number of clusters.

- [`shoal_silhouette()`](https://belian-earth.github.io/petalcluster/reference/shoal_silhouette.md)
  : Silhouette Widths
- [`shoal_metrics()`](https://belian-earth.github.io/petalcluster/reference/shoal_metrics.md)
  : Internal Cluster Validity Indices

## Methods

Shared print and plot methods, and prediction for centroid models.

- [`plot(`*`<shoal_clustering>`*`)`](https://belian-earth.github.io/petalcluster/reference/plot.shoal.md)
  : Plot clustering results
- [`shoal_palette()`](https://belian-earth.github.io/petalcluster/reference/shoal_palette.md)
  : Default cluster palette
- [`print(`*`<shoal_clustering>`*`)`](https://belian-earth.github.io/petalcluster/reference/print.shoal.md)
  [`print(`*`<shoal_hdbscan>`*`)`](https://belian-earth.github.io/petalcluster/reference/print.shoal.md)
  [`print(`*`<shoal_evoc>`*`)`](https://belian-earth.github.io/petalcluster/reference/print.shoal.md)
  [`print(`*`<shoal_kmeans>`*`)`](https://belian-earth.github.io/petalcluster/reference/print.shoal.md)
  [`print(`*`<shoal_gmm>`*`)`](https://belian-earth.github.io/petalcluster/reference/print.shoal.md)
  : Print a clustering result
- [`predict(`*`<shoal_kmeans>`*`)`](https://belian-earth.github.io/petalcluster/reference/predict.shoal_kmeans.md)
  : Assign New Observations to Fitted Clusters
- [`predict(`*`<shoal_gmm>`*`)`](https://belian-earth.github.io/petalcluster/reference/predict.shoal_gmm.md)
  : Predict Mixture Membership
- [`logLik(`*`<shoal_gmm>`*`)`](https://belian-earth.github.io/petalcluster/reference/logLik.shoal_gmm.md)
  : Log-Likelihood of a Fitted Mixture

## Configuration

Runtime settings.

- [`shoal_threads()`](https://belian-earth.github.io/petalcluster/reference/shoal_threads.md)
  : Threads used by the Rust backends

## Data

Bundled datasets.

- [`rings`](https://belian-earth.github.io/petalcluster/reference/rings.md)
  : Concentric rings with noise
- [`newsgroups`](https://belian-earth.github.io/petalcluster/reference/newsgroups.md)
  : Sentence embeddings of newsgroup posts
