use extendr_api::prelude::*;
use petal_clustering::{Dbscan, Fit, HDbscan};
use petal_neighbors::distance::{Cosine, Euclidean};

mod convert;
mod dist;
mod gmm;
mod hclust;
mod kmeans;
mod metrics;
use convert::{
    assignment_vector, partial_labels_from_list, rmatrix_to_array2, rmatrix_to_array2_linfa,
};

#[extendr]
fn rust_dbscan(x: RMatrix<f64>, eps: f64, min_samples: i32, metric: &str) -> Integers {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;

    let clusters = match metric {
        "euclidean" => {
            let mut model = Dbscan::new(eps, min_samples, Euclidean::default());
            model.fit(&data, None).0
        }
        "cosine" => {
            let mut model = Dbscan::new(eps, min_samples, Cosine::default());
            model.fit(&data, None).0
        }
        _ => panic!("Unknown metric: {metric}"),
    };

    Integers::from_values(assignment_vector(&clusters, n_points))
}

#[extendr]
fn rust_hdbscan(
    x: RMatrix<f64>,
    alpha: f64,
    min_samples: i32,
    min_cluster_size: i32,
    metric: &str,
    boruvka: bool,
    partial_labels: Nullable<List>,
) -> List {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;
    let min_cluster_size = min_cluster_size as usize;

    let labels = match partial_labels {
        Nullable::NotNull(pl) => Some(partial_labels_from_list(pl)),
        Nullable::Null => None,
    };

    let (clusters, scores) = match metric {
        "euclidean" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Euclidean::default(),
                boruvka,
            };
            let (clusters, _noise, scores) = model.fit(&data, labels.as_ref());
            (clusters, scores)
        }
        "cosine" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Cosine::default(),
                boruvka,
            };
            let (clusters, _noise, scores) = model.fit(&data, labels.as_ref());
            (clusters, scores)
        }
        _ => panic!("Unknown metric: {metric}"),
    };

    let assignment = assignment_vector(&clusters, n_points);
    let outlier_scores: Vec<Rfloat> = scores.iter().map(|&s| Rfloat::from(s)).collect();

    list!(cluster = assignment, outlier_scores = outlier_scores)
}

/// Condensed lower-triangle distance matrix, in R's `dist` layout.
#[extendr]
fn rust_dist(x: RMatrix<f64>, metric: &str, p: f64) -> Doubles {
    let data = rmatrix_to_array2(x);
    let values = dist::condensed(&data, dist::Metric::from_name(metric, p));
    values.into_iter().map(Rfloat::from).collect()
}

/// Hierarchical clustering over a condensed dissimilarity matrix.
///
/// Returns the `merge`, `height` and `order` components of an `hclust` object.
/// `d` arrives as an owned copy because kodama destroys the matrix it is given.
#[extendr]
fn rust_hclust(d: Vec<f64>, n: i32, method: &str) -> List {
    let n = n as usize;

    let expected = n * (n - 1) / 2;
    if d.len() != expected {
        panic!("expected {expected} dissimilarities for {n} observations, got {}", d.len());
    }
    // kodama panics on NaN; the R side rejects these first, so reaching here is a bug.
    if d.iter().any(|v| v.is_nan()) {
        panic!("dissimilarity matrix contains NaN");
    }

    let out = hclust::hclust(d, n, hclust::method_from_name(method));
    let n_steps = out.height.len();

    let merge = RMatrix::new_matrix(n_steps, 2, |r, c| out.merge[c * n_steps + r]);

    list!(merge = merge, height = out.height, order = out.order)
}

/// k-means clustering.
///
/// Returns the assignment vector, centroids, inertia and cluster sizes.
#[extendr]
fn rust_kmeans(
    x: RMatrix<f64>,
    k: i32,
    init: &str,
    n_runs: i32,
    max_iter: i32,
    tolerance: f64,
    seed: f64,
) -> Result<List> {
    let data = rmatrix_to_array2_linfa(x);

    let fit = kmeans::fit(
        data,
        k as usize,
        kmeans::init_from_name(init),
        n_runs as usize,
        max_iter as u64,
        tolerance,
        seed as u64,
    )
    .map_err(|e| Error::Other(format!("k-means failed: {e}")))?;

    let n_clusters = fit.centroids.nrows();
    let n_features = fit.centroids.ncols();

    let cluster: Vec<Rint> = fit
        .assignments
        .iter()
        .map(|&c| Rint::from((c + 1) as i32)) // 1-indexed for R
        .collect();

    let centroids = RMatrix::new_matrix(n_clusters, n_features, |r, c| fit.centroids[[r, c]]);

    Ok(list!(
        cluster = cluster,
        centroids = centroids,
        inertia = fit.inertia,
        sizes = fit.sizes
    ))
}

/// Assign observations to their nearest centroid. Backs `predict()`.
#[extendr]
fn rust_nearest_centroid(x: RMatrix<f64>, centroids: RMatrix<f64>) -> Integers {
    let data = rmatrix_to_array2_linfa(x);
    let centroids = rmatrix_to_array2_linfa(centroids);

    kmeans::nearest_centroid(&data, &centroids)
        .into_iter()
        .map(Rint::from)
        .collect()
}

/// Gaussian mixture model.
///
/// Returns only the fitted parameters; responsibilities, assignments and the
/// log-likelihood are derived from them on the R side.
#[extendr]
fn rust_gmm(
    x: RMatrix<f64>,
    k: i32,
    init: &str,
    n_runs: i32,
    max_iter: i32,
    tolerance: f64,
    reg_covariance: f64,
    seed: f64,
) -> Result<List> {
    let data = rmatrix_to_array2_linfa(x);

    let fit = gmm::fit(
        data,
        k as usize,
        gmm::init_from_name(init),
        n_runs as u64,
        max_iter as u64,
        tolerance,
        reg_covariance,
        seed as u64,
    )
    .map_err(|e| Error::Other(format!("Gaussian mixture fit failed: {e}")))?;

    let n_clusters = fit.means.nrows();
    let n_features = fit.means.ncols();
    let means = RMatrix::new_matrix(n_clusters, n_features, |r, c| fit.means[[r, c]]);

    Ok(list!(
        weights = fit.weights,
        means = means,
        covariances = fit.covariances
    ))
}

/// Per-observation silhouette widths from a condensed distance matrix.
///
/// `cluster` arrives 1-indexed from R and is returned to R the same way.
#[extendr]
fn rust_silhouette(d: Vec<f64>, n: i32, cluster: Vec<i32>, k: i32) -> List {
    let n = n as usize;
    let k = k as usize;
    let zero_based: Vec<usize> = cluster.iter().map(|&c| (c - 1) as usize).collect();

    let (widths, neighbours) = metrics::silhouette(&d, n, &zero_based, k);

    // usize::MAX marks "no neighbouring cluster"; that becomes NA in R.
    let neighbour: Vec<Rint> = neighbours
        .into_iter()
        .map(|nb| {
            if nb == usize::MAX {
                Rint::na()
            } else {
                Rint::from((nb + 1) as i32)
            }
        })
        .collect();

    list!(width = widths, neighbour = neighbour)
}

/// Calinski-Harabasz and Davies-Bouldin indices.
#[extendr]
fn rust_cluster_indices(x: RMatrix<f64>, cluster: Vec<i32>, k: i32) -> List {
    let data = rmatrix_to_array2(x);
    let n = data.nrows();
    let p = data.ncols();
    let k = k as usize;

    // Row-major copy: the metrics walk observations, not features.
    let mut flat = Vec::with_capacity(n * p);
    for i in 0..n {
        for f in 0..p {
            flat.push(data[[i, f]]);
        }
    }

    let zero_based: Vec<usize> = cluster.iter().map(|&c| (c - 1) as usize).collect();

    list!(
        calinski_harabasz = metrics::calinski_harabasz(&flat, n, p, &zero_based, k),
        davies_bouldin = metrics::davies_bouldin(&flat, n, p, &zero_based, k)
    )
}

extendr_module! {
    mod shoal;
    fn rust_dbscan;
    fn rust_hdbscan;
    fn rust_dist;
    fn rust_hclust;
    fn rust_kmeans;
    fn rust_nearest_centroid;
    fn rust_gmm;
    fn rust_silhouette;
    fn rust_cluster_indices;
}
