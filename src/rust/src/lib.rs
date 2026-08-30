use extendr_api::prelude::*;
use petal_clustering::{Dbscan, Fit, HDbscan};
use petal_neighbors::distance::{Cosine, Euclidean};

mod convert;
mod dist;
mod hclust;
use convert::{assignment_vector, partial_labels_from_list, rmatrix_to_array2};

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

extendr_module! {
    mod shoal;
    fn rust_dbscan;
    fn rust_hdbscan;
    fn rust_dist;
    fn rust_hclust;
}
