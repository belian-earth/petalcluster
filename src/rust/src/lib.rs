use extendr_api::prelude::*;
use petal_clustering::{Dbscan, Fit, HDbscan, Optics};
use petal_neighbors::distance::{Cosine, Euclidean};

mod convert;
use convert::{
    clusters_to_list, clusters_to_list_with_scores, partial_labels_from_list, rmatrix_to_array2,
};

#[extendr]
fn rust_dbscan(x: RMatrix<f64>, eps: f64, min_samples: i32, metric: &str) -> List {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;

    match metric {
        "euclidean" => {
            let mut model = Dbscan::new(eps, min_samples, Euclidean::default());
            let (clusters, noise) = model.fit(&data, None);
            clusters_to_list(&clusters, &noise, n_points)
        }
        "cosine" => {
            let mut model = Dbscan::new(eps, min_samples, Cosine::default());
            let (clusters, noise) = model.fit(&data, None);
            clusters_to_list(&clusters, &noise, n_points)
        }
        _ => panic!("Unknown metric: {metric}"),
    }
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

    match metric {
        "euclidean" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Euclidean::default(),
                boruvka,
            };
            let (clusters, noise, scores) = model.fit(&data, labels.as_ref());
            clusters_to_list_with_scores(&clusters, &noise, &scores, n_points)
        }
        "cosine" => {
            let mut model = HDbscan {
                alpha,
                min_samples,
                min_cluster_size,
                metric: Cosine::default(),
                boruvka,
            };
            let (clusters, noise, scores) = model.fit(&data, labels.as_ref());
            clusters_to_list_with_scores(&clusters, &noise, &scores, n_points)
        }
        _ => panic!("Unknown metric: {metric}"),
    }
}

#[extendr]
fn rust_optics(x: RMatrix<f64>, eps: f64, min_samples: i32, metric: &str) -> List {
    let data = rmatrix_to_array2(x);
    let n_points = data.nrows();
    let min_samples = min_samples as usize;

    match metric {
        "euclidean" => {
            let mut model = Optics::new(eps, min_samples, Euclidean::default());
            let (clusters, noise) = model.fit(&data, None);
            clusters_to_list(&clusters, &noise, n_points)
        }
        "cosine" => {
            let mut model = Optics::new(eps, min_samples, Cosine::default());
            let (clusters, noise) = model.fit(&data, None);
            clusters_to_list(&clusters, &noise, n_points)
        }
        _ => panic!("Unknown metric: {metric}"),
    }
}

extendr_module! {
    mod petalcluster;
    fn rust_dbscan;
    fn rust_hdbscan;
    fn rust_optics;
}
