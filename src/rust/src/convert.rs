use extendr_api::prelude::*;
use ndarray::{Array2, ShapeBuilder};
use std::collections::HashMap;

/// Convert an extendr RMatrix<f64> to an ndarray Array2<f64>.
///
/// R stores matrices in column-major order. We read the raw data and reshape
/// using Fortran (column-major) layout so no transposition is needed.
pub fn rmatrix_to_array2(x: RMatrix<f64>) -> Array2<f64> {
    let nrow = x.nrows();
    let ncol = x.ncols();
    let data: Vec<f64> = x.data().to_vec();
    // R is column-major (Fortran order)
    Array2::from_shape_vec((nrow, ncol).f(), data).expect("shape mismatch in matrix conversion")
}

/// Convert clustering output (clusters map + noise indices) into an R list.
///
/// Returns a list with:
///   - cluster: integer vector (1-indexed, NA for noise)
///   - n_clusters: integer scalar
///   - n_noise: integer scalar
pub fn clusters_to_list(
    clusters: &HashMap<usize, Vec<usize>>,
    noise: &[usize],
    n_points: usize,
) -> List {
    let n_clusters = clusters.len() as i32;
    let n_noise = noise.len() as i32;

    // Build assignment vector, default NA
    let mut assignment = vec![Rint::na(); n_points];

    // Renumber clusters to 1..k (sorted by original key for determinism)
    let mut keys: Vec<usize> = clusters.keys().copied().collect();
    keys.sort_unstable();

    for (new_id, key) in keys.iter().enumerate() {
        let r_id = Rint::from((new_id + 1) as i32); // 1-indexed
        if let Some(indices) = clusters.get(key) {
            for &idx in indices {
                if idx < n_points {
                    assignment[idx] = r_id;
                }
            }
        }
    }

    list!(
        cluster = assignment,
        n_clusters = n_clusters,
        n_noise = n_noise
    )
}

/// Convert clustering output with outlier scores (HDBSCAN) into an R list.
pub fn clusters_to_list_with_scores(
    clusters: &HashMap<usize, Vec<usize>>,
    noise: &[usize],
    outlier_scores: &[f64],
    n_points: usize,
) -> List {
    let n_clusters = clusters.len() as i32;
    let n_noise = noise.len() as i32;

    let mut assignment = vec![Rint::na(); n_points];

    let mut keys: Vec<usize> = clusters.keys().copied().collect();
    keys.sort_unstable();

    for (new_id, key) in keys.iter().enumerate() {
        let r_id = Rint::from((new_id + 1) as i32);
        if let Some(indices) = clusters.get(key) {
            for &idx in indices {
                if idx < n_points {
                    assignment[idx] = r_id;
                }
            }
        }
    }

    let scores: Vec<Rfloat> = outlier_scores.iter().map(|&s| Rfloat::from(s)).collect();

    list!(
        cluster = assignment,
        n_clusters = n_clusters,
        n_noise = n_noise,
        outlier_scores = scores
    )
}

/// Convert an R named list of partial labels to a HashMap<usize, Vec<usize>>.
///
/// The R list has names like "0", "1", ... (cluster IDs) and values are
/// integer vectors of 1-indexed point indices. We convert to 0-indexed.
pub fn partial_labels_from_list(labels: List) -> HashMap<usize, Vec<usize>> {
    let mut map = HashMap::new();
    let names: Vec<String> = labels
        .names()
        .unwrap_or_default()
        .map(|s| s.to_string())
        .collect();

    for (i, name) in names.iter().enumerate() {
        let cluster_id: usize = name.parse().expect("partial_labels names must be integer strings");
        let indices_robj = labels.elt(i).expect("invalid list element");
        let indices: Vec<usize> = indices_robj
            .as_integer_slice()
            .expect("partial_labels values must be integer vectors")
            .iter()
            .map(|&idx| (idx - 1) as usize) // R 1-indexed to Rust 0-indexed
            .collect();
        map.insert(cluster_id, indices);
    }

    map
}
