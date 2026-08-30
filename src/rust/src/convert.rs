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

/// Build a 1-indexed cluster assignment vector, `NA` for noise.
///
/// Clusters are renumbered to 1..k, sorted by their original key so the
/// labelling is deterministic across runs. Cluster and noise counts are derived
/// on the R side from this vector, so they are not returned here.
pub fn assignment_vector(clusters: &HashMap<usize, Vec<usize>>, n_points: usize) -> Vec<Rint> {
    let mut assignment = vec![Rint::na(); n_points];

    let mut keys: Vec<usize> = clusters.keys().copied().collect();
    keys.sort_unstable();

    for (new_id, key) in keys.iter().enumerate() {
        let r_id = Rint::from((new_id + 1) as i32); // 1-indexed for R
        if let Some(indices) = clusters.get(key) {
            for &idx in indices {
                if idx < n_points {
                    assignment[idx] = r_id;
                }
            }
        }
    }

    assignment
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
