//! Glue between R and the in-tree `evoc-core` port (see `src/rust/evoc-core`
//! and the parity suite in `evoc-port/`). Only conversion lives here; the
//! algorithm is entirely evoc-core's.

use evoc_core::pipeline::{evoc, EvocParams, EvocResult};
use extendr_api::prelude::*;

/// R column-major f64 matrix -> row-major f32, the layout evoc-core expects.
///
/// Fails if any value falls outside the single-precision range: `as f32`
/// saturates such values to infinity, which becomes NaN once rows are
/// recentred and would panic deep inside the kd-tree. The R side screens
/// non-finite doubles, so only magnitude can trip this.
pub fn rmatrix_to_rowmajor_f32(x: &RMatrix<f64>) -> Result<(Vec<f32>, usize, usize)> {
    let n = x.nrows();
    let p = x.ncols();
    let data = x.data();
    let mut out = Vec::with_capacity(n * p);
    for r in 0..n {
        for c in 0..p {
            let v = data[c * n + r] as f32;
            if !v.is_finite() {
                return Err(Error::Other(
                    "`x` contains values too large for single precision; rescale it first."
                        .to_string(),
                ));
            }
            out.push(v);
        }
    }
    Ok((out, n, p))
}

/// One 0-based label vector (-1 for noise) -> 1-indexed with `NA` noise.
fn layer_to_r(layer: &[i64]) -> Vec<Rint> {
    layer
        .iter()
        .map(|&l| if l < 0 { Rint::na() } else { Rint::from(l as i32 + 1) })
        .collect()
}

pub fn run(
    x: RMatrix<f64>,
    n_neighbors: usize,
    noise_level: f64,
    min_cluster_size: i64,
    min_samples: usize,
    n_epochs: usize,
    dim: Option<usize>,
    min_similarity_threshold: f64,
    max_layers: usize,
    n_label_prop_iter: usize,
    seed: u64,
) -> Result<EvocResult> {
    let (data, _n, p) = rmatrix_to_rowmajor_f32(&x)?;
    let params = EvocParams {
        n_neighbors,
        noise_level: noise_level as f32,
        base_min_cluster_size: min_cluster_size,
        min_samples,
        n_epochs,
        node_embedding_dim: dim,
        min_similarity_threshold,
        max_layers,
        n_label_prop_iter,
        seed,
    };
    Ok(evoc(&data, p, &params))
}

pub fn result_to_list(result: &EvocResult, n: usize) -> List {
    let layers = List::from_values(
        result
            .layers
            .iter()
            .map(|layer| Integers::from_values(layer_to_r(layer))),
    );
    let strengths = List::from_values(result.strengths.iter().map(|s| {
        s.iter()
            .map(|&v| Rfloat::from(f64::from(v)))
            .collect::<Doubles>()
    }));
    let persistence: Doubles = result
        .persistence_scores
        .iter()
        .map(|&v| Rfloat::from(v))
        .collect();

    let c = result.n_embedding_components;
    let embedding =
        RMatrix::new_matrix(n, c, |r, col| f64::from(result.embedding[r * c + col]));

    list!(
        layers = layers,
        strengths = strengths,
        persistence = persistence,
        embedding = embedding
    )
}
