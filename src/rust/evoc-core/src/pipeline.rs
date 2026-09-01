//! Top-level EVoC pipeline — port of `evoc_clusters` from `clustering.py`
//! with the reference's defaults: cosine kNN via NN-Descent, fuzzy graph
//! construction, label-propagation initialisation, node embedding, and
//! multi-layer density clustering of the embedded points.

use crate::graph::neighbor_graph_matrix;
use crate::labelprop::{label_propagation_init, LabelPropParams};
use crate::layers::{build_cluster_layers, LayerParams};
use crate::nndescent::nn_descent;
use crate::embedding::node_embedding;
use crate::rng::Rng;

pub struct EvocParams {
    pub n_neighbors: usize,
    pub noise_level: f32,
    pub base_min_cluster_size: i64,
    pub min_samples: usize,
    pub n_epochs: usize,
    /// Embedding dimension; `None` uses the reference's
    /// `min(max(n_neighbors / 4, 4), 15)`.
    pub node_embedding_dim: Option<usize>,
    pub min_similarity_threshold: f64,
    pub max_layers: usize,
    pub n_label_prop_iter: usize,
    pub seed: u64,
}

impl Default for EvocParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            noise_level: 0.5,
            base_min_cluster_size: 5,
            min_samples: 5,
            n_epochs: 50,
            node_embedding_dim: None,
            min_similarity_threshold: 0.2,
            max_layers: 10,
            n_label_prop_iter: 20,
            seed: 42,
        }
    }
}

pub struct EvocResult {
    /// Cluster layers, finest (most clusters) first; -1 marks noise.
    pub layers: Vec<Vec<i64>>,
    pub strengths: Vec<Vec<f32>>,
    pub persistence_scores: Vec<f64>,
    pub knn_indices: Vec<Vec<i64>>,
    pub knn_dists: Vec<Vec<f32>>,
    /// The learned node embedding, `n x n_embedding_components` row-major.
    pub embedding: Vec<f32>,
    pub n_embedding_components: usize,
}

/// Cluster `data` (`n x dims`, row-major) with the full EVoC pipeline.
/// Cosine distance is assumed, as the reference assumes for float input;
/// rows are L2-normalised internally, matching `knn_graph`.
pub fn evoc(data: &[f32], dims: usize, params: &EvocParams) -> EvocResult {
    assert!(dims > 0 && data.len() % dims == 0, "data must be n x dims row-major");
    let n = data.len() / dims;
    assert!(n > params.n_neighbors, "need more points than n_neighbors");
    assert!(params.n_neighbors > 0, "n_neighbors must be positive");
    assert!(
        params.base_min_cluster_size >= 2,
        "base_min_cluster_size must be at least 2: the condensed tree treats a \
         size-1 cluster as a point and mislabels its sibling"
    );
    assert!(
        params.node_embedding_dim != Some(0),
        "node_embedding_dim must be positive"
    );

    // Accumulated in f64: squaring in f32 overflows above ~1.8e19 and would
    // silently zero the row.
    let mut normed = data.to_vec();
    for row in normed.chunks_exact_mut(dims) {
        let norm = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v = (f64::from(*v) / norm) as f32;
            }
        }
    }

    let knn = nn_descent(&normed, dims, params.n_neighbors, params.seed);
    let graph =
        neighbor_graph_matrix(params.n_neighbors as f32, &knn.indices, &knn.distances, true);

    let n_components = params
        .node_embedding_dim
        .unwrap_or_else(|| (params.n_neighbors / 4).max(4).min(15));

    let mut rng = Rng::new(params.seed ^ 0x9D8F_3C1A_5B72_E604);
    let approx_n_parts =
        ((8.0 * (n as f64).sqrt()) as usize).clamp(256, 16384);
    let init = label_propagation_init(
        &graph,
        data,
        dims,
        &LabelPropParams {
            n_label_prop_iter: params.n_label_prop_iter,
            n_embedding_epochs: 50,
            approx_n_parts,
            n_components,
            scaling: 0.5,
            noise_level: params.noise_level,
            base_init_threshold: 64,
        },
        &mut rng,
    );

    let embedding = node_embedding(
        &graph,
        n_components,
        params.n_epochs,
        Some(init),
        0.1,
        1.0,
        params.noise_level,
        &mut rng,
    );

    let emb_rows: Vec<Vec<f32>> = embedding.chunks_exact(n_components).map(<[f32]>::to_vec).collect();
    let clusters = build_cluster_layers(
        &emb_rows,
        &LayerParams {
            min_samples: params.min_samples,
            base_min_cluster_size: params.base_min_cluster_size,
            min_similarity_threshold: params.min_similarity_threshold,
            max_layers: params.max_layers,
        },
    );

    EvocResult {
        layers: clusters.layers,
        strengths: clusters.strengths,
        persistence_scores: clusters.persistence_scores,
        knn_indices: knn.indices,
        knn_dists: knn.distances,
        embedding,
        n_embedding_components: n_components,
    }
}
