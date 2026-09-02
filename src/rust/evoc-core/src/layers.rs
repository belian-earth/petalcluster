//! `build_cluster_layers` — the deterministic clustering half of EVoC,
//! chaining MST, linkage, condensation, extraction and persistence exactly as
//! `clustering.py` does (minus the `base_n_clusters` branch, which the shoal
//! integration does not use yet).

use crate::condense::{
    condense_tree, extract_leaves, get_cluster_label_vector,
    get_point_membership_strength_vector, CondensedTree,
};
use crate::linkage::mst_to_linkage_tree;
use crate::mst::mutual_reachability_mst;
use crate::persistence::{
    compute_total_persistence, find_peaks, min_cluster_size_barcode, select_diverse_peaks,
};

pub struct ClusterLayers {
    /// Finest first, per the reference's final sort by descending cluster count.
    pub layers: Vec<Vec<i64>>,
    pub strengths: Vec<Vec<f32>>,
    pub persistence_scores: Vec<f64>,
}

pub struct LayerParams {
    pub min_samples: usize,
    pub base_min_cluster_size: i64,
    pub min_similarity_threshold: f64,
    pub max_layers: usize,
}

impl Default for LayerParams {
    fn default() -> Self {
        Self {
            min_samples: 5,
            base_min_cluster_size: 10,
            min_similarity_threshold: 0.2,
            max_layers: 10,
        }
    }
}

pub fn build_cluster_layers(embedding: &[Vec<f32>], params: &LayerParams) -> ClusterLayers {
    let sorted_mst = mutual_reachability_mst(embedding, params.min_samples);
    cluster_layers_from_mst(&sorted_mst, embedding.len(), params)
}

/// The deterministic clustering chain from an already-built canonical MST.
/// Split out so parity tests can drive it from a fixture's stored tree, making
/// the comparison independent of which equally-minimal MST was found.
pub fn cluster_layers_from_mst(
    sorted_mst: &[(u32, u32, f64)],
    n_samples: usize,
    params: &LayerParams,
) -> ClusterLayers {
    let hierarchy = mst_to_linkage_tree(sorted_mst);
    let condensed = condense_tree(&hierarchy, params.base_min_cluster_size);

    let leaves = extract_leaves(&condensed);
    let base_clusters = get_cluster_label_vector(&condensed, &leaves, 0.0, n_samples);
    let base_strengths = get_point_membership_strength_vector(&condensed, &leaves, &base_clusters);

    let mut layers = vec![base_clusters];
    let mut strengths = vec![base_strengths];
    let mut persistence_scores = vec![0.0f64];

    // The persistence machinery runs on the cluster-only tree.
    let mask: Vec<bool> = condensed.child.iter().map(|&c| c >= n_samples as i64).collect();
    let cluster_tree: CondensedTree = condensed.masked(&mask);

    if !cluster_tree.is_empty() && *cluster_tree.child.last().unwrap() >= n_samples as i64 {
        let barcode = min_cluster_size_barcode(
            &cluster_tree,
            n_samples as i64,
            params.base_min_cluster_size,
        );
        let (sizes, total_persistence) =
            compute_total_persistence(&barcode.births, &barcode.deaths, &barcode.lambda_deaths);
        let peaks = find_peaks(&total_persistence);
        let selected = select_diverse_peaks(
            &peaks,
            &total_persistence,
            &sizes,
            &barcode.births,
            &barcode.deaths,
            params.min_similarity_threshold,
            params.max_layers.saturating_sub(1), // one slot reserved for the base layer
        );

        for peak in selected {
            let best_birth = sizes[peak as usize];
            let persistence = f64::from(total_persistence[peak as usize]);
            let selected_clusters: Vec<i64> = (0..barcode.births.len())
                .filter(|&i| barcode.births[i] <= best_birth && barcode.deaths[i] > best_birth)
                .map(|i| i as i64 + n_samples as i64)
                .collect();

            let labels =
                get_cluster_label_vector(&condensed, &selected_clusters, 0.0, n_samples);
            let layer_strengths =
                get_point_membership_strength_vector(&condensed, &selected_clusters, &labels);

            layers.push(labels);
            strengths.push(layer_strengths);
            persistence_scores.push(persistence);
        }
    }

    // Final sort: most clusters first. The reference uses a reversed unstable
    // argsort; with distinct counts (the common case) order is well-defined.
    let mut order: Vec<usize> = (0..layers.len()).collect();
    order.sort_by(|&a, &b| {
        let ka = layers[a].iter().max().copied().unwrap_or(-1) + 1;
        let kb = layers[b].iter().max().copied().unwrap_or(-1) + 1;
        kb.cmp(&ka).then(a.cmp(&b))
    });

    ClusterLayers {
        layers: order.iter().map(|&i| layers[i].clone()).collect(),
        strengths: order.iter().map(|&i| strengths[i].clone()).collect(),
        persistence_scores: order.iter().map(|&i| persistence_scores[i]).collect(),
    }
}
