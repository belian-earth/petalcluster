use std::time::Instant;
use evoc_core::graph::neighbor_graph_matrix;
use evoc_core::labelprop::{label_propagation_init, LabelPropParams};
use evoc_core::layers::{build_cluster_layers, LayerParams};
use evoc_core::nndescent::nn_descent;
use evoc_core::embedding::node_embedding;
use evoc_core::rng::Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dims: usize = args[2].parse().unwrap();
    let data: Vec<f32> = std::fs::read(&args[1]).unwrap()
        .chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let n = data.len() / dims;
    println!("n={n} d={dims}");

    let t = Instant::now();
    let knn = nn_descent(&data, dims, 15, 42);
    println!("  nn_descent      {:6.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let graph = neighbor_graph_matrix(15.0, &knn.indices, &knn.distances, true);
    println!("  fuzzy graph     {:6.2}s", t.elapsed().as_secs_f64());

    let mut rng = Rng::new(42);
    let approx = ((8.0 * (n as f64).sqrt()) as usize).clamp(256, 16384);
    let t = Instant::now();
    let init = label_propagation_init(&graph, &data, dims, &LabelPropParams {
        n_label_prop_iter: 20, n_embedding_epochs: 50, approx_n_parts: approx,
        n_components: 4, scaling: 0.5, noise_level: 0.5, base_init_threshold: 64,
    }, &mut rng);
    println!("  labelprop init  {:6.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let emb = node_embedding(&graph, 4, 50, Some(init), 0.1, 1.0, 0.5, &mut rng);
    println!("  node embedding  {:6.2}s", t.elapsed().as_secs_f64());

    let rows: Vec<Vec<f32>> = emb.chunks_exact(4).map(<[f32]>::to_vec).collect();
    let t = Instant::now();
    let cl = build_cluster_layers(&rows, &LayerParams { min_samples: 5, base_min_cluster_size: 5, min_similarity_threshold: 0.2, max_layers: 10 });
    println!("  cluster layers  {:6.2}s ({} layers)", t.elapsed().as_secs_f64(), cl.layers.len());
}
