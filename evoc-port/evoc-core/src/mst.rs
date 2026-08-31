//! Mutual-reachability minimum spanning tree over an embedding.
//!
//! The reference builds this with a kd-tree and parallel Borůvka
//! (`boruvka.py`); this port uses brute-force Prim, which finds the same tree
//! (up to ties) in O(n²) — acceptable for the embedding sizes the fixtures
//! cover, and a placeholder for a dual-tree implementation later.
//!
//! Conventions mirrored from the reference:
//! - Core distance of a point is its `min_samples`-th nearest *other*
//!   neighbour (the reference queries `k = min_samples + 1` including self and
//!   takes the last column).
//! - All work happens on squared Euclidean distances in `f32`, with a single
//!   square root applied to the final edge weights, exactly as
//!   `parallel_boruvka` does.
//! - Output is canonical: each edge stored `(min, max)`, sorted by
//!   `(weight, u, v)`. The reference's emission order is not stable under tied
//!   weights, so fixtures store this canonical form.

/// Squared Euclidean distance, accumulated in f32 like the reference kd-tree.
fn rdist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Core distances in squared space: the `min_samples`-th smallest rdist to
/// another point (self counts as the 0th neighbour, at distance zero).
fn core_rdistances(data: &[Vec<f32>], min_samples: usize) -> Vec<f32> {
    let n = data.len();
    data.iter()
        .map(|p| {
            let mut dists: Vec<f32> = data.iter().map(|q| rdist(p, q)).collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dists[min_samples.min(n - 1)]
        })
        .collect()
}

/// Canonical mutual-reachability MST: rows of `(u, v, weight)` with `u < v`,
/// sorted by `(weight, u, v)`.
pub fn mutual_reachability_mst(data: &[Vec<f32>], min_samples: usize) -> Vec<(u32, u32, f64)> {
    let n = data.len();
    if n < 2 {
        return Vec::new();
    }
    let core = core_rdistances(data, min_samples);

    // Prim's algorithm over the implicit complete mutual-reachability graph.
    let mut in_tree = vec![false; n];
    let mut best_dist = vec![f32::INFINITY; n];
    let mut best_from = vec![0u32; n];
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(n - 1);

    let mut current = 0usize;
    in_tree[0] = true;
    for _ in 1..n {
        let cur_point = &data[current];
        let cur_core = core[current];
        let mut next = usize::MAX;
        let mut next_dist = f32::INFINITY;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let d = rdist(cur_point, &data[j]).max(cur_core).max(core[j]);
            if d < best_dist[j] {
                best_dist[j] = d;
                best_from[j] = current as u32;
            }
            if best_dist[j] < next_dist {
                next_dist = best_dist[j];
                next = j;
            }
        }
        edges.push((best_from[next], next as u32, next_dist));
        in_tree[next] = true;
        current = next;
    }

    let mut canonical: Vec<(u32, u32, f64)> = edges
        .into_iter()
        .map(|(a, b, w)| (a.min(b), a.max(b), f64::from(w.sqrt())))
        .collect();
    canonical.sort_by(|x, y| {
        x.2.partial_cmp(&y.2)
            .unwrap()
            .then(x.0.cmp(&y.0))
            .then(x.1.cmp(&y.1))
    });
    canonical
}
