//! Mutual-reachability minimum spanning tree over an embedding.
//!
//! Two implementations of the same contract:
//!
//! - [`mutual_reachability_mst`] — kd-tree Borůvka, the same structure as the
//!   reference's `parallel_boruvka`: core distances from a tree kNN query, a
//!   kNN initialisation pass that collapses most components before the first
//!   round, then component-aware tree queries per round. Per-point work runs
//!   on rayon; each round's winning edges are reduced sequentially in index
//!   order, so the result is deterministic regardless of thread scheduling
//!   (the reference races on its pruning bound instead).
//! - [`mutual_reachability_mst_brute`] — O(n²) Prim, kept as the oracle the
//!   property tests compare against.
//!
//! Conventions mirrored from the reference: core distance is the
//! `min_samples`-th nearest *other* neighbour; all work happens on squared
//! Euclidean f32 distances with one square root on the final weights; output
//! is canonical (`u < v`, sorted by weight then endpoints). Under tied weights
//! multiple equally-minimal trees exist, so the binding invariant across
//! implementations is the sorted weight sequence.

use crate::disjoint_set::RankDisjointSet;
use crate::kdtree::{rdist, KdTree};
use rayon::prelude::*;

const LEAF_SIZE: usize = 40; // reference default

fn canonicalise(edges: Vec<(u32, u32, f32)>) -> Vec<(u32, u32, f64)> {
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

/// Kd-tree Borůvka MST. `data` is row-major `n x dims`.
pub fn mutual_reachability_mst(data: &[Vec<f32>], min_samples: usize) -> Vec<(u32, u32, f64)> {
    let n = data.len();
    if n < 2 {
        return Vec::new();
    }
    let dims = data[0].len();
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    let tree = KdTree::build(&flat, dims, LEAF_SIZE);

    // Core distances: k = min_samples + 1 including self, last column.
    let k = (min_samples + 1).min(n);
    let core: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = &flat[i * dims..(i + 1) * dims];
            let knn = tree.knn_rdist(&flat, p, k);
            knn[k - 1]
        })
        .collect();

    let mut ds = RankDisjointSet::new(n);
    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(n - 1);
    let mut n_components = n;

    // kNN initialisation, as the reference does: when core[i] >= core[j] for a
    // kNN j of i, the mutual reachability to i's first such neighbour is
    // exactly max(core[i], d(i, j)) and no closer point can exist, so it is a
    // valid Borůvka candidate edge.
    let init_candidates: Vec<Option<(u32, f32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = &flat[i * dims..(i + 1) * dims];
            let mut nbrs: Vec<(f32, u32)> = tree
                .knn_rdist_with_idx(&flat, p, k)
                .into_iter()
                .filter(|&(_, j)| j != i as u32)
                .collect();
            nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            for (d, j) in nbrs {
                if core[i] >= core[j as usize] {
                    return Some((j, d.max(core[i])));
                }
            }
            None
        })
        .collect();

    for (i, cand) in init_candidates.iter().enumerate() {
        if let Some((j, w)) = cand {
            let (a, b) = (ds.find(i as i32), ds.find(*j as i32));
            if a != b {
                ds.union_by_rank(a, b);
                edges.push((i as u32, *j, *w));
                n_components -= 1;
            }
        }
    }

    // Borůvka rounds over the remaining components.
    let mut point_components = vec![0i32; n];
    let mut node_components = vec![-1i32; tree.n_nodes()];

    while n_components > 1 {
        for (i, pc) in point_components.iter_mut().enumerate() {
            *pc = ds.find(i as i32);
        }
        // Children come after parents in preorder, so reverse iteration
        // resolves leaves before internal nodes.
        for node in (0..tree.n_nodes()).rev() {
            node_components[node] = match tree.children(node) {
                None => {
                    let pts = tree.node_points(node);
                    let first = point_components[pts[0] as usize];
                    if pts.iter().all(|&p| point_components[p as usize] == first) {
                        first
                    } else {
                        -1
                    }
                }
                Some((l, r)) => {
                    let (cl, cr) = (node_components[l], node_components[r]);
                    if cl == cr {
                        cl
                    } else {
                        -1
                    }
                }
            };
        }

        // Shared per-component pruning bounds. Stale reads only cost pruning
        // efficiency; the winning edge per component is chosen in the
        // deterministic reduce below.
        let bounds: Vec<std::sync::atomic::AtomicU32> = (0..n)
            .map(|_| std::sync::atomic::AtomicU32::new(f32::INFINITY.to_bits()))
            .collect();

        let candidates: Vec<(f32, i64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                use std::sync::atomic::Ordering::Relaxed;
                let comp = point_components[i];
                let slot = &bounds[comp as usize];
                let bound = f32::from_bits(slot.load(Relaxed));
                if core[i] > bound {
                    return (f32::INFINITY, -1); // cannot beat a teammate
                }
                let p = &flat[i * dims..(i + 1) * dims];
                let best = tree.component_nn(
                    &flat, p, core[i], comp, &core, &point_components, &node_components, bound,
                );
                if best.1 >= 0 {
                    slot.fetch_min(best.0.to_bits(), Relaxed);
                }
                best
            })
            .collect();

        // Deterministic reduce: minimum per component, ties to the lowest
        // point index; merge in ascending component order.
        let mut best_edge: Vec<(f32, i64, i64)> = vec![(f32::INFINITY, -1, -1); n];
        for (i, &(d, j)) in candidates.iter().enumerate() {
            if j < 0 {
                continue;
            }
            let c = point_components[i] as usize;
            if d < best_edge[c].0 {
                best_edge[c] = (d, i as i64, j);
            }
        }
        let before = n_components;
        for &(w, i, j) in best_edge.iter() {
            if i < 0 {
                continue;
            }
            let (a, b) = (ds.find(i as i32), ds.find(j as i32));
            if a != b {
                ds.union_by_rank(a, b);
                edges.push((i as u32, j as u32, w));
                n_components -= 1;
            }
        }
        assert!(
            n_components < before,
            "Borůvka round made no progress; component-aware query is broken"
        );
    }

    canonicalise(edges)
}

/// Core distances by brute force, squared space.
fn core_rdistances_brute(data: &[Vec<f32>], min_samples: usize) -> Vec<f32> {
    let n = data.len();
    data.iter()
        .map(|p| {
            let mut dists: Vec<f32> = data.iter().map(|q| rdist(p, q)).collect();
            let k = min_samples.min(n - 1);
            let (_, kth, _) = dists.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
            *kth
        })
        .collect()
}

/// O(n²) Prim over the mutual-reachability graph — the property-test oracle.
pub fn mutual_reachability_mst_brute(
    data: &[Vec<f32>],
    min_samples: usize,
) -> Vec<(u32, u32, f64)> {
    let n = data.len();
    if n < 2 {
        return Vec::new();
    }
    let core = core_rdistances_brute(data, min_samples);

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

    canonicalise(edges)
}
