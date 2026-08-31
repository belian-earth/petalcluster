//! Label-propagation initialisation for the node embedding — port of
//! `label_propagation.py`'s `label_propagation_init` and its helpers.
//!
//! The reference recursively coarsens the fuzzy graph by label propagation,
//! embeds the coarse graph, and upscales with its `partition_expander` until
//! the graph is small enough (< 64 vertices) for a direct PCA layout of the
//! data. The sparse algebra scipy performs (`R_norm^T G R`, `G ∘ G^T`,
//! normalised expander products) is specialised here to the one-hot structure
//! of the reduction map, which collapses each product to per-partition sums.
//!
//! Randomness (seed placement, outlier fallback labels) comes from a
//! different generator than numpy's, so this stage is validated by the
//! ARI-bounded end-to-end fixtures rather than bitwise.

use crate::embedding::node_embedding;
use crate::graph::{transpose, Csr};
use crate::rng::Rng;
use std::collections::HashMap;

pub struct LabelPropParams {
    pub n_label_prop_iter: usize,
    /// Epochs for the coarse embedding at each level; the reference leaves
    /// this at its default (50) regardless of the top-level epoch count.
    pub n_embedding_epochs: usize,
    pub approx_n_parts: usize,
    pub n_components: usize,
    pub scaling: f32,
    pub noise_level: f32,
    pub base_init_threshold: usize,
}

impl Default for LabelPropParams {
    fn default() -> Self {
        Self {
            n_label_prop_iter: 20,
            n_embedding_epochs: 50,
            approx_n_parts: 512,
            n_components: 2,
            scaling: 0.1,
            noise_level: 0.5,
            base_init_threshold: 64,
        }
    }
}

/// One propagation sweep. Nodes that already carry a label are frozen; an
/// unlabelled node takes the label with the largest summed edge weight among
/// its neighbours, but only when that sum reaches the reference's threshold
/// of 1.0 (`max_vote` starts at 1). On a tie at the current maximum the
/// later label in first-seen order wins, as it does for the reference's
/// insertion-ordered dict when the node is unlabelled.
fn label_prop_iteration(g: &Csr, labels: &[i64]) -> Vec<i64> {
    let mut result = labels.to_vec();
    for i in 0..g.n {
        if labels[i] >= 0 {
            continue;
        }
        // Insertion-ordered vote accumulation, like numba's typed dict.
        let mut votes: Vec<(i64, f32)> = Vec::new();
        for k in g.indptr[i]..g.indptr[i + 1] {
            let l = labels[g.indices[k] as usize];
            match votes.iter_mut().find(|(vl, _)| *vl == l) {
                Some(entry) => entry.1 += g.data[k],
                None => votes.push((l, g.data[k])),
            }
        }
        let mut max_vote = 1.0f32;
        for &(l, v) in &votes {
            if l == -1 {
                continue;
            }
            if v > max_vote {
                max_vote = v;
                result[i] = l;
            } else if v == max_vote {
                result[i] = l;
            }
        }
    }
    result
}

/// Depth-first search from each still-unlabelled node for a labelled
/// neighbour, giving up after 64 pops and falling back to a random existing
/// label. Sequential, like the reference (its kernel is compiled without
/// `parallel=True`), so labels assigned early are visible to later nodes.
fn label_outliers(g: &Csr, labels: &mut [i64], seed: u64) {
    let max_label = labels.iter().copied().max().unwrap_or(-1);
    for i in 0..g.n {
        if labels[i] >= 0 {
            continue;
        }
        let mut queue = vec![i];
        let mut unlabelled = true;
        let mut n_iter = 0;
        while unlabelled && n_iter < 64 && !queue.is_empty() {
            n_iter += 1;
            let current = queue.pop().unwrap();
            for k in g.indptr[current]..g.indptr[current + 1] {
                let j = g.indices[k] as usize;
                if labels[j] >= 0 {
                    labels[i] = labels[j];
                    unlabelled = false;
                    break;
                }
                queue.push(j);
            }
        }
        if (n_iter >= 64 || unlabelled) && max_label >= 0 {
            let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            labels[i] = rng.below(max_label as usize + 1) as i64;
        }
    }
}

/// Compress labels to a dense `0..n_parts` range, keeping their sorted order.
fn remap_labels(labels: &mut [i64]) -> usize {
    let mut unique: Vec<i64> = labels.iter().copied().filter(|&l| l >= 0).collect();
    unique.sort_unstable();
    unique.dedup();
    let mapping: HashMap<i64, i64> = unique
        .iter()
        .enumerate()
        .map(|(idx, &l)| (l, idx as i64))
        .collect();
    let mut next_label = unique.len() as i64;
    for l in labels.iter_mut() {
        if *l < 0 {
            *l = next_label;
            next_label += 1;
        } else {
            *l = mapping[l];
        }
    }
    next_label as usize
}

/// Port of `label_prop_loop`: seed `approx_n_parts` labels at random nodes,
/// propagate for `n_iter` sweeps, then resolve leftovers and remap.
/// Returns the partition vector and the number of partitions.
fn label_prop_loop(
    g: &Csr,
    rng: &mut Rng,
    n_iter: usize,
    approx_n_parts: usize,
) -> (Vec<i64>, usize) {
    let mut labels = vec![-1i64; g.n];
    let outlier_seed = rng.next_u64();
    for i in 0..approx_n_parts {
        let pos = rng.below(g.n);
        labels[pos] = i as i64;
    }
    for _ in 0..n_iter {
        labels = label_prop_iteration(g, &labels);
    }
    label_outliers(g, &mut labels, outlier_seed);
    let n_parts = remap_labels(&mut labels);
    (labels, n_parts)
}

/// Direct PCA layout for the base of the recursion, standing in for
/// sklearn's `PCA(n_components).fit_transform(data)` followed by the
/// reference's global recentre/rescale. Computed by power iteration with
/// deflation; component signs are arbitrary, which the embedding does not
/// care about.
fn pca_embed(data: &[f32], n: usize, dims: usize, n_components: usize) -> Vec<f32> {
    let mut means = vec![0.0f64; dims];
    for row in data.chunks_exact(dims) {
        for (m, &v) in means.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }
    let mut centered: Vec<f64> = data
        .chunks_exact(dims)
        .flat_map(|row| row.iter().zip(&means).map(|(&v, &m)| v as f64 - m))
        .collect();

    let n_eff = n_components.min(n).min(dims);
    let mut result = vec![0.0f32; n * n_components];
    for comp in 0..n_eff {
        let mut rng = Rng::new(0x5CA1_AB1E ^ comp as u64);
        let mut v: Vec<f64> = (0..dims).map(|_| rng.gauss() as f64).collect();
        let norm0 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm0;
        }
        let mut scores = vec![0.0f64; n];
        for _ in 0..300 {
            for (i, row) in centered.chunks_exact(dims).enumerate() {
                scores[i] = row.iter().zip(&v).map(|(&a, &b)| a * b).sum();
            }
            let mut w = vec![0.0f64; dims];
            for (i, row) in centered.chunks_exact(dims).enumerate() {
                for (wj, &a) in w.iter_mut().zip(row) {
                    *wj += a * scores[i];
                }
            }
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                v.iter_mut().for_each(|x| *x = 0.0);
                break;
            }
            let mut converged = true;
            for (vj, wj) in v.iter_mut().zip(&w) {
                let new = wj / norm;
                if (new - *vj).abs() > 1e-10 {
                    converged = false;
                }
                *vj = new;
            }
            if converged {
                break;
            }
        }
        for (i, row) in centered.chunks_exact(dims).enumerate() {
            scores[i] = row.iter().zip(&v).map(|(&a, &b)| a * b).sum();
        }
        for (i, row) in centered.chunks_exact_mut(dims).enumerate() {
            for (rj, &vj) in row.iter_mut().zip(&v) {
                *rj -= scores[i] * vj;
            }
        }
        for i in 0..n {
            result[i * n_components + comp] = scores[i] as f32;
        }
    }

    // The reference recentres by the global (not per-column) mean and scales
    // the global range to 2.
    let mean = result.iter().map(|&v| v as f64).sum::<f64>() / result.len().max(1) as f64;
    for v in result.iter_mut() {
        *v -= mean as f32;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in result.iter() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let half_range = (hi - lo) / 2.0;
    if half_range > 1e-12 {
        for v in result.iter_mut() {
            *v /= half_range;
        }
    }
    result
}

/// `R_norm^T G R` specialised to the one-hot reduction map: aggregate every
/// edge of `G` into its partition pair, scale rows by `1/sqrt(count)`, and
/// clip into `[0, 1]`.
fn reduce_graph(g: &Csr, partition: &[i64], counts: &[usize]) -> Csr {
    let inv_sqrt: Vec<f32> = counts.iter().map(|&c| 1.0 / (c as f32).sqrt()).collect();
    let mut triplets: Vec<(u32, u32, f32)> = Vec::with_capacity(g.data.len());
    for i in 0..g.n {
        let p = partition[i] as u32;
        for k in g.indptr[i]..g.indptr[i + 1] {
            let q = partition[g.indices[k] as usize] as u32;
            triplets.push((p, q, g.data[k]));
        }
    }
    triplets.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    let n_parts = counts.len();
    let mut indptr = vec![0usize; n_parts + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    let mut t = 0;
    for p in 0..n_parts as u32 {
        while t < triplets.len() && triplets[t].0 == p {
            let q = triplets[t].1;
            let mut sum = 0.0f32;
            while t < triplets.len() && triplets[t].0 == p && triplets[t].1 == q {
                sum += triplets[t].2;
                t += 1;
            }
            let val = (sum * inv_sqrt[p as usize]).clamp(0.0, 1.0);
            if val != 0.0 {
                indices.push(q);
                data.push(val);
            }
        }
        indptr[p as usize + 1] = indices.len();
    }
    Csr { indptr, indices, data, n: n_parts }
}

/// Port of `label_propagation_init` with the reference's defaults:
/// `recursive_init=True`, `base_init="pca"`, `upscaling="partition_expander"`.
/// `data` is the matrix PCA falls back to at the base of the recursion; each
/// level passes down per-partition means, exactly as `data_reducer @ data`
/// evaluates to for the reference's L1-normalised map.
pub fn label_propagation_init(
    graph: &Csr,
    data: &[f32],
    dims: usize,
    params: &LabelPropParams,
    rng: &mut Rng,
) -> Vec<f32> {
    let n = graph.n;
    let c = params.n_components;

    if n < params.base_init_threshold {
        return pca_embed(data, n, dims, c);
    }

    let (partition, n_parts) =
        label_prop_loop(graph, rng, params.n_label_prop_iter, params.approx_n_parts);
    let mut counts = vec![0usize; n_parts];
    for &p in &partition {
        counts[p as usize] += 1;
    }

    // data_reducer @ data: per-partition means.
    let mut reduced_data = vec![0.0f32; n_parts * dims];
    for (i, row) in data.chunks_exact(dims).enumerate() {
        let p = partition[i] as usize;
        for (acc, &v) in reduced_data[p * dims..(p + 1) * dims].iter_mut().zip(row) {
            *acc += v;
        }
    }
    for (p, chunk) in reduced_data.chunks_exact_mut(dims).enumerate() {
        for v in chunk.iter_mut() {
            *v /= counts[p] as f32;
        }
    }

    let reduced_graph = reduce_graph(graph, &partition, &counts);

    let reduced_init = label_propagation_init(
        &reduced_graph,
        &reduced_data,
        dims,
        &LabelPropParams {
            approx_n_parts: params.approx_n_parts / 4,
            n_embedding_epochs: params.n_embedding_epochs.min(255),
            ..*params
        },
        rng,
    );

    let coarse_epochs = params.n_embedding_epochs.min(255);
    let reduced_layout = node_embedding(
        &reduced_graph,
        c,
        coarse_epochs,
        Some(reduced_init),
        0.001 * coarse_epochs as f32,
        1.0,
        params.noise_level,
        rng,
    );

    // partition_expander upscaling:
    //   (normalize_l1((G ∘ G^T) R_norm) @ layout + normalize_l1(R_norm) @ layout) / 2
    // The second term is a plain lookup of each node's partition row; the
    // first re-weights it by the node's mutual edges. An all-zero expander
    // row stays zero, as sklearn's normalize leaves it.
    let gt = transpose(graph);
    let inv_sqrt: Vec<f32> = counts.iter().map(|&cnt| 1.0 / (cnt as f32).sqrt()).collect();
    let mut result = vec![0.0f32; n * c];
    let mut row_weights: Vec<(u32, f32)> = Vec::new();
    for i in 0..n {
        row_weights.clear();
        let (mut a, a_end) = (graph.indptr[i], graph.indptr[i + 1]);
        let (mut b, b_end) = (gt.indptr[i], gt.indptr[i + 1]);
        while a < a_end && b < b_end {
            let (ca, cb) = (graph.indices[a], gt.indices[b]);
            if ca < cb {
                a += 1;
            } else if cb < ca {
                b += 1;
            } else {
                let j = ca as usize;
                let p = partition[j] as u32;
                let w = graph.data[a] * gt.data[b] * inv_sqrt[p as usize];
                match row_weights.iter_mut().find(|(rp, _)| *rp == p) {
                    Some(entry) => entry.1 += w,
                    None => row_weights.push((p, w)),
                }
                a += 1;
                b += 1;
            }
        }
        let total: f32 = row_weights.iter().map(|&(_, w)| w.abs()).sum();
        let out = &mut result[i * c..(i + 1) * c];
        if total > 0.0 {
            for &(p, w) in &row_weights {
                let scale = w / total;
                for (o, &l) in out
                    .iter_mut()
                    .zip(&reduced_layout[p as usize * c..(p as usize + 1) * c])
                {
                    *o += scale * l;
                }
            }
        }
        let p = partition[i] as usize;
        for (o, &l) in out.iter_mut().zip(&reduced_layout[p * c..(p + 1) * c]) {
            *o = (*o + l) / 2.0;
        }
    }

    // scaling * (result - column means)
    let mut col_means = vec![0.0f64; c];
    for row in result.chunks_exact(c) {
        for (m, &v) in col_means.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    for m in col_means.iter_mut() {
        *m /= n as f64;
    }
    for row in result.chunks_exact_mut(c) {
        for (v, &m) in row.iter_mut().zip(&col_means) {
            *v = params.scaling * (*v - m as f32);
        }
    }
    result
}
