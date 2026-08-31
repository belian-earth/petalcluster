//! Node embedding by stochastic gradient descent — port of
//! `node_embedding.py`'s reproducible path (`reproducible_flag=True`, the
//! `node_embedding_epoch_repr` kernel).
//!
//! The reference kernel is block-synchronous by design: within a block the
//! embedding is read-only, every write lands in a per-node `updates` row or a
//! per-edge epoch counter, and the block's updates are applied only after the
//! whole block is processed. Rows of the CSR graph are disjoint per node and
//! `node_order` is a permutation, so all parallel writes are to disjoint
//! locations — the result is bitwise independent of thread scheduling here,
//! which is exactly the property the reference's `_repr` variant exists for.
//! Epoch counters are stored as `AtomicU32` f32 bit patterns purely to make
//! that disjoint mutation safe Rust; every access is `Relaxed` and
//! contention-free.
//!
//! The RNG differs from numpy's, so this stage is validated by the
//! ARI-bounded end-to-end fixtures rather than bitwise.

use crate::graph::Csr;
use crate::rng::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Port of `make_epochs_per_sample`: edges are sampled in proportion to
/// weight, with the heaviest edge sampled every epoch.
pub fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    weights
        .iter()
        .map(|&w| n_epochs as f32 / (n_epochs as f32 * (w / w_max)).max(1.0))
        .collect()
}

#[inline]
fn rdist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

#[inline]
fn load(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Relaxed))
}

#[inline]
fn store(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Relaxed);
}

/// Learn a `n x n_components` embedding (row-major) of the fuzzy graph.
///
/// Mirrors `node_embedding(...)` with `reproducible_flag=True`: per-epoch
/// learning-rate decay `initial_alpha * (1 - n/n_epochs)` applied at epoch
/// end, a repulsion schedule `gamma` rising linearly from 0.5 to 1.5,
/// momentum decay `updates *= (1 - alpha)^2 * 0.5`, and a negative-sample
/// budget stretched by 1.5 on this path.
pub fn node_embedding(
    graph: &Csr,
    n_components: usize,
    n_epochs: usize,
    initial_embedding: Option<Vec<f32>>,
    initial_alpha: f32,
    negative_sample_rate: f32,
    noise_level: f32,
    rng: &mut Rng,
) -> Vec<f32> {
    let n = graph.n;
    let dim = n_components;
    let mut embedding = initial_embedding.unwrap_or_else(|| {
        let mut e = vec![0.0f32; n * dim];
        for v in e.iter_mut() {
            *v = rng.gauss() * 0.25;
        }
        e
    });
    assert_eq!(embedding.len(), n * dim);
    if n_epochs == 0 || graph.data.is_empty() {
        return embedding;
    }

    let epochs_per_sample = make_epochs_per_sample(&graph.data, n_epochs);
    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate * 1.5)
        .collect();
    let epoch_of_next_sample: Vec<AtomicU32> = epochs_per_sample
        .iter()
        .map(|&e| AtomicU32::new(e.to_bits()))
        .collect();
    let epoch_of_next_negative_sample: Vec<AtomicU32> = epochs_per_negative_sample
        .iter()
        .map(|&e| AtomicU32::new(e.to_bits()))
        .collect();

    // Per-epoch RNG values are drawn up front, as the reference draws
    // `rng_val` before its epoch loop.
    let rng_vals: Vec<u32> = (0..n_epochs).map(|_| rng.next_u32() & 0x7FFF_FFFF).collect();

    let mut node_order: Vec<u32> = (0..n as u32).collect();
    let block_size = (n / 8).max(1024);
    let mut updates = vec![0.0f32; n * dim];
    let mut delta = vec![0.0f32; n * dim];
    let mut alpha = initial_alpha;

    for epoch in 0..n_epochs {
        let gamma = if n_epochs > 1 {
            0.5 + epoch as f32 / (n_epochs - 1) as f32
        } else {
            0.5
        };
        let rng_state = rng_vals[epoch] as u64;
        let epoch_f = epoch as f32;

        let mut block_start = 0;
        while block_start < n {
            let block_end = (block_start + block_size).min(n);
            let idxs = &node_order[block_start..block_end];

            // Gradient pass: embedding is frozen; each node writes only its
            // own delta row and its own CSR row's epoch counters.
            let embedding_ref = &embedding;
            idxs.par_iter()
                .zip(delta[block_start * dim..block_end * dim].par_chunks_mut(dim))
                .for_each(|(&from_node, drow)| {
                    drow.fill(0.0);
                    let f = from_node as usize;
                    let current = &embedding_ref[f * dim..(f + 1) * dim];
                    for raw in graph.indptr[f]..graph.indptr[f + 1] {
                        let next_sample = load(&epoch_of_next_sample[raw]);
                        if next_sample > epoch_f {
                            continue;
                        }
                        let to = graph.indices[raw] as usize;
                        let other = &embedding_ref[to * dim..(to + 1) * dim];
                        let dist_squared = rdist(current, other);
                        if dist_squared > 0.0 {
                            let dist = dist_squared.sqrt();
                            let grad_coeff = (-2.0 * noise_level * dist - 2.0)
                                / (2.0 * dist_squared - 0.5 * dist + 1.0);
                            for d in 0..dim {
                                drow[d] += grad_coeff * (current[d] - other[d]) * alpha;
                            }
                        }
                        store(&epoch_of_next_sample[raw], next_sample + epochs_per_sample[raw]);

                        let next_neg = load(&epoch_of_next_negative_sample[raw]);
                        let n_neg_samples =
                            (((epoch_f - next_neg) / epochs_per_negative_sample[raw]) as i64)
                                .max(0);
                        for p in 0..n_neg_samples {
                            let k = ((raw as u64)
                                .wrapping_mul(epoch as u64 + p as u64 + 1)
                                .wrapping_mul(rng_state)
                                % n as u64) as usize;
                            let to = node_order[k] as usize;
                            let other = &embedding_ref[to * dim..(to + 1) * dim];
                            let dist_squared = rdist(current, other);
                            if dist_squared > 1e-2 {
                                let grad_coeff =
                                    gamma * 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared);
                                for d in 0..dim {
                                    let grad_d = (grad_coeff * (current[d] - other[d]))
                                        .clamp(-4.0, 4.0);
                                    drow[d] += grad_d * alpha;
                                }
                            }
                        }
                        store(
                            &epoch_of_next_negative_sample[raw],
                            next_neg + n_neg_samples as f32 * epochs_per_negative_sample[raw],
                        );
                    }
                });

            // Apply pass: fold each node's fresh delta into its momentum row,
            // then move the node by the full (momentum-carrying) update.
            for (slot, &from_node) in idxs.iter().enumerate() {
                let f = from_node as usize;
                let drow = &delta[(block_start + slot) * dim..(block_start + slot + 1) * dim];
                let urow = &mut updates[f * dim..(f + 1) * dim];
                let erow = &mut embedding[f * dim..(f + 1) * dim];
                for d in 0..dim {
                    urow[d] += drow[d];
                    erow[d] += urow[d];
                }
            }

            block_start = block_end;
        }

        let momentum = (1.0 - alpha) * (1.0 - alpha) * 0.5;
        for u in updates.iter_mut() {
            *u *= momentum;
        }
        rng.shuffle(&mut node_order);
        alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);
    }

    embedding
}
