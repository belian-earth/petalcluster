//! NN-Descent approximate kNN for unit-normalised float embeddings — the role
//! of the reference's `float_nndescent.py`.
//!
//! Behaviour-faithful rather than line-faithful: NN-Descent is stochastic and
//! its output is validated by recall and by the ARI-bounded end-to-end
//! fixtures, so this is the standard algorithm — random initialisation, then
//! local joins over sampled candidates with the new/old split of Dong et al.
//! (a pair is joined only when at least one side is a newly discovered
//! neighbour, which is what makes later iterations cheap) — with the
//! reference's metric and parameters: distance is the negated dot product
//! internally (input must be L2-normalised), reported as `max(-log2(dot), 0)`;
//! iterations default to `max(5, round(log2 n))`; `max_candidates =
//! min(60, 1.5k)`; early exit when an iteration changes fewer than
//! `delta * n * k` entries.
//!
//! Proposed neighbour updates are computed in parallel but applied
//! sequentially in point order, so the result is deterministic for a given
//! seed regardless of thread count.

use crate::rng::Rng;
use rayon::prelude::*;

const EXP_NEG_INF: f32 = 1e-8; // similarity floor, as the reference uses

/// Eight-lane accumulation so LLVM can vectorise; a single f32 accumulator
/// would pin the reduction to strict serial order and scalar code.
#[inline]
fn neg_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let (xa, xb) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for l in 0..8 {
            lanes[l] += xa[l] * xb[l];
        }
    }
    let mut acc = lanes.iter().sum::<f32>();
    for (&x, &y) in a[chunks * 8..].iter().zip(&b[chunks * 8..]) {
        acc += x * y;
    }
    if acc > 0.0 {
        -acc
    } else {
        -EXP_NEG_INF
    }
}

#[derive(Clone, Copy)]
struct Entry {
    d: f32,
    idx: i32,
    new: bool,
}

/// Per-point neighbour list kept sorted ascending by distance (k is small).
struct Heap {
    k: usize,
    entries: Vec<Entry>,
}

impl Heap {
    fn new(n: usize, k: usize) -> Self {
        Heap {
            k,
            entries: vec![Entry { d: f32::INFINITY, idx: -1, new: false }; n * k],
        }
    }

    #[inline]
    fn row(&self, i: usize) -> &[Entry] {
        &self.entries[i * self.k..(i + 1) * self.k]
    }

    #[inline]
    fn row_mut(&mut self, i: usize) -> &mut [Entry] {
        &mut self.entries[i * self.k..(i + 1) * self.k]
    }

    /// Insert unless present; returns true if the row changed.
    fn push(row: &mut [Entry], d: f32, idx: i32) -> bool {
        let worst = row.len() - 1;
        if d >= row[worst].d {
            return false;
        }
        if row.iter().any(|e| e.idx == idx) {
            return false;
        }
        let mut pos = worst;
        while pos > 0 && row[pos - 1].d > d {
            row[pos] = row[pos - 1];
            pos -= 1;
        }
        row[pos] = Entry { d, idx, new: true };
        true
    }
}

pub struct KnnGraph {
    /// `n x k`, self first; -1 padding never occurs for n > k.
    pub indices: Vec<Vec<i64>>,
    /// Transformed distances `max(-log2(dot), 0)`, matching the reference.
    pub distances: Vec<Vec<f32>>,
}

/// Deduplicate, then cap by a deterministic shuffle. Kept sorted so the join
/// loop and the explored-marking pass can binary search.
fn sample_candidates(list: &mut Vec<i32>, cap: usize, rng: &mut Rng) {
    list.sort_unstable();
    list.dedup();
    if list.len() > cap {
        rng.shuffle(list);
        list.truncate(cap);
        list.sort_unstable();
    }
}

/// `data` must be row-major unit vectors.
pub fn nn_descent(data: &[f32], dims: usize, k: usize, seed: u64) -> KnnGraph {
    let n = data.len() / dims;
    let point = |i: usize| &data[i * dims..(i + 1) * dims];
    let n_iters = ((n as f32).log2().round() as usize).max(5);
    let max_candidates = (((k as f32) * 1.5) as usize).min(60).max(1);
    let delta = 0.001f32;

    let mut heap = Heap::new(n, k);

    // Random initialisation, deterministic per point. Every entry starts new.
    (0..n)
        .into_par_iter()
        .zip(heap.entries.par_chunks_mut(k))
        .for_each(|(i, row)| {
            let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0xA24BAED4963EE407));
            Heap::push(row, neg_dot(point(i), point(i)), i as i32);
            for _ in 0..k * 2 {
                let j = rng.below(n);
                if j != i {
                    Heap::push(row, neg_dot(point(i), point(j)), j as i32);
                }
            }
        });

    for iter in 0..n_iters {
        // Candidate sampling with the new/old split: every edge of the
        // current graph feeds both endpoints' candidate lists, partitioned by
        // whether the heap entry has been through a join before.
        let mut new_cand: Vec<Vec<i32>> = vec![Vec::new(); n];
        let mut old_cand: Vec<Vec<i32>> = vec![Vec::new(); n];
        for i in 0..n {
            for e in heap.row(i) {
                if e.idx < 0 || e.idx as usize == i {
                    continue;
                }
                let j = e.idx as usize;
                if e.new {
                    new_cand[i].push(e.idx);
                    new_cand[j].push(i as i32);
                } else {
                    old_cand[i].push(e.idx);
                    old_cand[j].push(i as i32);
                }
            }
        }
        for i in 0..n {
            let mut rng = Rng::new(
                seed ^ (iter as u64).wrapping_mul(0x9E6C63D0876A9F4B)
                    ^ (i as u64).wrapping_mul(0xC2B2AE3D27D4EB4F),
            );
            sample_candidates(&mut new_cand[i], max_candidates, &mut rng);
            sample_candidates(&mut old_cand[i], max_candidates, &mut rng);
        }
        // Entries sampled as new candidates count as explored from now on.
        for i in 0..n {
            let cand = &new_cand[i];
            for e in heap.row_mut(i) {
                if e.new && cand.binary_search(&e.idx).is_ok() {
                    e.new = false;
                }
            }
        }

        // Local joins: new x new and new x old pairs of each point's
        // candidates may improve either side's list. Compute proposals in
        // parallel, apply sequentially in index order for determinism.
        // Proposals no better than the target's worst neighbour *at iteration
        // start* are dropped in the parallel pass — the snapshot makes the
        // filter deterministic, and the sequential push re-checks against the
        // live row anyway.
        let worst_at_start: Vec<f32> = (0..n)
            .map(|i| heap.row(i)[k - 1].d)
            .collect();
        let proposals: Vec<Vec<(u32, f32, i32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let news = &new_cand[i];
                let olds = &old_cand[i];
                let mut out = Vec::new();
                let propose = |a: i32, d: f32, b: i32, out: &mut Vec<(u32, f32, i32)>| {
                    if d < worst_at_start[a as usize] {
                        out.push((a as u32, d, b));
                    }
                };
                for (a_pos, &a) in news.iter().enumerate() {
                    for &b in &news[a_pos + 1..] {
                        let d = neg_dot(point(a as usize), point(b as usize));
                        propose(a, d, b, &mut out);
                        propose(b, d, a, &mut out);
                    }
                    for &b in olds.iter() {
                        if a == b {
                            continue;
                        }
                        let d = neg_dot(point(a as usize), point(b as usize));
                        propose(a, d, b, &mut out);
                        propose(b, d, a, &mut out);
                    }
                }
                out
            })
            .collect();

        let mut n_changed = 0usize;
        for plist in &proposals {
            for &(tgt, d, idx) in plist {
                if Heap::push(heap.row_mut(tgt as usize), d, idx) {
                    n_changed += 1;
                }
            }
        }

        if (n_changed as f32) < delta * (n * k) as f32 {
            break;
        }
    }

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let row = heap.row(i);
        indices.push(row.iter().map(|e| e.idx as i64).collect());
        distances.push(
            row.iter()
                .map(|e| {
                    // d = -similarity; distance = max(-log2(similarity), 0)
                    (-(-e.d).log2()).max(0.0)
                })
                .collect(),
        );
    }
    KnnGraph { indices, distances }
}

/// Exact brute-force variant with the same metric — recall oracle for tests.
pub fn exact_knn(data: &[f32], dims: usize, k: usize) -> KnnGraph {
    let n = data.len() / dims;
    let point = |i: usize| &data[i * dims..(i + 1) * dims];
    let rows: Vec<(Vec<i64>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut all: Vec<(f32, i64)> = (0..n)
                .map(|j| (neg_dot(point(i), point(j)), j as i64))
                .collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            all.truncate(k);
            (
                all.iter().map(|&(_, j)| j).collect(),
                all.iter().map(|&(d, _)| (-(-d).log2()).max(0.0)).collect(),
            )
        })
        .collect();
    KnnGraph {
        indices: rows.iter().map(|r| r.0.clone()).collect(),
        distances: rows.iter().map(|r| r.1.clone()).collect(),
    }
}
