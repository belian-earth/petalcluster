//! Exact k-nearest-neighbour search: a brute-force scan for any metric, and
//! a kd-tree (`kdtree.rs`) for every metric but Canberra and binary.
//!
//! Both paths score candidates with the same metric kernels as `dist.rs`,
//! so the distances returned agree with `shoal_dist()` to the bit, and both
//! keep candidates in the same `(distance, index)` order, so their results
//! are identical. Queries are independent and run in parallel; each keeps
//! only its `k` best candidates, so the memory is `O(m k)` where a distance
//! matrix would be `O(n^2)`.
//!
//! Beyond about ten dimensions the scan beats kd-tree search by a wide
//! margin (at 20,000 points in 16 dimensions, 2.5 s on one thread against
//! 8 s for dbscan's tree), and it is the only option for the metrics a
//! kd-tree cannot bound. In two or three dimensions the tree is the faster
//! by a large factor. The R side picks between them by dimension.

use crate::dist::Metric;
use crate::kdtree::{Angular, Euclid, KdTree, Manhat, Maxim, Minkow, TreeMetric};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Search result, row-major: query `i` owns entries `i * k .. (i + 1) * k`.
pub struct Knn {
    /// One-based data row indices, so they index the caller's matrix as is.
    pub index: Vec<i32>,
    pub dist: Vec<f64>,
    /// Whether every distance computed (not only those kept) was finite.
    pub finite: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Search {
    Brute,
    KdTree,
}

impl Search {
    /// Names are validated on the R side; an unknown name here is a bug.
    pub fn from_name(name: &str) -> Self {
        match name {
            "brute" => Search::Brute,
            "kdtree" => Search::KdTree,
            _ => panic!("Unknown search: {name}"),
        }
    }
}

/// The `k` best candidates seen so far by one query, kept sorted ascending
/// by `(distance, index)`.
///
/// Insertion into a sorted array beats a heap at the sizes `k` takes in
/// practice: the common case is a candidate worse than the current worst,
/// which is one comparison against the last element and no insertion.
pub struct Nearest {
    k: usize,
    items: Vec<(f64, u32)>,
}

impl Nearest {
    pub fn new(k: usize) -> Self {
        Nearest {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// The current `k`-th best distance, once `k` candidates are held.
    /// Until then nothing can be pruned.
    #[inline(always)]
    pub fn worst(&self) -> Option<f64> {
        if self.items.len() == self.k {
            Some(self.items[self.k - 1].0)
        } else {
            None
        }
    }

    /// Strict ordering on `(distance, index)`, so ties are broken by the
    /// lower data row and the result is fully determined by the input.
    /// `total_cmp` gives a total order in which NaN sorts last; a NaN
    /// distance is reported through `Knn::finite` rather than ranked.
    #[inline(always)]
    fn before(a: &(f64, u32), b: &(f64, u32)) -> bool {
        match a.0.total_cmp(&b.0) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => a.1 < b.1,
        }
    }

    #[inline(always)]
    pub fn offer(&mut self, d: f64, j: u32) {
        // `+ 0.0` folds a negative zero into a positive one; `total_cmp`
        // would otherwise rank -0.0 before 0.0 rather than by index.
        let cand = (d + 0.0, j);
        if self.items.len() == self.k {
            if !Self::before(&cand, &self.items[self.k - 1]) {
                return;
            }
            self.items.pop();
        }
        let pos = self.items.partition_point(|e| Self::before(e, &cand));
        self.items.insert(pos, cand);
    }

    fn clear(&mut self) {
        self.items.clear();
    }

    fn write(&self, idx_row: &mut [i32], dist_row: &mut [f64]) {
        for (slot, &(d, j)) in self.items.iter().enumerate() {
            idx_row[slot] = j as i32 + 1;
            dist_row[slot] = d;
        }
    }
}

/// Exact `k` nearest neighbours of each query row among the rows of `data`.
///
/// `data` is `n x p` row-major. `query`, when given, is `m x p` row-major
/// and every data row is a candidate for it; when `None` the data rows are
/// their own queries and each row is excluded from its own neighbours.
/// `k` must be at most `n` (query given) or `n - 1` (self search), and a
/// kd-tree search needs a metric it can bound; the R side enforces both.
pub fn search(
    data: &[f64],
    n: usize,
    p: usize,
    query: Option<&[f64]>,
    k: usize,
    metric: Metric,
    search: Search,
) -> Knn {
    debug_assert_eq!(data.len(), n * p);
    match search {
        Search::Brute => brute_force(data, n, p, query, k, metric),
        Search::KdTree => match metric {
            Metric::Euclidean => tree_search(data, n, p, query, k, metric, Euclid),
            Metric::Manhattan => tree_search(data, n, p, query, k, metric, Manhat),
            Metric::Maximum => tree_search(data, n, p, query, k, metric, Maxim),
            Metric::Minkowski(q) => tree_search(data, n, p, query, k, metric, Minkow(q)),
            Metric::Cosine | Metric::Correlation => {
                tree_search(data, n, p, query, k, metric, Angular(metric))
            }
            Metric::Canberra | Metric::Binary => panic!("kd-tree search on an unsupported metric"),
        },
    }
}

/// The tree: built once, then every query descends it. Compiled once per
/// `TreeMetric`, so the metric is resolved by the `match` in `search`.
#[inline(always)]
fn tree_search<M: TreeMetric>(
    data: &[f64],
    n: usize,
    p: usize,
    query: Option<&[f64]>,
    k: usize,
    metric: Metric,
    m: M,
) -> Knn {
    // Cosine and correlation search a transformed copy; every other metric
    // borrows the data as is.
    let rows = metric.tree_rows(data, p);
    let tree = KdTree::build(&rows, p);
    let ordered = tree.reorder(data);
    let tree_queries = query.map(|q| metric.tree_rows(q, p));
    let tq_all: &[f64] = tree_queries.as_deref().unwrap_or(&rows);
    per_query(data, n, p, query, k, |i, q, exclude_self, near, stack| {
        let tq = &tq_all[i * p..(i + 1) * p];
        let exclude = if exclude_self { i as u32 } else { u32::MAX };
        tree.query(&ordered, m, tq, q, exclude, near, stack)
    })
}

/// The scan: every query against every data row.
pub fn brute_force(data: &[f64], n: usize, p: usize, query: Option<&[f64]>, k: usize, metric: Metric) -> Knn {
    // Dispatch on the metric once, so each variant gets its own
    // monomorphised inner loop instead of a match per pair.
    match metric {
        Metric::Euclidean => scan(data, n, p, query, k, |a, b| Metric::Euclidean.compute(a, b)),
        Metric::Maximum => scan(data, n, p, query, k, |a, b| Metric::Maximum.compute(a, b)),
        Metric::Manhattan => scan(data, n, p, query, k, |a, b| Metric::Manhattan.compute(a, b)),
        Metric::Canberra => scan(data, n, p, query, k, |a, b| Metric::Canberra.compute(a, b)),
        Metric::Binary => scan(data, n, p, query, k, |a, b| Metric::Binary.compute(a, b)),
        Metric::Minkowski(q) => scan(data, n, p, query, k, |a, b| Metric::Minkowski(q).compute(a, b)),
        Metric::Cosine => scan(data, n, p, query, k, |a, b| Metric::Cosine.compute(a, b)),
        Metric::Correlation => scan(data, n, p, query, k, |a, b| Metric::Correlation.compute(a, b)),
    }
}

#[inline(always)]
fn scan<F>(data: &[f64], n: usize, p: usize, query: Option<&[f64]>, k: usize, f: F) -> Knn
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    per_query(data, n, p, query, k, |i, a, exclude_self, near, _| {
        let mut finite = true;
        // Iterators rather than indexing: no bounds check per pair.
        for (j, b) in data.chunks_exact(p).enumerate() {
            if exclude_self && j == i {
                continue;
            }
            let v = f(a, b);
            finite &= v.is_finite();
            near.offer(v, j as u32);
        }
        finite
    })
}

/// Run `one(i, q, exclude_self, near, scratch)` for every query row in
/// parallel, writing each row's neighbours into the result. `one` returns
/// whether the distances it computed were all finite. `scratch` is a
/// per-thread vector the tree search uses as its stack.
#[inline(always)]
fn per_query<F>(data: &[f64], n: usize, p: usize, query: Option<&[f64]>, k: usize, one: F) -> Knn
where
    F: Fn(usize, &[f64], bool, &mut Nearest, &mut Vec<usize>) -> bool + Sync,
{
    let (queries, exclude_self) = match query {
        Some(q) => (q, false),
        None => (data, true),
    };
    let m = if p == 0 { 0 } else { queries.len() / p };
    debug_assert!(k <= n.saturating_sub(usize::from(exclude_self)));

    let mut index = vec![0i32; m * k];
    let mut dist = vec![0f64; m * k];

    let finite = index
        .par_chunks_mut(k)
        .zip(dist.par_chunks_mut(k))
        .enumerate()
        .map_init(
            || (Nearest::new(k), Vec::with_capacity(64)),
            |(near, scratch), (i, (idx_row, dist_row))| {
                near.clear();
                let finite = one(i, &queries[i * p..(i + 1) * p], exclude_self, near, scratch);
                near.write(idx_row, dist_row);
                finite
            },
        )
        .reduce(|| true, |x, y| x && y);

    Knn { index, dist, finite }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<f64> {
        // Six points in the plane, with three coincident at the origin.
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 5.0, 5.0]
    }

    #[test]
    fn self_search_excludes_the_row_and_breaks_ties_by_index() {
        let data = grid();
        for s in [Search::Brute, Search::KdTree] {
            let r = search(&data, 6, 2, None, 3, Metric::Euclidean, s);
            assert!(r.finite);
            // Row 1 (origin): the two other origin rows, then either unit
            // point (row 4 first, by index).
            assert_eq!(&r.index[0..3], &[2, 3, 4]);
            assert_eq!(&r.dist[0..3], &[0.0, 0.0, 1.0]);
            // Row 6: row 4 at sqrt(41), then rows 1 and 2 at sqrt(50), by index.
            assert_eq!(&r.index[15..18], &[4, 1, 2]);
        }
    }

    #[test]
    fn query_search_keeps_the_identical_row() {
        let data = grid();
        let q = [1.0, 0.0];
        for s in [Search::Brute, Search::KdTree] {
            let r = search(&data, 6, 2, Some(&q), 2, Metric::Euclidean, s);
            assert_eq!(r.index, vec![4, 1]);
            assert_eq!(r.dist, vec![0.0, 1.0]);
        }
    }

    #[test]
    fn non_finite_distances_are_reported_even_when_not_kept() {
        // Correlation is undefined for a constant row; that row sorts last
        // and is not among the neighbours, but the flag still trips.
        let data = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.5, 3.0, 1.0, 2.0, 5.0, 5.0, 5.0];
        let r = brute_force(&data, 4, 3, None, 1, Metric::Correlation);
        assert!(!r.finite);
    }
}
