//! An axis-aligned kd-tree over f64 points for exact k-nearest-neighbour
//! search under the Minkowski family of metrics, and under cosine and
//! correlation through a transformed copy of the data.
//!
//! Bounding rectangles per node, widest-dimension median splits, flat
//! preorder node storage (the left child is `parent + 1`). A query descends
//! nearer child first and skips any node whose rectangle cannot hold a
//! point closer than the current `k`-th best.
//!
//! The result is identical to the brute-force scan in `knn.rs`, tie order
//! included. Two things make that hold. Leaf points are scored with the very
//! same `Metric::compute` on the original rows, so distances are
//! bit-identical. And the rectangle bound never exceeds the computed
//! distance of any point inside. For the Minkowski family the bound is
//! computed in the metric's own summation order, term by term no larger,
//! so IEEE rounding (monotone in each operand) keeps it at or below; for
//! cosine the bound comes from a different formula and is slackened by a
//! constant to absorb the rounding gap. Pruning only on a strictly larger
//! bound then never discards a point that ties the current worst, which is
//! exactly the point the index rule might prefer.
//!
//! Cosine distance `1 - cos` between rows `a` and `b` is half the squared
//! Euclidean distance between `a / |a|` and `b / |b|`, so the tree is built
//! on the unit-normalised rows and the Euclidean bound is halved and
//! squared. Correlation is cosine after centring each row on its own mean,
//! so it uses the same tree on centred rows. Canberra and binary have no
//! such bound and stay on the scan.
//!
//! The build is parallel: subtrees above `PARALLEL_MIN` points are built by
//! `rayon::join` into local node vectors and stitched into the preorder
//! layout with an index offset.
//!
//! Three layout choices matter once the tree outgrows L2 cache, which at a
//! million points it does. Nodes are 12 bytes and every bounding rectangle
//! lives in one flat vector indexed by node, so a visit is one predictable
//! read rather than a pointer chase into a separate allocation. The rows
//! are copied into tree order after the build, so scoring a leaf reads
//! contiguous memory instead of one random row per candidate. And the
//! query is generic over the metric's score and bound functions, so the
//! metric is resolved once per query and the inner loops carry no `match`.

use crate::dist::Metric;
use crate::knn::Nearest;
use rayon::prelude::*;

/// Points per leaf. Below this the scan over a leaf is cheaper than the
/// bound computations descending further would save.
const LEAF_SIZE: usize = 16;

/// Subtrees smaller than this are built sequentially; splitting them
/// across threads would cost more in scheduling than it saves.
const PARALLEL_MIN: usize = 4096;

/// Absolute slack subtracted from the cosine bound. The bound and the
/// scored distance are computed by different formulas, each with rounding
/// error of order `p * eps`; cosine distances lie in `[0, 2]`, so this
/// covers rows of tens of thousands of columns with room to spare, and
/// weakens pruning only within that margin of the k-th best distance.
const COSINE_SLACK: f64 = 1e-10;

pub struct KdTree {
    dims: usize,
    nodes: Vec<Node>,
    /// Bounding rectangles, `2 * dims` per node in node order: `dims` lows
    /// then `dims` highs.
    bounds: Vec<f64>,
    /// Permutation of point indices; each node owns a contiguous slice of it.
    idx: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Node {
    start: u32,
    end: u32,
    /// 0 for leaves; otherwise the flat index of the right child.
    right: u32,
}

#[inline(always)]
fn point(data: &[f64], dims: usize, i: u32) -> &[f64] {
    &data[i as usize * dims..(i as usize + 1) * dims]
}

/// Per-dimension distance from `q` to the rectangle, zero inside it.
#[inline(always)]
fn gap(lo: f64, hi: f64, v: f64) -> f64 {
    if v < lo {
        lo - v
    } else if v > hi {
        v - hi
    } else {
        0.0
    }
}

impl Metric {
    /// Whether the kd-tree can bound this metric. Mirrors the R-side check.
    pub fn kd_tree_supported(self) -> bool {
        !matches!(self, Metric::Canberra | Metric::Binary)
    }

    /// The rows the tree is built on: the data itself, or for cosine and
    /// correlation a unit-normalised (and centred) copy. Rows of zero norm
    /// become NaN, which the build tolerates and the scan reports as a
    /// non-finite distance exactly as the brute-force path does.
    pub fn tree_rows<'a>(self, data: &'a [f64], dims: usize) -> std::borrow::Cow<'a, [f64]> {
        match self {
            Metric::Cosine | Metric::Correlation => {
                let centre = matches!(self, Metric::Correlation);
                let mut out = data.to_vec();
                out.par_chunks_mut(dims).for_each(|row| {
                    if centre {
                        let mean = row.iter().sum::<f64>() / dims as f64;
                        row.iter_mut().for_each(|v| *v -= mean);
                    }
                    let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                    row.iter_mut().for_each(|v| *v /= norm);
                });
                std::borrow::Cow::Owned(out)
            }
            _ => std::borrow::Cow::Borrowed(data),
        }
    }

}

/// A metric the tree can search under, as a type so that `KdTree::query`
/// is compiled once per metric with no dispatch in its inner loops.
///
/// `score` is `Metric::compute` itself. `bound` is the smallest possible
/// distance from a tree-space query to any point inside a rectangle,
/// computed term by term in the metric's own summation order; see the
/// module notes.
pub trait TreeMetric: Copy + Sync {
    fn score(self, a: &[f64], b: &[f64]) -> f64;
    /// `rect` is `dims` lows then `dims` highs; `tq` the tree-space query.
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64;
}

#[inline(always)]
fn gaps<'a>(rect: &'a [f64], dims: usize, tq: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    let (lo, hi) = rect.split_at(dims);
    lo.iter().zip(hi).zip(tq).map(|((&l, &h), &v)| gap(l, h, v))
}

#[derive(Clone, Copy)]
pub struct Euclid;
#[derive(Clone, Copy)]
pub struct Manhat;
#[derive(Clone, Copy)]
pub struct Maxim;
#[derive(Clone, Copy)]
pub struct Minkow(pub f64);
/// Cosine or correlation: `Metric::Cosine` or `Metric::Correlation`, scored
/// on the original rows and bounded on the transformed ones.
#[derive(Clone, Copy)]
pub struct Angular(pub Metric);

impl TreeMetric for Euclid {
    #[inline(always)]
    fn score(self, a: &[f64], b: &[f64]) -> f64 {
        Metric::Euclidean.compute(a, b)
    }
    #[inline(always)]
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64 {
        gaps(rect, dims, tq).map(|g| g * g).sum::<f64>().sqrt()
    }
}

impl TreeMetric for Manhat {
    #[inline(always)]
    fn score(self, a: &[f64], b: &[f64]) -> f64 {
        Metric::Manhattan.compute(a, b)
    }
    #[inline(always)]
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64 {
        gaps(rect, dims, tq).sum()
    }
}

impl TreeMetric for Maxim {
    #[inline(always)]
    fn score(self, a: &[f64], b: &[f64]) -> f64 {
        Metric::Maximum.compute(a, b)
    }
    #[inline(always)]
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64 {
        gaps(rect, dims, tq).fold(0.0_f64, f64::max)
    }
}

impl TreeMetric for Minkow {
    #[inline(always)]
    fn score(self, a: &[f64], b: &[f64]) -> f64 {
        Metric::Minkowski(self.0).compute(a, b)
    }
    #[inline(always)]
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64 {
        gaps(rect, dims, tq).map(|g| g.powf(self.0)).sum::<f64>().powf(1.0 / self.0)
    }
}

impl TreeMetric for Angular {
    #[inline(always)]
    fn score(self, a: &[f64], b: &[f64]) -> f64 {
        self.0.compute(a, b)
    }
    // Not clamped at zero: the exact kernel can return a distance a few
    // ulps below zero for parallel rows, and a bound of exactly zero would
    // prune it.
    #[inline(always)]
    fn bound(self, rect: &[f64], dims: usize, tq: &[f64]) -> f64 {
        0.5 * gaps(rect, dims, tq).map(|g| g * g).sum::<f64>() - COSINE_SLACK
    }
}

impl KdTree {
    /// Build over `rows` (`n x dims`, row-major), which for cosine and
    /// correlation are the transformed rows from `Metric::tree_rows`.
    pub fn build(rows: &[f64], dims: usize) -> KdTree {
        let n = rows.len() / dims;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let (nodes, bounds) = build_subtree(rows, dims, &mut idx, 0);
        KdTree {
            dims,
            nodes,
            bounds,
            idx,
        }
    }

    /// `data` copied into tree order: position `t` holds row `idx[t]`, so
    /// the rows of a leaf are contiguous. This is what `query` scores
    /// against. Costs one copy of the data, made in parallel.
    pub fn reorder(&self, data: &[f64]) -> Vec<f64> {
        let dims = self.dims;
        let mut out = vec![0.0; data.len()];
        out.par_chunks_mut(dims)
            .zip(self.idx.par_iter())
            .for_each(|(dst, &i)| dst.copy_from_slice(point(data, dims, i)));
        out
    }

    #[inline(always)]
    fn rect(&self, node: usize) -> &[f64] {
        let w = 2 * self.dims;
        &self.bounds[node * w..(node + 1) * w]
    }

    /// Nearest neighbours of one query, accumulated into `near`.
    ///
    /// `tq` is the query in tree space and `q` in data space (the same
    /// slice except for cosine and correlation); `ordered` is the original
    /// data in tree order from `reorder`, which the leaf points are scored
    /// against. `exclude` is the index of the query itself in a
    /// self-search, or `u32::MAX` when every point is a candidate. `stack`
    /// is scratch, reused across queries. Returns whether every distance
    /// computed was finite.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn query<M: TreeMetric>(
        &self,
        ordered: &[f64],
        m: M,
        tq: &[f64],
        q: &[f64],
        exclude: u32,
        near: &mut Nearest,
        stack: &mut Vec<usize>,
    ) -> bool {
        let dims = self.dims;
        let mut finite = true;
        stack.clear();
        stack.push(0);
        while let Some(node) = stack.pop() {
            // Re-checked on arrival: the k-th best may have tightened since
            // this node was pushed as the farther child.
            if let Some(worst) = near.worst() {
                if m.bound(self.rect(node), dims, tq) > worst {
                    continue;
                }
            }
            let nd = self.nodes[node];
            if nd.right == 0 {
                let (start, end) = (nd.start as usize, nd.end as usize);
                let rows = ordered[start * dims..end * dims].chunks_exact(dims);
                for (&j, b) in self.idx[start..end].iter().zip(rows) {
                    if j == exclude {
                        continue;
                    }
                    let v = m.score(q, b);
                    finite &= v.is_finite();
                    near.offer(v, j);
                }
                continue;
            }
            // Nearer child on top of the stack, so it is searched first.
            let (l, r) = (node + 1, nd.right as usize);
            let dl = m.bound(self.rect(l), dims, tq);
            let dr = m.bound(self.rect(r), dims, tq);
            if dl <= dr {
                stack.push(r);
                stack.push(l);
            } else {
                stack.push(l);
                stack.push(r);
            }
        }
        finite
    }
}

/// Bounding rectangle of the points in `idx`, appended to `out`.
fn push_bounding_box(rows: &[f64], dims: usize, idx: &[u32], out: &mut Vec<f64>) {
    let at = out.len();
    out.resize(at + 2 * dims, 0.0);
    let (lo, hi) = out[at..].split_at_mut(dims);
    lo.fill(f64::INFINITY);
    hi.fill(f64::NEG_INFINITY);
    for &i in idx {
        for (d, &v) in point(rows, dims, i).iter().enumerate() {
            if v < lo[d] {
                lo[d] = v;
            }
            if v > hi[d] {
                hi[d] = v;
            }
        }
    }
}

/// Build the subtree over `idx`, whose first entry sits at absolute
/// position `offset` in the tree's permutation. Returns its nodes in
/// preorder with `right` indices relative to the subtree root at 0, and
/// their rectangles in the same order.
fn build_subtree(rows: &[f64], dims: usize, idx: &mut [u32], offset: usize) -> (Vec<Node>, Vec<f64>) {
    let n = idx.len();
    let mut bounds = Vec::with_capacity(2 * dims * (2 * n / LEAF_SIZE + 1));
    push_bounding_box(rows, dims, idx, &mut bounds);
    let root = Node {
        start: offset as u32,
        end: (offset + n) as u32,
        right: 0,
    };
    if n <= LEAF_SIZE {
        return (vec![root], bounds);
    }

    // Split the widest dimension at its median; both halves stay non-empty
    // because n > LEAF_SIZE >= 2. Coincident points make a zero-width box,
    // which is still split (by index order) so the recursion terminates.
    let mut split = 0;
    let mut width = f64::NEG_INFINITY;
    for d in 0..dims {
        let w = bounds[dims + d] - bounds[d];
        if w > width {
            width = w;
            split = d;
        }
    }
    let mid = n / 2;
    idx.select_nth_unstable_by(mid, |&a, &c| {
        let x = rows[a as usize * dims + split];
        let y = rows[c as usize * dims + split];
        x.total_cmp(&y)
    });

    let (left_idx, right_idx) = idx.split_at_mut(mid);
    let ((left, left_b), (right, right_b)) = if n >= PARALLEL_MIN {
        rayon::join(
            || build_subtree(rows, dims, left_idx, offset),
            || build_subtree(rows, dims, right_idx, offset + mid),
        )
    } else {
        (
            build_subtree(rows, dims, left_idx, offset),
            build_subtree(rows, dims, right_idx, offset + mid),
        )
    };

    // Preorder: root, then the left subtree at 1, then the right subtree
    // after it. Child pointers inside each subtree shift by where it lands;
    // rectangles simply follow in the same order.
    let right_at = 1 + left.len();
    let mut nodes = Vec::with_capacity(1 + left.len() + right.len());
    nodes.push(Node {
        right: right_at as u32,
        ..root
    });
    nodes.extend(left.into_iter().map(|nd| shift(nd, 1)));
    nodes.extend(right.into_iter().map(|nd| shift(nd, right_at)));
    bounds.extend_from_slice(&left_b);
    bounds.extend_from_slice(&right_b);
    (nodes, bounds)
}

#[inline]
fn shift(nd: Node, by: usize) -> Node {
    if nd.right == 0 {
        nd
    } else {
        Node {
            right: nd.right + by as u32,
            ..nd
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::{brute_force, search, Knn, Search};

    fn tree_knn(data: &[f64], n: usize, p: usize, query: Option<&[f64]>, k: usize, metric: Metric) -> Knn {
        search(data, n, p, query, k, metric, Search::KdTree)
    }

    /// Deterministic pseudo-random data with plenty of exact ties: values
    /// are drawn from a small integer grid.
    fn grid_data(n: usize, p: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n * p)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 7) as f64 + 1.0
            })
            .collect()
    }

    fn assert_same(a: &Knn, b: &Knn, label: &str) {
        let first = a.index.iter().zip(&b.index).position(|(x, y)| x != y);
        assert!(first.is_none(), "{label}: index differs first at entry {:?}", first);
        // Bitwise, so a NaN from an undefined correlation compares equal.
        let same_bits = a.dist.iter().zip(&b.dist).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same_bits, "{label}: distances differ");
        assert_eq!(a.finite, b.finite, "{label}: finite flag differs");
    }

    #[test]
    fn tree_matches_brute_force_including_tie_order() {
        let (n, p) = (400, 3);
        let data = grid_data(n, p, 12345);
        let q = grid_data(37, p, 999);
        for metric in [
            Metric::Euclidean,
            Metric::Manhattan,
            Metric::Maximum,
            Metric::Minkowski(3.0),
            Metric::Cosine,
            Metric::Correlation,
        ] {
            for k in [1, 5, 40] {
                let label = format!("{:?} k={k}", metric);
                assert_same(
                    &brute_force(&data, n, p, None, k, metric),
                    &tree_knn(&data, n, p, None, k, metric),
                    &format!("self {label}"),
                );
                assert_same(
                    &brute_force(&data, n, p, Some(&q), k, metric),
                    &tree_knn(&data, n, p, Some(&q), k, metric),
                    &format!("query {label}"),
                );
            }
        }
    }

    #[test]
    fn parallel_build_matches_on_a_large_input() {
        // Well past PARALLEL_MIN, so several joins happen and the stitched
        // preorder layout is exercised.
        let (n, p) = (40_000, 2);
        let data = grid_data(n, p, 42);
        for metric in [Metric::Euclidean, Metric::Cosine] {
            assert_same(
                &brute_force(&data, n, p, None, 6, metric),
                &tree_knn(&data, n, p, None, 6, metric),
                &format!("large {:?}", metric),
            );
        }
    }

    #[test]
    fn tree_handles_k_equal_to_every_other_point() {
        let (n, p) = (50, 2);
        let data = grid_data(n, p, 7);
        assert_same(
            &brute_force(&data, n, p, None, n - 1, Metric::Euclidean),
            &tree_knn(&data, n, p, None, n - 1, Metric::Euclidean),
            "all but self",
        );
    }

    #[test]
    fn cosine_reports_a_zero_row_as_non_finite_on_both_paths() {
        let mut data = grid_data(30, 3, 3);
        data[9..12].fill(0.0);
        let a = brute_force(&data, 30, 3, None, 2, Metric::Cosine);
        let b = tree_knn(&data, 30, 3, None, 2, Metric::Cosine);
        assert!(!a.finite);
        assert!(!b.finite);
    }
}
