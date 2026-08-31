//! A lean axis-aligned kd-tree over f32 points: batch kNN for core distances
//! and the component-aware nearest-neighbour queries Borůvka needs.
//!
//! Same role as the reference's `numba_kdtree.py`, not a line port: bounding
//! rectangles per node, widest-dimension median splits, flat preorder node
//! storage (left child is `parent + 1`, so reverse iteration visits children
//! before parents).

pub struct KdTree {
    pub dims: usize,
    nodes: Vec<Node>,
    /// Permutation of point indices; each node owns a contiguous slice of it.
    pub idx: Vec<u32>,
}

struct Node {
    start: u32,
    end: u32,
    /// 0 for leaves; otherwise the flat index of the right child
    /// (the left child is always `self + 1`).
    right: u32,
    /// Bounding box, `dims` lows then `dims` highs.
    bounds: Box<[f32]>,
}

#[inline]
fn point(data: &[f32], dims: usize, i: u32) -> &[f32] {
    &data[i as usize * dims..(i as usize + 1) * dims]
}

/// Squared Euclidean distance, f32 sequential accumulation (matches the
/// reference's rdist).
#[inline]
pub fn rdist(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Smallest possible rdist from `p` to any point inside `bounds`.
#[inline]
fn min_rdist(bounds: &[f32], dims: usize, p: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for d in 0..dims {
        let lo = bounds[d];
        let hi = bounds[dims + d];
        let v = p[d];
        let excess = if v < lo {
            lo - v
        } else if v > hi {
            v - hi
        } else {
            0.0
        };
        acc += excess * excess;
    }
    acc
}

impl KdTree {
    pub fn build(data: &[f32], dims: usize, leaf_size: usize) -> KdTree {
        let n = data.len() / dims;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / leaf_size.max(1) + 1);
        build_node(data, dims, leaf_size, &mut idx, 0, n, &mut nodes);
        KdTree { dims, nodes, idx }
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Iterate a node's point indices.
    #[inline]
    pub fn node_points(&self, node: usize) -> &[u32] {
        let nd = &self.nodes[node];
        &self.idx[nd.start as usize..nd.end as usize]
    }

    #[inline]
    pub fn children(&self, node: usize) -> Option<(usize, usize)> {
        let r = self.nodes[node].right;
        if r == 0 {
            None
        } else {
            Some((node + 1, r as usize))
        }
    }

    /// Squared distances to the `k` nearest points including the query point
    /// itself, ascending. Used for core distances.
    pub fn knn_rdist(&self, data: &[f32], p: &[f32], k: usize) -> Vec<f32> {
        // k is tiny (min_samples + 1); a sorted insert beats a heap.
        let mut best = vec![f32::INFINITY; k];
        self.knn_recurse(data, p, &mut best, 0);
        best
    }

    fn knn_recurse(&self, data: &[f32], p: &[f32], best: &mut [f32], node: usize) {
        if min_rdist(&self.nodes[node].bounds, self.dims, p) > best[best.len() - 1] {
            return;
        }
        match self.children(node) {
            None => {
                for &j in self.node_points(node) {
                    let d = rdist(p, point(data, self.dims, j));
                    let worst = best.len() - 1;
                    if d < best[worst] {
                        let mut pos = worst;
                        while pos > 0 && best[pos - 1] > d {
                            best[pos] = best[pos - 1];
                            pos -= 1;
                        }
                        best[pos] = d;
                    }
                }
            }
            Some((l, r)) => {
                // Nearer child first for tighter pruning.
                let dl = min_rdist(&self.nodes[l].bounds, self.dims, p);
                let dr = min_rdist(&self.nodes[r].bounds, self.dims, p);
                let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
                self.knn_recurse(data, p, best, first);
                self.knn_recurse(data, p, best, second);
            }
        }
    }

    /// As [`knn_rdist`](Self::knn_rdist), also returning the point indices.
    pub fn knn_rdist_with_idx(&self, data: &[f32], p: &[f32], k: usize) -> Vec<(f32, u32)> {
        let mut best = vec![(f32::INFINITY, u32::MAX); k];
        self.knn_idx_recurse(data, p, &mut best, 0);
        best.retain(|&(d, _)| d.is_finite());
        best
    }

    fn knn_idx_recurse(&self, data: &[f32], p: &[f32], best: &mut [(f32, u32)], node: usize) {
        if min_rdist(&self.nodes[node].bounds, self.dims, p) > best[best.len() - 1].0 {
            return;
        }
        match self.children(node) {
            None => {
                for &j in self.node_points(node) {
                    let d = rdist(p, point(data, self.dims, j));
                    let worst = best.len() - 1;
                    if d < best[worst].0 {
                        let mut pos = worst;
                        while pos > 0 && best[pos - 1].0 > d {
                            best[pos] = best[pos - 1];
                            pos -= 1;
                        }
                        best[pos] = (d, j);
                    }
                }
            }
            Some((l, r)) => {
                let dl = min_rdist(&self.nodes[l].bounds, self.dims, p);
                let dr = min_rdist(&self.nodes[r].bounds, self.dims, p);
                let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
                self.knn_idx_recurse(data, p, best, first);
                self.knn_idx_recurse(data, p, best, second);
            }
        }
    }

    /// Nearest neighbour of `p` outside its own component, under mutual
    /// reachability (squared space). Returns `(rdist, index)`.
    ///
    /// `bound` is the best distance any point of this component has found so
    /// far this round — pruning only, never affects which edge wins.
    #[allow(clippy::too_many_arguments)]
    pub fn component_nn(
        &self,
        data: &[f32],
        p: &[f32],
        my_core: f32,
        my_component: i32,
        core: &[f32],
        point_components: &[i32],
        node_components: &[i32],
        bound: f32,
    ) -> (f32, i64) {
        let mut best = (bound.min(f32::INFINITY), -1i64);
        self.component_recurse(
            data,
            p,
            my_core,
            my_component,
            core,
            point_components,
            node_components,
            0,
            &mut best,
        );
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn component_recurse(
        &self,
        data: &[f32],
        p: &[f32],
        my_core: f32,
        my_component: i32,
        core: &[f32],
        point_components: &[i32],
        node_components: &[i32],
        node: usize,
        best: &mut (f32, i64),
    ) {
        // Whole node inside my component, or provably too far.
        if node_components[node] == my_component
            || min_rdist(&self.nodes[node].bounds, self.dims, p) > best.0
        {
            return;
        }
        match self.children(node) {
            None => {
                for &j in self.node_points(node) {
                    if point_components[j as usize] == my_component {
                        continue;
                    }
                    let cj = core[j as usize];
                    if cj > best.0 {
                        continue; // mutual reachability is at least core[j]
                    }
                    let d = rdist(p, point(data, self.dims, j)).max(my_core).max(cj);
                    if d < best.0 || (d == best.0 && best.1 == -1) {
                        *best = (d, j as i64);
                    }
                }
            }
            Some((l, r)) => {
                let dl = min_rdist(&self.nodes[l].bounds, self.dims, p);
                let dr = min_rdist(&self.nodes[r].bounds, self.dims, p);
                let (first, second) = if dl <= dr { (l, r) } else { (r, l) };
                self.component_recurse(
                    data, p, my_core, my_component, core, point_components,
                    node_components, first, best,
                );
                self.component_recurse(
                    data, p, my_core, my_component, core, point_components,
                    node_components, second, best,
                );
            }
        }
    }
}

fn build_node(
    data: &[f32],
    dims: usize,
    leaf_size: usize,
    idx: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    let mut bounds = vec![f32::INFINITY; 2 * dims].into_boxed_slice();
    for b in bounds[dims..].iter_mut() {
        *b = f32::NEG_INFINITY;
    }
    for &i in &idx[start..end] {
        let pt = point(data, dims, i);
        for d in 0..dims {
            bounds[d] = bounds[d].min(pt[d]);
            bounds[dims + d] = bounds[dims + d].max(pt[d]);
        }
    }

    let me = nodes.len();
    nodes.push(Node {
        start: start as u32,
        end: end as u32,
        right: 0,
        bounds,
    });

    if end - start > leaf_size {
        let split_dim = (0..dims)
            .max_by(|&a, &b| {
                let wa = nodes[me].bounds[dims + a] - nodes[me].bounds[a];
                let wb = nodes[me].bounds[dims + b] - nodes[me].bounds[b];
                wa.partial_cmp(&wb).unwrap()
            })
            .unwrap();
        let mid = (end - start) / 2;
        idx[start..end].select_nth_unstable_by(mid, |&a, &b| {
            point(data, dims, a)[split_dim]
                .partial_cmp(&point(data, dims, b)[split_dim])
                .unwrap()
        });
        build_node(data, dims, leaf_size, idx, start, start + mid, nodes);
        let right = build_node(data, dims, leaf_size, idx, start + mid, end, nodes);
        nodes[me].right = right as u32;
    }
    me
}
