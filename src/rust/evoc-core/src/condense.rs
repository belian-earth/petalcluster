//! Condensed tree construction and cluster extraction — ports of
//! `condense_tree`, `extract_leaves`, `get_cluster_label_vector` and
//! `get_point_membership_strength_vector` from `cluster_trees.py`.

use crate::disjoint_set::RankDisjointSet;
use crate::linkage::LinkageRow;
use std::collections::{HashMap, HashSet};

/// Column-oriented condensed tree, same layout as the reference's namedtuple.
/// `lambda_val` is f32 there, so it is f32 here.
#[derive(Clone, Debug, Default)]
pub struct CondensedTree {
    pub parent: Vec<i64>,
    pub child: Vec<i64>,
    pub lambda_val: Vec<f32>,
    pub child_size: Vec<i64>,
}

impl CondensedTree {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Rows where the mask is true — `mask_condensed_tree` in the reference.
    pub fn masked(&self, mask: &[bool]) -> CondensedTree {
        fn filt<T: Copy>(v: &[T], mask: &[bool]) -> Vec<T> {
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(&x, _)| x)
                .collect()
        }
        CondensedTree {
            parent: filt(&self.parent, mask),
            child: filt(&self.child, mask),
            lambda_val: filt(&self.lambda_val, mask),
            child_size: filt(&self.child_size, mask),
        }
    }
}

/// Level-order traversal of the single-linkage hierarchy from `bfs_root`.
fn bfs_from_hierarchy(hierarchy: &[LinkageRow], bfs_root: i64, num_points: i64) -> Vec<i64> {
    let mut to_process = vec![bfs_root];
    let mut result = Vec::new();

    while !to_process.is_empty() {
        result.extend_from_slice(&to_process);
        let mut next = Vec::new();
        for &n in &to_process {
            if n >= num_points {
                let i = (n - num_points) as usize;
                next.push(hierarchy[i][0] as i64);
                next.push(hierarchy[i][1] as i64);
            }
        }
        to_process = next;
    }
    result
}

struct CondenseState {
    parents: Vec<i64>,
    children: Vec<i64>,
    lambdas: Vec<f32>,
    sizes: Vec<i64>,
    ignore: Vec<bool>,
    idx: usize,
}

/// Fold every point beneath `branch_node` up into `parent_node` at
/// `lambda_value`, marking traversed sub-clusters as dead.
fn eliminate_branch(
    st: &mut CondenseState,
    branch_node: i64,
    parent_node: i64,
    lambda_value: f32,
    hierarchy: &[LinkageRow],
    num_points: i64,
) {
    if branch_node < num_points {
        st.parents[st.idx] = parent_node;
        st.children[st.idx] = branch_node;
        st.lambdas[st.idx] = lambda_value;
        st.idx += 1;
    } else {
        for sub_node in bfs_from_hierarchy(hierarchy, branch_node, num_points) {
            if sub_node < num_points {
                st.children[st.idx] = sub_node;
                st.parents[st.idx] = parent_node;
                st.lambdas[st.idx] = lambda_value;
                st.idx += 1;
            } else {
                st.ignore[sub_node as usize] = true;
            }
        }
    }
}

pub fn condense_tree(hierarchy: &[LinkageRow], min_cluster_size: i64) -> CondensedTree {
    let root = 2 * hierarchy.len() as i64;
    let num_points = hierarchy.len() as i64 + 1;
    let mut next_label = num_points + 1;

    let node_list = bfs_from_hierarchy(hierarchy, root, num_points);

    let mut relabel = vec![0i64; root as usize + 1];
    relabel[root as usize] = num_points;

    let mut st = CondenseState {
        parents: vec![1; root as usize],
        children: vec![0; root as usize],
        lambdas: vec![0.0; root as usize],
        sizes: vec![1; root as usize],
        ignore: vec![false; root as usize + 1],
        idx: 0,
    };

    for node in node_list {
        if st.ignore[node as usize] || node < num_points {
            continue;
        }

        let parent_node = relabel[node as usize];
        let row = hierarchy[(node - num_points) as usize];
        let left = row[0] as i64;
        let right = row[1] as i64;
        let d = row[2];
        let lambda_value = if d > 0.0 { (1.0 / d) as f32 } else { f32::INFINITY };

        let left_count = if left >= num_points {
            hierarchy[(left - num_points) as usize][3] as i64
        } else {
            1
        };
        let right_count = if right >= num_points {
            hierarchy[(right - num_points) as usize][3] as i64
        } else {
            1
        };

        // Branch order mirrors the reference (it is performance-ordered there).
        if left < num_points && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            st.parents[st.idx] = parent_node;
            st.children[st.idx] = left;
            st.lambdas[st.idx] = lambda_value;
            st.idx += 1;
        } else if left_count < min_cluster_size && right_count >= min_cluster_size {
            relabel[right as usize] = parent_node;
            eliminate_branch(&mut st, left, parent_node, lambda_value, hierarchy, num_points);
        } else if left_count >= min_cluster_size && right_count < min_cluster_size {
            relabel[left as usize] = parent_node;
            eliminate_branch(&mut st, right, parent_node, lambda_value, hierarchy, num_points);
        } else if left_count < min_cluster_size && right_count < min_cluster_size {
            eliminate_branch(&mut st, left, parent_node, lambda_value, hierarchy, num_points);
            eliminate_branch(&mut st, right, parent_node, lambda_value, hierarchy, num_points);
        } else {
            relabel[left as usize] = next_label;
            st.parents[st.idx] = parent_node;
            st.children[st.idx] = next_label;
            st.lambdas[st.idx] = lambda_value;
            st.sizes[st.idx] = left_count;
            next_label += 1;
            st.idx += 1;

            relabel[right as usize] = next_label;
            st.parents[st.idx] = parent_node;
            st.children[st.idx] = next_label;
            st.lambdas[st.idx] = lambda_value;
            st.sizes[st.idx] = right_count;
            next_label += 1;
            st.idx += 1;
        }
    }

    let idx = st.idx;
    CondensedTree {
        parent: st.parents[..idx].to_vec(),
        child: st.children[..idx].to_vec(),
        lambda_val: st.lambdas[..idx].to_vec(),
        child_size: st.sizes[..idx].to_vec(),
    }
}

/// Clusters that never split into further clusters.
pub fn extract_leaves(tree: &CondensedTree) -> Vec<i64> {
    if tree.is_empty() {
        return Vec::new();
    }
    let n_nodes = *tree.parent.iter().max().unwrap() + 1;
    let n_points = *tree.parent.iter().min().unwrap();

    let mut leaf = vec![true; n_nodes as usize];
    for l in leaf.iter_mut().take(n_points as usize) {
        *l = false;
    }
    for (&parent, &child_size) in tree.parent.iter().zip(&tree.child_size) {
        if child_size > 1 {
            leaf[parent as usize] = false;
        }
    }
    (0..n_nodes).filter(|&i| leaf[i as usize]).collect()
}

fn single_cluster_label_vector(
    tree: &CondensedTree,
    cluster: i64,
    cluster_selection_epsilon: f64,
    n_samples: usize,
) -> Vec<i64> {
    if tree.is_empty() {
        return vec![-1; n_samples];
    }
    let mut result = vec![-1i64; n_samples];
    let max_lambda = tree
        .parent
        .iter()
        .zip(&tree.lambda_val)
        .filter(|(&p, _)| p == cluster)
        .map(|(_, &l)| l)
        .fold(f32::NEG_INFINITY, f32::max);

    for i in 0..tree.len() {
        let n = tree.child[i];
        let cur_lambda = tree.lambda_val[i];
        if cluster_selection_epsilon > 0.0 {
            result[n as usize] = if f64::from(cur_lambda) >= 1.0 / cluster_selection_epsilon {
                0
            } else {
                -1
            };
        } else if cur_lambda >= max_lambda {
            result[n as usize] = 0;
        }
    }
    result
}

/// Assign each sample to one of `clusters` (or -1) — the union-find sweep from
/// the reference. Row order must be parent-before-child, which
/// `condense_tree` guarantees; representatives are then exactly the selected
/// cluster ids.
pub fn get_cluster_label_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    cluster_selection_epsilon: f64,
    n_samples: usize,
) -> Vec<i64> {
    if clusters.len() == 1 {
        return single_cluster_label_vector(tree, clusters[0], cluster_selection_epsilon, n_samples);
    }
    if tree.is_empty() {
        return vec![-1; n_samples];
    }

    let root_cluster = *tree.parent.iter().min().unwrap();
    let mut result = vec![-1i64; n_samples];

    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let label_map: HashMap<i64, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(n, &c)| (c, n as i64))
        .collect();

    let max_parent = *tree.parent.iter().max().unwrap();
    let max_child = *tree.child.iter().max().unwrap();
    let mut ds = RankDisjointSet::new((max_parent.max(max_child) + 1) as usize);
    let cluster_set: HashSet<i64> = clusters.iter().copied().collect();

    for n in 0..tree.len() {
        let child = tree.child[n];
        let parent = tree.parent[n];
        if !cluster_set.contains(&child) {
            ds.union_by_rank(parent as i32, child as i32);
        }
    }

    for (n, r) in result.iter_mut().enumerate() {
        let cluster = i64::from(ds.find(n as i32));
        *r = if cluster <= root_cluster {
            -1
        } else {
            label_map[&cluster]
        };
    }
    result
}

/// Per-cluster death lambdas: the largest lambda of any singleton child.
fn max_lambdas(tree: &CondensedTree, clusters: &HashSet<i64>) -> HashMap<i64, f32> {
    let mut result: HashMap<i64, f32> = clusters.iter().map(|&c| (c, 0.0)).collect();
    for n in 0..tree.len() {
        let cluster = tree.parent[n];
        if tree.child_size[n] == 1 {
            if let Some(v) = result.get_mut(&cluster) {
                *v = v.max(tree.lambda_val[n]);
            }
        }
    }
    result
}

pub fn get_point_membership_strength_vector(
    tree: &CondensedTree,
    clusters: &[i64],
    labels: &[i64],
) -> Vec<f32> {
    let mut result = vec![0.0f32; labels.len()];
    let cluster_set: HashSet<i64> = clusters.iter().copied().collect();
    let deaths = max_lambdas(tree, &cluster_set);
    let root_cluster = *tree.parent.iter().min().unwrap();

    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort_unstable();

    for n in 0..tree.len() {
        let point = tree.child[n];
        if point >= root_cluster || labels[point as usize] < 0 {
            continue;
        }
        let cluster = sorted_clusters[labels[point as usize] as usize];
        let max_lambda = deaths[&cluster];
        if max_lambda == 0.0 || !tree.lambda_val[n].is_finite() {
            result[point as usize] = 1.0;
        } else {
            let lambda_val = tree.lambda_val[n].min(max_lambda);
            result[point as usize] = lambda_val / max_lambda;
        }
    }
    result
}
