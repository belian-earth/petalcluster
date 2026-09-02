//! Single-linkage tree from a sorted MST — port of `mst_to_linkage_tree` and
//! its `LinkageMergeData` machinery in `cluster_trees.py`.

/// One merge step: `(component_a, component_b, delta, merged_size)`, where
/// `component_a > component_b`, matching the reference's column convention.
pub type LinkageRow = [f64; 4];

struct LinkageMerge {
    parent: Vec<i64>,
    size: Vec<i64>,
    next: i64,
}

impl LinkageMerge {
    fn new(base_size: usize) -> Self {
        let mut size = vec![1i64; 2 * base_size - 1];
        for s in size.iter_mut().skip(base_size) {
            *s = 0;
        }
        Self {
            parent: vec![-1; 2 * base_size - 1],
            size,
            next: base_size as i64,
        }
    }

    fn find(&mut self, mut node: i64) -> i64 {
        let relabel = node;
        while self.parent[node as usize] != -1 && self.parent[node as usize] != node {
            node = self.parent[node as usize];
        }
        self.parent[node as usize] = node;

        // Path compression pass, exactly as the reference does it.
        let mut relabel = relabel;
        while self.parent[relabel as usize] != node {
            let next_relabel = self.parent[relabel as usize];
            self.parent[relabel as usize] = node;
            relabel = next_relabel;
        }
        node
    }

    fn join(&mut self, left: i64, right: i64) {
        self.size[self.next as usize] = self.size[left as usize] + self.size[right as usize];
        self.parent[left as usize] = self.next;
        self.parent[right as usize] = self.next;
        self.next += 1;
    }
}

/// `sorted_mst` rows are `(u, v, weight)` in ascending weight order.
pub fn mst_to_linkage_tree(sorted_mst: &[(u32, u32, f64)]) -> Vec<LinkageRow> {
    let n_samples = sorted_mst.len() + 1;
    let mut merge = LinkageMerge::new(n_samples);
    let mut result = Vec::with_capacity(sorted_mst.len());

    for &(left, right, delta) in sorted_mst {
        let left_component = merge.find(left as i64);
        let right_component = merge.find(right as i64);

        let (a, b) = if left_component > right_component {
            (left_component, right_component)
        } else {
            (right_component, left_component)
        };

        result.push([
            a as f64,
            b as f64,
            delta,
            (merge.size[left_component as usize] + merge.size[right_component as usize]) as f64,
        ]);
        merge.join(left_component, right_component);
    }

    result
}
