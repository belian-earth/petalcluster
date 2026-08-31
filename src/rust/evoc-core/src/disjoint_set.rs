//! Union-find, mirroring evoc's `disjoint_set.py` rank variant.
//!
//! `find` uses the same path-halving walk as the Python. Representative
//! identity matters downstream: `get_cluster_label_vector` looks label
//! representatives up in a map keyed by selected cluster ids, which is only
//! valid because rows are unioned in parent-before-child order — preserve
//! that ordering when calling.

pub struct RankDisjointSet {
    parent: Vec<i32>,
    rank: Vec<i32>,
}

impl RankDisjointSet {
    pub fn new(n_elements: usize) -> Self {
        Self {
            parent: (0..n_elements as i32).collect(),
            rank: vec![0; n_elements],
        }
    }

    pub fn find(&mut self, mut x: i32) -> i32 {
        while self.parent[x as usize] != x {
            let next = self.parent[x as usize];
            self.parent[x as usize] = self.parent[next as usize];
            x = next;
        }
        x
    }

    pub fn union_by_rank(&mut self, x: i32, y: i32) {
        let mut x = self.find(x);
        let mut y = self.find(y);

        if x == y {
            return;
        }
        if self.rank[x as usize] < self.rank[y as usize] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent[y as usize] = x;
        if self.rank[x as usize] == self.rank[y as usize] {
            self.rank[x as usize] += 1;
        }
    }
}
