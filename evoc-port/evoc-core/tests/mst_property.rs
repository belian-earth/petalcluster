//! The kd-tree Borůvka and the Prim oracle must agree on the MST weight
//! sequence — the invariant all minimum spanning trees of a graph share —
//! across shapes, sizes, dimensions and degenerate duplicate-heavy data.

use evoc_core::mst::{mutual_reachability_mst, mutual_reachability_mst_brute};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unif(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn random_points(rng: &mut Rng, n: usize, d: usize, clustered: bool) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let centre = if clustered { (i % 5) as f32 * 3.0 } else { 0.0 };
            (0..d).map(|_| centre + rng.unif() * 2.0 - 1.0).collect()
        })
        .collect()
}

fn assert_same_weights(data: &[Vec<f32>], min_samples: usize, what: &str) {
    let fast = mutual_reachability_mst(data, min_samples);
    let brute = mutual_reachability_mst_brute(data, min_samples);
    assert_eq!(fast.len(), brute.len(), "{what}: edge count");
    for (i, (f, b)) in fast.iter().zip(&brute).enumerate() {
        assert!(
            (f.2 - b.2).abs() <= 1e-6 * b.2.abs().max(1.0),
            "{what}: weight[{i}] fast={} brute={}",
            f.2,
            b.2
        );
    }
}

#[test]
fn matches_prim_across_shapes() {
    let mut rng = Rng(0xE0C);
    for &(n, d, clustered, min_samples) in &[
        (50usize, 2usize, false, 5usize),
        (300, 2, true, 5),
        (300, 4, false, 5),
        (500, 4, true, 5),
        (500, 8, true, 10),
        (800, 3, true, 1),
        (200, 4, true, 25),
    ] {
        let data = random_points(&mut rng, n, d, clustered);
        assert_same_weights(&data, min_samples, &format!("n={n} d={d} cl={clustered} ms={min_samples}"));
    }
}

#[test]
fn matches_prim_with_exact_duplicates() {
    let mut rng = Rng(7);
    let mut data = random_points(&mut rng, 240, 4, true);
    for i in 0..30 {
        data[i + 30] = data[i].clone(); // 30 duplicated points, zero distances
    }
    assert_same_weights(&data, 5, "duplicates");
}

#[test]
fn tiny_inputs() {
    let mut rng = Rng(3);
    for n in [2usize, 3, 5, 41] {
        let data = random_points(&mut rng, n, 3, false);
        assert_same_weights(&data, 5.min(n - 1), &format!("tiny n={n}"));
    }
}
