//! Full-pipeline wall-clock benchmark on external data:
//!   cargo run --release --example bench_full -- <data.bin> <truth.bin> <dims>
use std::collections::HashMap;
use std::time::Instant;

use evoc_core::pipeline::{evoc, EvocParams};

fn ari(a: &[i64], b: &[i64]) -> f64 {
    let n = a.len() as f64;
    let mut c: HashMap<(i64, i64), f64> = HashMap::new();
    let mut r: HashMap<i64, f64> = HashMap::new();
    let mut co: HashMap<i64, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *c.entry((x, y)).or_default() += 1.0;
        *r.entry(x).or_default() += 1.0;
        *co.entry(y).or_default() += 1.0;
    }
    let c2 = |x: f64| x * (x - 1.0) / 2.0;
    let sij: f64 = c.values().map(|&v| c2(v)).sum();
    let sa: f64 = r.values().map(|&v| c2(v)).sum();
    let sb: f64 = co.values().map(|&v| c2(v)).sum();
    let exp = sa * sb / c2(n);
    let mx = (sa + sb) / 2.0;
    if (mx - exp).abs() < 1e-12 { return 1.0; }
    (sij - exp) / (mx - exp)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dims: usize = args[3].parse().unwrap();
    let data: Vec<f32> = std::fs::read(&args[1])
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let truth: Vec<i64> = std::fs::read(&args[2])
        .unwrap()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let n = data.len() / dims;

    let t = Instant::now();
    let res = evoc(&data, dims, &EvocParams::default());
    let dt = t.elapsed().as_secs_f64();
    let best = res
        .layers
        .iter()
        .map(|l| ari(&truth, l))
        .fold(f64::NEG_INFINITY, f64::max);
    let ks: Vec<i64> = res.layers.iter().map(|l| l.iter().max().copied().unwrap_or(-1) + 1).collect();
    println!("rust  n={n:<7} d={dims}: {dt:6.2}s  best_ari={best:.3}  layers k={ks:?}");
}
