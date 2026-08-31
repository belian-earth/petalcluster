//! End-to-end validation of the full pipeline, stochastic stages included.
//!
//! The `*__end_to_end` fixtures record, for embedding-like inputs, the
//! reference's max-over-layers ARI across a 10-seed sweep and two bounds
//! derived from it: a per-seed floor (`min_ari_bound`, the reference's worst
//! seed minus 0.02 — the reference has tail draws of its own) and a mean
//! bound (`mean_ari_bound`, its average minus 0.04 — the check that catches a
//! systematically weaker port). The port must clear the floor on each of the
//! five tested seeds and the mean bound on their average — per-seed, not
//! universal: both the reference and the port are stochastic pipelines with
//! occasional tail draws below their typical score, which is exactly why the
//! mean carries the contract. Same-seed determinism and NN-Descent recall are
//! checked on the same data.

use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

use evoc_core::nndescent::{exact_knn, nn_descent};
use evoc_core::pipeline::{evoc, EvocParams};

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

/// Standard-alphabet base64 decode; enough for the fixture payloads.
fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("bad base64 byte {c}"),
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|&&b| b == b'=').count();
        let mut acc = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            let v = if b == b'=' { 0 } else { val(b) };
            acc |= v << (18 - 6 * i);
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    out
}

struct E2eCase {
    data: Vec<f32>,
    dims: usize,
    truth: Vec<i64>,
    base_min_cluster_size: i64,
    min_ari_bound: f64,
    mean_ari_bound: f64,
}

fn load_e2e(name: &str) -> E2eCase {
    let fx = load(&format!("{name}__end_to_end"));
    let input = &fx["input"];
    let shape = input["shape"].as_array().unwrap();
    let (n, dims) = (
        shape[0].as_u64().unwrap() as usize,
        shape[1].as_u64().unwrap() as usize,
    );
    let bytes = b64_decode(input["data_b64_f32_rowmajor"].as_str().unwrap());
    assert_eq!(bytes.len(), n * dims * 4);
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let truth: Vec<i64> = fx["reference"]["truth"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    E2eCase {
        data,
        dims,
        truth,
        base_min_cluster_size: input["base_min_cluster_size"].as_i64().unwrap(),
        min_ari_bound: fx["reference"]["min_ari_bound"].as_f64().unwrap(),
        mean_ari_bound: fx["reference"]["mean_ari_bound"].as_f64().unwrap(),
    }
}

/// Adjusted Rand index. Noise labels (-1) are treated as ordinary labels,
/// matching how the fixture ARIs were computed with sklearn.
fn ari(a: &[i64], b: &[i64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mut contingency: HashMap<(i64, i64), f64> = HashMap::new();
    let mut rows: HashMap<i64, f64> = HashMap::new();
    let mut cols: HashMap<i64, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *contingency.entry((x, y)).or_default() += 1.0;
        *rows.entry(x).or_default() += 1.0;
        *cols.entry(y).or_default() += 1.0;
    }
    let comb2 = |x: f64| x * (x - 1.0) / 2.0;
    let sum_ij: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_a: f64 = rows.values().map(|&v| comb2(v)).sum();
    let sum_b: f64 = cols.values().map(|&v| comb2(v)).sum();
    let expected = sum_a * sum_b / comb2(n);
    let max_index = (sum_a + sum_b) / 2.0;
    if (max_index - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max_index - expected)
}

fn run_case(name: &str) {
    let case = load_e2e(name);
    let seeds = [1u64, 2, 3, 4, 5];
    let mut bests = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let params = EvocParams {
            base_min_cluster_size: case.base_min_cluster_size,
            seed,
            ..EvocParams::default()
        };
        let result = evoc(&case.data, case.dims, &params);
        assert!(!result.layers.is_empty());
        let best = result
            .layers
            .iter()
            .map(|layer| ari(&case.truth, layer))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            best >= case.min_ari_bound,
            "{name} seed {seed}: best layer ARI {best:.4} below reference floor {:.4} \
             (layers: {:?})",
            case.min_ari_bound,
            result
                .layers
                .iter()
                .map(|l| l.iter().max().copied().unwrap_or(-1) + 1)
                .collect::<Vec<_>>()
        );
        bests.push(best);
    }
    let mean = bests.iter().sum::<f64>() / bests.len() as f64;
    assert!(
        mean >= case.mean_ari_bound,
        "{name}: mean best-layer ARI {mean:.4} over seeds {seeds:?} below reference mean bound \
         {:.4} (per seed: {bests:?})",
        case.mean_ari_bound
    );
}

#[test]
fn end_to_end_topics() {
    run_case("emb_topics");
}

#[test]
fn end_to_end_nested() {
    run_case("emb_nested");
}

#[test]
fn end_to_end_deterministic_per_seed() {
    let case = load_e2e("emb_topics");
    let params = EvocParams {
        base_min_cluster_size: case.base_min_cluster_size,
        seed: 7,
        ..EvocParams::default()
    };
    let a = evoc(&case.data, case.dims, &params);
    let b = evoc(&case.data, case.dims, &params);
    assert_eq!(a.embedding, b.embedding, "embedding must be bitwise reproducible");
    assert_eq!(a.layers, b.layers);
    assert_eq!(a.strengths, b.strengths);
    assert_eq!(a.persistence_scores, b.persistence_scores);
}

#[test]
fn nn_descent_recall() {
    let case = load_e2e("emb_topics");
    let k = 15;
    let approx = nn_descent(&case.data, case.dims, k, 42);
    let exact = exact_knn(&case.data, case.dims, k);
    let mut hits = 0usize;
    let mut total = 0usize;
    for (a_row, e_row) in approx.indices.iter().zip(&exact.indices) {
        for idx in e_row {
            total += 1;
            if a_row.contains(idx) {
                hits += 1;
            }
        }
    }
    let recall = hits as f64 / total as f64;
    assert!(recall >= 0.95, "NN-Descent recall@{k} = {recall:.4}, want >= 0.95");
}
