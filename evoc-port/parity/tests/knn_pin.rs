//! Pins NN-Descent's exact output on the end-to-end fixtures.
//!
//! The recall and ARI tests bound quality; this one bounds *change*. The
//! join is restructured from time to time for memory or speed (it is applied
//! in blocks, for instance), and every such change must be output-preserving.
//! If a change is meant to alter the neighbour graph, update the hashes here
//! deliberately, and re-run the end-to-end suite to re-establish the ARI
//! bounds.

use std::path::PathBuf;

use evoc_core::nndescent::nn_descent;

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a' + 26),
            b'0'..=b'9' => u32::from(c - b'0' + 52),
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

fn load(name: &str) -> (Vec<f32>, usize) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(format!("{name}__end_to_end.json"));
    let fx: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let dims = fx["input"]["shape"][1].as_u64().unwrap() as usize;
    let bytes = b64_decode(fx["input"]["data_b64_f32_rowmajor"].as_str().unwrap());
    let data = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, dims)
}

fn knn_hashes(name: &str) -> (u64, u64) {
    let (data, dims) = load(name);
    let g = nn_descent(&data, dims, 15, 42);
    (
        fnv(g.indices.iter().flatten().flat_map(|v| v.to_le_bytes())),
        fnv(g.distances.iter().flatten().flat_map(|v| v.to_bits().to_le_bytes())),
    )
}

#[test]
fn nn_descent_output_is_pinned() {
    assert_eq!(knn_hashes("emb_topics"), (0x1c7733ec48d7dd8b, 0x8afb4f98b174ee2f), "emb_topics kNN changed");
    assert_eq!(knn_hashes("emb_nested"), (0x18a7eb4c520be115, 0x204c6a4083750aca), "emb_nested kNN changed");
}
