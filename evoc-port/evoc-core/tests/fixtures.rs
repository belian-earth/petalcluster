//! Parity tests against fixtures generated from the Python reference
//! (see ../generate_fixtures.py). Integer outputs must match exactly; f32
//! outputs are compared at 1e-5 relative (the reference runs them under numba
//! fastmath); f64 pass-through values at 1e-12.

use serde_json::Value;
use std::path::PathBuf;

use evoc_core::condense::{
    condense_tree, extract_leaves, get_cluster_label_vector,
    get_point_membership_strength_vector, CondensedTree,
};
use evoc_core::graph::neighbor_graph_matrix;
use evoc_core::layers::{cluster_layers_from_mst, LayerParams};
use evoc_core::linkage::mst_to_linkage_tree;
use evoc_core::mst::mutual_reachability_mst;
use evoc_core::persistence::{
    compute_total_persistence, find_peaks, min_cluster_size_barcode, select_diverse_peaks,
};

const CASES: &[&str] = &["tiny", "blobs2", "blobs3_noise", "uniform", "dupes", "nested"];

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn scalar_f64(x: &Value) -> f64 {
    match x {
        Value::String(s) => match s.as_str() {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => panic!("unexpected float sentinel {other:?}"),
        },
        _ => x.as_f64().unwrap(),
    }
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(scalar_f64).collect()
}

fn f32s(v: &Value) -> Vec<f32> {
    f64s(v).into_iter().map(|x| x as f32).collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn mat_f32(v: &Value) -> Vec<Vec<f32>> {
    v.as_array().unwrap().iter().map(f32s).collect()
}

fn mat_f64(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64s).collect()
}

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let scale = w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol * scale || (g.is_infinite() && w.is_infinite()),
            "{what}[{i}]: got {g}, want {w}"
        );
    }
}

fn mst_rows(v: &Value) -> Vec<(u32, u32, f64)> {
    mat_f64(v)
        .into_iter()
        .map(|r| (r[0] as u32, r[1] as u32, r[2]))
        .collect()
}

// ---------------------------------------------------------------------- S4

#[test]
fn boruvka_mst_matches() {
    // All minimum spanning trees of a graph share the same sorted weight
    // sequence, and mutual-reachability graphs are heavily degenerate (many
    // pairs weigh exactly a point's core distance), so distinct equally-minimal
    // trees are common. The weight multiset is therefore the exact MST
    // invariant to test; endpoints are not.
    for case in CASES {
        let fx = load(&format!("{case}__boruvka_mst"));
        let embedding = mat_f32(&fx["input"]["embedding"]);
        let min_samples = fx["input"]["min_samples"].as_u64().unwrap() as usize;
        let want = mst_rows(&fx["output"]["sorted_mst"]);

        let got = mutual_reachability_mst(&embedding, min_samples);
        assert_eq!(got.len(), want.len(), "{case}: edge count");

        let mut gw: Vec<f64> = got.iter().map(|e| e.2).collect();
        let mut ww: Vec<f64> = want.iter().map(|e| e.2).collect();
        gw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ww.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (i, (g, w)) in gw.iter().zip(&ww).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "{case}: weight[{i}] got {g} want {w}"
            );
        }
    }
}

// ---------------------------------------------------------------------- S5

#[test]
fn linkage_matches() {
    for case in CASES {
        let fx = load(&format!("{case}__linkage"));
        let sorted_mst = mst_rows(&fx["input"]["sorted_mst"]);
        let want = mat_f64(&fx["output"]["linkage"]);

        let got = mst_to_linkage_tree(&sorted_mst);
        assert_eq!(got.len(), want.len(), "{case}: row count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g[0], w[0], "{case}: row {i} col0");
            assert_eq!(g[1], w[1], "{case}: row {i} col1");
            assert!((g[2] - w[2]).abs() <= 1e-12 * w[2].abs().max(1.0), "{case}: row {i} delta");
            assert_eq!(g[3], w[3], "{case}: row {i} size");
        }
    }
}

// ----------------------------------------------------------------- S6 + S7

fn load_linkage(fx: &Value) -> Vec<[f64; 4]> {
    mat_f64(&fx["input"]["linkage"])
        .into_iter()
        .map(|r| [r[0], r[1], r[2], r[3]])
        .collect()
}

#[test]
fn condensed_tree_and_extraction_match() {
    for case in CASES {
        for mcs in [3i64, 10] {
            let fx = load(&format!("{case}__condensed_mcs{mcs}"));
            let linkage = load_linkage(&fx);
            let n_samples = fx["input"]["n_samples"].as_u64().unwrap() as usize;
            let out = &fx["output"];

            let tree = condense_tree(&linkage, mcs);
            assert_eq!(tree.parent, i64s(&out["parent"]), "{case}/mcs{mcs}: parent");
            assert_eq!(tree.child, i64s(&out["child"]), "{case}/mcs{mcs}: child");
            assert_eq!(tree.child_size, i64s(&out["child_size"]), "{case}/mcs{mcs}: child_size");
            assert_close(&tree.lambda_val, &f32s(&out["lambda_val"]), 1e-5, "lambda_val");

            let leaves = extract_leaves(&tree);
            assert_eq!(leaves, i64s(&out["leaves"]), "{case}/mcs{mcs}: leaves");

            let labels = get_cluster_label_vector(&tree, &leaves, 0.0, n_samples);
            assert_eq!(labels, i64s(&out["labels"]), "{case}/mcs{mcs}: labels");

            let strengths = get_point_membership_strength_vector(&tree, &leaves, &labels);
            assert_close(&strengths, &f32s(&out["strengths"]), 1e-5, "strengths");
        }
    }
}

// ---------------------------------------------------------------------- S8

#[test]
fn persistence_chain_matches() {
    for case in CASES {
        let name = format!("{case}__persistence");
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures")
            .join(format!("{name}.json"));
        if !path.exists() {
            continue; // degenerate cluster tree: reference skipped the chain
        }
        let fx = load(&name);
        let linkage = load_linkage(&fx);
        let n_samples = fx["input"]["n_samples"].as_u64().unwrap() as i64;
        let mcs = fx["input"]["min_cluster_size"].as_i64().unwrap();
        let out = &fx["output"];

        let tree = condense_tree(&linkage, mcs);
        let mask: Vec<bool> = tree.child.iter().map(|&c| c >= n_samples).collect();
        let cluster_tree: CondensedTree = tree.masked(&mask);

        let barcode = min_cluster_size_barcode(&cluster_tree, n_samples, mcs);
        assert_close(&barcode.births, &f32s(&out["births"]), 1e-5, "births");
        assert_close(&barcode.deaths, &f32s(&out["deaths"]), 1e-5, "deaths");
        assert_close(&barcode.lambda_deaths, &f32s(&out["lambda_deaths"]), 1e-5, "lambda_deaths");
        let want_parents: Vec<i32> = i64s(&out["parents"]).iter().map(|&x| x as i32).collect();
        assert_eq!(barcode.parents, want_parents, "{case}: parents");

        let (sizes, total) =
            compute_total_persistence(&barcode.births, &barcode.deaths, &barcode.lambda_deaths);
        assert_close(&sizes, &f32s(&out["sizes"]), 1e-5, "sizes");
        assert_close(&total, &f32s(&out["total_persistence"]), 1e-4, "total_persistence");

        let peaks = find_peaks(&total);
        assert_eq!(peaks, i64s(&out["peaks"]), "{case}: peaks");

        let selected = select_diverse_peaks(
            &peaks, &total, &sizes, &barcode.births, &barcode.deaths, 0.2, 9,
        );
        assert_eq!(selected, i64s(&out["selected_peaks"]), "{case}: selected peaks");
    }
}

// ---------------------------------------------------------------------- S3

#[test]
fn cluster_layers_match() {
    // Driven from the fixture's stored canonical MST, so the comparison is a
    // pure function of shared input — independent of MST tie-breaking.
    for case in CASES {
        let fx = load(&format!("{case}__cluster_layers"));
        let sorted_mst = mst_rows(&fx["input"]["sorted_mst"]);
        let n_samples = fx["input"]["n_samples"].as_u64().unwrap() as usize;
        let params = LayerParams {
            min_samples: fx["input"]["min_samples"].as_u64().unwrap() as usize,
            base_min_cluster_size: fx["input"]["base_min_cluster_size"].as_i64().unwrap(),
            min_similarity_threshold: fx["input"]["min_similarity_threshold"].as_f64().unwrap(),
            max_layers: fx["input"]["max_layers"].as_u64().unwrap() as usize,
        };

        let got = cluster_layers_from_mst(&sorted_mst, n_samples, &params);
        let want_layers: Vec<Vec<i64>> =
            fx["output"]["layers"].as_array().unwrap().iter().map(i64s).collect();
        let want_scores = f64s(&fx["output"]["persistence_scores"]);

        assert_eq!(got.layers.len(), want_layers.len(), "{case}: layer count");
        for (i, (g, w)) in got.layers.iter().zip(&want_layers).enumerate() {
            assert_eq!(g, w, "{case}: layer {i} labels");
        }
        for (i, (g, w)) in got.persistence_scores.iter().zip(&want_scores).enumerate() {
            assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "{case}: score {i}");
        }
        let want_strengths: Vec<Vec<f32>> =
            fx["output"]["strengths"].as_array().unwrap().iter().map(f32s).collect();
        for (i, (g, w)) in got.strengths.iter().zip(&want_strengths).enumerate() {
            assert_close(g, w, 1e-5, &format!("{case}: strengths layer {i}"));
        }
    }
}

// ----------------------------------------------------------------- S1 + S2

#[test]
fn fuzzy_graph_matches() {
    for case in CASES {
        for sym in ["sym", "asym"] {
            let fx = load(&format!("{case}__fuzzy_graph_{sym}"));
            let knn_inds: Vec<Vec<i64>> =
                fx["input"]["knn_inds"].as_array().unwrap().iter().map(i64s).collect();
            let knn_dists = mat_f32(&fx["input"]["knn_dists"]);
            let n_neighbors = fx["input"]["n_neighbors"].as_f64().unwrap() as f32;

            let got = neighbor_graph_matrix(n_neighbors, &knn_inds, &knn_dists, sym == "sym");

            let want_indptr: Vec<usize> =
                i64s(&fx["output"]["indptr"]).iter().map(|&x| x as usize).collect();
            let want_indices: Vec<u32> =
                i64s(&fx["output"]["indices"]).iter().map(|&x| x as u32).collect();
            let want_data = f32s(&fx["output"]["data"]);

            assert_eq!(got.indptr, want_indptr, "{case}/{sym}: indptr");
            assert_eq!(got.indices, want_indices, "{case}/{sym}: indices");
            assert_close(&got.data, &want_data, 1e-4, &format!("{case}/{sym}: data"));
        }
    }
}
