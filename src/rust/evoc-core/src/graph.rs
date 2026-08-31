//! Fuzzy graph construction from a kNN graph — ports of `smooth_knn_dist`,
//! `compute_membership_strengths` and `neighbor_graph_matrix` from
//! `graph_construction.py` (the UMAP fuzzy simplicial set machinery).
//!
//! All arithmetic is f32 to match the reference; it runs those kernels under
//! numba `fastmath`, so fixtures compare at a small tolerance rather than
//! bitwise.

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;

/// Per-point bandwidths (sigma) and connection distances (rho) via binary
/// search to hit `log2(k) * bandwidth` total membership.
pub fn smooth_knn_dist(distances: &[Vec<f32>], k: f32, n_iter: usize, bandwidth: f32) -> (Vec<f32>, Vec<f32>) {
    let n = distances.len();
    let n_cols = if n > 0 { distances[0].len() } else { 0 };
    let target = k.log2() * bandwidth;

    let mut total = 0.0f32;
    for row in distances {
        for &d in row {
            total += d;
        }
    }
    let mean_distances = total / (n * n_cols) as f32;

    let mut rho = vec![0.0f32; n];
    let mut sigma = vec![0.0f32; n];

    for i in 0..n {
        let mut lo = 0.0f32;
        let mut hi = f32::INFINITY;
        let mut mid = 1.0f32;

        let ith = &distances[i];
        if let Some(&first_nonzero) = ith.iter().find(|&&d| d > 0.0) {
            rho[i] = first_nonzero;
        }

        for _ in 0..n_iter {
            let mut psum = 0.0f32;
            for &dist in ith.iter().skip(1) {
                let d = dist - rho[i];
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi == f32::INFINITY {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }
        sigma[i] = mid;

        if rho[i] > 0.0 {
            let mean_ith = ith.iter().sum::<f32>() / n_cols as f32;
            if sigma[i] < MIN_K_DIST_SCALE * mean_ith {
                sigma[i] = MIN_K_DIST_SCALE * mean_ith;
            }
        } else if sigma[i] < MIN_K_DIST_SCALE * mean_distances {
            sigma[i] = MIN_K_DIST_SCALE * mean_distances;
        }
    }

    (sigma, rho)
}

/// A CSR matrix with sorted column indices per row, mirroring what the
/// fixtures store.
#[derive(Debug, Clone)]
pub struct Csr {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
    pub n: usize,
}

/// COO triplets -> CSR with sorted columns, dropping explicit zeros
/// (scipy's `eliminate_zeros` on the reference path).
fn coo_to_csr(n: usize, rows: &[u32], cols: &[u32], vals: &[f32]) -> Csr {
    let mut entries: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for ((&r, &c), &v) in rows.iter().zip(cols).zip(vals) {
        if v != 0.0 {
            entries[r as usize].push((c, v));
        }
    }
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for row in entries.iter_mut() {
        row.sort_by_key(|&(c, _)| c);
        for &(c, v) in row.iter() {
            indices.push(c);
            data.push(v);
        }
        indptr.push(indices.len());
    }
    Csr { indptr, indices, data, n }
}

pub(crate) fn transpose(m: &Csr) -> Csr {
    let mut rows = Vec::with_capacity(m.data.len());
    let mut cols = Vec::with_capacity(m.data.len());
    let mut vals = Vec::with_capacity(m.data.len());
    for r in 0..m.n {
        for k in m.indptr[r]..m.indptr[r + 1] {
            rows.push(m.indices[k]);
            cols.push(r as u32);
            vals.push(m.data[k]);
        }
    }
    coo_to_csr(m.n, &rows, &cols, &vals)
}

/// Fuzzy union symmetrisation: `A + A^T - A o A^T`, elementwise, computed per
/// row as a sorted merge. Zero results are eliminated, as scipy does.
fn symmetrize(a: &Csr) -> Csr {
    let t = transpose(a);
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for r in 0..a.n {
        let (mut i, ia_end) = (a.indptr[r], a.indptr[r + 1]);
        let (mut j, it_end) = (t.indptr[r], t.indptr[r + 1]);
        while i < ia_end || j < it_end {
            let ca = if i < ia_end { a.indices[i] } else { u32::MAX };
            let ct = if j < it_end { t.indices[j] } else { u32::MAX };
            let (col, va, vt) = if ca < ct {
                let v = (ca, a.data[i], 0.0);
                i += 1;
                v
            } else if ct < ca {
                let v = (ct, 0.0, t.data[j]);
                j += 1;
                v
            } else {
                let v = (ca, a.data[i], t.data[j]);
                i += 1;
                j += 1;
                v
            };
            // Same association as scipy: (A + A^T) - (A o A^T).
            let val = (va + vt) - va * vt;
            if val != 0.0 {
                indices.push(col);
                data.push(val);
            }
        }
        indptr.push(indices.len());
    }
    Csr { indptr, indices, data, n: a.n }
}

/// Port of `neighbor_graph_matrix`: kNN graph in, weighted (optionally
/// symmetrised) fuzzy graph out.
pub fn neighbor_graph_matrix(
    n_neighbors: f32,
    knn_indices: &[Vec<i64>],
    knn_dists: &[Vec<f32>],
    symmetrize_graph: bool,
) -> Csr {
    let n = knn_indices.len();
    let (sigmas, rhos) = smooth_knn_dist(knn_dists, n_neighbors, 64, 1.0);

    let mut rows = Vec::with_capacity(n * knn_indices[0].len());
    let mut cols = Vec::with_capacity(rows.capacity());
    let mut vals = Vec::with_capacity(rows.capacity());

    for i in 0..n {
        for (j, &idx) in knn_indices[i].iter().enumerate() {
            if idx == -1 {
                continue;
            }
            let val = if idx == i as i64 {
                0.0
            } else if knn_dists[i][j] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-((knn_dists[i][j] - rhos[i]) / sigmas[i])).exp()
            };
            rows.push(i as u32);
            cols.push(idx as u32);
            vals.push(val);
        }
    }

    let csr = coo_to_csr(n, &rows, &cols, &vals);
    if symmetrize_graph {
        symmetrize(&csr)
    } else {
        csr
    }
}
