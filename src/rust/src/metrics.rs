use rayon::prelude::*;

/// Read element (i, j) of a condensed lower-triangle distance matrix.
///
/// The layout is R's `dist`: column-major lower triangle, no diagonal.
#[inline]
fn condensed_at(d: &[f64], n: usize, i: usize, j: usize) -> f64 {
    if i == j {
        return 0.0;
    }
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    d[n * lo - lo * (lo + 1) / 2 + (hi - lo - 1)]
}

/// Per-observation silhouette widths.
///
/// For each point: `a` is the mean distance to the rest of its own cluster and
/// `b` the smallest mean distance to any other cluster, giving
/// `(b - a) / max(a, b)`. A point alone in its cluster has width 0 by
/// convention, since it has no within-cluster distances to average.
///
/// `cluster` is zero-based. Returns one width per observation, plus the index
/// of each point's nearest neighbouring cluster (also zero-based, `usize::MAX`
/// where there is none).
pub fn silhouette(d: &[f64], n: usize, cluster: &[usize], k: usize) -> (Vec<f64>, Vec<usize>) {
    let mut sizes = vec![0usize; k];
    for &c in cluster {
        sizes[c] += 1;
    }

    let results: Vec<(f64, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let own = cluster[i];

            // Mean distance from i to every cluster, including its own.
            let mut totals = vec![0.0f64; k];
            for j in 0..n {
                if j != i {
                    totals[cluster[j]] += condensed_at(d, n, i, j);
                }
            }

            if sizes[own] <= 1 {
                return (0.0, usize::MAX);
            }

            let a = totals[own] / (sizes[own] - 1) as f64;

            let mut b = f64::INFINITY;
            let mut neighbour = usize::MAX;
            for c in 0..k {
                if c == own || sizes[c] == 0 {
                    continue;
                }
                let mean = totals[c] / sizes[c] as f64;
                if mean < b {
                    b = mean;
                    neighbour = c;
                }
            }

            if !b.is_finite() {
                return (0.0, usize::MAX); // only one non-empty cluster
            }

            let denom = if a > b { a } else { b };
            let width = if denom == 0.0 { 0.0 } else { (b - a) / denom };
            (width, neighbour)
        })
        .collect();

    let widths = results.iter().map(|&(w, _)| w).collect();
    let neighbours = results.iter().map(|&(_, nb)| nb).collect();
    (widths, neighbours)
}

/// Cluster centroids and sizes, from data laid out row-major as `n x p`.
fn centroids(data: &[f64], n: usize, p: usize, cluster: &[usize], k: usize) -> (Vec<f64>, Vec<usize>) {
    let mut sums = vec![0.0f64; k * p];
    let mut sizes = vec![0usize; k];

    for i in 0..n {
        let c = cluster[i];
        sizes[c] += 1;
        for f in 0..p {
            sums[c * p + f] += data[i * p + f];
        }
    }

    for c in 0..k {
        if sizes[c] > 0 {
            for f in 0..p {
                sums[c * p + f] /= sizes[c] as f64;
            }
        }
    }

    (sums, sizes)
}

/// Calinski-Harabasz index: between-cluster dispersion per degree of freedom
/// over within-cluster dispersion per degree of freedom. Higher is better.
pub fn calinski_harabasz(data: &[f64], n: usize, p: usize, cluster: &[usize], k: usize) -> f64 {
    if k < 2 || n <= k {
        return f64::NAN;
    }

    let (cent, sizes) = centroids(data, n, p, cluster, k);

    let mut grand = vec![0.0f64; p];
    for i in 0..n {
        for f in 0..p {
            grand[f] += data[i * p + f];
        }
    }
    for g in grand.iter_mut() {
        *g /= n as f64;
    }

    let mut between = 0.0f64;
    for c in 0..k {
        let mut sq = 0.0f64;
        for f in 0..p {
            let diff = cent[c * p + f] - grand[f];
            sq += diff * diff;
        }
        between += sizes[c] as f64 * sq;
    }

    let mut within = 0.0f64;
    for i in 0..n {
        let c = cluster[i];
        for f in 0..p {
            let diff = data[i * p + f] - cent[c * p + f];
            within += diff * diff;
        }
    }

    if within == 0.0 {
        return f64::INFINITY;
    }

    (between / (k - 1) as f64) / (within / (n - k) as f64)
}

/// Davies-Bouldin index: mean over clusters of the worst-case ratio of combined
/// within-cluster scatter to between-centroid distance. Lower is better.
pub fn davies_bouldin(data: &[f64], n: usize, p: usize, cluster: &[usize], k: usize) -> f64 {
    if k < 2 {
        return f64::NAN;
    }

    let (cent, sizes) = centroids(data, n, p, cluster, k);

    // Mean distance from each point to its own centroid.
    let mut scatter = vec![0.0f64; k];
    for i in 0..n {
        let c = cluster[i];
        let mut sq = 0.0f64;
        for f in 0..p {
            let diff = data[i * p + f] - cent[c * p + f];
            sq += diff * diff;
        }
        scatter[c] += sq.sqrt();
    }
    for c in 0..k {
        if sizes[c] > 0 {
            scatter[c] /= sizes[c] as f64;
        }
    }

    let mut total = 0.0f64;
    let mut counted = 0usize;

    for a in 0..k {
        if sizes[a] == 0 {
            continue;
        }
        let mut worst = 0.0f64;
        for b in 0..k {
            if a == b || sizes[b] == 0 {
                continue;
            }
            let mut sq = 0.0f64;
            for f in 0..p {
                let diff = cent[a * p + f] - cent[b * p + f];
                sq += diff * diff;
            }
            let separation = sq.sqrt();
            if separation > 0.0 {
                let ratio = (scatter[a] + scatter[b]) / separation;
                if ratio > worst {
                    worst = ratio;
                }
            }
        }
        total += worst;
        counted += 1;
    }

    if counted == 0 {
        f64::NAN
    } else {
        total / counted as f64
    }
}
