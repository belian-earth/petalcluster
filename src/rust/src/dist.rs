use rayon::prelude::*;

/// Pairwise distance metrics.
///
/// The metrics R also provides follow `stats::dist()` exactly, including its
/// treatment of degenerate terms, so the two are numerically comparable.
/// `Cosine` matches `petal_neighbors::distance::Cosine`, so it agrees with the
/// `metric = "cosine"` option on the density-based algorithms.
#[derive(Clone, Copy, Debug)]
pub enum Metric {
    Euclidean,
    Maximum,
    Manhattan,
    Canberra,
    Binary,
    Minkowski(f64),
    Cosine,
    Correlation,
}

impl Metric {
    /// Resolve a metric by name. `p` is only consulted for Minkowski.
    ///
    /// Names are validated on the R side; an unknown name here is a bug.
    pub fn from_name(name: &str, p: f64) -> Self {
        match name {
            "euclidean" => Metric::Euclidean,
            "maximum" => Metric::Maximum,
            "manhattan" => Metric::Manhattan,
            "canberra" => Metric::Canberra,
            "binary" => Metric::Binary,
            "minkowski" => Metric::Minkowski(p),
            "cosine" => Metric::Cosine,
            "correlation" => Metric::Correlation,
            _ => panic!("Unknown metric: {name}"),
        }
    }

    #[inline(always)]
    pub(crate) fn compute(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Metric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt(),

            Metric::Maximum => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0_f64, f64::max),

            Metric::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .sum(),

            // Follows R's R_canberra: the denominator is |x| + |y|, as the C code
            // computes it (the R documentation writes |x + y|; the two differ on
            // signed data), so every term lies in [0, 1].
            // Terms whose numerator and denominator are both zero are dropped,
            // and the total is rescaled by the number of terms that survived.
            Metric::Canberra => {
                let nc = a.len();
                let mut count = 0usize;
                let mut dist = 0.0_f64;

                for (&x, &y) in a.iter().zip(b.iter()) {
                    let sum = x.abs() + y.abs();
                    let diff = (x - y).abs();
                    if sum > f64::MIN_POSITIVE || diff > f64::MIN_POSITIVE {
                        let dev = diff / sum;
                        if !dev.is_nan() {
                            dist += dev;
                            count += 1;
                        } else if !diff.is_finite() && diff == sum {
                            // R's limit convention: Inf/Inf counts as 1
                            // (the `(dev = 1., TRUE)` reassignment in R_canberra).
                            dist += 1.0;
                            count += 1;
                        }
                    }
                }

                if count == 0 {
                    return f64::NAN;
                }
                if count != nc {
                    dist /= count as f64 / nc as f64;
                }
                dist
            }

            // Follows R's R_bin: the proportion of positions where exactly one
            // value is non-zero, among positions where at least one is.
            Metric::Binary => {
                let mut count = 0usize;
                let mut dist = 0usize;

                for (&x, &y) in a.iter().zip(b.iter()) {
                    let on1 = x != 0.0;
                    let on2 = y != 0.0;
                    if on1 || on2 {
                        count += 1;
                        if !(on1 && on2) {
                            dist += 1;
                        }
                    }
                }

                if count == 0 {
                    0.0
                } else {
                    dist as f64 / count as f64
                }
            }

            Metric::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p),

            Metric::Cosine => {
                let mut dot = 0.0_f64;
                let mut n1 = 0.0_f64;
                let mut n2 = 0.0_f64;
                for (&x, &y) in a.iter().zip(b.iter()) {
                    dot += x * y;
                    n1 += x * x;
                    n2 += y * y;
                }
                1.0 - dot / (n1.sqrt() * n2.sqrt())
            }

            // 1 - Pearson correlation across the two observations' features.
            Metric::Correlation => {
                let n = a.len() as f64;
                let ma = a.iter().sum::<f64>() / n;
                let mb = b.iter().sum::<f64>() / n;

                let mut sab = 0.0_f64;
                let mut saa = 0.0_f64;
                let mut sbb = 0.0_f64;
                for (&x, &y) in a.iter().zip(b.iter()) {
                    let dx = x - ma;
                    let dy = y - mb;
                    sab += dx * dy;
                    saa += dx * dx;
                    sbb += dy * dy;
                }

                let denom = (saa * sbb).sqrt();
                if denom == 0.0 {
                    f64::NAN
                } else {
                    1.0 - sab / denom
                }
            }
        }
    }
}

/// Fill `out` with the condensed lower triangle of the pairwise distance
/// matrix of `data` (`n x p`, row-major).
///
/// The layout is column-major without the diagonal, which is what R's `dist`
/// class uses: for `n = 4` the order is (1,2) (1,3) (1,4) (2,3) (2,4) (3,4),
/// so row `i` owns the contiguous run of `n - 1 - i` entries for `j > i`.
/// kodama uses the same convention, so the output can be handed to it
/// directly.
///
/// `out` is the buffer that will be used directly afterwards, an R vector
/// for `shoal_dist()` or kodama's scratch for `shoal_hclust()` on raw data,
/// so nothing is copied: the run for each row is split off as its own
/// mutable slice and the rows are filled in parallel. `out.len()` must be
/// `n * (n - 1) / 2`.
///
/// Returns whether every distance is finite. Each row is checked as soon as
/// it is written, while it is still in cache; a separate pass over the
/// result on the R side costs more than the distances do on narrow data.
pub fn condensed_into<T>(data: &[f64], n: usize, p: usize, metric: Metric, out: &mut [T]) -> bool
where
    T: From<f64> + Send,
{
    debug_assert_eq!(data.len(), n * p);
    debug_assert_eq!(out.len(), n * (n - 1) / 2);
    if n < 2 {
        return true;
    }

    // Disjoint per-row slices of the output, in row order.
    let mut rows: Vec<&mut [T]> = Vec::with_capacity(n - 1);
    let mut rest = out;
    for i in 0..n - 1 {
        let (head, tail) = rest.split_at_mut(n - 1 - i);
        rows.push(head);
        rest = tail;
    }

    // Dispatch on the metric once, so each variant gets its own
    // monomorphised inner loop instead of a match per pair.
    match metric {
        Metric::Euclidean => fill(data, n, p, rows, |a, b| Metric::Euclidean.compute(a, b)),
        Metric::Maximum => fill(data, n, p, rows, |a, b| Metric::Maximum.compute(a, b)),
        Metric::Manhattan => fill(data, n, p, rows, |a, b| Metric::Manhattan.compute(a, b)),
        Metric::Canberra => fill(data, n, p, rows, |a, b| Metric::Canberra.compute(a, b)),
        Metric::Binary => fill(data, n, p, rows, |a, b| Metric::Binary.compute(a, b)),
        Metric::Minkowski(q) => fill(data, n, p, rows, |a, b| Metric::Minkowski(q).compute(a, b)),
        Metric::Cosine => fill(data, n, p, rows, |a, b| Metric::Cosine.compute(a, b)),
        Metric::Correlation => fill(data, n, p, rows, |a, b| Metric::Correlation.compute(a, b)),
    }
}

/// Fill each row's run of the condensed matrix, rows in parallel, and report
/// whether every value written is finite.
#[inline(always)]
fn fill<T, F>(data: &[f64], n: usize, p: usize, rows: Vec<&mut [T]>, f: F) -> bool
where
    T: From<f64> + Send,
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    rows.into_par_iter()
        .enumerate()
        .map(|(i, row)| {
            let a = &data[i * p..(i + 1) * p];
            // Iterators rather than indexing: no bounds check per pair.
            let others = data[(i + 1) * p..].chunks_exact(p);
            let mut finite = true;
            for (slot, b) in row.iter_mut().zip(others) {
                let v = f(a, b);
                finite &= v.is_finite();
                *slot = T::from(v);
            }
            finite
        })
        .reduce(|| true, |x, y| x && y)
}
