use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

/// Pairwise distance metrics.
///
/// The metrics R also provides follow `stats::dist()` exactly, including its
/// treatment of degenerate terms, so the two are numerically comparable.
/// `Cosine` matches `petal_neighbors::distance::Cosine`, so it agrees with the
/// `metric = "cosine"` option on the density-based algorithms.
#[derive(Clone, Copy)]
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

    fn compute(self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
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

            // Follows R's R_canberra: terms whose numerator and denominator are
            // both zero are dropped, and the total is rescaled by the number of
            // terms that survived.
            Metric::Canberra => {
                let nc = a.len();
                let mut count = 0usize;
                let mut dist = 0.0_f64;

                for (&x, &y) in a.iter().zip(b.iter()) {
                    let sum = (x + y).abs();
                    let diff = (x - y).abs();
                    if sum > f64::MIN_POSITIVE || diff > f64::MIN_POSITIVE {
                        let dev = diff / sum;
                        if !dev.is_nan() || (!diff.is_finite() && diff == sum) {
                            dist += dev;
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

/// Compute the condensed lower triangle of the pairwise distance matrix.
///
/// The result is laid out column-major and excludes the diagonal, which is the
/// layout R's `dist` class uses: for `n = 4` the order is
/// (1,2) (1,3) (1,4) (2,3) (2,4) (3,4). kodama uses this same convention, so
/// the output can be handed to it directly.
pub fn condensed(data: &Array2<f64>, metric: Metric) -> Vec<f64> {
    let n = data.nrows();
    if n < 2 {
        return Vec::new();
    }

    // Row `i` contributes distances to every j > i, in ascending j. Emitting
    // rows in order and flattening reproduces R's layout exactly.
    let rows: Vec<Vec<f64>> = (0..n - 1)
        .into_par_iter()
        .map(|i| {
            let a = data.row(i);
            (i + 1..n).map(|j| metric.compute(&a, &data.row(j))).collect()
        })
        .collect();

    rows.into_iter().flatten().collect()
}
