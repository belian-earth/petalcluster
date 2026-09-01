use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::{KMeans, KMeansInit};
use ndarray_linfa::{Array1, Array2};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Resolve an initialisation strategy by name.
///
/// Names are validated on the R side; an unknown name here is a bug.
pub fn init_from_name(name: &str) -> KMeansInit<f64> {
    match name {
        "kmeans++" => KMeansInit::KMeansPlusPlus,
        "kmeans_parallel" => KMeansInit::KMeansPara,
        "random" => KMeansInit::Random,
        _ => panic!("Unknown initialisation method: {name}"),
    }
}

/// A fitted k-means model, reduced to the pieces R needs.
pub struct KMeansFit {
    /// Zero-based cluster index per observation.
    pub assignments: Array1<usize>,
    /// `k x n_features` centroid matrix.
    pub centroids: Array2<f64>,
    /// Sum of squared distances of each point to its assigned centroid.
    pub inertia: f64,
    /// Number of observations assigned to each cluster.
    pub sizes: Vec<i32>,
}

/// Fit k-means.
///
/// `seed` makes the run reproducible: linfa takes an explicit RNG, so the same
/// seed and parameters always give the same partition.
pub fn fit(
    data: Array2<f64>,
    k: usize,
    init: KMeansInit<f64>,
    n_runs: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> Result<KMeansFit, String> {
    let rng = Xoshiro256Plus::seed_from_u64(seed);
    let dataset = DatasetBase::from(data);

    let model = KMeans::params_with_rng(k, rng)
        .n_runs(n_runs)
        .max_n_iterations(max_iter)
        .tolerance(tolerance)
        .init_method(init)
        .fit(&dataset)
        .map_err(|e| e.to_string())?;

    let assignments: Array1<usize> = model.predict(dataset.records());
    let centroids = model.centroids().to_owned();

    // Not taken from the model: linfa's `inertia()` is the *mean* squared
    // distance, measured one iteration before the final centroid update, and
    // `cluster_count()` comes from whichever run happened last rather than
    // the best one. Both are derived here from the assignments actually
    // returned, so the three components always describe the same partition.
    let mut sizes = vec![0i32; k];
    let mut inertia = 0.0f64;
    for (row, &c) in dataset.records().rows().into_iter().zip(assignments.iter()) {
        sizes[c] += 1;
        inertia += row
            .iter()
            .zip(centroids.row(c).iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>();
    }

    Ok(KMeansFit {
        assignments,
        centroids,
        inertia,
        sizes,
    })
}

/// Assign observations to their nearest centroid, by squared Euclidean distance.
///
/// Used by `predict()` on a fitted model. Kept independent of linfa so a saved
/// model is just its centroids -- there is no Rust-side state to keep alive
/// between calls.
pub fn nearest_centroid(data: &Array2<f64>, centroids: &Array2<f64>) -> Vec<i32> {
    data.rows()
        .into_iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (idx, centroid) in centroids.rows().into_iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = idx;
                }
            }
            (best + 1) as i32 // 1-indexed for R
        })
        .collect()
}
