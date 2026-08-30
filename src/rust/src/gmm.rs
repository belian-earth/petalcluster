use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{GaussianMixtureModel, GmmInitMethod};
use ndarray_linfa::Array2;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// Resolve an initialisation strategy by name.
///
/// Names are validated on the R side; an unknown name here is a bug.
pub fn init_from_name(name: &str) -> GmmInitMethod {
    match name {
        "kmeans" => GmmInitMethod::KMeans,
        "random" => GmmInitMethod::Random,
        _ => panic!("Unknown initialisation method: {name}"),
    }
}

/// A fitted mixture, reduced to its parameters.
///
/// Only the parameters cross into R. Responsibilities, hard assignments and the
/// log-likelihood are all derived from these on the R side, so training and
/// prediction go through exactly one implementation.
pub struct GmmFit {
    /// Mixing proportions, one per component.
    pub weights: Vec<f64>,
    /// `k x n_features` component means.
    pub means: Array2<f64>,
    /// Covariances flattened in R's column-major order for `dim = c(k, p, p)`.
    pub covariances: Vec<f64>,
}

/// Fit a Gaussian mixture by expectation-maximisation.
pub fn fit(
    data: Array2<f64>,
    k: usize,
    init: GmmInitMethod,
    n_runs: u64,
    max_iter: u64,
    tolerance: f64,
    reg_covariance: f64,
    seed: u64,
) -> Result<GmmFit, String> {
    let rng = Xoshiro256Plus::seed_from_u64(seed);
    let dataset = DatasetBase::from(data);

    let model = GaussianMixtureModel::params_with_rng(k, rng)
        .n_runs(n_runs)
        .max_n_iterations(max_iter)
        .tolerance(tolerance)
        .reg_covariance(reg_covariance)
        .init_method(init)
        .fit(&dataset)
        .map_err(|e| e.to_string())?;

    let covariances = model.covariances();
    let n_features = covariances.shape()[1];

    // R arrays are column-major, so for dim = c(k, p, p) the first index varies
    // fastest. Emitting in this order lets R reshape without a permutation.
    let mut flat = Vec::with_capacity(k * n_features * n_features);
    for b in 0..n_features {
        for a in 0..n_features {
            for j in 0..k {
                flat.push(covariances[[j, a, b]]);
            }
        }
    }

    Ok(GmmFit {
        weights: model.weights().to_vec(),
        means: model.means().to_owned(),
        covariances: flat,
    })
}
