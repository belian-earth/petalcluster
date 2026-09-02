#' Gaussian Mixture Model
#'
#' Fits a mixture of multivariate Gaussians by expectation-maximisation, via the
#' Rust \href{https://github.com/rust-ml/linfa}{linfa} toolkit.
#'
#' Unlike the other algorithms here, a GMM is generative: it gives each
#' observation a probability of belonging to each component rather than a single
#' label. `posterior` holds those responsibilities; `cluster` is their row-wise
#' maximum, provided for consistency with the rest of the package.
#'
#' Clusters are elliptical rather than spherical, so a GMM handles correlated
#' and differently-scaled features that k-means would split badly.
#'
#' # Covariance structure
#'
#' Only full covariance matrices are supported: each component gets its own
#' unconstrained matrix. The constrained families `mclust` offers (spherical,
#' diagonal, tied) are not available upstream.
#'
#' # Choosing k
#'
#' A `logLik()` method is provided, so [stats::AIC()] and [stats::BIC()] work
#' directly on a fitted model and can be compared across `k`. Unlike the
#' within-cluster sum of squares reported by [shoal_kmeans()], these penalise
#' parameter count and so have an interior optimum.
#'
#' @param x A numeric matrix or data frame. Data frames are coerced to a matrix
#'   using their numeric columns (non-numeric columns are dropped).
#' @param k Number of mixture components. Required, since there is no sensible
#'   default for the central modelling decision.
#' @param init Initialisation method: `"kmeans"` (default) or `"random"`.
#' @param n_runs Number of restarts, keeping the best fit. Default `1L`:
#'   unlike [shoal_kmeans()]'s 10, a single run is the norm for EM (sklearn does
#'   the same) because the k-means initialisation already starts close and each
#'   run is expensive. Raise it for small, multimodal problems.
#' @param max_iter Maximum EM iterations per run. Default `100L`.
#' @param tolerance Convergence threshold on the log-likelihood. Default `1e-3`.
#' @param reg_covariance Value added to the diagonal of each covariance matrix
#'   to keep it positive definite. Default `1e-6`.
#' @param seed Non-negative whole-number seed for initialisation. Stored and
#'   passed as a double, so values beyond the integer range are safe. Default `1L`.
#'
#' @returns An object of class `c("shoal_gmm", "shoal_clustering")`: a list with
#'   components `cluster`, `n_clusters`, `n_noise` (always `0`), `data`,
#'   `algorithm`, `params`, `posterior` (an `n x k` matrix of responsibilities),
#'   `weights`, `means`, `covariances` (a `k x p x p` array) and `loglik`.
#'
#' @seealso [predict.shoal_gmm()], [logLik.shoal_gmm()].
#'
#' @examples
#' fit <- shoal_gmm(as.matrix(iris[, 1:4]), k = 3L)
#' fit
#' BIC(fit)
#' head(fit$posterior)
#'
#' @export
shoal_gmm <- function(x, k, init = c("kmeans", "random"),
                      n_runs = 1L, max_iter = 100L, tolerance = 1e-3,
                      reg_covariance = 1e-6, seed = 1L) {
  rlang::check_required(k)
  x <- check_numeric_matrix(x)
  check_positive_integer(k)
  init <- rlang::arg_match(init)
  check_positive_integer(n_runs)
  check_positive_integer(max_iter)
  check_positive_number(tolerance)
  check_positive_number(reg_covariance)
  check_count(seed)

  if (k > nrow(x)) {
    cli::cli_abort("{.arg k} ({k}) cannot exceed the number of rows ({nrow(x)}).")
  }

  result <- rust_gmm(
    x, as.integer(k), init, as.integer(n_runs), as.integer(max_iter),
    as.double(tolerance), as.double(reg_covariance), as.double(seed)
  )

  p <- ncol(x)
  covariances <- array(result$covariances, dim = c(k, p, p))
  colnames(result$means) <- colnames(x)

  dens <- gmm_log_density(x, result$weights, result$means, covariances)

  new_clustering(
    cluster = dens$cluster,
    data = x,
    algorithm = "Gaussian Mixture",
    subclass = "shoal_gmm",
    params = list(
      k = as.integer(k),
      init = init,
      n_runs = as.integer(n_runs),
      # Kept as double: a whole-number seed can exceed the integer range.
      seed = as.numeric(seed)
    ),
    posterior = dens$posterior,
    weights = result$weights,
    means = result$means,
    covariances = covariances,
    loglik = dens$loglik
  )
}

#' Weighted log-density of each observation under each mixture component
#'
#' The single place the mixture density is evaluated. Fitting, prediction and
#' [logLik()] all route through it, so a fitted model and a prediction on the
#' same data cannot disagree.
#'
#' @param x Numeric matrix of observations.
#' @param weights Mixing proportions.
#' @param means `k x p` matrix of component means.
#' @param covariances `k x p x p` array of component covariances.
#'
#' @returns A list with `cluster` (the most likely component per observation),
#'   `posterior` (`n x k` responsibilities) and `loglik` (the total).
#'
#' @noRd
gmm_log_density <- function(x, weights, means, covariances) {
  n <- nrow(x)
  p <- ncol(x)
  k <- length(weights)

  log_joint <- matrix(NA_real_, nrow = n, ncol = k)
  const <- p * log(2 * pi)

  for (j in seq_len(k)) {
    sigma <- matrix(covariances[j, , ], nrow = p, ncol = p)
    ch <- tryCatch(chol(sigma), error = function(e) {
      cli::cli_abort(c(
        "Component {j} has a singular covariance matrix.",
        "i" = "Increase {.arg reg_covariance}, or fit fewer components."
      ))
    })

    # Mahalanobis distance via the Cholesky factor, avoiding an explicit inverse.
    centred <- t(x) - means[j, ]
    quad <- colSums(backsolve(ch, centred, transpose = TRUE)^2)
    log_det <- 2 * sum(log(diag(ch)))

    log_joint[, j] <- log(weights[j]) - 0.5 * (quad + log_det + const)
  }

  # Row-wise log-sum-exp, shifted by the row maximum for numerical stability.
  best <- max.col(log_joint, ties.method = "first")
  row_max <- log_joint[cbind(seq_len(n), best)]
  row_logsum <- row_max + log(rowSums(exp(log_joint - row_max)))

  list(
    cluster = best,
    posterior = exp(log_joint - row_logsum),
    loglik = sum(row_logsum)
  )
}

#' Predict Mixture Membership
#'
#' @param object A fitted [shoal_gmm()] model.
#' @param newdata A numeric matrix or data frame with the same columns as the
#'   data the model was fitted to. Omit it to return the training assignment.
#' @param type `"class"` (default) for the most likely component, or
#'   `"posterior"` for the full `n x k` matrix of responsibilities.
#' @param ... Ignored.
#'
#' @returns An integer vector of component IDs, or an `n x k` numeric matrix of
#'   posterior probabilities.
#'
#' @examples
#' fit <- shoal_gmm(as.matrix(iris[1:100, 1:4]), k = 2L)
#' predict(fit, as.matrix(iris[101:150, 1:4]))
#' head(predict(fit, type = "posterior"))
#'
#' @export
predict.shoal_gmm <- function(object, newdata = NULL, type = c("class", "posterior"), ...) {
  type <- rlang::arg_match(type)

  if (is.null(newdata)) {
    return(switch(type, class = object$cluster, posterior = object$posterior))
  }

  newdata <- check_numeric_matrix(newdata, na_action = "error")
  check_newdata_columns(newdata, object$means)

  dens <- gmm_log_density(newdata, object$weights, object$means, object$covariances)

  switch(type, class = dens$cluster, posterior = dens$posterior)
}

#' Log-Likelihood of a Fitted Mixture
#'
#' Implementing this gives [stats::AIC()] and [stats::BIC()] for free, which is
#' how the number of components is normally chosen.
#'
#' The degrees of freedom count the free parameters of a full-covariance
#' mixture: `k * p` means, `k * p * (p + 1) / 2` distinct covariance entries and
#' `k - 1` independent mixing proportions.
#'
#' @param object A fitted [shoal_gmm()] model.
#' @param ... Ignored.
#'
#' @returns An object of class `"logLik"`, with `df` and `nobs` attributes.
#'
#' @examples
#' fit <- shoal_gmm(as.matrix(iris[, 1:4]), k = 3L)
#' logLik(fit)
#' AIC(fit)
#' BIC(fit)
#'
#' @export
logLik.shoal_gmm <- function(object, ...) {
  k <- length(object$weights)
  p <- ncol(object$means)

  structure(
    object$loglik,
    df = k * p + k * p * (p + 1) / 2 + (k - 1),
    nobs = nrow(object$data),
    class = "logLik"
  )
}
